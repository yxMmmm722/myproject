from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import copy
import json
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _raft_model(self):
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model

    @staticmethod
    def _safe_corr_from_sums(sum_x, sum_y, sum_x2, sum_y2, sum_xy, count):
        count = np.asarray(count, dtype=np.float64)
        safe_count = np.maximum(count, 1.0)
        mean_x = np.asarray(sum_x, dtype=np.float64) / safe_count
        mean_y = np.asarray(sum_y, dtype=np.float64) / safe_count
        ex2 = np.asarray(sum_x2, dtype=np.float64) / safe_count
        ey2 = np.asarray(sum_y2, dtype=np.float64) / safe_count
        exy = np.asarray(sum_xy, dtype=np.float64) / safe_count
        var_x = np.maximum(ex2 - mean_x ** 2, 0.0)
        var_y = np.maximum(ey2 - mean_y ** 2, 0.0)
        cov_xy = exy - mean_x * mean_y
        denom = np.sqrt(var_x * var_y)
        corr = np.zeros_like(cov_xy, dtype=np.float64)
        valid = denom > 1e-12
        corr[valid] = cov_xy[valid] / denom[valid]
        return corr

    @staticmethod
    def _build_branch_rows(channel_names, mse_arr, mae_arr):
        rows = []
        for idx, name in enumerate(channel_names):
            rows.append(
                {
                    'channel_idx': int(idx),
                    'channel_name': str(name),
                    'mse': float(mse_arr[idx]),
                    'mae': float(mae_arr[idx]),
                }
            )
        return rows

    @staticmethod
    def _decompose_future_mg(y, period_num):
        mg = []
        for g in period_num:
            cur = y.unfold(dimension=1, size=g, step=g).mean(dim=-1)
            cur = cur.repeat_interleave(repeats=g, dim=1)
            cur = cur - cur[:, -1:, :]
            mg.append(cur)
        return torch.stack(mg, dim=0)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        model.prepare_dataset(train_data, vali_data, test_data)
        
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, split_mode='valid'):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.model == 'RAFT':
                    outputs = self.model(batch_x, index, mode=split_mode)
                else:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
#         early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        best_valid_loss = float('inf')
        best_model = None
            
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.model == 'RAFT':
                    outputs = self.model(batch_x, index, mode='train')
                else:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, split_mode='valid')
            test_loss = self.vali(test_data, test_loader, criterion, split_mode='test')

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            # We do not use early stopping
            
            if vali_loss < best_valid_loss:
                best_model = copy.deepcopy(self.model)
                best_valid_loss = vali_loss
                
        best_model_path = path + '/' + 'checkpoint.pth'
        torch.save(best_model.state_dict(), best_model_path)
#         self.model.load_state_dict(torch.load(best_model_path))

#         return self.model
        return best_model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        base_sq_sum = None
        base_abs_sum = None
        base_elem_count = 0.0
        retrieval_sq_sum = None
        retrieval_abs_sum = None
        retrieval_elem_count = 0.0
        final_sq_sum = None
        final_abs_sum = None
        final_elem_count = 0.0
        period_sq_sum = None
        period_abs_sum = None
        period_elem_count = 0.0
        wave_sq_sum = 0.0
        wave_abs_sum = 0.0
        wave_elem_count = 0.0
        wave_sq_by_g = None
        wave_abs_by_g = None
        wave_elem_by_g = None
        wave_sq_by_c = None
        wave_abs_by_c = None
        wave_elem_by_c = None
        base_err_sum = None
        retrieval_err_sum = None
        base_err_sq_sum = None
        retrieval_err_sq_sum = None
        base_retrieval_err_cross_sum = None
        residual_sum = None
        residual_sq_sum = None
        retrieval_delta_sum = None
        retrieval_delta_sq_sum = None
        residual_delta_cross_sum = None
        complement_elem_count = 0.0
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        raft_model = self._raft_model() if self.args.model == 'RAFT' else None
        retrieval_future_quality = None
        if raft_model is not None and hasattr(raft_model, 'get_retrieval_future_quality'):
            retrieval_future_quality = raft_model.get_retrieval_future_quality()
        collect_wave_quality_fallback = bool(
            raft_model is not None
            and retrieval_future_quality is None
            and not bool(getattr(self.args, 'meta_only_retrieval', False))
        )
        self.model.eval()
        with torch.no_grad():
            for i, (index, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.model == 'RAFT':
                    outputs = raft_model(batch_x, index, mode='test')
                else:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                batch_y_scaled_cpu = batch_y.detach().cpu()
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y
                if raft_model is not None:
                    retrieval_bank = getattr(raft_model, 'retrieval_dict', {}).get('test', None)
                    period_num = getattr(raft_model, 'period_num', None)
                    if collect_wave_quality_fallback and retrieval_bank is not None and period_num is not None:
                        retrieval_index = index.long().to(retrieval_bank.device)
                        wave_pred = retrieval_bank[:, retrieval_index].float()  # G, B, P, C
                        true_future_t = batch_y_scaled_cpu.float()
                        true_mg = self._decompose_future_mg(true_future_t, period_num)[:, :, :, f_dim:]
                        wave_err = wave_pred[:, :, :, f_dim:].cpu().numpy() - true_mg.cpu().numpy()

                        wave_sq_sum += float((wave_err ** 2).sum())
                        wave_abs_sum += float(np.abs(wave_err).sum())
                        wave_elem_count += float(wave_err.size)

                        gsz_dbg = wave_err.shape[0]
                        ch_dbg = wave_err.shape[3]
                        if wave_sq_by_g is None:
                            wave_sq_by_g = np.zeros((gsz_dbg,), dtype=np.float64)
                            wave_abs_by_g = np.zeros((gsz_dbg,), dtype=np.float64)
                            wave_elem_by_g = np.zeros((gsz_dbg,), dtype=np.float64)
                            wave_sq_by_c = np.zeros((ch_dbg,), dtype=np.float64)
                            wave_abs_by_c = np.zeros((ch_dbg,), dtype=np.float64)
                            wave_elem_by_c = np.zeros((ch_dbg,), dtype=np.float64)

                        wave_sq_by_g += (wave_err ** 2).sum(axis=(1, 2, 3))
                        wave_abs_by_g += np.abs(wave_err).sum(axis=(1, 2, 3))
                        wave_elem_by_g += np.full((gsz_dbg,), wave_err.shape[1] * wave_err.shape[2] * wave_err.shape[3], dtype=np.float64)
                        wave_sq_by_c += (wave_err ** 2).sum(axis=(0, 1, 2))
                        wave_abs_by_c += np.abs(wave_err).sum(axis=(0, 1, 2))
                        wave_elem_by_c += np.full((ch_dbg,), wave_err.shape[0] * wave_err.shape[1] * wave_err.shape[2], dtype=np.float64)

                    if hasattr(raft_model, 'get_last_debug'):
                        debug = raft_model.get_last_debug()
                    else:
                        debug = None

                    if isinstance(debug, dict):
                        pred_stack = debug.get('pred_stack', None)
                        retrieval_agg = debug.get('retrieval_agg', None)
                        base_pred = debug.get('base_pred', None)
                        if pred_stack is not None and retrieval_agg is not None and base_pred is not None:
                            pred_stack_np = pred_stack.detach().cpu().numpy()
                            retrieval_agg_np = retrieval_agg.detach().cpu().numpy()
                            base_pred_np = base_pred.detach().cpu().numpy()

                            if test_data.scale and self.args.inverse:
                                bsz_dbg, gsz_dbg, pred_len_dbg, channels_dbg = pred_stack_np.shape
                                pred_stack_np = test_data.inverse_transform(
                                    pred_stack_np.transpose(1, 0, 2, 3).reshape(gsz_dbg * bsz_dbg * pred_len_dbg, channels_dbg)
                                ).reshape(gsz_dbg, bsz_dbg, pred_len_dbg, channels_dbg).transpose(1, 0, 2, 3)

                                retrieval_shape = retrieval_agg_np.shape
                                retrieval_agg_np = test_data.inverse_transform(
                                    retrieval_agg_np.reshape(retrieval_shape[0] * retrieval_shape[1], retrieval_shape[2])
                                ).reshape(retrieval_shape)

                                base_shape = base_pred_np.shape
                                base_pred_np = test_data.inverse_transform(
                                    base_pred_np.reshape(base_shape[0] * base_shape[1], base_shape[2])
                                ).reshape(base_shape)

                            pred_stack_np = pred_stack_np[:, :, :, f_dim:]
                            retrieval_agg_np = retrieval_agg_np[:, :, f_dim:]
                            base_pred_np = base_pred_np[:, :, f_dim:]

                            channels_dbg = base_pred_np.shape[-1]
                            if base_sq_sum is None:
                                base_sq_sum = np.zeros((channels_dbg,), dtype=np.float64)
                                base_abs_sum = np.zeros((channels_dbg,), dtype=np.float64)
                                retrieval_sq_sum = np.zeros((channels_dbg,), dtype=np.float64)
                                retrieval_abs_sum = np.zeros((channels_dbg,), dtype=np.float64)
                                period_sq_sum = np.zeros((channels_dbg, pred_stack_np.shape[1]), dtype=np.float64)
                                period_abs_sum = np.zeros((channels_dbg, pred_stack_np.shape[1]), dtype=np.float64)
                                final_sq_sum = np.zeros((channels_dbg,), dtype=np.float64)
                                final_abs_sum = np.zeros((channels_dbg,), dtype=np.float64)
                                base_err_sum = np.zeros((channels_dbg,), dtype=np.float64)
                                retrieval_err_sum = np.zeros((channels_dbg,), dtype=np.float64)
                                base_err_sq_sum = np.zeros((channels_dbg,), dtype=np.float64)
                                retrieval_err_sq_sum = np.zeros((channels_dbg,), dtype=np.float64)
                                base_retrieval_err_cross_sum = np.zeros((channels_dbg,), dtype=np.float64)
                                residual_sum = np.zeros((channels_dbg,), dtype=np.float64)
                                residual_sq_sum = np.zeros((channels_dbg,), dtype=np.float64)
                                retrieval_delta_sum = np.zeros((channels_dbg,), dtype=np.float64)
                                retrieval_delta_sq_sum = np.zeros((channels_dbg,), dtype=np.float64)
                                residual_delta_cross_sum = np.zeros((channels_dbg,), dtype=np.float64)

                            period_err = pred_stack_np - true[:, None, :, :]
                            period_sq_sum += ((period_err ** 2).sum(axis=(0, 2))).transpose(1, 0)
                            period_abs_sum += (np.abs(period_err).sum(axis=(0, 2))).transpose(1, 0)
                            period_elem_count += float(period_err.shape[0] * period_err.shape[2])

                            base_err = base_pred_np - true
                            retrieval_err = retrieval_agg_np - true
                            base_sq_sum += (base_err ** 2).sum(axis=(0, 1))
                            base_abs_sum += np.abs(base_err).sum(axis=(0, 1))
                            base_elem_count += float(base_err.shape[0] * base_err.shape[1])
                            retrieval_sq_sum += (retrieval_err ** 2).sum(axis=(0, 1))
                            retrieval_abs_sum += np.abs(retrieval_err).sum(axis=(0, 1))
                            retrieval_elem_count += float(retrieval_err.shape[0] * retrieval_err.shape[1])

                            base_err_sum += base_err.sum(axis=(0, 1))
                            retrieval_err_sum += retrieval_err.sum(axis=(0, 1))
                            base_err_sq_sum += (base_err ** 2).sum(axis=(0, 1))
                            retrieval_err_sq_sum += (retrieval_err ** 2).sum(axis=(0, 1))
                            base_retrieval_err_cross_sum += (base_err * retrieval_err).sum(axis=(0, 1))

                            residual = true - base_pred_np
                            retrieval_delta = retrieval_agg_np - base_pred_np
                            residual_sum += residual.sum(axis=(0, 1))
                            residual_sq_sum += (residual ** 2).sum(axis=(0, 1))
                            retrieval_delta_sum += retrieval_delta.sum(axis=(0, 1))
                            retrieval_delta_sq_sum += (retrieval_delta ** 2).sum(axis=(0, 1))
                            residual_delta_cross_sum += (residual * retrieval_delta).sum(axis=(0, 1))
                            complement_elem_count += float(retrieval_err.shape[0] * retrieval_err.shape[1])

                final_err = pred - true
                if final_sq_sum is None:
                    final_sq_sum = np.zeros((final_err.shape[-1],), dtype=np.float64)
                    final_abs_sum = np.zeros((final_err.shape[-1],), dtype=np.float64)
                final_sq_sum += (final_err ** 2).sum(axis=(0, 1))
                final_abs_sum += np.abs(final_err).sum(axis=(0, 1))
                final_elem_count += float(final_err.shape[0] * final_err.shape[1])

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        feature_names = getattr(test_data, 'feature_names', None)
        if feature_names is None:
            channel_names = [f'channel_{i}' for i in range(preds.shape[-1])]
        else:
            channel_names = [str(name) for name in feature_names[f_dim:]]
        
        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1,1)
                y = trues[i].reshape(-1,1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = -999
            

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))

        final_channel_mse = final_sq_sum / max(final_elem_count, 1.0)
        final_channel_mae = final_abs_sum / max(final_elem_count, 1.0)

        summary_json = {
            'setting': str(setting),
            'retrieval_mode_active': 'meta' if bool(getattr(self.args, 'meta_only_retrieval', False)) else 'wave',
            'topm': int(getattr(self.args, 'topm', 0)),
            'retrieval_temperature': float(getattr(self.args, 'retrieval_temperature', 0.1)),
            'channels': channel_names,
            'final_metrics': {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'mspe': float(mspe),
                'dtw': float(dtw),
                'per_channel': self._build_branch_rows(channel_names, final_channel_mse, final_channel_mae),
            },
        }

        base_channel_mse = None
        retrieval_channel_mse = None
        if base_sq_sum is not None and base_elem_count > 0:
            base_channel_mse = base_sq_sum / base_elem_count
            base_channel_mae = base_abs_sum / base_elem_count
            summary_json['base_branch'] = {
                'mse_overall': float(base_channel_mse.mean()),
                'mae_overall': float(base_channel_mae.mean()),
                'per_channel': self._build_branch_rows(channel_names, base_channel_mse, base_channel_mae),
            }

        if retrieval_sq_sum is not None and retrieval_elem_count > 0:
            retrieval_channel_mse = retrieval_sq_sum / retrieval_elem_count
            retrieval_channel_mae = retrieval_abs_sum / retrieval_elem_count
            summary_json['retrieval_agg_branch'] = {
                'mse_overall': float(retrieval_channel_mse.mean()),
                'mae_overall': float(retrieval_channel_mae.mean()),
                'per_channel': self._build_branch_rows(channel_names, retrieval_channel_mse, retrieval_channel_mae),
            }

        if retrieval_future_quality is not None:
            rfq = copy.deepcopy(retrieval_future_quality)
            if 'per_channel' in rfq and len(rfq['per_channel']) == len(channel_names):
                for idx, name in enumerate(channel_names):
                    rfq['per_channel'][idx]['channel_name'] = str(name)
            summary_json['retrieval_future_quality'] = rfq
        elif wave_elem_count > 0:
            wave_mse = wave_sq_sum / wave_elem_count
            wave_mae = wave_abs_sum / wave_elem_count
            per_period = []
            if wave_sq_by_g is not None:
                for g_i, g in enumerate(getattr(raft_model, 'period_num', [])):
                    denom = max(wave_elem_by_g[g_i], 1.0)
                    per_period.append(
                        {
                            'period_idx': int(g_i),
                            'period_g': int(g),
                            'wave_mse': float(wave_sq_by_g[g_i] / denom),
                            'wave_mae': float(wave_abs_by_g[g_i] / denom),
                        }
                    )
            per_channel = []
            if wave_sq_by_c is not None:
                for c_i, name in enumerate(channel_names):
                    denom = max(wave_elem_by_c[c_i], 1.0)
                    per_channel.append(
                        {
                            'channel_idx': int(c_i),
                            'channel_name': str(name),
                            'wave_mse': float(wave_sq_by_c[c_i] / denom),
                            'wave_mae': float(wave_abs_by_c[c_i] / denom),
                        }
                    )
            summary_json['retrieval_future_quality'] = {
                'split': 'eval',
                'wave_mse': float(wave_mse),
                'wave_mae': float(wave_mae),
                'per_period': per_period,
                'per_channel': per_channel,
            }

        if base_channel_mse is not None or retrieval_channel_mse is not None:
            fusion_effect = {'overall': {}, 'per_channel': []}
            if base_channel_mse is not None:
                gain_over_base = base_channel_mse - final_channel_mse
                fusion_effect['overall']['gain_over_base_mse'] = float(base_channel_mse.mean() - mse)
            else:
                gain_over_base = None
            if retrieval_channel_mse is not None:
                gain_over_retrieval = retrieval_channel_mse - final_channel_mse
                fusion_effect['overall']['gain_over_retrieval_mse'] = float(retrieval_channel_mse.mean() - mse)
            else:
                gain_over_retrieval = None
            if base_channel_mse is not None and retrieval_channel_mse is not None:
                best_component_channel_mse = np.minimum(base_channel_mse, retrieval_channel_mse)
                fusion_effect['overall']['gain_over_best_component_mse'] = float(best_component_channel_mse.mean() - mse)
            else:
                best_component_channel_mse = None

            for idx, name in enumerate(channel_names):
                item = {
                    'channel_idx': int(idx),
                    'channel_name': str(name),
                    'final_mse': float(final_channel_mse[idx]),
                    'final_mae': float(final_channel_mae[idx]),
                }
                if base_channel_mse is not None:
                    item['base_mse'] = float(base_channel_mse[idx])
                    item['gain_over_base_mse'] = float(gain_over_base[idx])
                if retrieval_channel_mse is not None:
                    item['retrieval_agg_mse'] = float(retrieval_channel_mse[idx])
                    item['gain_over_retrieval_mse'] = float(gain_over_retrieval[idx])
                if best_component_channel_mse is not None:
                    item['gain_over_best_component_mse'] = float(best_component_channel_mse[idx] - final_channel_mse[idx])
                fusion_effect['per_channel'].append(item)
            summary_json['fusion_effect'] = fusion_effect

        if period_sq_sum is not None and period_elem_count > 0:
            period_num = [int(g) for g in getattr(raft_model, 'period_num', [])]
            per_period_channel_mse = period_sq_sum / period_elem_count
            per_period_channel_mae = period_abs_sum / period_elem_count
            best_period_idx = np.argmin(per_period_channel_mse, axis=1)
            best_period_mse = per_period_channel_mse[np.arange(len(channel_names)), best_period_idx]
            scale_diag = {
                'period_num': period_num,
                'channels': channel_names,
                'per_period_channel_mse': per_period_channel_mse.tolist(),
                'per_period_channel_mae': per_period_channel_mae.tolist(),
                'best_period_idx': best_period_idx.astype(int).tolist(),
                'best_period_g': [int(period_num[idx]) for idx in best_period_idx],
                'best_period_mse': best_period_mse.tolist(),
            }
            if retrieval_channel_mse is not None:
                scale_diag['retrieval_agg_channel_mse'] = retrieval_channel_mse.tolist()
                scale_diag['retrieval_agg_channel_mae'] = retrieval_channel_mae.tolist()
                scale_diag['retrieval_agg_mse_overall'] = float(retrieval_channel_mse.mean())
                scale_diag['retrieval_agg_mae_overall'] = float(retrieval_channel_mae.mean())
            scale_diag['final_channel_mse'] = final_channel_mse.tolist()
            scale_diag['final_channel_mae'] = final_channel_mae.tolist()
            scale_diag['final_mse_overall'] = float(final_channel_mse.mean())
            scale_diag['final_mae_overall'] = float(final_channel_mae.mean())
            if base_channel_mse is not None:
                scale_diag['base_channel_mse'] = base_channel_mse.tolist()
                scale_diag['base_channel_mae'] = base_channel_mae.tolist()
                scale_diag['base_mse_overall'] = float(base_channel_mse.mean())
                scale_diag['base_mae_overall'] = float(base_channel_mae.mean())
            summary_json['scale_diagnostics'] = scale_diag

        if complement_elem_count > 0:
            per_channel_count = np.full((len(channel_names),), complement_elem_count, dtype=np.float64)
            err_corr_by_channel = self._safe_corr_from_sums(
                base_err_sum,
                retrieval_err_sum,
                base_err_sq_sum,
                retrieval_err_sq_sum,
                base_retrieval_err_cross_sum,
                per_channel_count,
            )
            correction_align_by_channel = self._safe_corr_from_sums(
                residual_sum,
                retrieval_delta_sum,
                residual_sq_sum,
                retrieval_delta_sq_sum,
                residual_delta_cross_sum,
                per_channel_count,
            )
            overall_count = complement_elem_count * len(channel_names)
            err_corr_overall = self._safe_corr_from_sums(
                np.array([base_err_sum.sum()]),
                np.array([retrieval_err_sum.sum()]),
                np.array([base_err_sq_sum.sum()]),
                np.array([retrieval_err_sq_sum.sum()]),
                np.array([base_retrieval_err_cross_sum.sum()]),
                np.array([overall_count]),
            )[0]
            correction_align_overall = self._safe_corr_from_sums(
                np.array([residual_sum.sum()]),
                np.array([retrieval_delta_sum.sum()]),
                np.array([residual_sq_sum.sum()]),
                np.array([retrieval_delta_sq_sum.sum()]),
                np.array([residual_delta_cross_sum.sum()]),
                np.array([overall_count]),
            )[0]
            summary_json['complementarity'] = {
                'overall': {
                    'base_retrieval_error_corr': float(err_corr_overall),
                    'correction_alignment_corr': float(correction_align_overall),
                },
                'per_channel': [
                    {
                        'channel_idx': int(idx),
                        'channel_name': str(name),
                        'base_retrieval_error_corr': float(err_corr_by_channel[idx]),
                        'correction_alignment_corr': float(correction_align_by_channel[idx]),
                    }
                    for idx, name in enumerate(channel_names)
                ],
            }

        with open(os.path.join(folder_path, 'retrieval_effect_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary_json, f, indent=2, ensure_ascii=False)

        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
