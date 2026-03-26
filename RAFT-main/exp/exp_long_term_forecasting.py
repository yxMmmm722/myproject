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
import json
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    @staticmethod
    def _unpack_batch(batch):
        index, batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:5]
        meta_data = batch[5] if len(batch) > 5 else None
        return index, batch_x, batch_y, batch_x_mark, batch_y_mark, meta_data

    def _raft_model(self):
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model

    @staticmethod
    def _save_retrieval_compare_viz(cmp_stats, out_dir, tag):
        if cmp_stats is None:
            return
        os.makedirs(out_dir, exist_ok=True)
        m = int(cmp_stats.get("m", 0))
        if m <= 0:
            return

        rank_ratio = np.asarray(cmp_stats.get("rank_match_ratio", [])[:m], dtype=np.float32)
        overlap_hist = np.asarray(cmp_stats.get("overlap_hist_ratio", [])[: m + 1], dtype=np.float32)

        np.save(os.path.join(out_dir, f"{tag}_rank_match_ratio.npy"), rank_ratio)
        np.save(os.path.join(out_dir, f"{tag}_overlap_hist_ratio.npy"), overlap_hist)
        with open(os.path.join(out_dir, f"{tag}_summary.json"), "w", encoding="utf-8") as f:
            json.dump(cmp_stats, f, indent=2, ensure_ascii=False)

        try:
            import matplotlib.pyplot as plt

            x_rank = np.arange(1, m + 1)
            plt.figure(figsize=(8, 3.2))
            plt.bar(x_rank, rank_ratio, color="#2563eb")
            plt.ylim(0.0, 1.0)
            plt.xlabel("Rank (1..m)")
            plt.ylabel("Match Ratio")
            plt.title(f"Wave vs Meta Rank-Match ({tag})")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{tag}_rank_match.png"), dpi=150)
            plt.close()

            x_overlap = np.arange(0, m + 1)
            plt.figure(figsize=(8, 3.2))
            plt.bar(x_overlap, overlap_hist, color="#16a34a")
            plt.ylim(0.0, 1.0)
            plt.xlabel("Overlap Count in Top-m")
            plt.ylabel("Frequency Ratio")
            plt.title(f"Wave vs Meta Top-m Overlap Histogram ({tag})")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{tag}_overlap_hist.png"), dpi=150)
            plt.close()
        except Exception as e:
            print(f"[RetrievalCompare] skip plot save due to matplotlib error: {e}")

    @staticmethod
    def _save_retrieval_case_viz(case_data, out_dir, tag):
        if case_data is None:
            return
        os.makedirs(out_dir, exist_ok=True)

        query_history = np.asarray(case_data.get("query_history", []), dtype=np.float32)
        wave_hist = np.asarray(case_data.get("wave_histories", []), dtype=np.float32)
        meta_hist = np.asarray(case_data.get("meta_histories", []), dtype=np.float32)
        wave_fut = np.asarray(case_data.get("wave_futures", []), dtype=np.float32)
        meta_fut = np.asarray(case_data.get("meta_futures", []), dtype=np.float32)
        true_fut = np.asarray(case_data.get("true_future", []), dtype=np.float32)
        pred_fut = np.asarray(case_data.get("pred_future", []), dtype=np.float32)
        wave_pred_fut = np.asarray(case_data.get("wave_pred_future", []), dtype=np.float32)
        meta_pred_fut = np.asarray(case_data.get("meta_pred_future", []), dtype=np.float32)

        np.save(os.path.join(out_dir, f"{tag}_query_history.npy"), query_history)
        np.save(os.path.join(out_dir, f"{tag}_wave_histories.npy"), wave_hist)
        np.save(os.path.join(out_dir, f"{tag}_meta_histories.npy"), meta_hist)
        np.save(os.path.join(out_dir, f"{tag}_wave_futures.npy"), wave_fut)
        np.save(os.path.join(out_dir, f"{tag}_meta_futures.npy"), meta_fut)
        if true_fut.size > 0:
            np.save(os.path.join(out_dir, f"{tag}_true_future.npy"), true_fut)
        if pred_fut.size > 0:
            np.save(os.path.join(out_dir, f"{tag}_pred_future.npy"), pred_fut)
        if wave_pred_fut.size > 0:
            np.save(os.path.join(out_dir, f"{tag}_wave_pred_future.npy"), wave_pred_fut)
        if meta_pred_fut.size > 0:
            np.save(os.path.join(out_dir, f"{tag}_meta_pred_future.npy"), meta_pred_fut)

        meta_info = {
            "period_idx": int(case_data.get("period_idx", -1)),
            "period_g": int(case_data.get("period_g", -1)),
            "channel_idx": int(case_data.get("channel_idx", -1)),
            "topm": int(case_data.get("topm", 0)),
            "wave_topm_idx": list(map(int, case_data.get("wave_topm_idx", []))),
            "meta_topm_idx": list(map(int, case_data.get("meta_topm_idx", []))),
        }
        with open(os.path.join(out_dir, f"{tag}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta_info, f, indent=2, ensure_ascii=False)

        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 8))
            fig.suptitle(
                f"Wave vs Meta Top-m Case | period_g={meta_info['period_g']} channel={meta_info['channel_idx']} topm={meta_info['topm']}",
                fontsize=12,
            )

            ax = axes[0, 0]
            if query_history.size > 0:
                ax.plot(query_history, color="black", linewidth=2.0, label="query history")
            for i in range(wave_hist.shape[0]):
                ax.plot(wave_hist[i], alpha=0.35, linewidth=1.0)
            ax.set_title("Wave Top-m Histories")
            ax.legend(loc="best")

            ax = axes[0, 1]
            if query_history.size > 0:
                ax.plot(query_history, color="black", linewidth=2.0, label="query history")
            for i in range(meta_hist.shape[0]):
                ax.plot(meta_hist[i], alpha=0.35, linewidth=1.0)
            ax.set_title("Meta Top-m Histories")
            ax.legend(loc="best")

            ax = axes[1, 0]
            if true_fut.size > 0:
                ax.plot(true_fut, color="black", linewidth=2.0, label="true future")
            if wave_pred_fut.size > 0:
                ax.plot(wave_pred_fut, color="#2563eb", linewidth=1.8, label="wave pred")
            if meta_pred_fut.size > 0:
                ax.plot(meta_pred_fut, color="#dc2626", linewidth=1.8, label="meta pred")
            elif pred_fut.size > 0:
                ax.plot(pred_fut, color="#dc2626", linewidth=1.8, label="pred future")
            for i in range(wave_fut.shape[0]):
                ax.plot(wave_fut[i], alpha=0.35, linewidth=1.0)
            ax.set_title("Wave Top-m Futures")
            ax.legend(loc="best")

            ax = axes[1, 1]
            if true_fut.size > 0:
                ax.plot(true_fut, color="black", linewidth=2.0, label="true future")
            if wave_pred_fut.size > 0:
                ax.plot(wave_pred_fut, color="#2563eb", linewidth=1.8, label="wave pred")
            if meta_pred_fut.size > 0:
                ax.plot(meta_pred_fut, color="#dc2626", linewidth=1.8, label="meta pred")
            elif pred_fut.size > 0:
                ax.plot(pred_fut, color="#dc2626", linewidth=1.8, label="pred future")
            for i in range(meta_fut.shape[0]):
                ax.plot(meta_fut[i], alpha=0.35, linewidth=1.0)
            ax.set_title("Meta Top-m Futures")
            ax.legend(loc="best")

            # Unify axis ranges/ticks across left-right panels (histories and futures).
            def _collect_series(*arrs):
                out = []
                for a in arrs:
                    arr = np.asarray(a, dtype=np.float32)
                    if arr.size == 0:
                        continue
                    if arr.ndim == 1:
                        out.append(arr)
                    else:
                        out.extend([arr[i] for i in range(arr.shape[0])])
                return out

            hist_series = _collect_series(query_history, wave_hist, meta_hist)
            fut_series = _collect_series(true_fut, wave_fut, meta_fut, pred_fut, wave_pred_fut, meta_pred_fut)

            def _limits(series_list):
                if len(series_list) == 0:
                    return -1.0, 1.0
                mn = min(float(np.min(s)) for s in series_list)
                mx = max(float(np.max(s)) for s in series_list)
                if not np.isfinite(mn) or not np.isfinite(mx):
                    return -1.0, 1.0
                if abs(mx - mn) < 1e-8:
                    pad = max(abs(mx), 1.0) * 0.1 + 1e-3
                    return mn - pad, mx + pad
                pad = 0.05 * (mx - mn)
                return mn - pad, mx + pad

            hist_ymin, hist_ymax = _limits(hist_series)
            fut_ymin, fut_ymax = _limits(fut_series)

            if query_history.ndim == 1 and query_history.size > 0:
                hist_len = int(query_history.shape[0])
            elif wave_hist.ndim == 2 and wave_hist.size > 0:
                hist_len = int(wave_hist.shape[1])
            elif meta_hist.ndim == 2 and meta_hist.size > 0:
                hist_len = int(meta_hist.shape[1])
            else:
                hist_len = 720

            if true_fut.ndim == 1 and true_fut.size > 0:
                fut_len = int(true_fut.shape[0])
            elif wave_fut.ndim == 2 and wave_fut.size > 0:
                fut_len = int(wave_fut.shape[1])
            elif meta_fut.ndim == 2 and meta_fut.size > 0:
                fut_len = int(meta_fut.shape[1])
            elif wave_pred_fut.ndim == 1 and wave_pred_fut.size > 0:
                fut_len = int(wave_pred_fut.shape[0])
            elif meta_pred_fut.ndim == 1 and meta_pred_fut.size > 0:
                fut_len = int(meta_pred_fut.shape[0])
            elif pred_fut.ndim == 1 and pred_fut.size > 0:
                fut_len = int(pred_fut.shape[0])
            else:
                fut_len = 96
            hist_xticks = np.linspace(0, max(hist_len - 1, 1), 7, dtype=int)
            fut_xticks = np.linspace(0, max(fut_len - 1, 1), 7, dtype=int)
            hist_yticks = np.linspace(hist_ymin, hist_ymax, 7)
            fut_yticks = np.linspace(fut_ymin, fut_ymax, 7)

            for ax in [axes[0, 0], axes[0, 1]]:
                ax.set_xlim(0, max(hist_len - 1, 1))
                ax.set_ylim(hist_ymin, hist_ymax)
                ax.set_xticks(hist_xticks)
                ax.set_yticks(hist_yticks)
            for ax in [axes[1, 0], axes[1, 1]]:
                ax.set_xlim(0, max(fut_len - 1, 1))
                ax.set_ylim(fut_ymin, fut_ymax)
                ax.set_xticks(fut_xticks)
                ax.set_yticks(fut_yticks)

            for ax in axes.reshape(-1):
                ax.grid(alpha=0.2)

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{tag}_panel.png"), dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"[RetrievalCase] skip plot save due to matplotlib error: {e}")

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
        amp_enabled = self.args.use_amp and self.args.use_gpu
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                index, batch_x, batch_y, batch_x_mark, batch_y_mark, meta_data = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        if self.args.model == 'RAFT':
                            outputs = self.model(
                                batch_x,
                                index,
                                mode=split_mode,
                                meta_data=meta_data,
                            )
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.model == 'RAFT':
                        outputs = self.model(
                            batch_x,
                            index,
                            mode=split_mode,
                            meta_data=meta_data,
                        )
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())
        total_loss = float(np.average(total_loss))
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
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        amp_enabled = self.args.use_amp and self.args.use_gpu
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            if self.args.model == 'RAFT':
                raft_model = self._raft_model()
                if hasattr(raft_model, "refresh_retrieval_bank"):
                    raft_model.refresh_retrieval_bank()
                if hasattr(raft_model, "reset_retrieval_compare_stats"):
                    raft_model.reset_retrieval_compare_stats()

            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                index, batch_x, batch_y, batch_x_mark, batch_y_mark, meta_data = self._unpack_batch(batch)
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
                if self.args.use_amp:
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        if self.args.model == 'RAFT':
                            outputs = self.model(
                                batch_x,
                                index,
                                mode='train',
                                meta_data=meta_data,
                            )
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.model == 'RAFT':
                        outputs = self.model(
                            batch_x,
                            index,
                            mode='train',
                            meta_data=meta_data,
                        )
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

                if amp_enabled:
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
            if self.args.model == 'RAFT':
                raft_model = self._raft_model()
                if hasattr(raft_model, "get_retrieval_compare_stats"):
                    cmp_stats = raft_model.get_retrieval_compare_stats()
                    if cmp_stats is not None:
                        print(
                            "[RetrievalCompare][Epoch {}] overlap@m={:.4f}, exact_set={:.4f}, exact_order={:.4f}, calls={}".format(
                                epoch + 1,
                                cmp_stats["overlap_ratio"],
                                cmp_stats["exact_set_ratio"],
                                cmp_stats["exact_order_ratio"],
                                cmp_stats["calls"],
                            )
                        )
                        self._save_retrieval_compare_viz(
                            cmp_stats,
                            out_dir=os.path.join(path, "retrieval_compare"),
                            tag=f"epoch_{epoch + 1}",
                        )

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        amp_enabled = self.args.use_amp and self.args.use_gpu
        cmp_stats_test = None
        case_saved = False
        case_data_pending = None
        with torch.no_grad():
            raft_model = None
            if self.args.model == 'RAFT':
                raft_model = self._raft_model()
                if hasattr(raft_model, "reset_retrieval_compare_stats"):
                    raft_model.reset_retrieval_compare_stats()
            for i, batch in enumerate(test_loader):
                index, batch_x, batch_y, batch_x_mark, batch_y_mark, meta_data = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if (
                    self.args.model == 'RAFT'
                    and (not case_saved)
                    and case_data_pending is None
                    and getattr(self.args, "save_retrieval_cases", False)
                    and raft_model is not None
                    and hasattr(raft_model, "export_wave_meta_topm_case")
                ):
                    case_data = raft_model.export_wave_meta_topm_case(
                        x=batch_x,
                        index=index,
                        meta_data=meta_data,
                        sample_idx=getattr(self.args, "retrieval_case_sample_idx", 0),
                        period_idx=getattr(self.args, "retrieval_case_period_idx", -1),
                        channel_idx=getattr(self.args, "retrieval_case_channel_idx", -1),
                        train=False,
                    )
                    if case_data is not None:
                        sample_idx = int(getattr(self.args, "retrieval_case_sample_idx", 0))
                        sample_idx = max(0, min(sample_idx, batch_x.shape[0] - 1))
                        channel_idx = int(case_data.get("channel_idx", -1))
                        period_idx = int(case_data.get("period_idx", -1))
                        batch_y_future = batch_y[:, -self.args.pred_len:, :]
                        true_future = None
                        if (
                            hasattr(raft_model, "rt")
                            and hasattr(raft_model.rt, "decompose_mg")
                            and 0 <= period_idx < len(getattr(raft_model.rt, "period_num", []))
                            and 0 <= channel_idx < batch_y_future.shape[-1]
                        ):
                            y_mg, _ = raft_model.rt.decompose_mg(batch_y_future)
                            true_future = y_mg[period_idx, sample_idx, :, channel_idx].detach().cpu().numpy()
                        elif 0 <= channel_idx < batch_y_future.shape[-1]:
                            true_future = batch_y_future[sample_idx, :, channel_idx].detach().cpu().numpy()

                        if true_future is not None:
                            case_data["true_future"] = true_future
                        case_data_pending = {
                            "case_data": case_data,
                            "sample_idx": sample_idx,
                        }

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        if self.args.model == 'RAFT':
                            outputs = self.model(
                                batch_x,
                                index,
                                mode='test',
                                meta_data=meta_data,
                            )
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.model == 'RAFT':
                        outputs = self.model(
                            batch_x,
                            index,
                            mode='test',
                            meta_data=meta_data,
                        )
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if (
                    self.args.model == 'RAFT'
                    and (not case_saved)
                    and case_data_pending is not None
                    and raft_model is not None
                ):
                    case_data = case_data_pending["case_data"]
                    sample_idx = int(case_data_pending["sample_idx"])
                    channel_idx = int(case_data.get("channel_idx", -1))
                    period_idx = int(case_data.get("period_idx", -1))
                    rt = getattr(raft_model, "rt", None)

                    def _extract_future_curve(pred_tensor):
                        pred_tensor = pred_tensor[:, -self.args.pred_len:, :]
                        if (
                            rt is not None
                            and hasattr(rt, "decompose_mg")
                            and 0 <= period_idx < len(getattr(rt, "period_num", []))
                            and 0 <= channel_idx < pred_tensor.shape[-1]
                        ):
                            pred_mg, _ = rt.decompose_mg(pred_tensor)
                            return pred_mg[period_idx, sample_idx, :, channel_idx].detach().cpu().numpy()
                        if 0 <= channel_idx < pred_tensor.shape[-1]:
                            return pred_tensor[sample_idx, :, channel_idx].detach().cpu().numpy()
                        return None

                    # Current run-mode prediction (for backward compatibility).
                    pred_future = _extract_future_curve(outputs)
                    if pred_future is not None:
                        case_data["pred_future"] = pred_future

                    # Explicitly compare wave-only retrieval prediction vs meta-only retrieval prediction.
                    if rt is not None and hasattr(rt, "meta_only_retrieval"):
                        orig_meta_only = bool(rt.meta_only_retrieval)
                        orig_compare = bool(getattr(rt, "compare_retrieval_topk", False))
                        try:
                            if hasattr(rt, "compare_retrieval_topk"):
                                rt.compare_retrieval_topk = False

                            rt.meta_only_retrieval = False
                            if self.args.use_amp:
                                with torch.cuda.amp.autocast(enabled=amp_enabled):
                                    wave_outputs = self.model(
                                        batch_x,
                                        index,
                                        mode='test',
                                        meta_data=meta_data,
                                    )
                            else:
                                wave_outputs = self.model(
                                    batch_x,
                                    index,
                                    mode='test',
                                    meta_data=meta_data,
                                )
                            wave_pred_future = _extract_future_curve(wave_outputs)
                            if wave_pred_future is not None:
                                case_data["wave_pred_future"] = wave_pred_future

                            rt.meta_only_retrieval = True
                            if self.args.use_amp:
                                with torch.cuda.amp.autocast(enabled=amp_enabled):
                                    meta_outputs = self.model(
                                        batch_x,
                                        index,
                                        mode='test',
                                        meta_data=meta_data,
                                    )
                            else:
                                meta_outputs = self.model(
                                    batch_x,
                                    index,
                                    mode='test',
                                    meta_data=meta_data,
                                )
                            meta_pred_future = _extract_future_curve(meta_outputs)
                            if meta_pred_future is not None:
                                case_data["meta_pred_future"] = meta_pred_future
                        finally:
                            rt.meta_only_retrieval = orig_meta_only
                            if hasattr(rt, "compare_retrieval_topk"):
                                rt.compare_retrieval_topk = orig_compare

                    self._save_retrieval_case_viz(
                        case_data=case_data,
                        out_dir=os.path.join(folder_path, "retrieval_cases"),
                        tag="test_case_0",
                    )
                    case_saved = True
                    case_data_pending = None

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
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

        if self.args.model == 'RAFT':
            raft_model = self._raft_model()
            if hasattr(raft_model, "get_retrieval_compare_stats"):
                cmp_stats = raft_model.get_retrieval_compare_stats()
                if cmp_stats is not None:
                    cmp_stats_test = cmp_stats
                    print(
                        "[RetrievalCompare][Test] overlap@m={:.4f}, exact_set={:.4f}, exact_order={:.4f}, calls={}".format(
                            cmp_stats["overlap_ratio"],
                            cmp_stats["exact_set_ratio"],
                            cmp_stats["exact_order_ratio"],
                            cmp_stats["calls"],
                        )
                    )

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
        if cmp_stats_test is not None:
            self._save_retrieval_compare_viz(
                cmp_stats_test,
                out_dir=os.path.join(folder_path, "retrieval_compare"),
                tag="test",
            )
        
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
