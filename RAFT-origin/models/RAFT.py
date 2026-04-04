import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Retrieval import RetrievalTool

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.device = torch.device(
            f'cuda:{configs.gpu}' if (getattr(configs, 'use_gpu', False) and torch.cuda.is_available()) else 'cpu'
        )
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        # Series decomposition block from Autoformer
#         self.decompsition = series_decomp(configs.moving_avg)
#         self.individual = individual
        self.channels = configs.enc_in

        self.linear_x = nn.Linear(self.seq_len, self.pred_len)
        
        self.n_period = configs.n_period
        self.topm = configs.topm
        self.compare_retrieval_future_quality = bool(getattr(configs, 'compare_retrieval_future_quality', False))
        self.retrieval_future_quality = None
        
        self.rt = RetrievalTool(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            channels=self.channels,
            n_period=self.n_period,
            temperature=getattr(configs, 'retrieval_temperature', 0.1),
            topm=self.topm,
            meta_only_retrieval=getattr(configs, 'meta_only_retrieval', False),
        )
        
        self.period_num = self.rt.period_num[-1 * self.n_period:]
        
        module_list = [
            nn.Linear(self.pred_len // g, self.pred_len)
            for g in self.period_num
        ]
        self.retrieval_pred = nn.ModuleList(module_list)
        self.linear_pred = nn.Linear(2 * self.pred_len, self.pred_len)
        self._last_base_pred = None
        self._last_debug = None

#         if self.task_name == 'classification':
#             self.projection = nn.Linear(
#                 configs.enc_in * configs.seq_len, configs.num_class)

    def prepare_dataset(self, train_data, valid_data, test_data):
        self.rt.prepare_dataset(train_data)
        
        self.retrieval_dict = {}
        
        print('Doing Train Retrieval')
        train_rt = self.rt.retrieve_all(train_data, train=True, device=self.device)

        print('Doing Valid Retrieval')
        valid_rt = self.rt.retrieve_all(valid_data, train=False, device=self.device)

        print('Doing Test Retrieval')
        test_rt = self.rt.retrieve_all(test_data, train=False, device=self.device)

        if self.compare_retrieval_future_quality:
            print('Evaluating retrieval-only future quality (wave vs meta) on test split')
            self.retrieval_future_quality = self.rt.evaluate_wave_meta_retrieval_quality(
                data=test_data,
                device=self.device,
                train=False,
            )

        del self.rt
        torch.cuda.empty_cache()
            
        self.retrieval_dict['train'] = train_rt.detach()
        self.retrieval_dict['valid'] = valid_rt.detach()
        self.retrieval_dict['test'] = test_rt.detach()

    def get_last_base_pred(self):
        return self._last_base_pred

    def get_last_debug(self):
        return self._last_debug

    def get_retrieval_future_quality(self):
        return self.retrieval_future_quality

    def encoder(self, x, index, mode):
        bsz, seq_len, channels = x.shape
        assert(seq_len == self.seq_len, channels == self.channels)
        
        x_offset = x[:, -1:, :].detach()
        x_norm = x - x_offset

        x_pred_from_x = self.linear_x(x_norm.permute(0, 2, 1)).permute(0, 2, 1) # B, P, C
        self._last_base_pred = (x_pred_from_x + x_offset).detach()
        
        retrieval_bank = self.retrieval_dict[mode]
        retrieval_index = index.long().to(retrieval_bank.device)
        pred_from_retrieval = retrieval_bank[:, retrieval_index].to(x.device) # G, B, P, C
        
        retrieval_pred_list = []
        
        # Compress repeating dimensions
        for i, pr in enumerate(pred_from_retrieval):
            assert((bsz, self.pred_len, channels) == pr.shape)
            g = self.period_num[i]
            pr = pr.reshape(bsz, self.pred_len // g, g, channels)
            pr = pr[:, :, 0, :]
            
            pr = self.retrieval_pred[i](pr.permute(0, 2, 1)).permute(0, 2, 1)
            pr = pr.reshape(bsz, self.pred_len, self.channels)
            
            retrieval_pred_list.append(pr)

        pred_stack = torch.stack(retrieval_pred_list, dim=1)  # B, G, P, C
        retrieval_agg = pred_stack.sum(dim=1)  # B, P, C

        x_offset_period = x_offset.unsqueeze(1)  # B, 1, 1, C
        self._last_debug = {
            'pred_stack': (pred_stack + x_offset_period).detach(),
            'retrieval_agg': (retrieval_agg + x_offset).detach(),
            'base_pred': (x_pred_from_x + x_offset).detach(),
        }
        
        pred = torch.cat([x_pred_from_x, retrieval_agg], dim=1)
        pred = self.linear_pred(pred.permute(0, 2, 1)).permute(0, 2, 1).reshape(bsz, self.pred_len, self.channels)
        
        pred = pred + x_offset
        
        return pred

    def forecast(self, x_enc, index, mode):
        # Encoder
        return self.encoder(x_enc, index, mode)

    def imputation(self, x_enc, index, mode):
        # Encoder
        return self.encoder(x_enc, index, mode)

    def anomaly_detection(self, x_enc, index, mode):
        # Encoder
        return self.encoder(x_enc, index, mode)

    def classification(self, x_enc, index, mode):
        # Encoder
        enc_out = self.encoder(x_enc, index, mode)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, index, mode='train'):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, index, mode)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, index, mode)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, index, mode)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, index, mode)
            return dec_out  # [B, N]
        return None
