import torch
import torch.nn as nn

from layers.Retrieval import RetrievalTool


class Model(nn.Module):
    def __init__(self, configs, individual=False):
        super(Model, self).__init__()
        self.device = torch.device(f"cuda:{configs.gpu}" if (configs.use_gpu and torch.cuda.is_available()) else "cpu")
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len if self.task_name in ["classification", "anomaly_detection", "imputation"] else configs.pred_len
        self.channels = configs.enc_in

        self.retrieval_cache_device = getattr(configs, "retrieval_cache_device", "gpu")
        self.learnable_alpha = getattr(configs, "learnable_alpha", False)
        self.online_retrieval = getattr(configs, "online_retrieval", False) or self.learnable_alpha
        self.refresh_context_each_epoch = getattr(configs, "refresh_context_each_epoch", True)
        if self.retrieval_cache_device == "gpu" and not torch.cuda.is_available():
            self.retrieval_cache_device = "cpu"

        self.linear_x = nn.Linear(self.seq_len, self.pred_len)

        self.n_period = configs.n_period
        self.topm = configs.topm

        self.rt = RetrievalTool(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            channels=self.channels,
            n_period=self.n_period,
            topm=self.topm,
            alpha=getattr(configs, "retrieval_alpha", 0.7),
            coarse_k=getattr(configs, "retrieval_coarse_k", 80),
            context_dim=getattr(configs, "context_dim", getattr(configs, "meta_embed_dim", 64)),
            gate_hidden_dim=getattr(configs, "gate_hidden_dim", 128),
            use_gated_aggregation=getattr(configs, "use_gated_aggregation", True),
            learnable_alpha=getattr(configs, "learnable_alpha", False),
            train_context_encoder=not getattr(configs, "freeze_context_encoder", False),
            meta_only_retrieval=getattr(configs, "meta_only_retrieval", False),
            compare_retrieval_topk=getattr(
                configs,
                "compare_retrieval_topm",
                getattr(configs, "compare_retrieval_topk", False),
            ),
        )

        self.period_num = self.rt.period_num[-1 * self.n_period:]
        self.retrieval_pred = nn.ModuleList(
            [nn.Linear(self.pred_len // g, self.pred_len) for g in self.period_num]
        )
        self.period_attn_dim = getattr(configs, "period_router_hidden_dim", 128)
        self.period_query_proj = nn.Linear(self.pred_len, self.period_attn_dim)
        self.period_key_proj = nn.Linear(self.pred_len, self.period_attn_dim)
        self.period_meta_proj = nn.Linear(4, self.period_attn_dim)
        self.period_attn_dropout = nn.Dropout(getattr(configs, "period_attn_dropout", 0.0))
        self.period_attn_scale = self.period_attn_dim ** -0.5
        # One-shot base/retrieval fusion after cross-scale retrieval aggregation.
        self.linear_pred = nn.Linear(2 * self.pred_len, self.pred_len)
        self.linear_pred_per_period = nn.ModuleList(
            [nn.Linear(self.pred_len, self.pred_len) for _ in self.period_num]
        )

    def _resolve_cache_device(self, cache_mode):
        if cache_mode == "gpu":
            return self.device
        return torch.device("cpu")

    def prepare_dataset(self, train_data, valid_data, test_data):
        self.retrieval_dict = {}
        retrieval_cache_device = self._resolve_cache_device(self.retrieval_cache_device)

        if self.online_retrieval:
            print("Preparing Online Retrieval Bank")
            self.rt.prepare_dataset(train_data, cache_device=retrieval_cache_device)
        else:
            self.rt.prepare_dataset(train_data, cache_device=retrieval_cache_device)

            print("Doing Train Retrieval")
            train_rt = self.rt.retrieve_all(train_data, train=True, device=self.device)

            print("Doing Valid Retrieval")
            valid_rt = self.rt.retrieve_all(valid_data, train=False, device=self.device)

            print("Doing Test Retrieval")
            test_rt = self.rt.retrieve_all(test_data, train=False, device=self.device)

        if not self.online_retrieval:
            self.retrieval_dict["train"] = train_rt.detach().to(retrieval_cache_device)
            self.retrieval_dict["valid"] = valid_rt.detach().to(retrieval_cache_device)
            self.retrieval_dict["test"] = test_rt.detach().to(retrieval_cache_device)

        if not self.online_retrieval:
            del self.rt
        torch.cuda.empty_cache()

    def refresh_retrieval_bank(self):
        if not self.online_retrieval or not self.refresh_context_each_epoch:
            return
        if not hasattr(self, "rt"):
            return
        refresh_device = self._resolve_cache_device(self.retrieval_cache_device)
        self.rt.refresh_context_pool(device=refresh_device)

    def reset_retrieval_compare_stats(self):
        if not self.online_retrieval:
            return
        if not hasattr(self, "rt"):
            return
        if hasattr(self.rt, "reset_retrieval_compare_stats"):
            self.rt.reset_retrieval_compare_stats()

    def get_retrieval_compare_stats(self):
        if not self.online_retrieval:
            return None
        if not hasattr(self, "rt"):
            return None
        if hasattr(self.rt, "get_retrieval_compare_stats"):
            return self.rt.get_retrieval_compare_stats()
        return None

    def export_wave_meta_topm_case(self, x, index, meta_data=None, sample_idx=0, period_idx=-1, channel_idx=-1, train=False):
        if not self.online_retrieval:
            return None
        if not hasattr(self, "rt"):
            return None
        if not hasattr(self.rt, "export_wave_meta_topm_case"):
            return None
        return self.rt.export_wave_meta_topm_case(
            x=x,
            index=index,
            meta_query=meta_data,
            sample_idx=sample_idx,
            period_idx=period_idx,
            channel_idx=channel_idx,
            train=train,
        )

    def _extract_local_state_feature(self, meta_data, bsz, device):
        gsz = len(self.period_num)
        out = torch.zeros((bsz, gsz, 4), device=device)
        if not isinstance(meta_data, dict):
            return out

        local_state = meta_data.get("local_state_by_period", None)
        if local_state is None:
            return out

        if torch.is_tensor(local_state):
            local_state = local_state.float()
        else:
            local_state = torch.tensor(local_state, dtype=torch.float32)

        if local_state.dim() == 2:
            local_state = local_state.unsqueeze(0)
        if local_state.shape[0] == 1 and bsz > 1:
            local_state = local_state.repeat(bsz, 1, 1)
        if local_state.shape[0] > bsz:
            local_state = local_state[:bsz]
        if local_state.shape[0] < bsz:
            pad = torch.zeros((bsz - local_state.shape[0], local_state.shape[1], local_state.shape[2]))
            local_state = torch.cat([local_state, pad], dim=0)
        if local_state.shape[1] < gsz:
            pad = torch.zeros((bsz, gsz - local_state.shape[1], local_state.shape[2]))
            local_state = torch.cat([local_state, pad], dim=1)
        if local_state.shape[2] < 4:
            pad = torch.zeros((bsz, local_state.shape[1], 4 - local_state.shape[2]))
            local_state = torch.cat([local_state, pad], dim=2)

        local_state = local_state[:, :gsz, :4].to(device)
        return local_state

    def encoder(self, x, index, mode, meta_data=None):
        bsz, seq_len, channels = x.shape
        assert seq_len == self.seq_len and channels == self.channels

        x_offset = x[:, -1:, :].detach()
        x_norm = x - x_offset  # keep offset subtraction: x_hat = x - x^L

        x_pred_from_x = self.linear_x(x_norm.permute(0, 2, 1)).permute(0, 2, 1)  # B, P, C

        if self.online_retrieval:
            use_train_mask = mode == "train"
            pred_from_retrieval = self.rt.retrieve(
                x,
                index,
                meta_query=meta_data,
                train=use_train_mask,
            ).to(self.device)  # G, B, P, C
        else:
            retrieval_bank = self.retrieval_dict[mode]
            retrieval_index = index.long().to(retrieval_bank.device)
            pred_from_retrieval = retrieval_bank[:, retrieval_index].to(self.device)  # G, B, P, C

        retrieval_pred_list = []
        numeric_feature = x_pred_from_x.permute(0, 2, 1)  # B, C, P

        for i, pr in enumerate(pred_from_retrieval):
            assert pr.shape == (bsz, self.pred_len, channels)
            g = self.period_num[i]
            pr = pr.reshape(bsz, self.pred_len // g, g, channels)
            pr = pr[:, :, 0, :]
            pr = self.retrieval_pred[i](pr.permute(0, 2, 1)).permute(0, 2, 1)
            pr = pr.reshape(bsz, self.pred_len, self.channels)  # B, P, C

            retrieval_feature = pr.permute(0, 2, 1)  # B, C, P
            pred_i = self.linear_pred_per_period[i](retrieval_feature).permute(0, 2, 1).reshape(
                bsz, self.pred_len, self.channels
            )
            retrieval_pred_list.append(pred_i)

        pred_stack = torch.stack(retrieval_pred_list, dim=1)  # B, G, P, C

        # Channel-wise period routing: one period distribution per sample-channel pair.
        query_signal = x_pred_from_x.permute(0, 2, 1)  # B, C, P
        query_token = self.period_query_proj(
            query_signal.reshape(bsz * channels, self.pred_len)
        ).reshape(bsz, channels, self.period_attn_dim)  # B, C, D

        period_signal = pred_stack.permute(0, 3, 1, 2)  # B, C, G, P
        gsz = len(self.period_num)
        key_token = self.period_key_proj(
            period_signal.reshape(bsz * channels * gsz, self.pred_len)
        ).reshape(bsz, channels, gsz, self.period_attn_dim)  # B, C, G, D

        local_state_feature = self._extract_local_state_feature(meta_data, bsz, x.device)  # B, G, 4
        local_state_feature = local_state_feature.unsqueeze(1).expand(-1, channels, -1, -1)  # B, C, G, 4
        key_token = key_token + self.period_meta_proj(
            local_state_feature.reshape(bsz * channels * gsz, 4)
        ).reshape(bsz, channels, gsz, self.period_attn_dim)  # B, C, G, D

        period_logits = torch.matmul(
            query_token.unsqueeze(2), key_token.transpose(2, 3)
        ).squeeze(2) * self.period_attn_scale  # B, C, G
        period_weight = torch.softmax(period_logits, dim=2)  # B, C, G
        period_weight = self.period_attn_dropout(period_weight)
        period_weight = period_weight / period_weight.sum(dim=2, keepdim=True).clamp_min(1e-6)

        retrieval_agg = (period_signal * period_weight.unsqueeze(-1)).sum(dim=2).permute(0, 2, 1)  # B, P, C

        # Fuse once: aggregated multi-scale retrieval + base prediction.
        retrieval_feature = retrieval_agg.permute(0, 2, 1)  # B, C, P
        fusion_feature = torch.cat([numeric_feature, retrieval_feature], dim=2)  # B, C, 2P
        pred = self.linear_pred(fusion_feature).permute(0, 2, 1).reshape(bsz, self.pred_len, self.channels)
        pred = pred + x_offset
        return pred

    def forecast(self, x_enc, index, mode, meta_data=None):
        return self.encoder(
            x_enc,
            index,
            mode,
            meta_data=meta_data,
        )

    def imputation(self, x_enc, index, mode, meta_data=None):
        return self.encoder(
            x_enc,
            index,
            mode,
            meta_data=meta_data,
        )

    def anomaly_detection(self, x_enc, index, mode, meta_data=None):
        return self.encoder(
            x_enc,
            index,
            mode,
            meta_data=meta_data,
        )

    def classification(self, x_enc, index, mode, meta_data=None):
        enc_out = self.encoder(
            x_enc,
            index,
            mode,
            meta_data=meta_data,
        )
        output = enc_out.reshape(enc_out.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, index, mode="train", meta_data=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            dec_out = self.forecast(
                x_enc,
                index,
                mode,
                meta_data=meta_data,
            )
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == "imputation":
            return self.imputation(
                x_enc,
                index,
                mode,
                meta_data=meta_data,
            )
        if self.task_name == "anomaly_detection":
            return self.anomaly_detection(
                x_enc,
                index,
                mode,
                meta_data=meta_data,
            )
        if self.task_name == "classification":
            return self.classification(
                x_enc,
                index,
                mode,
                meta_data=meta_data,
            )
        return None
