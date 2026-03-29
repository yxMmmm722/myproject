import torch
import torch.nn as nn
import numpy as np

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
        self.text_encoder = None
        self.text_proj = None
        self.text_feature_dict = {}
        self.text_feature_by_period_dict = {}
        # One-shot base/retrieval fusion after cross-scale retrieval aggregation.
        self.linear_pred = nn.Linear(2 * self.pred_len, self.pred_len)
        self.linear_pred_per_period = nn.ModuleList(
            [nn.Linear(self.pred_len, self.pred_len) for _ in self.period_num]
        )

    def _resolve_cache_device(self, cache_mode):
        if cache_mode == "gpu":
            return self.device
        return torch.device("cpu")

    def _encode_texts(self, texts):
        # Text branch is disabled; keep no-op for backward compatibility.
        return torch.zeros((len(texts), 0))

    def _collect_texts(self, dataset):
        texts = []
        texts_by_period = []
        for i in range(len(dataset)):
            item = dataset[i]
            text = item[6] if len(item) > 6 else ""
            text = str(text)
            raw_period_text = item[7] if len(item) > 7 else None
            if raw_period_text is None:
                period_text = [text for _ in self.period_num]
            else:
                period_text = [str(x) for x in list(raw_period_text)]
                if len(period_text) < len(self.period_num):
                    period_text += [text for _ in range(len(self.period_num) - len(period_text))]
                period_text = period_text[: len(self.period_num)]
            texts.append(text)
            texts_by_period.append(period_text)
        return texts, texts_by_period

    def _encode_texts_by_period(self, texts_by_period):
        # Text branch is disabled; keep no-op for backward compatibility.
        return torch.zeros((len(self.period_num), len(texts_by_period), 0))

    def _normalize_period_text_batch(self, meta_text_by_period, bsz, fallback_text=None):
        if meta_text_by_period is None:
            if fallback_text is None:
                fallback_text = ["" for _ in range(bsz)]
            return [[str(fallback_text[b]) for _ in self.period_num] for b in range(bsz)]

        data = meta_text_by_period
        if torch.is_tensor(data):
            data = data.tolist()
        elif isinstance(data, np.ndarray):
            data = data.tolist()

        # Case-A: DataLoader default_collate on sample-level list[str]:
        #   period-major shape: [G][B]
        if isinstance(data, (list, tuple)) and len(data) == len(self.period_num):
            first = data[0] if len(data) > 0 else []
            if isinstance(first, (list, tuple)) and len(first) == bsz:
                sample_major = []
                for b in range(bsz):
                    sample_major.append([str(data[g][b]) for g in range(len(self.period_num))])
                return sample_major

        # Case-B: sample-major shape: [B][G]
        if isinstance(data, (list, tuple)) and len(data) == bsz:
            out = []
            for b in range(bsz):
                row = data[b]
                if isinstance(row, torch.Tensor):
                    row = row.tolist()
                if isinstance(row, np.ndarray):
                    row = row.tolist()
                if isinstance(row, (list, tuple)):
                    row = [str(x) for x in row]
                else:
                    row = [str(row)]
                if len(row) < len(self.period_num):
                    fill = row[-1] if len(row) > 0 else (str(fallback_text[b]) if fallback_text is not None else "")
                    row += [fill for _ in range(len(self.period_num) - len(row))]
                row = row[: len(self.period_num)]
                out.append(row)
            return out

        if fallback_text is None:
            fallback_text = ["" for _ in range(bsz)]
        return [[str(fallback_text[b]) for _ in self.period_num] for b in range(bsz)]

    def prepare_dataset(self, train_data, valid_data, test_data):
        self.retrieval_dict = {}
        retrieval_cache_device = self._resolve_cache_device(self.retrieval_cache_device)

        if self.online_retrieval:
            print("Preparing Online Retrieval Bank")
            self.rt.prepare_dataset(train_data, cache_device=retrieval_cache_device)
        else:
            self.rt.prepare_dataset(train_data, cache_device=retrieval_cache_device)

            print("Doing Train Retrieval")
            train_rt = self.rt.retrieve_all(train_data, train=True, device=self.device, return_texts=False)

            print("Doing Valid Retrieval")
            valid_rt = self.rt.retrieve_all(valid_data, train=False, device=self.device, return_texts=False)

            print("Doing Test Retrieval")
            test_rt = self.rt.retrieve_all(test_data, train=False, device=self.device, return_texts=False)

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

    def _get_text_feature(self, index, mode, bsz, meta_text=None):
        return torch.zeros((bsz, 0), device=self.device)

    def _get_text_feature_by_period(self, index, mode, bsz, meta_text=None, meta_text_by_period=None):
        return torch.zeros((len(self.period_num), bsz, 0), device=self.device)

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

    def encoder(self, x, index, mode, meta_data=None, meta_text=None, meta_text_by_period=None):
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
        query_signal = x_pred_from_x.mean(dim=2)  # B, P
        query_token = self.period_query_proj(query_signal).unsqueeze(1)  # B, 1, D

        period_signal = pred_stack.mean(dim=3)  # B, G, P
        key_token = self.period_key_proj(period_signal)  # B, G, D
        local_state_feature = self._extract_local_state_feature(meta_data, bsz, x.device)  # B, G, 4
        key_token = key_token + self.period_meta_proj(local_state_feature)  # B, G, D

        period_logits = torch.matmul(query_token, key_token.transpose(1, 2)).squeeze(1) * self.period_attn_scale  # B, G
        period_weight = torch.softmax(period_logits, dim=1)
        period_weight = self.period_attn_dropout(period_weight)
        period_weight = period_weight / period_weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
        retrieval_agg = (pred_stack * period_weight.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)  # B, P, C

        # Fuse once: aggregated multi-scale retrieval + base prediction.
        retrieval_feature = retrieval_agg.permute(0, 2, 1)  # B, C, P
        fusion_feature = torch.cat([numeric_feature, retrieval_feature], dim=2)  # B, C, 2P
        pred = self.linear_pred(fusion_feature).permute(0, 2, 1).reshape(bsz, self.pred_len, self.channels)
        pred = pred + x_offset
        return pred

    def forecast(self, x_enc, index, mode, meta_data=None, meta_text=None, meta_text_by_period=None):
        return self.encoder(
            x_enc,
            index,
            mode,
            meta_data=meta_data,
            meta_text=meta_text,
            meta_text_by_period=meta_text_by_period,
        )

    def imputation(self, x_enc, index, mode, meta_data=None, meta_text=None, meta_text_by_period=None):
        return self.encoder(
            x_enc,
            index,
            mode,
            meta_data=meta_data,
            meta_text=meta_text,
            meta_text_by_period=meta_text_by_period,
        )

    def anomaly_detection(self, x_enc, index, mode, meta_data=None, meta_text=None, meta_text_by_period=None):
        return self.encoder(
            x_enc,
            index,
            mode,
            meta_data=meta_data,
            meta_text=meta_text,
            meta_text_by_period=meta_text_by_period,
        )

    def classification(self, x_enc, index, mode, meta_data=None, meta_text=None, meta_text_by_period=None):
        enc_out = self.encoder(
            x_enc,
            index,
            mode,
            meta_data=meta_data,
            meta_text=meta_text,
            meta_text_by_period=meta_text_by_period,
        )
        output = enc_out.reshape(enc_out.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, index, mode="train", meta_data=None, meta_text=None, meta_text_by_period=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            dec_out = self.forecast(
                x_enc,
                index,
                mode,
                meta_data=meta_data,
                meta_text=meta_text,
                meta_text_by_period=meta_text_by_period,
            )
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == "imputation":
            return self.imputation(
                x_enc,
                index,
                mode,
                meta_data=meta_data,
                meta_text=meta_text,
                meta_text_by_period=meta_text_by_period,
            )
        if self.task_name == "anomaly_detection":
            return self.anomaly_detection(
                x_enc,
                index,
                mode,
                meta_data=meta_data,
                meta_text=meta_text,
                meta_text_by_period=meta_text_by_period,
            )
        if self.task_name == "classification":
            return self.classification(
                x_enc,
                index,
                mode,
                meta_data=meta_data,
                meta_text=meta_text,
                meta_text_by_period=meta_text_by_period,
            )
        return None
