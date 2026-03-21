import torch
import torch.nn as nn
import numpy as np

from layers.Retrieval import RetrievalTool
from layers.TextEncoder import FrozenTextEncoder
from utils.meta_text_dump import dump_meta_texts


class Model(nn.Module):
    def __init__(self, configs, individual=False):
        super(Model, self).__init__()
        self.device = torch.device(f"cuda:{configs.gpu}" if (configs.use_gpu and torch.cuda.is_available()) else "cpu")
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len if self.task_name in ["classification", "anomaly_detection", "imputation"] else configs.pred_len
        self.channels = configs.enc_in

        self.retrieval_cache_device = getattr(configs, "retrieval_cache_device", "gpu")
        self.text_cache_device = getattr(configs, "text_cache_device", self.retrieval_cache_device)
        self.learnable_alpha = getattr(configs, "learnable_alpha", False)
        self.online_retrieval = getattr(configs, "online_retrieval", False) or self.learnable_alpha
        self.refresh_context_each_epoch = getattr(configs, "refresh_context_each_epoch", True)
        if self.retrieval_cache_device == "gpu" and not torch.cuda.is_available():
            self.retrieval_cache_device = "cpu"
        if self.text_cache_device == "gpu" and not torch.cuda.is_available():
            self.text_cache_device = "cpu"

        self.text_batch_size = getattr(configs, "text_batch_size", 64)
        self.text_proj_dim = getattr(configs, "text_proj_dim", 64)
        self.model_id = getattr(configs, "model_id", "temp")
        self.save_meta_texts = getattr(configs, "save_meta_texts", False)
        self.meta_text_dump_dir = getattr(configs, "meta_text_dump_dir", "./meta_text_dumps")
        self.meta_text_max_samples = getattr(configs, "meta_text_max_samples", 200)

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
            learnable_alpha=getattr(configs, "learnable_alpha", False),
            train_context_encoder=not getattr(configs, "freeze_context_encoder", False),
        )

        self.period_num = self.rt.period_num[-1 * self.n_period:]
        self.retrieval_pred = nn.ModuleList(
            [nn.Linear(self.pred_len // g, self.pred_len) for g in self.period_num]
        )

        self.text_encoder = FrozenTextEncoder(
            model_name=getattr(configs, "text_encoder_name", "bert-base-uncased"),
            max_length=getattr(configs, "text_max_len", 32),
            require_transformer=getattr(configs, "require_text_encoder", False),
        )
        self.text_proj = nn.Linear(self.text_encoder.output_dim, self.text_proj_dim)
        self.linear_pred = nn.Linear(2 * self.pred_len + self.text_proj_dim, self.pred_len)
        self.linear_pred_per_period = nn.ModuleList(
            [nn.Linear(2 * self.pred_len + self.text_proj_dim, self.pred_len) for _ in self.period_num]
        )

    def _resolve_cache_device(self, cache_mode):
        if cache_mode == "gpu":
            return self.device
        return torch.device("cpu")

    def _encode_texts(self, texts):
        if len(texts) == 0:
            return torch.zeros((0, self.text_encoder.output_dim))

        unique_texts = list(dict.fromkeys(texts))
        unique_emb = self.text_encoder.encode(
            unique_texts,
            device=self.device,
            batch_size=self.text_batch_size,
        )
        text_to_emb = {text: emb for text, emb in zip(unique_texts, unique_emb)}
        ordered_emb = torch.stack([text_to_emb[text] for text in texts], dim=0)
        return ordered_emb

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
        if len(texts_by_period) == 0:
            return torch.zeros((len(self.period_num), 0, self.text_encoder.output_dim))

        period_embeds = []
        for g_idx in range(len(self.period_num)):
            cur_texts = [sample[g_idx] for sample in texts_by_period]
            cur_emb = self._encode_texts(cur_texts)  # [N, D]
            period_embeds.append(cur_emb)
        return torch.stack(period_embeds, dim=0)  # [G, N, D]

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
        self.text_feature_dict = {}
        self.text_feature_by_period_dict = {}
        retrieval_cache_device = self._resolve_cache_device(self.retrieval_cache_device)
        text_cache_device = self._resolve_cache_device(self.text_cache_device)

        if self.online_retrieval:
            print("Preparing Online Retrieval Bank")
            self.rt.prepare_dataset(train_data, cache_device=retrieval_cache_device)
            train_texts, train_texts_by_period = self._collect_texts(train_data)
            valid_texts, valid_texts_by_period = self._collect_texts(valid_data)
            test_texts, test_texts_by_period = self._collect_texts(test_data)
        else:
            self.rt.prepare_dataset(train_data, cache_device=retrieval_cache_device)

            print("Doing Train Retrieval")
            train_rt = self.rt.retrieve_all(train_data, train=True, device=self.device, return_texts=False)

            print("Doing Valid Retrieval")
            valid_rt = self.rt.retrieve_all(valid_data, train=False, device=self.device, return_texts=False)

            print("Doing Test Retrieval")
            test_rt = self.rt.retrieve_all(test_data, train=False, device=self.device, return_texts=False)

            train_texts, train_texts_by_period = self._collect_texts(train_data)
            valid_texts, valid_texts_by_period = self._collect_texts(valid_data)
            test_texts, test_texts_by_period = self._collect_texts(test_data)

        if not self.online_retrieval:
            self.retrieval_dict["train"] = train_rt.detach().to(retrieval_cache_device)
            self.retrieval_dict["valid"] = valid_rt.detach().to(retrieval_cache_device)
            self.retrieval_dict["test"] = test_rt.detach().to(retrieval_cache_device)

        self.text_feature_dict["train"] = self._encode_texts(train_texts).detach().to(text_cache_device)
        self.text_feature_dict["valid"] = self._encode_texts(valid_texts).detach().to(text_cache_device)
        self.text_feature_dict["test"] = self._encode_texts(test_texts).detach().to(text_cache_device)
        self.text_feature_by_period_dict["train"] = self._encode_texts_by_period(train_texts_by_period).detach().to(text_cache_device)
        self.text_feature_by_period_dict["valid"] = self._encode_texts_by_period(valid_texts_by_period).detach().to(text_cache_device)
        self.text_feature_by_period_dict["test"] = self._encode_texts_by_period(test_texts_by_period).detach().to(text_cache_device)

        if self.save_meta_texts:
            dataset_name = getattr(train_data, "dataset_name", "unknown_dataset")
            prompt_factory = getattr(train_data, "prompt_factory", None)
            template_catalog = prompt_factory.get_template_catalog() if prompt_factory is not None else {}
            json_path, txt_path = dump_meta_texts(
                dataset_name=dataset_name,
                model_id=self.model_id,
                output_dir=self.meta_text_dump_dir,
                split_texts={
                    "train": train_texts,
                    "valid": valid_texts,
                    "test": test_texts,
                },
                split_texts_by_period={
                    "train": train_texts_by_period,
                    "valid": valid_texts_by_period,
                    "test": test_texts_by_period,
                },
                max_samples=self.meta_text_max_samples,
                template_catalog=template_catalog,
            )
            print(f"Meta texts dumped to: {json_path} and {txt_path}")

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

    def _get_text_feature(self, index, mode, bsz, meta_text=None):
        if mode in self.text_feature_dict:
            text_bank = self.text_feature_dict[mode]
            text_index = index.long().to(text_bank.device)
            if text_index.numel() > 0:
                max_idx = int(text_index.max().item())
                min_idx = int(text_index.min().item())
                if min_idx >= 0 and max_idx < text_bank.shape[0]:
                    return text_bank[text_index].to(self.device)
            # Fallback: if index range mismatches cache split, use current batch texts.
            if meta_text is not None:
                if isinstance(meta_text, (list, tuple)):
                    return self._encode_texts([str(t) for t in meta_text]).to(self.device)
                return self._encode_texts([str(meta_text) for _ in range(bsz)]).to(self.device)

        if meta_text is not None:
            if isinstance(meta_text, (list, tuple)):
                return self._encode_texts([str(t) for t in meta_text]).to(self.device)
            return self._encode_texts([str(meta_text) for _ in range(bsz)]).to(self.device)

        return torch.zeros((bsz, self.text_encoder.output_dim), device=self.device)

    def _get_text_feature_by_period(self, index, mode, bsz, meta_text=None, meta_text_by_period=None):
        if mode in self.text_feature_by_period_dict:
            text_bank = self.text_feature_by_period_dict[mode]  # [G, N, D]
            text_index = index.long().to(text_bank.device)
            if text_index.numel() > 0:
                max_idx = int(text_index.max().item())
                min_idx = int(text_index.min().item())
                if min_idx >= 0 and max_idx < text_bank.shape[1]:
                    return text_bank[:, text_index, :].to(self.device)  # [G, B, D]
            # Fallback to on-the-fly encode if split mode/index mismatches.
            period_text_batch = self._normalize_period_text_batch(meta_text_by_period, bsz, fallback_text=None)
            return self._encode_texts_by_period(period_text_batch).to(self.device)

        fallback_text = None
        if meta_text is not None:
            if isinstance(meta_text, (list, tuple)):
                fallback_text = [str(t) for t in meta_text]
            else:
                fallback_text = [str(meta_text) for _ in range(bsz)]
            if len(fallback_text) < bsz:
                fill = fallback_text[-1] if len(fallback_text) > 0 else ""
                fallback_text += [fill for _ in range(bsz - len(fallback_text))]
            fallback_text = fallback_text[:bsz]

        period_text_batch = self._normalize_period_text_batch(meta_text_by_period, bsz, fallback_text=fallback_text)
        return self._encode_texts_by_period(period_text_batch).to(self.device)

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

        text_feature_by_period = self._get_text_feature_by_period(
            index,
            mode,
            bsz,
            meta_text=meta_text,
            meta_text_by_period=meta_text_by_period,
        )  # [G, B, D]
        text_feature_by_period = self.text_proj(
            text_feature_by_period.reshape(-1, text_feature_by_period.shape[-1])
        ).reshape(len(self.period_num), bsz, self.text_proj_dim)  # [G, B, D_t]

        for i, pr in enumerate(pred_from_retrieval):
            assert pr.shape == (bsz, self.pred_len, channels)
            g = self.period_num[i]
            pr = pr.reshape(bsz, self.pred_len // g, g, channels)
            pr = pr[:, :, 0, :]
            pr = self.retrieval_pred[i](pr.permute(0, 2, 1)).permute(0, 2, 1)
            pr = pr.reshape(bsz, self.pred_len, self.channels)  # B, P, C

            retrieval_feature = pr.permute(0, 2, 1)  # B, C, P
            text_feature_i = text_feature_by_period[i].unsqueeze(1).expand(-1, self.channels, -1)  # B, C, D_t
            fusion_feature = torch.cat([numeric_feature, retrieval_feature, text_feature_i], dim=2)  # B, C, 2P + D_t
            pred_i = self.linear_pred_per_period[i](fusion_feature).permute(0, 2, 1).reshape(
                bsz, self.pred_len, self.channels
            )
            retrieval_pred_list.append(pred_i)

        pred = torch.stack(retrieval_pred_list, dim=1).mean(dim=1)  # B, P, C
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
