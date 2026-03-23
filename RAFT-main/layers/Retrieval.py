import copy
import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class HierarchicalContextEncoder(nn.Module):
    def __init__(self, period_count: int, context_dim: int = 64, cat_dim: int = 16):
        super().__init__()
        self.period_count = period_count

        # Layer-1 and Layer-2 categorical embeddings.
        self.dataset_emb = nn.Embedding(2048, cat_dim)
        self.sensor_emb = nn.Embedding(256, cat_dim)
        self.location_emb = nn.Embedding(4096, cat_dim)
        self.hour_emb = nn.Embedding(24, cat_dim)
        self.weekday_emb = nn.Embedding(7, cat_dim)
        self.month_emb = nn.Embedding(13, cat_dim)
        self.holiday_emb = nn.Embedding(2, cat_dim)
        self.peak_emb = nn.Embedding(2, cat_dim)

        cat_total_dim = cat_dim * 8
        self.cat_proj = nn.Sequential(
            nn.Linear(cat_total_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )

        # Layer-3 numerical projection (mean, std, skewness, slope).
        self.local_proj = nn.Sequential(
            nn.Linear(4, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )
        self.norm = nn.LayerNorm(context_dim)

    def forward(self, meta_batch: Dict[str, torch.Tensor]):
        dataset = meta_batch["dataset_id"].long()
        sensor = meta_batch["sensor_type_id"].long()
        location = meta_batch["physical_location_id"].long()
        hour = meta_batch["hour"].long().clamp(0, 23)
        weekday = meta_batch["day_of_week"].long().clamp(0, 6)
        month = meta_batch["month"].long().clamp(1, 12)
        holiday = meta_batch["is_holiday"].long().clamp(0, 1)
        peak = meta_batch["peak_status_id"].long().clamp(0, 1)

        cat_context = torch.cat(
            [
                self.dataset_emb(dataset),
                self.sensor_emb(sensor),
                self.location_emb(location),
                self.hour_emb(hour),
                self.weekday_emb(weekday),
                self.month_emb(month),
                self.holiday_emb(holiday),
                self.peak_emb(peak),
            ],
            dim=1,
        )
        cat_context = self.cat_proj(cat_context)  # [B, D]

        local_state = meta_batch["local_state_by_period"].float()  # [B, G, 4]
        local_state = local_state[:, : self.period_count, :]
        bsz, gsz, _ = local_state.shape

        local_context = self.local_proj(local_state.reshape(-1, 4)).reshape(bsz, gsz, -1)  # [B, G, D]
        context = self.norm(cat_context.unsqueeze(1) + local_context)
        return context.permute(1, 0, 2)  # [G, B, D]


class RetrievalTool(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        n_period=3,
        temperature=0.1,
        topm=20,
        alpha=0.7,
        coarse_k=80,
        context_dim=64,
        gate_hidden_dim=128,
        use_gated_aggregation=True,
        with_dec=False,
        return_key=False,
        learnable_alpha=False,
        train_context_encoder=True,
    ):
        super().__init__()
        base_periods = [4, 2, 1]
        n_period = max(1, min(n_period, len(base_periods)))

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels

        self.n_period = n_period
        self.period_num = base_periods[:n_period]

        self.temperature = temperature
        self.topm = topm
        self.coarse_k = max(topm, coarse_k)
        self.train_context_encoder = bool(train_context_encoder)
        self.context_dim = int(context_dim)
        self.use_gated_aggregation = bool(use_gated_aggregation)

        self.learnable_alpha = learnable_alpha
        if self.learnable_alpha:
            # Sigmoid(alpha_logit) in (0, 1).
            alpha = min(max(alpha, 1e-4), 1.0 - 1e-4)
            self.alpha_logit = nn.Parameter(torch.logit(torch.tensor(float(alpha))))
        else:
            self.register_buffer("alpha_const", torch.tensor(float(alpha)))

        self.with_dec = with_dec
        self.return_key = return_key

        self.context_encoder = HierarchicalContextEncoder(
            period_count=self.n_period,
            context_dim=context_dim,
            cat_dim=16,
        )
        gate_in_dim = self.context_dim * 2 + 2
        self.candidate_gate = nn.Sequential(
            nn.Linear(gate_in_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 1),
        )
        if self.train_context_encoder:
            self.context_encoder.train()
        else:
            self.context_encoder.eval()

        self.local_state_mean = None
        self.local_state_std = None
        self.meta_pool_context = None
        self.train_meta_all = None
        self.low_mem_stream = False
        self.stream_batch_size = 512
        self.train_data_all = None
        self.train_data_all_mg = None
        self.y_data_all = None
        self.y_data_all_mg = None
        self.train_series_x = None
        self.train_series_y = None

    @staticmethod
    def _to_tensor(value, dtype=torch.float32):
        if torch.is_tensor(value):
            return value.to(dtype=dtype)
        if isinstance(value, np.ndarray):
            return torch.tensor(value, dtype=dtype)
        if isinstance(value, list):
            return torch.tensor(value, dtype=dtype)
        return torch.tensor(value, dtype=dtype)

    def _meta_from_records(self, records):
        bsz = len(records)
        local_state_list = []
        packed = {
            "dataset_id": [],
            "sensor_type_id": [],
            "physical_location_id": [],
            "hour": [],
            "day_of_week": [],
            "month": [],
            "is_holiday": [],
            "peak_status_id": [],
        }

        for meta in records:
            meta = meta or {}
            packed["dataset_id"].append(int(meta.get("dataset_id", 0)))
            packed["sensor_type_id"].append(int(meta.get("sensor_type_id", 0)))
            packed["physical_location_id"].append(int(meta.get("physical_location_id", 0)))
            packed["hour"].append(int(meta.get("hour", 0)))
            packed["day_of_week"].append(int(meta.get("day_of_week", 0)))
            packed["month"].append(int(meta.get("month", 1)))
            packed["is_holiday"].append(int(meta.get("is_holiday", 0)))
            packed["peak_status_id"].append(int(meta.get("peak_status_id", 0)))

            local_state = meta.get("local_state_by_period", np.zeros((3, 4), dtype=np.float32))
            local_state = np.asarray(local_state, dtype=np.float32)
            if local_state.ndim == 1:
                local_state = local_state.reshape(1, -1)
            if local_state.shape[0] < 3:
                pad = np.zeros((3 - local_state.shape[0], local_state.shape[1]), dtype=np.float32)
                local_state = np.concatenate([local_state, pad], axis=0)
            local_state_list.append(local_state[:3, :4])

        out = {k: torch.tensor(v, dtype=torch.long) for k, v in packed.items()}
        out["local_state_by_period"] = torch.tensor(np.stack(local_state_list, axis=0), dtype=torch.float32)
        return out

    def _meta_from_batch(self, meta_data, bsz):
        if meta_data is None:
            empty_records = [{} for _ in range(bsz)]
            return self._meta_from_records(empty_records)

        if isinstance(meta_data, dict):
            out = {}
            for key in [
                "dataset_id",
                "sensor_type_id",
                "physical_location_id",
                "hour",
                "day_of_week",
                "month",
                "is_holiday",
                "peak_status_id",
            ]:
                value = meta_data.get(key, 0)
                value = self._to_tensor(value, dtype=torch.float32).long()
                if value.dim() == 0:
                    value = value.unsqueeze(0)
                if value.shape[0] == 1 and bsz > 1:
                    value = value.repeat(bsz)
                if value.shape[0] > bsz:
                    value = value[:bsz]
                if value.shape[0] < bsz:
                    pad = torch.zeros((bsz - value.shape[0],), dtype=value.dtype)
                    value = torch.cat([value, pad], dim=0)
                out[key] = value

            local_state = meta_data.get("local_state_by_period", torch.zeros((bsz, 3, 4)))
            local_state = self._to_tensor(local_state, dtype=torch.float32)
            if local_state.dim() == 2:
                local_state = local_state.unsqueeze(0)
            if local_state.shape[0] == 1 and bsz > 1:
                local_state = local_state.repeat(bsz, 1, 1)
            if local_state.shape[0] > bsz:
                local_state = local_state[:bsz]
            if local_state.shape[0] < bsz:
                pad = torch.zeros((bsz - local_state.shape[0], local_state.shape[1], local_state.shape[2]))
                local_state = torch.cat([local_state, pad], dim=0)
            out["local_state_by_period"] = local_state[:, :3, :4]
            return out

        if isinstance(meta_data, list):
            return self._meta_from_records(meta_data)

        empty_records = [{} for _ in range(bsz)]
        return self._meta_from_records(empty_records)

    def _normalize_local_state(self, local_state):
        if self.local_state_mean is None or self.local_state_std is None:
            return local_state
        mean = self.local_state_mean.to(local_state.device)
        std = self.local_state_std.to(local_state.device)
        return (local_state - mean) / std

    def _normalize_meta_batch(self, meta_batch, device):
        norm_batch = {}
        for key, value in meta_batch.items():
            if key == "local_state_by_period":
                v = value.float()
                v = self._normalize_local_state(v)
                norm_batch[key] = v.to(device)
            else:
                norm_batch[key] = value.long().to(device)
        return norm_batch

    def _alpha_value(self, device):
        if self.learnable_alpha:
            return torch.sigmoid(self.alpha_logit).to(device)
        return self.alpha_const.to(device)

    def _encode_query_context(self, meta_query, bsz, device, require_grad=False):
        query_meta = self._meta_from_batch(meta_query, bsz)
        query_meta = self._normalize_meta_batch(query_meta, device=device)

        self.context_encoder.to(device)
        if require_grad and self.train_context_encoder:
            return self.context_encoder(query_meta)  # [G, B, D]
        with torch.no_grad():
            return self.context_encoder(query_meta)  # [G, B, D]

    def _gather_candidate_context(self, candidate_idx, device):
        if self.meta_pool_context is None:
            raise RuntimeError("Context pool is empty. Call prepare_dataset() before retrieval.")

        pool_ctx = self.meta_pool_context.to(device)  # [G, T, D]
        gsz, bsz, ksz = candidate_idx.shape
        ctx_dim = pool_ctx.shape[-1]
        pool_expand = pool_ctx.unsqueeze(1).expand(gsz, bsz, -1, -1)  # [G, B, T, D]
        gather_idx = candidate_idx.unsqueeze(-1).expand(-1, -1, -1, ctx_dim)
        return torch.gather(pool_expand, dim=2, index=gather_idx)  # [G, B, K, D]

    def refresh_context_pool(self, device=torch.device("cpu")):
        if self.train_meta_all is None:
            return

        was_training = self.context_encoder.training
        self.context_encoder.to(device)
        self.context_encoder.eval()
        with torch.no_grad():
            train_meta = self._normalize_meta_batch(self.train_meta_all, device=device)
            self.meta_pool_context = self.context_encoder(train_meta).detach().to(device)  # [G, T, D]

        if was_training:
            self.context_encoder.train()

    def prepare_dataset(self, train_data, cache_device=torch.device("cpu")):
        train_data_all = []
        y_data_all = []
        meta_records = []

        # Decide low-memory mode before collecting candidate banks.
        self.low_mem_stream = bool(
            cache_device.type == "cpu" and (self.channels >= 800 or (self.seq_len >= 720 and self.n_period >= 3))
        )
        self.stream_batch_size = 16 if self.low_mem_stream else 512

        for i in range(len(train_data)):
            td = train_data[i]
            if not self.low_mem_stream:
                train_data_all.append(td[1])

                if self.with_dec:
                    y_data_all.append(td[2][-(train_data.pred_len + train_data.label_len) :])
                else:
                    y_data_all.append(td[2][-train_data.pred_len :])

            meta_records.append(td[5] if len(td) > 5 else {})

        self.n_train = len(train_data_all)
        if self.low_mem_stream:
            self.n_train = len(train_data)
        else:
            self.n_train = len(train_data_all)

        if self.low_mem_stream:
            # Keep only original full series and build windows on-the-fly in retrieval.
            self.train_series_x = torch.tensor(np.asarray(train_data.data_x), dtype=torch.float16).to(cache_device)
            self.train_series_y = torch.tensor(np.asarray(train_data.data_y), dtype=torch.float16).to(cache_device)
            self.train_data_all = None
            self.train_data_all_mg = None
            self.y_data_all = None
            self.y_data_all_mg = None
        else:
            self.train_data_all = torch.tensor(np.stack(train_data_all, axis=0)).float().to(cache_device)
            self.train_data_all_mg, _ = self.decompose_mg(self.train_data_all)
            self.y_data_all = torch.tensor(np.stack(y_data_all, axis=0)).float().to(cache_device)
            self.y_data_all_mg, _ = self.decompose_mg(self.y_data_all)
            # Raw banks are not needed after decomposition in cached mode.
            self.train_data_all = None
            self.y_data_all = None
            self.train_series_x = None
            self.train_series_y = None

        self.train_meta_all = self._meta_from_records(meta_records)
        self.train_meta_all = {k: v.to(cache_device) for k, v in self.train_meta_all.items()}
        local_state = self.train_meta_all["local_state_by_period"].float()  # [T, 3, 4]
        self.local_state_mean = local_state.mean(dim=0, keepdim=True)
        self.local_state_std = local_state.std(dim=0, keepdim=True) + 1e-6

        self.refresh_context_pool(device=cache_device)
        if self.train_context_encoder:
            self.context_encoder.train()
        else:
            self.context_encoder.eval()

    def decompose_mg(self, data_all, remove_offset=True):
        mg = []
        for g in self.period_num:
            cur = data_all.unfold(dimension=1, size=g, step=g).mean(dim=-1)
            cur = cur.repeat_interleave(repeats=g, dim=1)
            mg.append(cur)

        mg = torch.stack(mg, dim=0)  # G, T, S, C

        if remove_offset:
            offset = []
            for i, data_p in enumerate(mg):
                cur_offset = data_p[:, -1:, :]
                mg[i] = data_p - cur_offset
                offset.append(cur_offset)
            offset = torch.stack(offset, dim=0)
        else:
            offset = None

        return mg, offset

    def periodic_batch_corr(self, data_all, key, in_bsz=512):
        _, bsz, _ = key.shape
        _, train_len, _ = data_all.shape

        bx = key - torch.mean(key, dim=2, keepdim=True)
        iters = math.ceil(train_len / in_bsz)

        sim = []
        for i in range(iters):
            start_idx = i * in_bsz
            end_idx = min((i + 1) * in_bsz, train_len)

            cur_data = data_all[:, start_idx:end_idx].to(key.device)
            ax = cur_data - torch.mean(cur_data, dim=2, keepdim=True)
            cur_sim = torch.bmm(F.normalize(bx, dim=2), F.normalize(ax, dim=2).transpose(-1, -2))
            sim.append(cur_sim)

        return torch.cat(sim, dim=2)

    def periodic_batch_corr_stream(self, key, in_bsz=32):
        _, bsz, _ = key.shape
        train_len = self.n_train
        bx = key - torch.mean(key, dim=2, keepdim=True)
        iters = math.ceil(train_len / in_bsz)

        sim = []
        for i in range(iters):
            start_idx = i * in_bsz
            end_idx = min((i + 1) * in_bsz, train_len)
            if self.train_series_x is None:
                raise RuntimeError("train_series_x is not initialized in low_mem_stream mode.")

            num = end_idx - start_idx
            cand = torch.arange(start_idx, end_idx, device=self.train_series_x.device).unsqueeze(1)
            off = torch.arange(self.seq_len, device=self.train_series_x.device).unsqueeze(0)
            win_idx = cand + off  # [num, S]
            win_idx_flat = win_idx.reshape(-1)
            cur_data = self.train_series_x.index_select(0, win_idx_flat).reshape(num, self.seq_len, self.channels)
            cur_data = cur_data.to(key.device).float()  # [num, S, C]
            cur_data_mg, _ = self.decompose_mg(cur_data)  # [G, T, S, C]
            ax = cur_data_mg.flatten(start_dim=2)
            ax = ax - torch.mean(ax, dim=2, keepdim=True)
            cur_sim = torch.bmm(F.normalize(bx, dim=2), F.normalize(ax, dim=2).transpose(-1, -2))
            sim.append(cur_sim)

        return torch.cat(sim, dim=2)

    def context_similarity(self, query_ctx, candidate_ctx):
        query_norm = F.normalize(query_ctx, dim=-1).unsqueeze(2)  # [G, B, 1, D]
        cand_norm = F.normalize(candidate_ctx, dim=-1)  # [G, B, K, D]
        return (query_norm * cand_norm).sum(dim=-1)  # [G, B, K]

    def retrieve(self, x, index, meta_query=None, train=True):
        index = index.to(x.device)

        bsz, seq_len, channels = x.shape
        assert seq_len == self.seq_len and channels == self.channels

        # Numerical branch with offset removal.
        x_mg, _ = self.decompose_mg(x)  # [G, B, S, C]
        x_key = x_mg.flatten(start_dim=2)  # [G, B, S*C]
        if self.train_data_all_mg is not None:
            sim_wave = self.periodic_batch_corr(
                self.train_data_all_mg.flatten(start_dim=2),  # [G, T, S*C]
                x_key,
            )  # [G, B, T]
        else:
            sim_wave = self.periodic_batch_corr_stream(
                x_key,
                in_bsz=self.stream_batch_size,
            )  # [G, B, T]

        if train:
            sliding_index = torch.arange(2 * (self.seq_len + self.pred_len) - 1).to(x.device)
            sliding_index = sliding_index.unsqueeze(0).repeat(len(index), 1)
            sliding_index = sliding_index + (index - self.seq_len - self.pred_len + 1).unsqueeze(1)
            sliding_index = torch.where(sliding_index >= 0, sliding_index, 0)
            sliding_index = torch.where(sliding_index < self.n_train, sliding_index, self.n_train - 1)

            self_mask = torch.zeros((bsz, self.n_train), device=x.device)
            self_mask = self_mask.scatter_(1, sliding_index, 1.0)
            self_mask = self_mask.unsqueeze(0).repeat(self.n_period, 1, 1)
            sim_wave = sim_wave.masked_fill_(self_mask.bool(), float("-inf"))

        # Stage-1 coarse retrieval by waveform similarity.
        coarse_k = min(self.coarse_k, self.n_train)
        coarse_wave, coarse_idx = torch.topk(sim_wave, coarse_k, dim=2)

        # Stage-2 contextual reranking inside waveform coarse candidates only.
        require_grad = train and self.train_context_encoder and torch.is_grad_enabled()
        query_ctx = self._encode_query_context(meta_query, bsz, x.device, require_grad=require_grad)  # [G, B, D]
        coarse_candidate_ctx = self._gather_candidate_context(coarse_idx, x.device)  # [G, B, K, D]
        coarse_context = self.context_similarity(query_ctx, coarse_candidate_ctx)  # [G, B, K]

        select_k = min(self.topm, coarse_k)
        selected_context, selected_local_idx = torch.topk(coarse_context, select_k, dim=2)
        selected_global_idx = torch.gather(coarse_idx, dim=2, index=selected_local_idx)
        selected_wave = torch.gather(coarse_wave, dim=2, index=selected_local_idx)
        selected_ctx = torch.gather(
            coarse_candidate_ctx,
            dim=2,
            index=selected_local_idx.unsqueeze(-1).expand(-1, -1, -1, coarse_candidate_ctx.shape[-1]),
        )  # [G, B, m, D]

        if self.use_gated_aggregation:
            query_expand = query_ctx.unsqueeze(2).expand(-1, -1, select_k, -1)  # [G, B, m, D]
            gate_input = torch.cat(
                [
                    query_expand,
                    selected_ctx,
                    selected_wave.unsqueeze(-1),
                    selected_context.unsqueeze(-1),
                ],
                dim=-1,
            )  # [G, B, m, 2D+2]
            gate_logits = self.candidate_gate(gate_input).squeeze(-1)  # [G, B, m]
            ranking_prob = F.softmax(gate_logits, dim=2)
        else:
            ranking_prob = F.softmax(selected_context / self.temperature, dim=2)

        if self.y_data_all_mg is not None:
            y_data_all = self.y_data_all_mg.flatten(start_dim=2).to(x.device)  # [G, T, P*C]
            y_expand = y_data_all.unsqueeze(1).expand(-1, bsz, -1, -1)  # [G, B, T, P*C]
            gather_idx = selected_global_idx.unsqueeze(-1).expand(-1, -1, -1, y_data_all.shape[-1])
            selected_y = torch.gather(y_expand, dim=2, index=gather_idx)  # [G, B, m, P*C]
            pred_from_retrieval = (ranking_prob.unsqueeze(-1) * selected_y).sum(dim=2)
            pred_from_retrieval = pred_from_retrieval.reshape(self.n_period, bsz, -1, channels)
        else:
            if self.train_series_y is None:
                raise RuntimeError("train_series_y is not initialized in low_mem_stream mode.")
            retrieval_list = []
            y_off = torch.arange(self.pred_len, device=self.train_series_y.device).unsqueeze(0)  # [1, P]
            for g_i, g in enumerate(self.period_num):
                cand_idx = selected_global_idx[g_i].reshape(-1).to(self.train_series_y.device).unsqueeze(1)  # [B*m, 1]
                y_idx = cand_idx + self.seq_len + y_off  # [B*m, P]
                y_idx_flat = y_idx.reshape(-1)
                selected_raw = self.train_series_y.index_select(0, y_idx_flat).reshape(
                    bsz * select_k, self.pred_len, channels
                ).to(x.device).float()  # [B*m, P, C]
                cur = selected_raw.unfold(dimension=1, size=g, step=g).mean(dim=-1)
                cur = cur.repeat_interleave(repeats=g, dim=1)
                cur = cur - cur[:, -1:, :]
                cur = cur.reshape(bsz, select_k, self.pred_len, channels)  # [B, m, P, C]
                weighted = (ranking_prob[g_i].unsqueeze(-1).unsqueeze(-1) * cur).sum(dim=1)  # [B, P, C]
                retrieval_list.append(weighted)
            pred_from_retrieval = torch.stack(retrieval_list, dim=0)  # [G, B, P, C]
        return pred_from_retrieval

    def retrieve_all(self, data, train=False, device=torch.device("cpu"), return_texts=False):
        assert (self.train_data_all_mg is not None) or (self.train_series_x is not None)

        rt_loader = DataLoader(
            data,
            batch_size=1024,
            shuffle=False,
            num_workers=0 if self.low_mem_stream else 8,
            drop_last=False,
        )

        retrievals = []
        all_texts = []
        with torch.no_grad():
            for batch in tqdm(rt_loader):
                index, batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:5]
                meta_data = batch[5] if len(batch) > 5 else None
                meta_text = batch[6] if len(batch) > 6 else None

                pred_from_retrieval = self.retrieve(
                    batch_x.float().to(device),
                    index,
                    meta_query=meta_data,
                    train=train,
                )
                retrievals.append(pred_from_retrieval.cpu())

                if return_texts:
                    if meta_text is None:
                        all_texts.extend([""] * batch_x.shape[0])
                    else:
                        all_texts.extend(list(meta_text))

        retrievals = torch.cat(retrievals, dim=1)
        if return_texts:
            return retrievals, all_texts
        return retrievals
