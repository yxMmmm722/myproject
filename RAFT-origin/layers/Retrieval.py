import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class RetrievalTool(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        n_period=3,
        temperature=0.1,
        topm=20,
        with_dec=False,
        return_key=False,
        meta_only_retrieval=False,
    ):
        super().__init__()

        base_periods = [4, 2, 1]
        n_period = max(1, min(int(n_period), len(base_periods)))

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels

        self.n_period = n_period
        self.period_num = base_periods[:n_period]

        self.temperature = temperature
        self.topm = topm
        self.with_dec = with_dec
        self.return_key = return_key
        self.meta_only_retrieval = bool(meta_only_retrieval)

        self.train_data_all_mg = None
        self.train_channel_state_mg = None
        self.train_time_key = None
        self.y_data_all_mg = None
        self.n_train = 0

    def prepare_dataset(self, train_data):
        train_data_all = []
        y_data_all = []
        train_mark_all = []

        for i in range(len(train_data)):
            td = train_data[i]
            train_data_all.append(td[1])
            train_mark_all.append(td[3])
            if self.with_dec:
                y_data_all.append(td[2][-(train_data.pred_len + train_data.label_len):])
            else:
                y_data_all.append(td[2][-train_data.pred_len:])

        train_data_all = torch.tensor(np.stack(train_data_all, axis=0)).float()
        self.train_data_all_mg, _ = self.decompose_mg(train_data_all)
        self.train_channel_state_mg = self._extract_channel_state(self.train_data_all_mg)
        train_mark_all = torch.tensor(np.stack(train_mark_all, axis=0)).float()
        self.train_time_key = self._extract_time_key(train_mark_all)

        y_data_all = torch.tensor(np.stack(y_data_all, axis=0)).float()
        self.y_data_all_mg, _ = self.decompose_mg(y_data_all)

        self.n_train = int(self.train_data_all_mg.shape[1])

    def decompose_mg(self, data_all, remove_offset=True):
        mg = []
        for g in self.period_num:
            cur = data_all.unfold(dimension=1, size=g, step=g).mean(dim=-1)
            cur = cur.repeat_interleave(repeats=g, dim=1)
            mg.append(cur)

        mg = torch.stack(mg, dim=0)  # G, N, S, C

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

    def _extract_channel_state(self, mg):
        # mg: [G, N, S, C] -> state: [G, N, C, 4]
        mean = mg.mean(dim=2)
        std = mg.std(dim=2, unbiased=False)
        if mg.shape[2] > 1:
            slope = (mg[:, :, -1, :] - mg[:, :, 0, :]) / float(mg.shape[2] - 1)
            abs_diff = torch.mean(torch.abs(mg[:, :, 1:, :] - mg[:, :, :-1, :]), dim=2)
        else:
            slope = torch.zeros_like(mean)
            abs_diff = torch.zeros_like(mean)
        return torch.stack([mean, std, slope, abs_diff], dim=-1)

    def _extract_time_key(self, mark):
        if mark is None:
            return None
        if not torch.is_tensor(mark):
            mark = torch.tensor(mark)
        mark = mark.float()
        if mark.dim() == 3:
            return mark[:, -1, :]
        if mark.dim() == 2:
            return mark
        raise ValueError(f"Unsupported mark shape for time key extraction: {tuple(mark.shape)}")

    def _meta_channel_similarity(self, query_channel_state, self_mask=None):
        # query_channel_state: [G, B, C, 4] -> similarity: [G, B, C, T]
        q = F.normalize(query_channel_state, dim=-1)
        pool = self.train_channel_state_mg.to(query_channel_state.device)
        p = F.normalize(pool, dim=-1)
        sim = torch.einsum("gbcd,gtcd->gbct", q, p)
        if self_mask is not None:
            sim = sim.masked_fill(self_mask.unsqueeze(2), float("-inf"))
        return sim

    def _build_time_filter_mask(self, meta_query, select_k, device):
        if meta_query is None or self.train_time_key is None or self.n_train <= 0:
            return None

        query_key = self._extract_time_key(meta_query).to(device)
        bank_key = self.train_time_key.to(device)

        horizon_ratio = min(1.0, float(self.pred_len) / max(float(self.seq_len), 1.0))
        keep_multiplier = int(round(8 + 16 * (1.0 - horizon_ratio)))
        min_keep = int(round(256 + 512 * (1.0 - horizon_ratio)))
        keep_k = min(self.n_train, max(select_k * keep_multiplier, min_keep))
        if keep_k >= self.n_train:
            return None

        dist = torch.cdist(query_key, bank_key, p=2)
        keep_idx = torch.topk(dist, k=keep_k, dim=1, largest=False).indices
        mask = torch.ones((query_key.shape[0], self.n_train), device=device, dtype=torch.bool)
        mask.scatter_(1, keep_idx, False)
        return mask.unsqueeze(0).expand(self.n_period, -1, -1)

    def periodic_batch_corr(self, data_all, key, in_bsz=512):
        _, _, _ = key.shape
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

    def _build_train_self_mask(self, index, bsz, device):
        sliding_index = torch.arange(2 * (self.seq_len + self.pred_len) - 1, device=device)
        sliding_index = sliding_index.unsqueeze(0).repeat(len(index), 1)
        sliding_index = sliding_index + (index - self.seq_len - self.pred_len + 1).unsqueeze(1)
        sliding_index = torch.where(sliding_index >= 0, sliding_index, 0)
        sliding_index = torch.where(sliding_index < self.n_train, sliding_index, self.n_train - 1)

        self_mask = torch.zeros((bsz, self.n_train), device=device)
        self_mask = self_mask.scatter_(1, sliding_index, 1.0)
        self_mask = self_mask.unsqueeze(0).repeat(self.n_period, 1, 1)
        return self_mask.bool()

    def _compute_wave_and_meta_topm(self, x, index, meta_query=None, train=True):
        bsz, seq_len, channels = x.shape
        assert seq_len == self.seq_len and channels == self.channels

        select_k = min(self.topm, self.n_train)
        self_mask = None
        if train:
            self_mask = self._build_train_self_mask(index, bsz, x.device)
        time_mask = self._build_time_filter_mask(meta_query, select_k, x.device)
        candidate_mask = self_mask
        if time_mask is not None:
            candidate_mask = time_mask if candidate_mask is None else (candidate_mask | time_mask)

        x_mg, _ = self.decompose_mg(x)  # [G, B, S, C]

        x_wave = x_mg.flatten(start_dim=2)  # [G, B, S*C]
        bank_wave = self.train_data_all_mg.flatten(start_dim=2).to(x.device)  # [G, T, S*C]
        sim_wave = self.periodic_batch_corr(bank_wave, x_wave)  # [G, B, T]
        if candidate_mask is not None:
            sim_wave = sim_wave.masked_fill(candidate_mask, float("-inf"))
        wave_score_raw, wave_idx_raw = torch.topk(sim_wave, k=select_k, dim=2)  # [G, B, k]
        wave_idx = wave_idx_raw.unsqueeze(2).expand(-1, -1, channels, -1)  # [G, B, C, k]
        wave_score = wave_score_raw.unsqueeze(2).expand(-1, -1, channels, -1)  # [G, B, C, k]

        query_channel_state = self._extract_channel_state(x_mg)  # [G, B, C, 4]
        meta_sim = self._meta_channel_similarity(query_channel_state, self_mask=candidate_mask)  # [G, B, C, T]
        meta_score, meta_idx = torch.topk(meta_sim, k=select_k, dim=3)  # [G, B, C, k]

        return {
            "select_k": select_k,
            "wave_idx": wave_idx,
            "wave_score": wave_score,
            "meta_idx": meta_idx,
            "meta_score": meta_score,
        }

    def retrieve(self, x, index, meta_query=None, train=True):
        bsz, seq_len, channels = x.shape
        assert seq_len == self.seq_len and channels == self.channels
        index = index.to(x.device)

        out = self._compute_wave_and_meta_topm(x=x, index=index, meta_query=meta_query, train=train)
        selected_idx = out["meta_idx"] if self.meta_only_retrieval else out["wave_idx"]
        base_score = out["meta_score"] if self.meta_only_retrieval else out["wave_score"]
        ranking_prob = F.softmax(base_score / self.temperature, dim=-1)  # [G, B, C, k]

        y_bank = self.y_data_all_mg.to(x.device).float()  # [G, T, P, C]
        retrieval_list = []
        for g_i in range(self.n_period):
            idx_g = selected_idx[g_i]  # [B, C, k]
            prob_g = ranking_prob[g_i]  # [B, C, k]
            pred_g = torch.zeros((bsz, self.pred_len, channels), device=x.device, dtype=y_bank.dtype)
            for c_i in range(channels):
                idx_bc = idx_g[:, c_i, :]  # [B, k]
                cand_c = y_bank[g_i, :, :, c_i][idx_bc]  # [B, k, P]
                w_c = prob_g[:, c_i, :].unsqueeze(-1)  # [B, k, 1]
                pred_g[:, :, c_i] = (w_c * cand_c).sum(dim=1)  # [B, P]
            retrieval_list.append(pred_g)

        pred_from_retrieval = torch.stack(retrieval_list, dim=0)  # [G, B, P, C]
        return pred_from_retrieval

    def _recommend_query_batch_size(self, max_batch):
        max_batch = max(1, int(max_batch))
        if self.n_train <= 0:
            return max_batch
        target_bytes = 512 * 1024 * 1024
        denom = max(1, self.n_period * self.channels * self.n_train * 4 * 2)
        suggested = max(1, int(target_bytes // denom))
        return max(1, min(max_batch, suggested))

    def retrieve_all(self, data, train=False, device=torch.device("cpu")):
        assert self.train_data_all_mg is not None and self.train_channel_state_mg is not None and self.y_data_all_mg is not None

        query_batch_size = self._recommend_query_batch_size(max_batch=1024)
        rt_loader = DataLoader(
            data,
            batch_size=query_batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False,
        )

        retrievals = []
        with torch.no_grad():
            for batch in tqdm(rt_loader):
                index, batch_x = batch[0], batch[1]
                batch_x_mark = batch[3] if len(batch) > 3 else None
                if batch_x_mark is not None:
                    batch_x_mark = batch_x_mark.float().to(device)
                pred_from_retrieval = self.retrieve(batch_x.float().to(device), index, meta_query=batch_x_mark, train=train)
                retrievals.append(pred_from_retrieval.cpu())

        retrievals = torch.cat(retrievals, dim=1)
        return retrievals

    @torch.no_grad()
    def evaluate_wave_meta_retrieval_quality(self, data, device=torch.device("cpu"), train=False):
        query_batch_size = self._recommend_query_batch_size(max_batch=512)
        loader = DataLoader(
            data,
            batch_size=query_batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False,
        )

        wave_sq_sum = 0.0
        wave_abs_sum = 0.0
        meta_sq_sum = 0.0
        meta_abs_sum = 0.0
        n_elem_total = 0

        period_count = self.n_period
        channel_count = self.channels
        wave_sq_by_g = np.zeros((period_count,), dtype=np.float64)
        wave_abs_by_g = np.zeros((period_count,), dtype=np.float64)
        meta_sq_by_g = np.zeros((period_count,), dtype=np.float64)
        meta_abs_by_g = np.zeros((period_count,), dtype=np.float64)
        n_elem_by_g = np.zeros((period_count,), dtype=np.float64)

        wave_sq_by_c = np.zeros((channel_count,), dtype=np.float64)
        wave_abs_by_c = np.zeros((channel_count,), dtype=np.float64)
        meta_sq_by_c = np.zeros((channel_count,), dtype=np.float64)
        meta_abs_by_c = np.zeros((channel_count,), dtype=np.float64)
        n_elem_by_c = np.zeros((channel_count,), dtype=np.float64)

        original_meta_only = bool(self.meta_only_retrieval)

        for batch in tqdm(loader):
            index, batch_x, batch_y = batch[0], batch[1], batch[2]
            batch_x_mark = batch[3] if len(batch) > 3 else None
            batch_x = batch_x.float().to(device)
            if batch_x_mark is not None:
                batch_x_mark = batch_x_mark.float().to(device)
            batch_y_future = batch_y[:, -self.pred_len:, :].float().to(device)
            true_mg, _ = self.decompose_mg(batch_y_future)  # [G, B, P, C]

            self.meta_only_retrieval = False
            wave_pred = self.retrieve(batch_x, index, meta_query=batch_x_mark, train=train)  # [G, B, P, C]

            self.meta_only_retrieval = True
            meta_pred = self.retrieve(batch_x, index, meta_query=batch_x_mark, train=train)  # [G, B, P, C]

            wave_err = wave_pred - true_mg
            meta_err = meta_pred - true_mg

            wave_sq_sum += float((wave_err ** 2).sum().item())
            wave_abs_sum += float(torch.abs(wave_err).sum().item())
            meta_sq_sum += float((meta_err ** 2).sum().item())
            meta_abs_sum += float(torch.abs(meta_err).sum().item())
            n_elem_total += int(wave_err.numel())

            wave_sq_by_c += (wave_err ** 2).sum(dim=(0, 1, 2)).detach().cpu().numpy()
            wave_abs_by_c += torch.abs(wave_err).sum(dim=(0, 1, 2)).detach().cpu().numpy()
            meta_sq_by_c += (meta_err ** 2).sum(dim=(0, 1, 2)).detach().cpu().numpy()
            meta_abs_by_c += torch.abs(meta_err).sum(dim=(0, 1, 2)).detach().cpu().numpy()
            n_elem_by_c += float(wave_err.shape[0] * wave_err.shape[1] * wave_err.shape[2])

            for g_i in range(period_count):
                we = wave_err[g_i]
                me = meta_err[g_i]
                wave_sq_by_g[g_i] += float((we ** 2).sum().item())
                wave_abs_by_g[g_i] += float(torch.abs(we).sum().item())
                meta_sq_by_g[g_i] += float((me ** 2).sum().item())
                meta_abs_by_g[g_i] += float(torch.abs(me).sum().item())
                n_elem_by_g[g_i] += float(we.numel())

        self.meta_only_retrieval = original_meta_only

        if n_elem_total <= 0:
            return None

        wave_mse = wave_sq_sum / n_elem_total
        wave_mae = wave_abs_sum / n_elem_total
        meta_mse = meta_sq_sum / n_elem_total
        meta_mae = meta_abs_sum / n_elem_total

        per_period = []
        for g_i, g in enumerate(self.period_num):
            denom = max(n_elem_by_g[g_i], 1.0)
            per_period.append(
                {
                    "period_idx": int(g_i),
                    "period_g": int(g),
                    "wave_mse": float(wave_sq_by_g[g_i] / denom),
                    "wave_mae": float(wave_abs_by_g[g_i] / denom),
                    "meta_mse": float(meta_sq_by_g[g_i] / denom),
                    "meta_mae": float(meta_abs_by_g[g_i] / denom),
                    "delta_mse_meta_minus_wave": float(meta_sq_by_g[g_i] / denom - wave_sq_by_g[g_i] / denom),
                }
            )

        per_channel = []
        for c_i in range(channel_count):
            denom = max(n_elem_by_c[c_i], 1.0)
            per_channel.append(
                {
                    "channel_idx": int(c_i),
                    "channel_name": f"channel_{c_i}",
                    "wave_mse": float(wave_sq_by_c[c_i] / denom),
                    "wave_mae": float(wave_abs_by_c[c_i] / denom),
                    "meta_mse": float(meta_sq_by_c[c_i] / denom),
                    "meta_mae": float(meta_abs_by_c[c_i] / denom),
                    "delta_mse_meta_minus_wave": float(meta_sq_by_c[c_i] / denom - wave_sq_by_c[c_i] / denom),
                }
            )

        return {
            "split": "eval",
            "wave_mse": float(wave_mse),
            "wave_mae": float(wave_mae),
            "meta_mse": float(meta_mse),
            "meta_mae": float(meta_mae),
            "delta_mse_meta_minus_wave": float(meta_mse - wave_mse),
            "delta_mae_meta_minus_wave": float(meta_mae - wave_mae),
            "per_period": per_period,
            "per_channel": per_channel,
        }
