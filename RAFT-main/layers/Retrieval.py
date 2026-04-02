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
        compare_retrieval_topk=False,
        compare_log_interval=100,
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
        self.meta_only_retrieval = bool(meta_only_retrieval)
        self.compare_retrieval_topk = bool(compare_retrieval_topk)
        self.compare_log_interval = int(compare_log_interval)
        self.with_dec = with_dec
        self.return_key = return_key

        self.low_mem_stream = False
        self.stream_batch_size = 512
        self.train_data_all = None
        self.train_data_all_mg = None
        self.train_channel_state_mg = None
        self.y_data_all = None
        self.y_data_all_mg = None
        self.train_series_x = None
        self.train_series_y = None
        self.reset_retrieval_compare_stats()

    def prepare_dataset(self, train_data, cache_device=torch.device("cpu")):
        train_data_all = []
        y_data_all = []

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
            self.train_channel_state_mg = None
            self.y_data_all = None
            self.y_data_all_mg = None
        else:
            self.train_data_all = torch.tensor(np.stack(train_data_all, axis=0)).float().to(cache_device)
            self.train_data_all_mg, _ = self.decompose_mg(self.train_data_all)
            self.train_channel_state_mg = self._extract_channel_state(self.train_data_all_mg).to(cache_device)
            self.y_data_all = torch.tensor(np.stack(y_data_all, axis=0)).float().to(cache_device)
            self.y_data_all_mg, _ = self.decompose_mg(self.y_data_all)
            # Raw banks are not needed after decomposition in cached mode.
            self.train_data_all = None
            self.y_data_all = None
            self.train_series_x = None
            self.train_series_y = None

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

    def _meta_channel_similarity(self, query_channel_state, self_mask=None, in_bsz=512):
        # query_channel_state: [G, B, C, 4] -> similarity: [G, B, C, T]
        q = F.normalize(query_channel_state, dim=-1)

        if self.train_channel_state_mg is not None:
            pool = self.train_channel_state_mg.to(query_channel_state.device)  # [G, T, C, 4]
            p = F.normalize(pool, dim=-1)
            sim = torch.einsum("gbcd,gtcd->gbct", q, p)
            if self_mask is not None:
                sim = sim.masked_fill(self_mask.unsqueeze(2), float("-inf"))
            return sim

        if self.train_series_x is None:
            raise RuntimeError("Neither cached channel-state bank nor stream series is available.")

        train_len = self.n_train
        iters = math.ceil(train_len / in_bsz)
        sim_parts = []
        for i in range(iters):
            start_idx = i * in_bsz
            end_idx = min((i + 1) * in_bsz, train_len)
            num = end_idx - start_idx

            cand = torch.arange(start_idx, end_idx, device=self.train_series_x.device).unsqueeze(1)
            off = torch.arange(self.seq_len, device=self.train_series_x.device).unsqueeze(0)
            win_idx = cand + off  # [num, S]
            win_idx_flat = win_idx.reshape(-1)
            cur_data = self.train_series_x.index_select(0, win_idx_flat).reshape(num, self.seq_len, self.channels)
            cur_data = cur_data.to(query_channel_state.device).float()  # [num, S, C]
            cur_mg, _ = self.decompose_mg(cur_data)  # [G, num, S, C]
            cur_state = self._extract_channel_state(cur_mg)  # [G, num, C, 4]
            p = F.normalize(cur_state, dim=-1)
            cur_sim = torch.einsum("gbcd,gtcd->gbct", q, p)  # [G, B, C, num]
            sim_parts.append(cur_sim)

        sim = torch.cat(sim_parts, dim=3)
        if self_mask is not None:
            sim = sim.masked_fill(self_mask.unsqueeze(2), float("-inf"))
        return sim

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

    def periodic_batch_corr_channelwise(self, data_all, key, in_bsz=512):
        # data_all: [G, T, S, C], key: [G, B, S, C] -> sim: [G, B, C, T]
        _, bsz, _, _ = key.shape
        _, train_len, _, _ = data_all.shape

        bx = key - torch.mean(key, dim=2, keepdim=True)
        bx = F.normalize(bx, dim=2)
        iters = math.ceil(train_len / in_bsz)

        sim = []
        for i in range(iters):
            start_idx = i * in_bsz
            end_idx = min((i + 1) * in_bsz, train_len)

            cur_data = data_all[:, start_idx:end_idx].to(key.device)
            ax = cur_data - torch.mean(cur_data, dim=2, keepdim=True)
            ax = F.normalize(ax, dim=2)
            cur_sim = torch.einsum("gbsc,gtsc->gbct", bx, ax)
            sim.append(cur_sim)

        return torch.cat(sim, dim=3)

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

    def periodic_batch_corr_stream_channelwise(self, key, in_bsz=32):
        # key: [G, B, S, C] -> sim: [G, B, C, T]
        _, bsz, _, _ = key.shape
        train_len = self.n_train
        bx = key - torch.mean(key, dim=2, keepdim=True)
        bx = F.normalize(bx, dim=2)
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
            ax = cur_data_mg - torch.mean(cur_data_mg, dim=2, keepdim=True)
            ax = F.normalize(ax, dim=2)
            cur_sim = torch.einsum("gbsc,gtsc->gbct", bx, ax)
            sim.append(cur_sim)

        return torch.cat(sim, dim=3)

    def reset_retrieval_compare_stats(self):
        self.compare_stat_calls = 0
        self.compare_overlap_sum = 0.0
        self.compare_exact_set_sum = 0.0
        self.compare_exact_order_sum = 0.0
        self.compare_pair_count = 0
        self.compare_rank_match_sum = np.zeros((self.topm,), dtype=np.float64)
        self.compare_overlap_hist = np.zeros((self.topm + 1,), dtype=np.float64)
        self.compare_first_example = None

    def get_retrieval_compare_stats(self):
        if self.compare_stat_calls <= 0:
            return None
        calls = float(self.compare_stat_calls)
        pair_cnt = max(float(self.compare_pair_count), 1.0)
        return {
            "calls": int(self.compare_stat_calls),
            "overlap_ratio": self.compare_overlap_sum / calls,
            "exact_set_ratio": self.compare_exact_set_sum / calls,
            "exact_order_ratio": self.compare_exact_order_sum / calls,
            "m": int(self.topm),
            "rank_match_ratio": (self.compare_rank_match_sum / pair_cnt).tolist(),
            "overlap_hist_ratio": (self.compare_overlap_hist / pair_cnt).tolist(),
            "first_example": self.compare_first_example,
        }

    def _update_retrieval_compare_stats(self, wave_idx, meta_idx):
        topm = min(self.topm, wave_idx.shape[-1], meta_idx.shape[-1])
        if topm <= 0:
            return
        wave_top = wave_idx[..., :topm]
        meta_top = meta_idx[..., :topm]

        # Flatten all leading dimensions as independent compare pairs.
        wave_flat = wave_top.reshape(-1, topm)  # [N, m]
        meta_flat = meta_top.reshape(-1, topm)  # [N, m]

        overlap_cnt = (wave_flat.unsqueeze(-1) == meta_flat.unsqueeze(-2)).any(dim=-1).float().sum(dim=-1)  # [N]
        overlap_ratio = overlap_cnt / float(topm)
        exact_set = (overlap_cnt == float(topm)).float()
        exact_order = (wave_flat == meta_flat).all(dim=-1).float()
        rank_match = (wave_flat == meta_flat).float().sum(dim=0)  # [m]

        self.compare_stat_calls += 1
        self.compare_overlap_sum += float(overlap_ratio.mean().item())
        self.compare_exact_set_sum += float(exact_set.mean().item())
        self.compare_exact_order_sum += float(exact_order.mean().item())
        self.compare_pair_count += int(wave_flat.shape[0])
        self.compare_rank_match_sum[:topm] += rank_match.detach().cpu().numpy()

        hist = torch.bincount(overlap_cnt.long().reshape(-1), minlength=topm + 1).float().cpu().numpy()
        self.compare_overlap_hist[: topm + 1] += hist
        if self.compare_first_example is None:
            self.compare_first_example = {
                "wave_topm": wave_flat[0].detach().cpu().tolist(),
                "meta_topm": meta_flat[0].detach().cpu().tolist(),
            }

        if self.compare_log_interval > 0 and (self.compare_stat_calls % self.compare_log_interval == 0):
            stats = self.get_retrieval_compare_stats()
            print(
                "[RetrievalCompare] "
                f"calls={stats['calls']} "
                f"overlap@m={stats['overlap_ratio']:.4f} "
                f"exact_set={stats['exact_set_ratio']:.4f} "
                f"exact_order={stats['exact_order_ratio']:.4f}"
            )

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

    def _compute_wave_and_meta_topm(self, x, index, train=True, force_wave=False):
        bsz, seq_len, channels = x.shape
        assert seq_len == self.seq_len and channels == self.channels
        select_k = min(self.topm, self.n_train)

        self_mask = None
        if train:
            self_mask = self._build_train_self_mask(index, bsz, x.device)  # [G, B, T]

        # Build per-channel query meta state for channel-independent meta retrieval.
        x_mg, _ = self.decompose_mg(x)  # [G, B, S, C]
        query_channel_state = self._extract_channel_state(x_mg)  # [G, B, C, 4]

        need_wave = force_wave or (not self.meta_only_retrieval) or self.compare_retrieval_topk
        sim_wave, wave_idx, wave_score = None, None, None
        if need_wave:
            # RAFT-origin style wave retrieval: block-wise similarity on flattened
            # multivariate windows (S*C), i.e. one shared top-m set per sample.
            x_wave = x_mg.flatten(start_dim=2)  # [G, B, S*C]
            if self.train_data_all_mg is not None:
                bank_wave = self.train_data_all_mg.flatten(start_dim=2)  # [G, T, S*C]
                sim_wave = self.periodic_batch_corr(
                    bank_wave,
                    x_wave,
                )  # [G, B, T]
            else:
                sim_wave = self.periodic_batch_corr_stream(
                    x_wave,
                    in_bsz=self.stream_batch_size,
                )  # [G, B, T]
            if self_mask is not None:
                sim_wave = sim_wave.masked_fill(self_mask, float("-inf"))
            wave_score, wave_idx = torch.topk(sim_wave, select_k, dim=2)  # [G, B, k]
            # Keep channel axis for compatibility with current downstream path.
            wave_score = wave_score.unsqueeze(2).expand(-1, -1, channels, -1)  # [G, B, C, k]
            wave_idx = wave_idx.unsqueeze(2).expand(-1, -1, channels, -1)  # [G, B, C, k]

        # Meta-context-only top-m.
        channel_context = self._meta_channel_similarity(
            query_channel_state,
            self_mask=self_mask,
            in_bsz=self.stream_batch_size,
        )  # [G, B, C, T]
        meta_sim = channel_context
        meta_score, meta_idx = torch.topk(meta_sim, select_k, dim=3)  # [G, B, C, k]

        return {
            "select_k": select_k,
            "wave_idx": wave_idx,
            "wave_score": wave_score,
            "meta_idx": meta_idx,
            "meta_score": meta_score,
        }

    def _gather_candidate_hist_future(self, candidate_idx_1d, period_idx, device):
        candidate_idx_1d = candidate_idx_1d.long()
        m = int(candidate_idx_1d.shape[0])
        if m <= 0:
            return torch.empty((0, self.seq_len, self.channels), device=device), torch.empty((0, self.pred_len, self.channels), device=device)

        if self.train_data_all_mg is not None and self.y_data_all_mg is not None:
            hist = self.train_data_all_mg[period_idx, candidate_idx_1d].to(device).float()  # [m, S, C]
            fut = self.y_data_all_mg[period_idx, candidate_idx_1d].to(device).float()  # [m, P, C]
            return hist, fut

        if self.train_series_x is None or self.train_series_y is None:
            raise RuntimeError("Neither cached bank nor stream bank is available for retrieval debug export.")

        cand = candidate_idx_1d.to(self.train_series_x.device).unsqueeze(1)  # [m, 1]

        # History windows.
        off_s = torch.arange(self.seq_len, device=self.train_series_x.device).unsqueeze(0)  # [1, S]
        x_idx = cand + off_s  # [m, S]
        raw_x = self.train_series_x.index_select(0, x_idx.reshape(-1)).reshape(m, self.seq_len, self.channels).to(device).float()
        x_mg, _ = self.decompose_mg(raw_x)
        hist = x_mg[period_idx]  # [m, S, C]

        # Future windows.
        off_p = torch.arange(self.pred_len, device=self.train_series_y.device).unsqueeze(0)  # [1, P]
        y_idx = cand + self.seq_len + off_p  # [m, P]
        raw_y = self.train_series_y.index_select(0, y_idx.reshape(-1)).reshape(m, self.pred_len, self.channels).to(device).float()
        g = self.period_num[period_idx]
        fut = raw_y.unfold(dimension=1, size=g, step=g).mean(dim=-1)
        fut = fut.repeat_interleave(repeats=g, dim=1)
        fut = fut - fut[:, -1:, :]
        return hist, fut

    @torch.no_grad()
    def export_wave_meta_topm_case(
        self,
        x,
        index,
        meta_query=None,
        sample_idx=0,
        period_idx=-1,
        channel_idx=-1,
        train=False,
    ):
        index = index.to(x.device)
        bsz = x.shape[0]
        sample_idx = max(0, min(int(sample_idx), bsz - 1))
        if period_idx < 0:
            period_idx = len(self.period_num) + int(period_idx)
        period_idx = max(0, min(int(period_idx), len(self.period_num) - 1))

        out = self._compute_wave_and_meta_topm(
            x=x,
            index=index,
            train=train,
            force_wave=True,
        )
        wave_idx = out["wave_idx"]
        meta_idx = out["meta_idx"]
        select_k = int(out["select_k"])

        if channel_idx < 0:
            channel_idx = self.channels + int(channel_idx)
        channel_idx = max(0, min(int(channel_idx), self.channels - 1))

        if wave_idx.dim() == 4:
            wave_sel = wave_idx[period_idx, sample_idx, channel_idx, :select_k]
        else:
            wave_sel = wave_idx[period_idx, sample_idx, :select_k]
        if meta_idx.dim() == 4:
            meta_sel = meta_idx[period_idx, sample_idx, channel_idx, :select_k]
        else:
            meta_sel = meta_idx[period_idx, sample_idx, :select_k]
        wave_hist, wave_fut = self._gather_candidate_hist_future(wave_sel, period_idx, x.device)
        meta_hist, meta_fut = self._gather_candidate_hist_future(meta_sel, period_idx, x.device)

        x_mg, _ = self.decompose_mg(x)
        query_hist = x_mg[period_idx, sample_idx, :, channel_idx].detach().cpu().numpy()

        return {
            "period_idx": int(period_idx),
            "period_g": int(self.period_num[period_idx]),
            "channel_idx": int(channel_idx),
            "topm": int(select_k),
            "query_history": query_hist,
            "wave_topm_idx": wave_sel.detach().cpu().numpy(),
            "meta_topm_idx": meta_sel.detach().cpu().numpy(),
            "wave_histories": wave_hist[:, :, channel_idx].detach().cpu().numpy(),
            "meta_histories": meta_hist[:, :, channel_idx].detach().cpu().numpy(),
            "wave_futures": wave_fut[:, :, channel_idx].detach().cpu().numpy(),
            "meta_futures": meta_fut[:, :, channel_idx].detach().cpu().numpy(),
        }

    def retrieve(self, x, index, meta_query=None, train=True):
        bsz, seq_len, channels = x.shape
        assert seq_len == self.seq_len and channels == self.channels
        index = index.to(x.device)

        out = self._compute_wave_and_meta_topm(
            x=x,
            index=index,
            train=train,
            force_wave=False,
        )
        select_k = int(out["select_k"])
        wave_idx = out["wave_idx"]
        wave_score = out["wave_score"]
        meta_idx = out["meta_idx"]
        meta_score = out["meta_score"]

        # Compare waveform-only top-k vs meta-only top-k.
        if self.compare_retrieval_topk and wave_idx is not None:
            self._update_retrieval_compare_stats(wave_idx, meta_idx)

        if self.meta_only_retrieval:
            selected_global_idx = meta_idx
            base_score = meta_score
        else:
            if wave_idx is None:
                raise RuntimeError("Waveform similarity is unavailable for waveform-only retrieval.")
            selected_global_idx = wave_idx
            base_score = wave_score
        ranking_prob = F.softmax(base_score / self.temperature, dim=-1)

        if self.y_data_all_mg is not None:
            y_bank = self.y_data_all_mg.to(x.device).float()  # [G, T, P, C]
            retrieval_list = []
            for g_i in range(self.n_period):
                idx_g = selected_global_idx[g_i]
                prob_g = ranking_prob[g_i]
                if idx_g.dim() == 2:
                    selected_y = y_bank[g_i][idx_g]  # [B, k, P, C]
                    weighted = (prob_g.unsqueeze(-1).unsqueeze(-1) * selected_y).sum(dim=1)  # [B, P, C]
                    retrieval_list.append(weighted)
                    continue

                pred_g = torch.zeros((bsz, self.pred_len, channels), device=x.device, dtype=y_bank.dtype)
                for c_i in range(channels):
                    idx_bc = idx_g[:, c_i, :]  # [B, k]
                    cand_c = y_bank[g_i, :, :, c_i][idx_bc]  # [B, k, P]
                    w_c = prob_g[:, c_i, :].unsqueeze(-1)  # [B, k, 1]
                    pred_g[:, :, c_i] = (w_c * cand_c).sum(dim=1)  # [B, P]
                retrieval_list.append(pred_g)
            pred_from_retrieval = torch.stack(retrieval_list, dim=0)  # [G, B, P, C]
        else:
            if self.train_series_y is None:
                raise RuntimeError("train_series_y is not initialized in low_mem_stream mode.")
            retrieval_list = []
            y_off = torch.arange(self.pred_len, device=self.train_series_y.device).unsqueeze(0)  # [1, P]
            for g_i, g in enumerate(self.period_num):
                idx_g = selected_global_idx[g_i]
                prob_g = ranking_prob[g_i]

                if idx_g.dim() == 2:
                    cand_idx = idx_g.reshape(-1).to(self.train_series_y.device).unsqueeze(1)  # [B*m, 1]
                    y_idx = cand_idx + self.seq_len + y_off  # [B*m, P]
                    y_idx_flat = y_idx.reshape(-1)
                    selected_raw = self.train_series_y.index_select(0, y_idx_flat).reshape(
                        bsz * select_k, self.pred_len, channels
                    ).to(x.device).float()  # [B*m, P, C]
                    cur = selected_raw.unfold(dimension=1, size=g, step=g).mean(dim=-1)
                    cur = cur.repeat_interleave(repeats=g, dim=1)
                    cur = cur - cur[:, -1:, :]
                    cur = cur.reshape(bsz, select_k, self.pred_len, channels)  # [B, m, P, C]
                    weighted = (prob_g.unsqueeze(-1).unsqueeze(-1) * cur).sum(dim=1)  # [B, P, C]
                    retrieval_list.append(weighted)
                    continue

                pred_g = torch.zeros((bsz, self.pred_len, channels), device=x.device, dtype=torch.float32)
                for c_i in range(channels):
                    cand_idx = idx_g[:, c_i, :].reshape(-1).to(self.train_series_y.device).unsqueeze(1)  # [B*m, 1]
                    y_idx = cand_idx + self.seq_len + y_off  # [B*m, P]
                    y_idx_flat = y_idx.reshape(-1)
                    selected_raw_c = self.train_series_y[:, c_i].index_select(0, y_idx_flat).reshape(
                        bsz * select_k, self.pred_len
                    ).to(x.device).float()  # [B*m, P]
                    cur = selected_raw_c.unfold(dimension=1, size=g, step=g).mean(dim=-1)
                    cur = cur.repeat_interleave(repeats=g, dim=1)
                    cur = cur - cur[:, -1:]
                    cur = cur.reshape(bsz, select_k, self.pred_len)  # [B, m, P]
                    weighted_c = (prob_g[:, c_i, :].unsqueeze(-1) * cur).sum(dim=1)  # [B, P]
                    pred_g[:, :, c_i] = weighted_c
                retrieval_list.append(pred_g)
            pred_from_retrieval = torch.stack(retrieval_list, dim=0)  # [G, B, P, C]
        return pred_from_retrieval

    def retrieve_all(self, data, train=False, device=torch.device("cpu")):
        assert (self.train_data_all_mg is not None) or (self.train_series_x is not None)

        rt_loader = DataLoader(
            data,
            batch_size=1024,
            shuffle=False,
            num_workers=0 if self.low_mem_stream else 8,
            drop_last=False,
        )

        retrievals = []
        with torch.no_grad():
            for batch in tqdm(rt_loader):
                index, batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:5]
                meta_data = batch[5] if len(batch) > 5 else None

                pred_from_retrieval = self.retrieve(
                    batch_x.float().to(device),
                    index,
                    meta_query=meta_data,
                    train=train,
                )
                retrievals.append(pred_from_retrieval.cpu())

        retrievals = torch.cat(retrievals, dim=1)
        return retrievals

    @torch.no_grad()
    def evaluate_wave_meta_retrieval_quality(self, data, device=torch.device("cpu"), train=False):
        """
        Quantify retrieval-only future quality over a full split.
        Compare wave-retrieved future vs meta-retrieved future against true future
        in the same multi-scale decomposition space.
        """
        loader = DataLoader(
            data,
            batch_size=512,
            shuffle=False,
            num_workers=0 if self.low_mem_stream else 4,
            drop_last=False,
        )

        wave_sq_sum = 0.0
        wave_abs_sum = 0.0
        meta_sq_sum = 0.0
        meta_abs_sum = 0.0
        n_elem_total = 0

        g_count = len(self.period_num)
        channel_count = int(self.channels)
        wave_sq_by_g = np.zeros((g_count,), dtype=np.float64)
        wave_abs_by_g = np.zeros((g_count,), dtype=np.float64)
        meta_sq_by_g = np.zeros((g_count,), dtype=np.float64)
        meta_abs_by_g = np.zeros((g_count,), dtype=np.float64)
        n_elem_by_g = np.zeros((g_count,), dtype=np.float64)
        wave_sq_by_c = np.zeros((channel_count,), dtype=np.float64)
        wave_abs_by_c = np.zeros((channel_count,), dtype=np.float64)
        meta_sq_by_c = np.zeros((channel_count,), dtype=np.float64)
        meta_abs_by_c = np.zeros((channel_count,), dtype=np.float64)
        n_elem_by_c = np.zeros((channel_count,), dtype=np.float64)

        original_meta_only = bool(self.meta_only_retrieval)

        for batch in tqdm(loader):
            index, batch_x, batch_y = batch[0], batch[1], batch[2]
            meta_data = batch[5] if len(batch) > 5 else None

            batch_x = batch_x.float().to(device)
            batch_y_future = batch_y[:, -self.pred_len :, :].float().to(device)
            true_mg, _ = self.decompose_mg(batch_y_future)  # [G, B, P, C]

            self.meta_only_retrieval = False
            wave_pred = self.retrieve(
                batch_x,
                index,
                meta_query=meta_data,
                train=train,
            )  # [G, B, P, C]

            self.meta_only_retrieval = True
            meta_pred = self.retrieve(
                batch_x,
                index,
                meta_query=meta_data,
                train=train,
            )  # [G, B, P, C]

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
            n_elem_by_c += np.full((channel_count,), wave_err.shape[0] * wave_err.shape[1] * wave_err.shape[2], dtype=np.float64)

            for g_i in range(g_count):
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
                }
            )

        feature_names = getattr(data, "feature_names", None)
        if feature_names is None or len(feature_names) != channel_count:
            feature_names = [f"channel_{i}" for i in range(channel_count)]

        per_channel = []
        for c_i, name in enumerate(feature_names):
            denom = max(n_elem_by_c[c_i], 1.0)
            wave_mse_c = float(wave_sq_by_c[c_i] / denom)
            meta_mse_c = float(meta_sq_by_c[c_i] / denom)
            wave_mae_c = float(wave_abs_by_c[c_i] / denom)
            meta_mae_c = float(meta_abs_by_c[c_i] / denom)
            per_channel.append(
                {
                    "channel_idx": int(c_i),
                    "channel_name": str(name),
                    "wave_mse": wave_mse_c,
                    "wave_mae": wave_mae_c,
                    "meta_mse": meta_mse_c,
                    "meta_mae": meta_mae_c,
                    "delta_mse_meta_minus_wave": float(meta_mse_c - wave_mse_c),
                    "delta_mae_meta_minus_wave": float(meta_mae_c - wave_mae_c),
                }
            )

        return {
            "split": "train" if train else "eval",
            "wave_mse": float(wave_mse),
            "wave_mae": float(wave_mae),
            "meta_mse": float(meta_mse),
            "meta_mae": float(meta_mae),
            "delta_mse_meta_minus_wave": float(meta_mse - wave_mse),
            "delta_mae_meta_minus_wave": float(meta_mae - wave_mae),
            "per_period": per_period,
            "per_channel": per_channel,
        }
