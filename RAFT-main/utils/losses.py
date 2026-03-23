import torch
import torch.nn as nn
import torch.nn.functional as F


class QDFLiteLoss(nn.Module):
    """
    Quadratic-Direct-Forecast style loss with light-weight covariance tracking.
    """

    def __init__(
        self,
        pred_len: int,
        qdf_beta: float = 0.7,
        qdf_warmup_epochs: int = 5,
        qdf_diff_weight: float = 0.15,
        qdf_level_weight: float = 0.05,
        qdf_ema_decay: float = 0.98,
        qdf_bandwidth: int = 32,
        qdf_update_interval: int = 1,
        qdf_eps: float = 1e-5,
    ):
        super().__init__()
        self.pred_len = int(pred_len)
        self.qdf_beta = float(qdf_beta)
        self.qdf_warmup_epochs = int(max(0, qdf_warmup_epochs))
        self.qdf_diff_weight = float(qdf_diff_weight)
        self.qdf_level_weight = float(qdf_level_weight)
        self.qdf_ema_decay = float(qdf_ema_decay)
        self.qdf_bandwidth = int(qdf_bandwidth)
        self.qdf_update_interval = int(max(1, qdf_update_interval))
        self.qdf_eps = float(qdf_eps)
        self.current_epoch = 0

        self.register_buffer("sigma", torch.eye(self.pred_len, dtype=torch.float32))
        self.register_buffer("update_step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("band_mask", self._build_band_mask(self.pred_len, self.qdf_bandwidth))

    @staticmethod
    def _build_band_mask(size: int, bandwidth: int):
        if bandwidth <= 0 or bandwidth >= size:
            return torch.ones((size, size), dtype=torch.float32)
        idx = torch.arange(size)
        dist = (idx[:, None] - idx[None, :]).abs()
        return (dist <= bandwidth).float()

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)

    def _beta(self):
        if self.qdf_warmup_epochs <= 0:
            return self.qdf_beta
        warm = min(1.0, float(self.current_epoch + 1) / float(self.qdf_warmup_epochs))
        return self.qdf_beta * warm

    def _apply_band(self, matrix: torch.Tensor):
        mask = self.band_mask.to(matrix.device, dtype=matrix.dtype)
        return matrix * mask

    def _normalize_trace(self, matrix: torch.Tensor):
        tr = matrix.diagonal().sum().clamp_min(self.qdf_eps)
        return matrix * (float(self.pred_len) / tr)

    @torch.no_grad()
    def _update_sigma(self, error: torch.Tensor):
        # error: [B, P, C] -> [B*C, P]
        flat = error.permute(0, 2, 1).reshape(-1, self.pred_len).float()
        if flat.shape[0] < 2:
            return
        flat = flat - flat.mean(dim=0, keepdim=True)
        cov = torch.matmul(flat.transpose(0, 1), flat) / max(flat.shape[0] - 1, 1)
        cov = self._apply_band(cov)
        cov = 0.5 * (cov + cov.transpose(0, 1))
        cov = cov + self.qdf_eps * torch.eye(self.pred_len, device=cov.device, dtype=cov.dtype)
        cov = self._normalize_trace(cov)

        sigma = self.sigma.to(cov.device)
        sigma = self.qdf_ema_decay * sigma + (1.0 - self.qdf_ema_decay) * cov
        sigma = self._apply_band(sigma)
        sigma = 0.5 * (sigma + sigma.transpose(0, 1))
        sigma = sigma + self.qdf_eps * torch.eye(self.pred_len, device=sigma.device, dtype=sigma.dtype)
        sigma = self._normalize_trace(sigma)
        self.sigma.copy_(sigma.to(self.sigma.device))

    def forward(self, pred: torch.Tensor, true: torch.Tensor):
        pred = pred.float()
        true = true.float()
        error = pred - true

        mse = F.mse_loss(pred, true)

        sigma = self.sigma.to(error.device, dtype=error.dtype)
        flat = error.permute(0, 2, 1).reshape(-1, self.pred_len)  # [B*C, P]
        quadratic = torch.einsum("bp,pq,bq->b", flat, sigma, flat).mean()

        if self.pred_len > 1:
            diff_pred = pred[:, 1:, :] - pred[:, :-1, :]
            diff_true = true[:, 1:, :] - true[:, :-1, :]
            diff_loss = F.smooth_l1_loss(diff_pred, diff_true, beta=0.1)
        else:
            diff_loss = torch.zeros((), device=pred.device, dtype=pred.dtype)

        level_loss = F.mse_loss(pred.mean(dim=1), true.mean(dim=1))

        beta = self._beta()
        loss = (1.0 - beta) * mse + beta * quadratic
        loss = loss + self.qdf_diff_weight * diff_loss + self.qdf_level_weight * level_loss

        if self.training:
            self.update_step += 1
            if int(self.update_step.item()) % self.qdf_update_interval == 0:
                self._update_sigma(error.detach())

        return loss
