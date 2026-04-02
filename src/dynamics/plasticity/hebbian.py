import torch
import torch.nn.functional as F
from torch import nn


class Hebbian(nn.Module):
    def __init__(self, ETA, DT, HEBB_TYPE="bcm", CORR_FRAC=0.1):
        super().__init__()
        self.DT = DT
        self.HEBB_TYPE = HEBB_TYPE
        self.ETA_DT = ETA * DT
        self.CORR_FRAC = CORR_FRAC
        self._kernel: torch.Tensor | None = None
        self._kernel_n: int = -1

    def _get_kernel(self, N: int, device: torch.device) -> torch.Tensor:
        if self._kernel is not None and self._kernel_n == N and self._kernel.device == device:
            return self._kernel
        neighborhood = int(N * self.CORR_FRAC)
        if neighborhood < 1:
            neighborhood = 1
        if neighborhood % 2 == 0:
            neighborhood += 1
        self._kernel = torch.ones(1, 1, neighborhood, device=device) / neighborhood
        self._kernel_n = N
        return self._kernel

    def spatial_sum(self, rates):
        N = rates.shape[-1]
        neighborhood = int(N * self.CORR_FRAC)
        if neighborhood < 1:
            return rates
        if neighborhood % 2 == 0:
            neighborhood += 1
        pad = neighborhood // 2
        kernel = self._get_kernel(N, rates.device)
        x = rates.unsqueeze(1)
        x_padded = torch.cat([x[..., -pad:], x, x[..., :pad]], dim=-1)
        summed = F.conv1d(x_padded, kernel, padding=0)
        return summed.squeeze(1)

    def hebbian_learning(self, pre, post, avg_pre=None, avg_post=None):
        if self.HEBB_TYPE == "cov":
            delta_pre = pre - avg_pre
            delta_post = post - avg_post
            return self.ETA_DT * (delta_pre.unsqueeze(2) * delta_post.unsqueeze(1))

        if self.HEBB_TYPE == "corr":
            spatial_pre = self.spatial_sum(pre)
            spatial_post = self.spatial_sum(post)
            mean_pre = spatial_pre.mean(dim=-1, keepdim=True)
            mean_post = spatial_post.mean(dim=-1, keepdim=True)
            std_pre = spatial_pre.std(dim=1, unbiased=False, keepdim=True) + 1e-9
            std_post = spatial_post.std(dim=1, unbiased=False, keepdim=True) + 1e-9
            spatial_pre = (spatial_pre - mean_pre) / std_pre
            spatial_post = (spatial_post - mean_post) / std_post
            return self.ETA_DT * spatial_pre.unsqueeze(2) * spatial_post.unsqueeze(1)

        wij = pre.unsqueeze(2) * post.unsqueeze(1)
        if self.HEBB_TYPE == "bcm":
            delta_pre = pre - avg_pre
            return self.ETA_DT * delta_pre.unsqueeze(2) * wij

        return self.ETA_DT * wij

    def forward(self, pre, post, avg_pre=None, avg_post=None):
        return self.hebbian_learning(pre, post, avg_pre, avg_post)
