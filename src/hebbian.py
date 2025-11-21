import torch
import torch.nn.functional as F


class Hebbian:
    """
    A Hebbian learning class
    """

    def __init__(self, ETA, DT, HEBB_TYPE="bcm", frac=0.1):
        self.DT = DT
        self.HEBB_TYPE = HEBB_TYPE
        self.ETA_DT = ETA * DT
        self.frac = frac


    def spatial_sum(self, rates):
        neighborhood = int(rates.shape[-1] * self.frac)
        kernel = torch.ones(1, 1, neighborhood) / neighborhood
        # rates: [N_BATCH, N_NEURONS]
        pad = neighborhood // 2
        rates_unsq = rates.unsqueeze(1)  # [N_BATCH, 1, N_NEURONS]
        # Apply moving average (sum)
        summed = F.conv1d(rates_unsq, kernel.to(rates.device), padding=pad)
        # summed.shape: [N_BATCH, 1, N_NEURONS]
        return summed.squeeze(1)  # [N_BATCH, N_NEURONS]


    def hebbian_learning(self, pre, post, avg_pre=None, avg_post=None, weights=None):

        if self.HEBB_TYPE == 'cov':
            delta_pre = pre - avg_pre
            delta_post = post - avg_post
            return self.ETA_DT * (delta_pre.unsqueeze(2) * delta_post.unsqueeze(1))

        if self.HEBB_TYPE == 'corr':
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

        if self.HEBB_TYPE == 'bcm':
            delta_pre = pre - avg_pre
            return self.ETA_DT * delta_pre.unsqueeze(2) * wij

        return self.ETA_DT * wij


    def forward(self, pre, post, avg_pre=None, avg_post=None):
        return self.hebbian_learning(pre, post, avg_pre, avg_post)


    def __call__(self, pre, post, avg_pre=None, avg_post=None):
        return self.forward(pre, post, avg_pre, avg_post)
