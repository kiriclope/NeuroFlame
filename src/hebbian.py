import torch
import torch.nn.functional as F


class Hebbian:
    """
    A Hebbian learning class
    """

    def __init__(self, ETA, DT, HEBB_TYPE="bcm", CORR_FRAC=0.1):
        self.DT = DT
        self.HEBB_TYPE = HEBB_TYPE
        self.ETA_DT = ETA * DT
        self.CORR_FRAC = CORR_FRAC

    # def spatial_sum(self, rates):
    #     # Key idea: for each neuron index =i=, build a window of indices =[i-pad, ..., i+pad]= modulo =N=, then average.
    #     # rates: [B, N]
    #     B, N = rates.shape

    #     neighborhood = int(N * self.CORR_FRAC)
    #     if neighborhood < 1:
    #         return rates

    #     # make kernel size odd
    #     if neighborhood % 2 == 0:
    #         neighborhood += 1
    #     pad = neighborhood // 2

    #     device = rates.device

    #     # indices: [N] -> 0..N-1
    #     base = torch.arange(N, device=device)

    #     # shifts: [-pad, ..., pad], shape [K]
    #     shifts = torch.arange(-pad, pad + 1, device=device)  # K = neighborhood

    #     # full index matrix: [N, K], positions around ring for each neuron
    #     idx = (base[:, None] + shifts[None, :]) % N

    #     # gather windows: [B, N, K]
    #     windows = rates[:, idx]

    #     # average over neighborhood: [B, N]
    #     return windows.mean(dim=-1)


    def spatial_sum(self, rates):
        # rates: [N_BATCH, N_NEURONS]
        N = rates.shape[-1]
        neighborhood = int(N * self.CORR_FRAC)
        if neighborhood < 1:
            return rates  # or handle as you like

        # Force odd kernel size so it’s symmetric
        if neighborhood % 2 == 0:
            neighborhood += 1

        pad = neighborhood // 2
        kernel = torch.ones(1, 1, neighborhood, device=rates.device) / neighborhood

        # [N_BATCH, 1, N_NEURONS]
        x = rates.unsqueeze(1)

        # circular pad: take last `pad` and first `pad` elements
        x_padded = torch.cat([x[..., -pad:], x, x[..., :pad]], dim=-1)

        # conv without padding; output length stays N_NEURONS
        summed = F.conv1d(x_padded, kernel, padding=0)

        return summed.squeeze(1)  # [N_BATCH, N_NEURONS]

    # def spatial_sum(self, rates):
    #     neighborhood = int(rates.shape[-1] * self.CORR_FRAC)
    #     kernel = torch.ones(1, 1, neighborhood) / neighborhood
    #     # rates: [N_BATCH, N_NEURONS]
    #     rates_unsq = rates.unsqueeze(1)  # [N_BATCH, 1, N_NEURONS]
    #     # Apply moving average (sum)
    #     summed = F.conv1d(rates_unsq, kernel.to(rates.device), padding='same')
    #     # summed.shape: [N_BATCH, 1, N_NEURONS]
    #     return summed.squeeze(1)  # [N_BATCH, N_NEURONS]


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
