import time
import torch
from torch import nn
import torch.nn.init as init


def normalize_tensor(tensor, idx, slices, Na):
    norm_tensor = tensor.clone()

    mask = tensor[:, slices[idx]] / Na[idx]
    norm_tensor[:, slices[idx]] = mask

    # mask = tensor[slice[idx]] / Na[idx]
    # norm_tensor[slice[idx]] = mask

    return norm_tensor


def clamp_tensor(tensor, idx, slices):
    # Create a mask for non-zero elements
    clamped_tensor = tensor.clone()
    if idx == 0:
        mask = tensor[slices[0]].clamp(min=0.0)
        clamped_tensor[slices[0]] = mask
    elif idx == 'lr':
        # mask = tensor[slice[0]].clamp(min=-1.0, max=1.0)
        mask = tensor[slices[0]].clamp(min=-1.0)
        clamped_tensor[slices[0]] = mask
    else:
        mask = tensor[slices[1]].clamp(max=0.0)
        clamped_tensor[slices[1]] = mask

    return clamped_tensor


def masked_normalize(tensor):
    # Create a mask for non-zero elements
    mask = tensor != 0
    normalized_tensor = tensor.clone()  # Create a clone to avoid in-place modification
    if mask.any():
        masked_tensor = tensor[mask]
        mean = masked_tensor.mean()
        std = (
            masked_tensor.std(unbiased=False) + 1e-6
        )  # Adding epsilon for numerical stability
        # Normalize only the non-zero elements and replace them in the clone
        normalized_tensor[mask] = (masked_tensor - mean) / std
    return normalized_tensor


class LowRankWeights(nn.Module):
    def __init__(
        self,
        N_NEURON,
        Na,
        slices,
        RANK=1,
        LR_MN=1,
        LR_KAPPA=0,
        LR_BIAS=1,
        LR_READOUT=1,
        LR_FIX_READ=0,
        LR_MASK=0,
        LR_CLASS=1,
        LR_GAUSS=0,
        LR_INI=.001,
        DEVICE="cuda",
    ):
        super().__init__()

        self.N_NEURON = N_NEURON
        self.slices = slices
        self.RANK = RANK
        self.LR_MN = LR_MN
        self.LR_KAPPA = LR_KAPPA
        self.LR_BIAS = LR_BIAS
        self.LR_READOUT = LR_READOUT
        self.LR_FIX_READ = LR_FIX_READ
        self.LR_CLASS = LR_CLASS

        self.LR_MASK = LR_MASK
        self.LR_GAUSS = LR_GAUSS
        self.LR_INI = LR_INI

        self.Na = Na
        self.device = DEVICE

        if self.LR_GAUSS==0:

            self.V = nn.Parameter(
                torch.randn((self.N_NEURON, int(self.RANK)), device=self.device) * self.LR_INI
            )

            if self.LR_MN:
                self.U = nn.Parameter(
                    torch.randn((self.N_NEURON, int(self.RANK)), device=self.device) * self.LR_INI
                )

                # with torch.no_grad():
                # self.U.copy_(self.V)
            else:
                self.U = (
                    torch.randn((self.N_NEURON, int(self.RANK)), device=self.device) * self.LR_INI
                )

        if self.LR_KAPPA == 1:
            self.lr_kappa = nn.Parameter(torch.rand(1, device=self.device))
        else:
            self.lr_kappa = torch.tensor(1.0, device=self.device)


        # Mask to train excitatory neurons only
        # self.lr_mask = torch.zeros((self.N_NEURON, self.N_NEURON), device=self.device)

        # if self.LR_MASK == 0:
        #     self.lr_mask[self.slices[0], self.slices[0]] = 1.0
        # if self.LR_MASK == 1:
        #     self.lr_mask[self.slices[1], self.slices[1]] = 1.0
        # if self.LR_MASK == -1:
        #     self.lr_mask = torch.ones(
        #         (self.N_NEURON, self.N_NEURON), device=self.device
        #     )

        # Linear readout for supervised learning
        if self.LR_READOUT:
            self.linear = nn.Linear(
                self.Na[0], self.LR_CLASS, device=self.device, bias=self.LR_BIAS
            )
            if self.LR_FIX_READ:
                for param in self.linear.parameters():
                    param.requires_grad = False

                # Initialize the weights with a Gaussian (normal) distribution
                init.normal_(self.linear.weight, mean=0.0, std=1.0)

                # Optionally initialize the biases as well
                if self.LR_BIAS:
                    init.normal_(self.linear.bias, mean=0.0, std=1.0)

        if self.LR_GAUSS:
            self.mu_M = nn.Parameter(torch.randn(1, self.RANK, device=self.device) * 0.01)
            self.log_sigma_M = nn.Parameter(torch.randn(1, self.RANK, device=self.device) * 0.1 - 1.0)
            self.mu_N = nn.Parameter(torch.randn(1, self.RANK, device=self.device) * 0.01)
            self.log_sigma_N = nn.Parameter(torch.randn(1, self.RANK, device=self.device) * 0.1 - 1.0)


    def forward(self, LR_NORM=0, LR_CLAMP=0):
        if LR_NORM:
            U_norm = self.U.norm(p='fro') + 1e-6
            V_norm = self.V.norm(p='fro') + 1e-6
            #     self.lr = self.lr_kappa * (
            #         masked_normalize(self.U) @ masked_normalize(self.V).T
            #     )

        else:
            U_norm = 1.0
            V_norm = 1.0

        if self.LR_GAUSS:
            epsilon_M = torch.randn((self.Na[0], self.RANK), device=self.device)
            epsilon_N = torch.randn((self.Na[0], self.RANK), device=self.device)

            # Reparameterize to get M and N
            self.U = self.mu_M + torch.exp(self.log_sigma_M) * epsilon_M
            self.V = self.mu_N + torch.exp(self.log_sigma_N) * epsilon_N

        if self.LR_MN:
            self.lr = self.lr_kappa * ((self.U / U_norm) @ (self.V.T / V_norm))
        else:
            self.lr = self.lr_kappa * (self.V @ self.V.T)

        # self.lr = self.lr_mask * self.lr
        # if LR_NORM:
        #     self.lr = normalize_tensor(self.lr, 0, self.slices, self.Na)

        if LR_CLAMP:
            self.lr = clamp_tensor(self.lr, 'lr', self.slices)

        return self.lr

def get_theta(a, b, IF_NORM=0):
    u, v = a, b

    if IF_NORM:
        u = a / torch.norm(a, p="fro")
        v = b / torch.norm(b, p="fro")

    return torch.atan2(v, u)


def get_idx(ksi, ksi1):
    theta = get_theta(ksi, ksi1, GM=0, IF_NORM=0)
    return theta.argsort()


def get_overlap(model, rates):
    ksi = model.PHI0.cpu().detach().numpy()
    return rates @ ksi.T / rates.shape[-1]
