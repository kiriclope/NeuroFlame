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
        LR_READOUT=1,
        LR_INI=.001,
        LR_UeqV=1,
        DEVICE="cuda",
    ):
        super().__init__()

        self.N_NEURON = N_NEURON
        self.slices = slices
        self.Na = Na

        self.RANK = RANK
        self.LR_MN = LR_MN
        self.LR_READOUT = LR_READOUT

        self.LR_INI = LR_INI
        self.LR_UeqV = LR_UeqV

        self.device = DEVICE

        self.V = nn.Parameter(
            torch.randn((self.N_NEURON, int(self.RANK)), device=self.device) * self.LR_INI
        )

        if self.LR_MN:
            self.U = nn.Parameter(
                torch.randn((self.N_NEURON, int(self.RANK)), device=self.device) * self.LR_INI
            )

            if self.LR_UeqV:
                with torch.no_grad():
                    self.U.copy_(self.V)
        else:
            self.U = torch.randn((self.N_NEURON, int(self.RANK)), device=self.device) * self.LR_INI

        # Linear readout for supervised learning
        if self.LR_READOUT:
            self.linear = nn.Linear(self.Na[0], 1, device=self.device, bias=1)


    def forward(self, LR_NORM=0, LR_CLAMP=0):
        if LR_NORM:
            U_norm = self.U.norm(p='fro') + 1e-6
            V_norm = self.V.norm(p='fro') + 1e-6
        else:
            U_norm = 1.0
            V_norm = 1.0

        if self.LR_MN:
            self.lr = (self.U / U_norm) @ (self.V.T / V_norm)
        else:
            self.lr = (self.V @ self.V.T)

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
