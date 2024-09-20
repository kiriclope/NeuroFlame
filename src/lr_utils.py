import time
import torch
from torch import nn
import torch.nn.init as init


def normalize_tensor(tensor, idx, slice, Na):
    norm_tensor = tensor.clone()

    mask = tensor[:, slice[idx]] / Na[idx]
    norm_tensor[:, slice[idx]] = mask

    # mask = tensor[slice[idx]] / Na[idx]
    # norm_tensor[slice[idx]] = mask

    return norm_tensor


def clamp_tensor(tensor, idx, slice):
    # Create a mask for non-zero elements
    clamped_tensor = tensor.clone()
    if idx == 0:
        mask = tensor[slice[0]].clamp(min=0.0)
        clamped_tensor[slice[0]] = mask
    elif idx == 'lr':
        # mask = tensor[slice[0]].clamp(min=-1.0, max=1.0)
        mask = tensor[slice[0]].clamp(min=-1.0)
        clamped_tensor[slice[0]] = mask
    else:
        mask = tensor[slice[1]].clamp(max=0.0)
        clamped_tensor[slice[1]] = mask

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
        DROP_RATE=0,
        LR_MASK=0,
        LR_CLASS=1,
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

        self.DROP_RATE = DROP_RATE
        self.LR_MASK = LR_MASK

        self.Na = Na
        self.device = DEVICE

        self.U = nn.Parameter(
            torch.randn((self.N_NEURON, int(self.RANK)), device=self.device) * 0.001
        )

        if self.LR_MN:
            self.V = nn.Parameter(
                torch.randn((self.N_NEURON, int(self.RANK)), device=self.device) * 0.001
            )
        else:
            self.V = (
                torch.randn((self.N_NEURON, int(self.RANK)), device=self.device) * 0.001
            )

        if self.LR_KAPPA == 1:
            self.lr_kappa = nn.Parameter(torch.rand(1, device=self.device))
        else:
            self.lr_kappa = torch.tensor(5.0, device=self.device)

        # Mask to train excitatory neurons only
        self.lr_mask = torch.zeros((self.N_NEURON, self.N_NEURON), device=self.device)

        if self.LR_MASK == 0:
            self.lr_mask[self.slices[0], self.slices[0]] = 1.0
        if self.LR_MASK == 1:
            self.lr_mask[self.slices[1], self.slices[1]] = 1.0
        if self.LR_MASK == -1:
            self.lr_mask = torch.ones(
                (self.N_NEURON, self.N_NEURON), device=self.device
            )

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


    def forward(self, LR_NORM=0, LR_CLAMP=0):
        if LR_NORM:
            self.lr = self.lr_kappa * (
                masked_normalize(self.U) @ masked_normalize(self.V).T
            )
        else:
            if self.LR_MN:
                self.lr = self.lr_kappa * (self.U @ self.V.T)
            else:
                self.lr = self.lr_kappa * (self.U @ self.U.T)

        self.lr = self.lr_mask * self.lr

        self.lr = normalize_tensor(self.lr, 0, self.slices, self.Na)

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
