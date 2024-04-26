import time
import torch
from torch import nn


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
    else:
        mask = tensor[slice[1]].clamp(max=0.0)

    clamped_tensor[slice[idx]] = mask

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
        slices,
        RANK=1,
        LR_BIAS=1,
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

        self.LR_BIAS = LR_BIAS
        self.LR_FIX_READ = LR_FIX_READ
        self.LR_CLASS = LR_CLASS

        self.DROP_RATE = DROP_RATE
        self.LR_MASK = LR_MASK

        self.Na0 = self.slices[0].shape[0]
        self.device = DEVICE

        self.U = nn.Parameter(
            torch.randn((self.N_NEURON, int(self.RANK)), device=self.device) * 0.01
        )

        if self.LR_MN:
            self.V = nn.Parameter(
                torch.randn((self.N_NEURON, int(self.RANK)), device=self.device) * 0.01
            )
        else:
            self.V = (
                torch.randn((self.N_NEURON, int(self.RANK)), device=self.device) * 0.01
            )

        if self.LR_KAPPA == 1:
            self.lr_kappa = nn.Parameter(torch.rand(1, device=self.device))
        else:
            self.lr_kappa = torch.tensor(1.0, device=self.device)

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
        self.linear = nn.Linear(
            self.Na0, self.LR_CLASS, device=self.device, bias=self.LR_BIAS
        )

        self.dropout = nn.Dropout(self.DROP_RATE)

        self.odors = torch.randn(
            (3, self.Na0),
            device=self.device,
        )

        if self.LR_FIX_READ:
            for param in self.linear.parameters():
                param.requires_grad = False

        def forward(self, LR_NORM=0):

            if self.LR_NORM:
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
            self.lr = normalize_tensor(self.lr, 1, self.slices, self.Na)


def initLR(model):
    # Low rank vector
    model.U = nn.Parameter(
        torch.randn((model.N_NEURON, int(model.RANK)), device=model.device) * 0.01
    )

    if model.LR_MN:
        model.V = nn.Parameter(
            torch.randn((model.N_NEURON, int(model.RANK)), device=model.device) * 0.01
        )
    else:
        model.V = (
            torch.randn((model.N_NEURON, int(model.RANK)), device=model.device) * 0.01
        )

    if model.LR_KAPPA == 1:
        model.lr_kappa = nn.Parameter(torch.rand(1, device=model.device))
    else:
        model.lr_kappa = torch.tensor(1.0, device=model.device)

    # Mask to train excitatory neurons only
    model.lr_mask = torch.zeros((model.N_NEURON, model.N_NEURON), device=model.device)

    if model.LR_MASK == 0:
        model.lr_mask[model.slices[0], model.slices[0]] = 1.0
    if model.LR_MASK == 1:
        model.lr_mask[model.slices[1], model.slices[1]] = 1.0
    if model.LR_MASK == -1:
        model.lr_mask = torch.ones(
            (model.N_NEURON, model.N_NEURON), device=model.device
        )

    # Linear readout for supervised learning
    model.linear = nn.Linear(
        model.Na[0], model.LR_CLASS, device=model.device, bias=model.LR_BIAS
    )

    model.dropout = nn.Dropout(model.DROP_RATE)

    model.odors = torch.randn(
        (3, model.Na[0]),
        device=model.device,
    )

    if model.LR_FIX_READ:
        for param in model.linear.parameters():
            param.requires_grad = False

    # Window where to evaluate loss
    if model.LR_EVAL_WIN == -1:
        model.lr_eval_win = -1
    else:
        model.lr_eval_win = int(model.LR_EVAL_WIN / model.DT / model.N_WINDOW)


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
