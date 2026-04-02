import torch
from torch import nn
from math import floor


def normalize_tensor(tensor, pop_idx, slices, scale):
    norm_tensor = tensor.clone()
    norm_tensor[:, slices[pop_idx]] = tensor[:, slices[pop_idx]] / scale[pop_idx]
    return norm_tensor


def clamp_tensor(tensor, idx, slices):
    clamped_tensor = tensor.clone()
    if idx == 0:
        clamped_tensor[slices[0]] = tensor[slices[0]].clamp(min=0.0)
    elif idx == "lr":
        clamped_tensor[slices[0]] = tensor[slices[0]].clamp(min=-1.0)
    else:
        clamped_tensor[slices[1]] = tensor[slices[1]].clamp(max=0.0)
    return clamped_tensor


def masked_normalize(tensor):
    mask = tensor != 0
    normalized_tensor = tensor.clone()
    if mask.any():
        masked_tensor = tensor[mask]
        mean = masked_tensor.mean()
        std = masked_tensor.std(unbiased=False) + 1e-6
        normalized_tensor[mask] = (masked_tensor - mean) / std
    return normalized_tensor


class LowRankWeights(nn.Module):
    def __init__(
        self, N_NEURON, RANK=1, LR_MN=1, LR_READOUT=0, LR_INI=0.001,
        LR_UeqV=1, DEVICE="cuda", LR_READOUT_DIM=1,
    ):
        super().__init__()
        self.N_NEURON = int(N_NEURON)
        self.RANK = int(RANK)
        self.LR_MN = int(LR_MN)
        self.LR_READOUT = int(LR_READOUT)
        self.LR_INI = LR_INI
        self.LR_UeqV = int(LR_UeqV)
        self.LR_READOUT_DIM = int(LR_READOUT_DIM)
        self.device = DEVICE

        self.V = nn.Parameter(
            torch.randn((self.N_NEURON, self.RANK), device=self.device) * self.LR_INI
        )
        if self.LR_MN:
            self.U = nn.Parameter(
                torch.randn((self.N_NEURON, self.RANK), device=self.device) * self.LR_INI
            )
            if self.LR_UeqV:
                with torch.no_grad():
                    self.U.copy_(self.V)
        else:
            self.U = None

        if self.LR_READOUT:
            self.readout_linear = nn.Linear(
                self.N_NEURON, self.LR_READOUT_DIM, device=self.device, bias=True,
            )
        self.lr = None

    def get_U(self):
        return self.U if self.LR_MN else self.V

    def get_V(self):
        return self.V

    def get_readout(self):
        return self.get_V()

    def resample_basis(self):
        return

    def forward(self, LR_NORM=0):
        U = self.get_U()
        V = self.get_V()
        U_norm, V_norm = 1.0, 1.0
        if LR_NORM:
            U_norm = U.norm(p="fro") + 1e-6
            V_norm = V.norm(p="fro") + 1e-6
        self.lr = (U / U_norm) @ (V.T / V_norm)
        return self.lr


class SupportLowRankWeights(nn.Module):
    def __init__(
        self, N_NEURON, RANK=1, LR_MN=1, LR_READOUT=0, LR_INI=0.001,
        LR_UeqV=1, DEVICE="cuda", N_SUPPORTS=1, SUPPORT_WEIGHTS=None,
        BASIS_DIM=None, TRAIN_BIASES=False, LR_READOUT_DIM=1,
        INIT_GAUSSIAN_BASIS=None,
    ):
        super().__init__()
        self.N_NEURON = int(N_NEURON)
        self.RANK = int(RANK)
        self.LR_MN = int(LR_MN)
        self.LR_READOUT = int(LR_READOUT)
        self.LR_INI = LR_INI
        self.LR_UeqV = int(LR_UeqV)
        self.LR_READOUT_DIM = int(LR_READOUT_DIM)
        self.N_SUPPORTS = int(N_SUPPORTS)
        self.BASIS_DIM = 2 * self.RANK if BASIS_DIM is None else int(BASIS_DIM)
        self.TRAIN_BIASES = TRAIN_BIASES
        self.device = DEVICE

        if INIT_GAUSSIAN_BASIS is None:
            basis = torch.randn((self.BASIS_DIM, self.N_NEURON), device=self.device)
        else:
            basis = INIT_GAUSSIAN_BASIS.to(self.device)

        self.gaussian_basis = nn.Parameter(basis, requires_grad=False)
        self.supports = nn.Parameter(
            torch.zeros((self.N_SUPPORTS, self.N_NEURON), device=self.device),
            requires_grad=False,
        )

        if SUPPORT_WEIGHTS is None:
            self.support_weights = nn.Parameter(
                torch.ones(self.N_SUPPORTS, device=self.device) / self.N_SUPPORTS,
                requires_grad=False,
            )
            l_support = self.N_NEURON // self.N_SUPPORTS
            for i in range(self.N_SUPPORTS):
                if i < self.N_SUPPORTS - 1:
                    self.supports.data[i, l_support * i : l_support * (i + 1)] = 1
                else:
                    self.supports.data[i, l_support * i :] = 1
        else:
            support_weights = torch.tensor(
                SUPPORT_WEIGHTS, device=self.device, dtype=torch.float32,
            )
            support_weights = support_weights / support_weights.sum()
            self.support_weights = nn.Parameter(support_weights, requires_grad=False)
            k = 0
            for i in range(self.N_SUPPORTS):
                if i < self.N_SUPPORTS - 1:
                    width = floor(float(self.support_weights[i]) * self.N_NEURON)
                    self.supports.data[i, k : k + width] = 1
                    k += width
                else:
                    self.supports.data[i, k:] = 1

        self.V_weights = nn.Parameter(
            torch.randn(
                (self.RANK, self.N_SUPPORTS, self.BASIS_DIM), device=self.device,
            ) * self.LR_INI
        )
        self.V_biases = nn.Parameter(
            torch.zeros((self.RANK, self.N_SUPPORTS), device=self.device),
            requires_grad=self.TRAIN_BIASES,
        )

        if self.LR_MN:
            self.U_weights = nn.Parameter(
                torch.randn(
                    (self.RANK, self.N_SUPPORTS, self.BASIS_DIM), device=self.device,
                ) * self.LR_INI
            )
            self.U_biases = nn.Parameter(
                torch.zeros((self.RANK, self.N_SUPPORTS), device=self.device),
                requires_grad=self.TRAIN_BIASES,
            )
            if self.LR_UeqV:
                with torch.no_grad():
                    self.U_weights.copy_(self.V_weights)
                    self.U_biases.copy_(self.V_biases)
        else:
            self.U_weights = None
            self.U_biases = None

        if self.LR_READOUT:
            self.readout_weights = nn.Parameter(
                torch.randn(
                    (self.LR_READOUT_DIM, self.N_SUPPORTS, self.BASIS_DIM),
                    device=self.device,
                ) * self.LR_INI
            )
            self.readout_biases = nn.Parameter(
                torch.zeros(
                    (self.LR_READOUT_DIM, self.N_SUPPORTS), device=self.device,
                ),
                requires_grad=self.TRAIN_BIASES,
            )
            self.readout_linear = nn.Linear(
                self.N_NEURON, self.LR_READOUT_DIM, device=self.device, bias=True,
            )

        self.U = None
        self.V = None
        self.readout = None
        self.lr = None
        self.define_proxy_parameters()

    def support_to_neurons(self, weights, biases=None):
        vec = weights @ self.gaussian_basis
        vec = torch.sum(vec * self.supports.unsqueeze(0), dim=1)
        if biases is not None:
            vec = vec + biases @ self.supports
        return vec.T

    def define_proxy_parameters(self):
        self.V = self.support_to_neurons(self.V_weights, self.V_biases)
        if self.LR_MN:
            self.U = self.support_to_neurons(self.U_weights, self.U_biases)
        else:
            self.U = self.V
        if self.LR_READOUT:
            self.readout = self.support_to_neurons(
                self.readout_weights, self.readout_biases,
            )
        else:
            self.readout = self.V

    def get_U(self):
        self.define_proxy_parameters()
        return self.U

    def get_V(self):
        self.define_proxy_parameters()
        return self.V

    def get_readout(self):
        self.define_proxy_parameters()
        return self.readout

    def resample_basis(self):
        with torch.no_grad():
            self.gaussian_basis.normal_()
        self.define_proxy_parameters()

    def forward(self, LR_NORM=0):
        U = self.get_U()
        V = self.get_V()
        U_norm, V_norm = 1.0, 1.0
        if LR_NORM:
            U_norm = U.norm(p="fro") + 1e-6
            V_norm = V.norm(p="fro") + 1e-6
        self.lr = (U / U_norm) @ (V.T / V_norm)
        return self.lr


def init_low_rank(
    N_NEURON, RANK=1, LR_MN=1, LR_READOUT=0, LR_INI=0.001,
    LR_UeqV=1, DEVICE="cuda", LR_TYPE="standard", LR_N_SUPPORTS=1,
    LR_SUPPORT_WEIGHTS=None, LR_BASIS_DIM=None, LR_TRAIN_BIASES=False,
    LR_READOUT_DIM=1, LR_INIT_GAUSSIAN_BASIS=None,
):
    if LR_TYPE in ["support", "gaussian_mixture", "gmm"]:
        return SupportLowRankWeights(
            N_NEURON=N_NEURON, RANK=RANK, LR_MN=LR_MN,
            LR_READOUT=LR_READOUT, LR_INI=LR_INI, LR_UeqV=LR_UeqV,
            DEVICE=DEVICE, N_SUPPORTS=LR_N_SUPPORTS,
            SUPPORT_WEIGHTS=LR_SUPPORT_WEIGHTS, BASIS_DIM=LR_BASIS_DIM,
            TRAIN_BIASES=LR_TRAIN_BIASES, LR_READOUT_DIM=LR_READOUT_DIM,
            INIT_GAUSSIAN_BASIS=LR_INIT_GAUSSIAN_BASIS,
        )
    return LowRankWeights(
        N_NEURON=N_NEURON, RANK=RANK, LR_MN=LR_MN,
        LR_READOUT=LR_READOUT, LR_INI=LR_INI, LR_UeqV=LR_UeqV,
        DEVICE=DEVICE, LR_READOUT_DIM=LR_READOUT_DIM,
    )


def get_theta(a, b, IF_NORM=0):
    u, v = a, b
    if IF_NORM:
        u = a / torch.norm(a, p="fro")
        v = b / torch.norm(b, p="fro")
    return torch.atan2(v, u)


def get_idx(ksi, ksi1):
    theta = get_theta(ksi, ksi1, IF_NORM=0)
    return theta.argsort()


def get_overlap(model, rates):
    ksi = model.PHI0
    if not isinstance(ksi, torch.Tensor):
        ksi = torch.as_tensor(ksi, device=rates.device, dtype=rates.dtype)
    else:
        ksi = ksi.to(device=rates.device, dtype=rates.dtype)
    return rates @ ksi.T / rates.shape[-1]
