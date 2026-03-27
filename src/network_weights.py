from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from src.connectivity import Connectivity
from src.lr_utils import init_low_rank, clamp_tensor, normalize_tensor


def recurrent_matmul(rates: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    if W.dim() == 2:
        return rates @ W
    if W.dim() == 3:
        return torch.bmm(rates.unsqueeze(1), W).squeeze(1)
    raise ValueError(f"Unsupported shapes: rates={rates.shape}, W={W.shape}")


@dataclass
class WeightOutputs:
    W_train: Optional[torch.Tensor]
    W_rec: torch.Tensor
    W_stp: Optional[list]


class WeightBuilder(nn.Module):
    """
    Owns persistent weight-related parameters/buffers and builds effective weights.
    Convention: W_rec[pre, post].
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device

        for k, v in vars(cfg).items():
            setattr(self, k, v)

        self._init_base_weights()
        self._init_trainable_weights()
        self._init_stp_buffers()
        self._init_low_rank_module()

    def _init_base_weights(self):
        W_rec = torch.zeros((self.N_NEURON, self.N_NEURON), device=self.device)

        for post in range(self.N_POP):
            for pre in range(self.N_POP):
                conn = Connectivity(
                    self.Na[post],
                    self.Na[pre],
                    self.Ka[pre],
                    device=self.device,
                )

                block = conn(
                    self.CON_TYPE,
                    self.PROBA_TYPE[post][pre],
                    kappa=self.KAPPA[post][pre],
                    phase=self.PHASE,
                    sigma=self.SIGMA[post][pre],
                    lr_mean=self.LR_MEAN,
                    lr_cov=self.LR_COV,
                    ksi=self.PHI0,
                )

                # block assumed [post, pre] -> store [pre, post]
                W_rec[self.slices[pre], self.slices[post]] = self.Jab[post][pre] * block.T

        self.register_buffer("W_rec_base", W_rec)

    def _init_trainable_weights(self):
        if self.TRAIN_EI:
            train_shape = (self.N_NEURON, self.N_NEURON)
        else:
            train_shape = (self.Na[0], self.Na[0])

        self.W_train_dense = None

        if not self.LR_TRAIN:
            self.W_train_dense = nn.Parameter(
                torch.randn(train_shape, device=self.device) * self.LR_INI
            )

        if self.TRAIN_EI:
            mask = torch.zeros(train_shape, device=self.device)
            for post in range(self.N_POP):
                for pre in range(self.N_POP):
                    mask[self.slices[pre], self.slices[post]] = self.IS_TRAIN[post][pre]
            self.register_buffer("train_mask", mask)
        else:
            self.train_mask = None

    def _init_low_rank_module(self):
        if self.LR_TRAIN:
            self.low_rank = init_low_rank(
                N_NEURON=self.Na[0],
                RANK=self.RANK,
                LR_MN=self.LR_MN,
                LR_READOUT=self.LR_READOUT,
                LR_INI=self.LR_INI,
                LR_UeqV=self.LR_UeqV,
                DEVICE=self.device,
                LR_TYPE=self.LR_TYPE,
                LR_N_SUPPORTS=self.LR_N_SUPPORTS,
                LR_SUPPORT_WEIGHTS=self.LR_SUPPORT_WEIGHTS,
                LR_BASIS_DIM=self.LR_BASIS_DIM,
                LR_TRAIN_BIASES=self.LR_TRAIN_BIASES,
                LR_READOUT_DIM=self.LR_READOUT_DIM,
                LR_INIT_GAUSSIAN_BASIS=self.LR_INIT_GAUSSIAN_BASIS,
            )
        else:
            self.low_rank = None

    def _init_stp_buffers(self):
        self.stp_block_index = []
        self.W_stp_blocks_base = []

        if not self.IF_STP:
            self.J_STP_param = None
            return

        J_stp = torch.as_tensor(self.J_STP, device=self.device, dtype=torch.float32)
        if getattr(self, "TRAIN_J_STP", False):
            self.J_STP_param = nn.Parameter(J_stp)
        else:
            self.register_buffer("J_STP_param", J_stp)

        for post in range(self.N_POP):
            for pre in range(self.N_POP):
                idx = pre + post * self.N_POP
                if not self.IS_STP[idx]:
                    continue

                pre_slice = self.slices[pre]
                post_slice = self.slices[post]

                block = self.W_rec_base[pre_slice, post_slice].clone() / torch.abs(self.Jab[post, pre])
                self.stp_block_index.append((pre, post, idx))
                self.W_stp_blocks_base.append(block)

    def build_trainable_matrix(self) -> Optional[torch.Tensor]:
        if self.LR_TRAIN:
            return self.low_rank(self.LR_NORM)
        return self.W_train_dense

    def build_recurrent(self, W_train: Optional[torch.Tensor]) -> torch.Tensor:
        W_rec = self.W_rec_base.clone()

        if W_train is None:
            return W_rec

        if self.TRAIN_EI:
            W_eff = W_train
            for pop_idx in range(self.N_POP):
                W_eff = normalize_tensor(W_eff, pop_idx, self.slices, self.train_scale)

            W_rec = W_rec + self.train_mask * W_eff

            if self.CLAMP:
                for pop_idx in range(self.N_POP):
                    W_rec = clamp_tensor(W_rec.T, pop_idx, self.slices).T
        else:
            ee = self.slices[0]
            W_rec[ee, ee] = W_rec[ee, ee] + self.GAIN * W_train / torch.sqrt(self.Ka[0])

            if self.CLAMP:
                W_rec = clamp_tensor(W_rec.T, 0, self.slices).T

        return W_rec

    def build_stp(self, W_train: Optional[torch.Tensor]) -> Optional[list]:
        if not self.IF_STP:
            return None

        W_stp = []
        fix_scale = torch.sqrt(self.Ka[0])

        for k, (pre, post, idx) in enumerate(self.stp_block_index):
            base_block = self.W_stp_blocks_base[k]

            if pre == 0 and post == 0 and W_train is not None and not self.TRAIN_EI:
                block = (
                    self.GAIN
                    * self.J_STP_param
                    * (
                        base_block / fix_scale
                        * (1.0 + W_train / self.train_scale[0])
                    )
                )
                if self.CLAMP:
                    block = clamp_tensor(block, 0, self.slices)
            else:
                block = self.W_STP[idx] * self.J_STP_param * base_block / torch.sqrt(self.Ka[pre])

            W_stp.append(block)

        return W_stp

    def expand_for_hebbian(self, W_rec: torch.Tensor, batch_size: int) -> torch.Tensor:
        if not self.IF_HEBB:
            return W_rec
        return W_rec.unsqueeze(0).repeat(batch_size, 1, 1)

    def build_all(self, batch_size: int) -> WeightOutputs:
        W_train = self.build_trainable_matrix()
        W_rec = self.build_recurrent(W_train)
        W_stp = self.build_stp(W_train)
        W_rec = self.expand_for_hebbian(W_rec, batch_size)
        return WeightOutputs(W_train=W_train, W_rec=W_rec, W_stp=W_stp)

    def get_readout(self):
        return None if self.low_rank is None else self.low_rank.get_readout()
