"""Recurrent weight construction and management."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.config.types import WeightConfig
from src.weights.connectivity import Connectivity
from src.weights.lr import clamp_tensor, init_low_rank, normalize_tensor


Tensor = torch.Tensor
STPBlockSpec = tuple[int, int, int]  # (pre_pop, post_pop, block_id)


def recurrent_matmul(rates: Tensor, W: Tensor) -> Tensor:
    if W.dim() == 2:
        return rates @ W
    if W.dim() == 3:
        return torch.bmm(rates.unsqueeze(1), W).squeeze(1)
    raise ValueError(f"W must be 2D or 3D, got {W.dim()}D")


@dataclass
class WeightOutputs:
    W_train: Tensor | None
    W_rec: Tensor
    W_stp_blocks: list[Tensor] | None


class WeightBuilder(nn.Module):
    def __init__(self, cfg: WeightConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.low_rank: nn.Module | None = None
        self.W_train_dense: nn.Parameter | None = None
        self.stp_block_specs: list[STPBlockSpec] = []

        self._init_base_weights()
        self._init_trainable_weights()
        self._init_stp_buffers()
        self._init_low_rank_module()

    @property
    def device(self):
        return self.cfg.geo.device

    def _init_base_weights(self) -> None:
        geo = self.cfg.geo
        conn = self.cfg.conn
        dtype = torch.float32
        W_rec = torch.zeros(
            (geo.N_NEURON, geo.N_NEURON), device=self.device, dtype=dtype,
        )
        for post in range(geo.N_POP):
            for pre in range(geo.N_POP):
                c = Connectivity(
                    geo.Na[post], geo.Na[pre], geo.Ka[pre], device=self.device,
                )
                block = c(
                    conn.CON_TYPE,
                    conn.PROBA_TYPE[post][pre],
                    kappa=conn.KAPPA[post][pre],
                    phase=conn.PHASE,
                    sigma=conn.SIGMA[post][pre],
                    lr_mean=conn.LR_MEAN,
                    lr_cov=conn.LR_COV,
                    ksi=conn.PHI0,
                )
                W_rec[geo.slices[pre], geo.slices[post]] = self.cfg.Jab[post, pre] * block.T
        self.register_buffer("W_rec_base", W_rec)

    def _init_trainable_weights(self) -> None:
        geo = self.cfg.geo
        t = self.cfg.trainable
        dtype = self.W_rec_base.dtype

        if t.TRAIN_EI:
            shape = (geo.N_NEURON, geo.N_NEURON)
            mask = torch.zeros(shape, device=self.device, dtype=dtype)
            for post in range(geo.N_POP):
                for pre in range(geo.N_POP):
                    mask[geo.slices[pre], geo.slices[post]] = t.IS_TRAIN[post][pre]
            self.register_buffer("train_mask", mask)
        else:
            shape = (geo.Na[0], geo.Na[0])

        if not t.LR_TRAIN:
            self.W_train_dense = nn.Parameter(
                torch.randn(shape, device=self.device, dtype=dtype) * t.LR_INI
            )

    def _init_low_rank_module(self) -> None:
        geo = self.cfg.geo
        t = self.cfg.trainable
        if t.LR_TRAIN:
            self.low_rank = init_low_rank(
                N_NEURON=geo.Na[0], RANK=t.RANK, LR_MN=t.LR_MN,
                LR_READOUT=t.LR_READOUT, LR_INI=t.LR_INI, LR_UeqV=t.LR_UeqV,
                DEVICE=self.device, LR_TYPE=t.LR_TYPE,
                LR_N_SUPPORTS=t.LR_N_SUPPORTS,
                LR_SUPPORT_WEIGHTS=t.LR_SUPPORT_WEIGHTS,
                LR_BASIS_DIM=t.LR_BASIS_DIM,
                LR_TRAIN_BIASES=t.LR_TRAIN_BIASES,
                LR_READOUT_DIM=t.LR_READOUT_DIM,
                LR_INIT_GAUSSIAN_BASIS=t.LR_INIT_GAUSSIAN_BASIS,
            )

    def _init_stp_buffers(self) -> None:
        geo = self.cfg.geo
        s = self.cfg.stp
        self.stp_block_specs = []

        if not s.IF_STP:
            self.J_STP_param = None
            return

        dtype = self.W_rec_base.dtype
        j_stp = torch.as_tensor(s.J_STP, device=self.device, dtype=dtype)

        if s.TRAIN_J_STP:
            self.J_STP_param = nn.Parameter(j_stp.clone())
        else:
            self.register_buffer("J_STP_param", j_stp.clone())

        for post in range(geo.N_POP):
            for pre in range(geo.N_POP):
                block_id = pre + post * geo.N_POP
                if not s.IS_STP[block_id]:
                    continue
                denom = torch.abs(torch.as_tensor(
                    self.cfg.Jab[post, pre], device=self.device, dtype=dtype,
                ))
                if denom == 0:
                    raise ValueError(
                        f"Jab[{post},{pre}] is zero; cannot normalise STP block"
                    )
                block = (
                    self.W_rec_base[geo.slices[pre], geo.slices[post]].clone() / denom
                )
                self.stp_block_specs.append((pre, post, block_id))
                self.register_buffer(
                    f"_W_stp_base_{len(self.stp_block_specs) - 1}", block,
                )

    def _get_stp_base(self, block_idx: int) -> Tensor:
        return getattr(self, f"_W_stp_base_{block_idx}")

    def _j_stp_value(self, block_id: int) -> Tensor:
        if self.J_STP_param is None:
            raise RuntimeError("J_STP_param is not initialized")
        if self.J_STP_param.dim() == 0:
            return self.J_STP_param
        return self.J_STP_param[block_id]

    def build_trainable_matrix(self) -> Tensor | None:
        t = self.cfg.trainable
        if t.LR_TRAIN:
            if self.low_rank is None:
                raise RuntimeError("LR_TRAIN=True but low_rank module is not initialized")
            return self.low_rank(t.LR_NORM)
        return self.W_train_dense

    def build_recurrent(self, W_train: Tensor | None) -> Tensor:
        geo = self.cfg.geo
        t = self.cfg.trainable
        W_rec = self.W_rec_base.clone()
        if W_train is None:
            return W_rec

        if t.TRAIN_EI:
            W_eff = W_train
            for pop_idx in range(geo.N_POP):
                W_eff = normalize_tensor(W_eff, pop_idx, geo.slices, t.train_scale)
            W_rec = W_rec + self.train_mask * W_eff
            if t.CLAMP:
                for pop_idx in range(geo.N_POP):
                    W_rec = clamp_tensor(W_rec.T, pop_idx, geo.slices).T
            return W_rec

        ee = geo.slices[0]
        k0 = torch.sqrt(torch.as_tensor(
            geo.Ka[0], device=self.device, dtype=W_rec.dtype,
        ))
        W_rec[ee, ee] = W_rec[ee, ee] + t.GAIN * W_train / k0
        if t.CLAMP:
            W_rec = clamp_tensor(W_rec.T, 0, geo.slices).T
        return W_rec

    def build_stp(self, W_train: Tensor | None) -> list[Tensor] | None:
        geo = self.cfg.geo
        t = self.cfg.trainable
        s = self.cfg.stp
        if not s.IF_STP:
            return None

        dtype = self.W_rec_base.dtype
        fix_scale = torch.sqrt(torch.as_tensor(
            geo.Ka[0], device=self.device, dtype=dtype,
        ))
        blocks: list[Tensor] = []

        for block_idx, (pre, post, block_id) in enumerate(self.stp_block_specs):
            base = self._get_stp_base(block_idx)
            j = self._j_stp_value(block_id)

            if pre == 0 and post == 0 and W_train is not None and not t.TRAIN_EI:
                block = (
                    t.GAIN * j * base / fix_scale
                    * (1.0 + W_train / t.train_scale[0])
                )
                if t.CLAMP:
                    block = block.clamp(min=0.0)
            else:
                k_pre = torch.sqrt(torch.as_tensor(
                    geo.Ka[pre], device=self.device, dtype=dtype,
                ))
                block = s.W_STP[block_id] * j * base / k_pre

            blocks.append(block)
        return blocks

    def build_all(self, batch_size: int, expand_hebbian: bool = False) -> WeightOutputs:
        W_train = self.build_trainable_matrix()
        W_rec = self.build_recurrent(W_train)
        W_stp = self.build_stp(W_train)
        if expand_hebbian:
            W_rec = W_rec.unsqueeze(0).expand(batch_size, -1, -1).clone()
        return WeightOutputs(W_train=W_train, W_rec=W_rec, W_stp_blocks=W_stp)

    def get_readout(self) -> Tensor | None:
        if self.low_rank is None:
            return None
        return self.low_rank.get_readout()
