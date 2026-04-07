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
        self._init_stp_buffers()       # STP before trainable — it zeros out blocks
        self._init_trainable_weights()
        self._init_low_rank_module()

    @property
    def device(self):
        return self.cfg.geo.device

    def _init_base_weights(self) -> None:
        geo = self.cfg.geo
        conn = self.cfg.conn
        dtype = torch.float32
        # Original stores as Wab_T (transposed): Wab_T[pre, post] = Jab[post,pre] * block.T
        # So final Wab_T[j,i] where j=pre, i=post for matmul: rates @ Wab_T
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
                ) # (post, pre)

                # print('pre', pre, 'post', post,
                #       'Jab', self.cfg.Jab[post, pre],
                #       'n_pre', torch.mean( 1.0 * block.sum(dim=1)),
                #       'n_post', torch.mean(1.0 * block.sum(dim=0)))

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
            # Zero out train_mask for STP blocks (as in original initSTP)
            for _block_idx, (pre, post, _block_id) in enumerate(self.stp_block_specs):
                mask[geo.slices[pre], geo.slices[post]] = 0.0
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

        self.J_STP_param = nn.Parameter(j_stp.clone())
        # if s.TRAIN_J_STP:
        #     self.J_STP_param = nn.Parameter(j_stp.clone())
        # else:
        #     self.register_buffer("J_STP_param", j_stp.clone())

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
                # Extract the block from W_rec_base (which is already transposed)
                # W_rec_base[pre, post] layout, same as original Wab_T[pre, post]
                block = self.W_rec_base[geo.slices[pre], geo.slices[post]].clone() / denom
                self.stp_block_specs.append((pre, post, block_id))
                self.register_buffer(
                    f"_W_stp_base_{len(self.stp_block_specs) - 1}", block,
                )

                # Zero out the STP block from W_rec_base (as in original initSTP)
                with torch.no_grad():
                    self.W_rec_base[geo.slices[pre], geo.slices[post]].zero_()

    def _get_stp_base(self, block_idx: int) -> Tensor:
        return getattr(self, f"_W_stp_base_{block_idx}")

    def _j_stp_value(self) -> Tensor:
        """Return J_STP scalar (original code uses self.J_STP for all blocks)."""
        if self.J_STP_param is None:
            raise RuntimeError("J_STP_param is not initialized")
        return self.J_STP_param

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

        # ee = geo.slices[0]
        # k0 = torch.sqrt(torch.as_tensor(
        #     geo.Ka[0], device=self.device, dtype=W_rec.dtype,
        # ))

        # W_rec[ee, ee] = W_rec[ee, ee] + t.GAIN * W_train / k0
        # if t.CLAMP:
        #     W_rec = clamp_tensor(W_rec.T, 0, geo.slices).T

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
        j_stp = self._j_stp_value()

        blocks: list[Tensor] = []
        for block_idx, (pre, post, block_id) in enumerate(self.stp_block_specs):
            base = self._get_stp_base(block_idx)

            if block_idx == 0 and pre == 0 and post == 0:
                # First EE block: original logic
                # W_stp_T[0] = GAIN * J_STP * (W_stp_T[0] / fix_scale
                #               * (1.0 + Wab_train[ee, ee] / train_scale[0]))
                if W_train is not None and not t.TRAIN_EI:
                    ee = geo.slices[0]
                    block = (
                        t.GAIN * j_stp
                        * (base / fix_scale
                           * (1.0 + W_train[ee, ee] / t.train_scale[0]))
                    )
                else:
                    block = t.GAIN * j_stp * base / fix_scale

                if t.CLAMP:
                    block = clamp_tensor(block, 0, geo.slices)
            else:
                # Other blocks: W_STP[block_id] * J_STP * W_stp_T[k] / sqrt(Ka[pre])
                w_stp_scale = s.W_STP[block_id] if s.W_STP is not None else 1.0
                k_pre = torch.sqrt(torch.as_tensor(
                    geo.Ka[pre], device=self.device, dtype=dtype,
                ))
                block = w_stp_scale * j_stp * base / k_pre

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
