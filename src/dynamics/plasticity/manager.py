"""Plasticity mechanism construction and Hebbian update logic."""
from __future__ import annotations

import torch
from torch import nn

from src.config.types import HebbianConfig, NetworkGeometry, STPModuleConfig
from src.dynamics.plasticity.hebbian import Hebbian
from src.dynamics.plasticity.stp import Plasticity
from src.state.containers import NetworkState
from src.weights.builder import STPBlockSpec

Tensor = torch.Tensor


class PlasticityManager(nn.Module):
    def __init__(
        self,
        geo: NetworkGeometry,
        stp_cfg: STPModuleConfig,
        hebb_cfg: HebbianConfig,
        W_rec_base: Tensor,
        stp_block_index: list[STPBlockSpec],
    ) -> None:
        super().__init__()
        self.geo = geo
        self.stp_cfg = stp_cfg
        self.hebb_cfg = hebb_cfg
        self.register_buffer("_W_rec_base", W_rec_base)
        self.stp_block_index = stp_block_index

        self.hebb: Hebbian | None = (
            Hebbian(hebb_cfg.ETA, hebb_cfg.DT, hebb_cfg.HEBB_TYPE, hebb_cfg.HEBB_FRAC)
            if hebb_cfg.IF_HEBB else None
        )

    @staticmethod
    def _restore_stp_state(
        mod: Plasticity, u_saved: Tensor, x_saved: Tensor, label: str,
    ) -> None:
        expected = tuple(mod.u_stp.shape)
        if tuple(u_saved.shape) != expected:
            raise RuntimeError(
                f"u_stp shape mismatch at {label}: {tuple(u_saved.shape)} != {expected}"
            )
        if tuple(x_saved.shape) != expected:
            raise RuntimeError(
                f"x_stp shape mismatch at {label}: {tuple(x_saved.shape)} != {expected}"
            )
        mod.u_stp = u_saved.to(device=mod.u_stp.device, dtype=mod.u_stp.dtype)
        mod.x_stp = x_saved.to(device=mod.x_stp.device, dtype=mod.x_stp.dtype)

    def make_stp_modules(
        self, state: NetworkState, batch_size: int, reset: bool,
    ) -> list[Plasticity] | None:
        sc = self.stp_cfg
        geo = self.geo
        if not sc.IF_STP:
            return None

        n_blocks = len(self.stp_block_index)
        if not reset:
            if state.u_stp_last is None or state.x_stp_last is None:
                raise RuntimeError("STP state missing on restore path")
            if len(state.u_stp_last) != n_blocks:
                raise RuntimeError("STP state length mismatch")

        stp_modules: list[Plasticity] = []
        for block_idx, (pre, _post, block_id) in enumerate(self.stp_block_index):
            mod = Plasticity(
                sc.USE[block_id], sc.TAU_FAC[block_id], sc.TAU_REC[block_id],
                sc.DT, (batch_size, geo.Na[pre]),
                device=geo.device,
                STP_TYPE=sc.STP_TYPE, IF_INIT=reset,
            )
            if not reset:
                self._restore_stp_state(
                    mod, state.u_stp_last[block_idx],
                    state.x_stp_last[block_idx], f"block {block_idx}",
                )
            stp_modules.append(mod)

        state.u_stp_last = [m.u_stp.detach().clone() for m in stp_modules]
        state.x_stp_last = [m.x_stp.detach().clone() for m in stp_modules]
        return stp_modules

    def make_ff_stp_module(
        self, state: NetworkState, batch_size: int, reset: bool,
    ) -> Plasticity | None:
        sc = self.stp_cfg
        geo = self.geo
        if not sc.IF_FF_STP:
            return None

        mod = Plasticity(
            sc.FF_USE, sc.TAU_FF_FAC, sc.TAU_FF_REC, sc.DT,
            (batch_size, geo.N_NEURON),
            device=geo.device,
            STP_TYPE=sc.STP_TYPE, IF_INIT=reset,
        )
        if not reset:
            if state.u_ff_stp_last is None or state.x_ff_stp_last is None:
                raise RuntimeError("FF-STP state missing on restore path")
            self._restore_stp_state(
                mod, state.u_ff_stp_last, state.x_ff_stp_last, "FF-STP",
            )

        state.u_ff_stp_last = mod.u_stp.detach().clone()
        state.x_ff_stp_last = mod.x_stp.detach().clone()
        return mod

    def apply_hebbian(
        self, hebb_rates: Tensor | None, rates: Tensor, W_rec: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        hc = self.hebb_cfg
        geo = self.geo
        if not hc.IF_HEBB:
            return W_rec, hebb_rates
        if self.hebb is None:
            raise RuntimeError("IF_HEBB=True but Hebbian module not initialized")
        if hebb_rates is None:
            raise RuntimeError("hebb_rates is None while IF_HEBB=True")
        if W_rec.dim() != 3:
            raise ValueError(
                f"Hebbian path expects batched W_rec, got shape {tuple(W_rec.shape)}"
            )

        scale = torch.sqrt(torch.as_tensor(
            geo.Ka[0], device=W_rec.device, dtype=W_rec.dtype,
        ))

        for pre in range(geo.N_POP):
            for post in range(geo.N_POP):
                block_id = pre + post * geo.N_POP
                if not hc.IS_HEBB[block_id]:
                    continue
                W_hebb = self.hebb(
                    rates[:, geo.slices[pre]], rates[:, geo.slices[post]],
                    hebb_rates[:, geo.slices[pre]], hebb_rates[:, geo.slices[post]],
                ) / scale
                base = self._W_rec_base[geo.slices[pre], geo.slices[post]].unsqueeze(0)
                W_rec[:, geo.slices[pre], geo.slices[post]] = base + W_hebb

        W_rec[:, geo.slices[0]] = W_rec[:, geo.slices[0]].clamp(min=0.0)
        if geo.N_POP > 1:
            W_rec[:, geo.slices[1]] = W_rec[:, geo.slices[1]].clamp(max=0.0)

        if hc.HEBB_TYPE == "bcm":
            assert hc.EXP_HEBB is not None, (
                "IF_HEBB=True with HEBB_TYPE='bcm' but EXP_HEBB is None"
            )
            hebb_rates = hebb_rates * hc.EXP_HEBB + rates * (1.0 - hc.EXP_HEBB)

        return W_rec, hebb_rates
