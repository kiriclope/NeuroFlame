"""State initialisation, persistence, and restore."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from src.config.types import StateConfig, Tensor
from src.dynamics.activation import Activation
from src.dynamics.plasticity.stp import Plasticity
from src.state.containers import NetworkState


def _terminal_mask(step: int, end_indices: Tensor, batch_size: int) -> Tensor:
    if end_indices.dim() == 1:
        if end_indices.shape[0] != batch_size:
            raise ValueError(
                f"end_indices shape {tuple(end_indices.shape)} incompatible "
                f"with batch_size={batch_size}"
            )
        return end_indices == step
    if end_indices.dim() == 2:
        if end_indices.shape[-1] != batch_size:
            raise ValueError(
                f"end_indices shape {tuple(end_indices.shape)} incompatible "
                f"with batch_size={batch_size}"
            )
        return end_indices[-1] == step
    raise ValueError(f"Unsupported end_indices shape: {tuple(end_indices.shape)}")


class StateManager(nn.Module):
    def __init__(self, cfg: StateConfig, activation: Activation) -> None:
        super().__init__()
        self.cfg = cfg
        self.activation = activation

        self.register_buffer("rates_last", None)
        self.register_buffer("rec_input_last", None)
        self.register_buffer("thresh_last", None)
        self.register_buffer("ff_state_last", None)
        self.register_buffer("hebb_rates_last", None)
        self.register_buffer("ff_adapt_thresh_last", None)
        self.register_buffer("end_mask", None)
        self.register_buffer("u_ff_stp_last", None)
        self.register_buffer("x_ff_stp_last", None)

        self._n_stp_blocks: int = 0
        self._stp_block_sizes: list[int] = []

    # ---- STP list helpers ---------------------------------------------

    def _clear_stp_buffers(self) -> None:
        names = [
            f"{prefix}{i}"
            for i in range(self._n_stp_blocks)
            for prefix in ("_u_stp_last_", "_x_stp_last_")
        ]
        for name in names:
            if name in self._buffers:
                delattr(self, name)
        self._n_stp_blocks = 0
        self._stp_block_sizes = []

    def _register_stp_list(self, u_list: list[Tensor], x_list: list[Tensor]) -> None:
        self._clear_stp_buffers()
        self._n_stp_blocks = len(u_list)
        self._stp_block_sizes = [u.shape[-1] for u in u_list]
        for i, (u, x) in enumerate(zip(u_list, x_list)):
            self.register_buffer(f"_u_stp_last_{i}", u.detach().clone())
            self.register_buffer(f"_x_stp_last_{i}", x.detach().clone())

    def _get_stp_lists(self) -> tuple[list[Tensor] | None, list[Tensor] | None]:
        if self._n_stp_blocks == 0:
            return None, None
        u_list, x_list = [], []
        for i in range(self._n_stp_blocks):
            u = self._buffers.get(f"_u_stp_last_{i}")
            x = self._buffers.get(f"_x_stp_last_{i}")
            if u is None or x is None:
                return None, None
            u_list.append(u)
            x_list.append(x)
        return u_list, x_list

    def _update_stp_indices(self, idx: Tensor, stp_modules: Sequence[Plasticity]) -> None:
        if self._n_stp_blocks == 0:
            self._register_stp_list(
                [torch.zeros_like(m.u_stp) for m in stp_modules],
                [torch.zeros_like(m.x_stp) for m in stp_modules],
            )
        for i, mod in enumerate(stp_modules):
            self._buffers[f"_u_stp_last_{i}"][idx] = mod.u_stp[idx].detach()
            self._buffers[f"_x_stp_last_{i}"][idx] = mod.x_stp[idx].detach()

    # ---- public API ---------------------------------------------------

    def initialize(self, ff_input: Tensor) -> NetworkState:
        c = self.cfg
        geo = c.geo
        if ff_input.dim() != 3:
            raise ValueError(f"ff_input must be 3D, got {tuple(ff_input.shape)}")

        batch_size = ff_input.shape[0]
        dtype = ff_input.dtype

        thresh = c.thresh[:batch_size].clone().to(device=geo.device, dtype=dtype)
        rec_input = torch.zeros(
            (int(c.IF_NMDA) + 1, batch_size, geo.N_NEURON),
            device=geo.device, dtype=dtype,
        )
        ff0 = ff_input[:, 0]
        rates = self.activation(ff0 + rec_input[0], func_name=c.TF_TYPE, thresh=thresh)

        return NetworkState(
            rates=rates,
            rec_input=rec_input,
            thresh=thresh,
            ff_state=torch.zeros_like(ff0),
            hebb_rates=rates.clone() if c.IF_HEBB else None,
            ff_adapt_thresh=torch.zeros_like(ff0) if c.IF_FF_ADAPT else None,
        )

    def restore_last_state(self) -> NetworkState:
        if any(
            x is None
            for x in (self.rates_last, self.rec_input_last,
                      self.thresh_last, self.ff_state_last)
        ):
            raise RuntimeError(
                "No saved state available. Call forward(init_state=True) first."
            )

        def _c(t: Tensor | None) -> Tensor | None:
            return None if t is None else t.clone()

        def _cl(lst: list[Tensor] | None) -> list[Tensor] | None:
            return None if lst is None else [x.clone() for x in lst]

        u_stp, x_stp = self._get_stp_lists()

        return NetworkState(
            rates=self.rates_last.clone(),
            rec_input=self.rec_input_last.clone(),
            thresh=self.thresh_last.clone(),
            ff_state=self.ff_state_last.clone(),
            hebb_rates=_c(self.hebb_rates_last),
            u_stp_last=_cl(u_stp),
            x_stp_last=_cl(x_stp),
            u_ff_stp_last=_c(self.u_ff_stp_last),
            x_ff_stp_last=_c(self.x_ff_stp_last),
            ff_adapt_thresh=_c(self.ff_adapt_thresh_last),
        )

    def init_last_state_buffers(self, state: NetworkState, batch_size: int) -> None:
        c = self.cfg
        geo = c.geo
        self.rates_last = torch.zeros_like(state.rates)
        self.rec_input_last = torch.zeros_like(state.rec_input)
        self.thresh_last = torch.zeros_like(state.thresh)
        self.ff_state_last = torch.zeros_like(state.ff_state)
        self.end_mask = torch.ones(
            (batch_size, 1), device=geo.device, dtype=state.rates.dtype,
        )
        self.hebb_rates_last = (
            torch.zeros_like(state.hebb_rates)
            if c.IF_HEBB and state.hebb_rates is not None else None
        )
        self.ff_adapt_thresh_last = (
            torch.zeros_like(state.ff_adapt_thresh)
            if c.IF_FF_ADAPT and state.ff_adapt_thresh is not None else None
        )
        self._clear_stp_buffers()
        self.u_ff_stp_last = None
        self.x_ff_stp_last = None

    def save_terminal_state(
        self,
        step: int,
        state: NetworkState,
        end_indices: Tensor,
        stp_modules: Sequence[Plasticity] | None = None,
        ff_stp_module: Plasticity | None = None,
    ) -> None:
        if self.end_mask is None:
            raise RuntimeError("Last-state buffers not initialized.")

        batch_size = state.rates.shape[0]
        mask = _terminal_mask(step, end_indices, batch_size)
        if not mask.any():
            return

        idx = mask.nonzero(as_tuple=True)[0]
        self.end_mask[idx, 0] = float("nan")
        self.rates_last[idx] = state.rates[idx].detach()
        self.rec_input_last[:, idx] = state.rec_input[:, idx].detach()
        self.thresh_last[idx] = state.thresh[idx].detach()
        self.ff_state_last[idx] = state.ff_state[idx].detach()

        if self.hebb_rates_last is not None and state.hebb_rates is not None:
            self.hebb_rates_last[idx] = state.hebb_rates[idx].detach()
        if self.ff_adapt_thresh_last is not None and state.ff_adapt_thresh is not None:
            self.ff_adapt_thresh_last[idx] = state.ff_adapt_thresh[idx].detach()

        if stp_modules is not None:
            self._update_stp_indices(idx, stp_modules)
        if ff_stp_module is not None:
            if self.u_ff_stp_last is None or self.x_ff_stp_last is None:
                self.u_ff_stp_last = torch.zeros_like(ff_stp_module.u_stp)
                self.x_ff_stp_last = torch.zeros_like(ff_stp_module.x_stp)
            self.u_ff_stp_last[idx] = ff_stp_module.u_stp[idx].detach()
            self.x_ff_stp_last[idx] = ff_stp_module.x_stp[idx].detach()
