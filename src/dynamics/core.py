"""Single-step recurrent dynamics."""
from __future__ import annotations

import torch
from torch import nn

from src.config.types import RecurrentConfig
from src.dynamics.activation import Activation
from src.dynamics.plasticity.stp import Plasticity
from src.state.containers import NetworkState
from src.weights.builder import STPBlockSpec, recurrent_matmul

Tensor = torch.Tensor


class DynamicsCore(nn.Module):
    def __init__(
        self, cfg: RecurrentConfig, stp_block_index: list[STPBlockSpec],
        activation: Activation,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stp_block_index = stp_block_index
        self.activation = activation

    def _compute_stp_hidden(
        self, rates: Tensor, batch_size: int,
        stp_modules: list[Plasticity], W_stp: list[Tensor],
    ) -> Tensor:
        geo = self.cfg.geo
        hidden = torch.zeros(
            (batch_size, geo.N_NEURON), device=rates.device, dtype=rates.dtype,
        )
        for block_idx, (pre, post, _) in enumerate(self.stp_block_index):
            stp_out = stp_modules[block_idx](rates[:, geo.slices[pre]])
            hidden[:, geo.slices[post]] += stp_out @ W_stp[block_idx]
        return hidden

    def step(
        self,
        state: NetworkState,
        ff_step: Tensor,
        W_rec: Tensor,
        W_stp: list[Tensor] | None,
        stp_modules: list[Plasticity] | None,
        ff_stp_module: Plasticity | None,
    ) -> None:
        c = self.cfg
        geo = c.geo
        rates = state.rates
        rec_input = state.rec_input
        batch_size = rates.shape[0]

        hidden = recurrent_matmul(rates, W_rec)

        hidden_stp: Tensor | None = None
        if c.IF_STP and stp_modules is not None and W_stp is not None:
            hidden_stp = self._compute_stp_hidden(rates, batch_size, stp_modules, W_stp)
            hidden = hidden + hidden_stp

        if c.IF_FF_STP and ff_stp_module is not None:
            k0 = torch.sqrt(torch.as_tensor(geo.Ka[0], device=ff_step.device, dtype=ff_step.dtype))
            ff_stp_hidden = torch.sign(ff_step) * (ff_stp_module(torch.abs(ff_step) / k0) * k0)
            ff_step = ff_step + ff_stp_hidden

        if c.IF_FF_DYN:
            ff_step = state.ff_state * c.EXP_FF + ff_step * (1.0 - c.EXP_FF)
            state.ff_state = ff_step

        if c.IF_BATCH_J:
            hidden[:, geo.slices[0]].add_(c.Jab_batch * (rates[:, geo.slices[0]] @ c.W_batch_T))

        if c.SYN_DYN:
            rec_input[0].mul_(c.EXP_DT_TAU_SYN).add_(hidden * (1.0 - c.EXP_DT_TAU_SYN))
        else:
            rec_input[0] = hidden

        if c.IF_FF_ADAPT:
            if state.ff_adapt_thresh is None:
                state.ff_adapt_thresh = torch.zeros_like(ff_step)
            adapted = torch.sign(ff_step) * torch.relu(torch.abs(ff_step) - state.ff_adapt_thresh)
            state.ff_adapt_thresh = (
                state.ff_adapt_thresh * c.EXP_FF_ADAPT
                + torch.abs(adapted) * c.A_FF_ADAPT * (1.0 - c.EXP_FF_ADAPT)
            )
            ff_step = adapted

        net_input = ff_step + rec_input[0]

        if c.IF_NMDA:
            exc_rates = rates[:, geo.slices[0]]
            if W_rec.dim() == 2:
                hidden_nmda = recurrent_matmul(exc_rates, W_rec[geo.slices[0]])
            elif W_rec.dim() == 3:
                hidden_nmda = recurrent_matmul(exc_rates, W_rec[:, geo.slices[0]])
            else:
                raise ValueError(f"Invalid W_rec dim for NMDA: {W_rec.dim()}")

            if hidden_stp is not None:
                hidden_nmda = hidden_nmda.clone()
                hidden_nmda[:, geo.slices[0]] += hidden_stp[:, geo.slices[0]]

            rec_input[1].mul_(c.EXP_DT_TAU_NMDA).add_(
                c.R_NMDA * hidden_nmda * (1.0 - c.EXP_DT_TAU_NMDA)
            )
            net_input = net_input + rec_input[1]

        non_linear = self.activation(net_input, func_name=c.TF_TYPE, thresh=state.thresh)

        if c.RATE_DYN:
            state.rates = rates * c.EXP_DT_TAU + non_linear * (1.0 - c.EXP_DT_TAU)
        else:
            state.rates = non_linear

        if c.IF_ADAPT:
            state.thresh[:, geo.slices[0]] = (
                state.thresh[:, geo.slices[0]] * c.EXP_ADAPT
                + state.rates[:, geo.slices[0]].detach() * c.A_ADAPT * (1.0 - c.EXP_ADAPT)
            )
