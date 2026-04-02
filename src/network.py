"""Top-level recurrent network model."""
from __future__ import annotations

import torch
from torch import nn

from src.config import Configuration
from src.dynamics import Activation, DynamicsCore
from src.dynamics.plasticity import PlasticityManager
from src.io import FFInputBuilder
from src.state import ForwardOutputs, OutputCollector, StateManager
from src.weights import WeightBuilder, WeightOutputs

Tensor = torch.Tensor


class Network(nn.Module):
    def __init__(self, conf_name, repo_root, **kwargs) -> None:
        super().__init__()

        self.config = Configuration(conf_name, repo_root)(**kwargs)
        c = self.config
        activation = Activation()

        self.weight_builder = WeightBuilder(c.weight_cfg)
        self.ff_input_builder = FFInputBuilder(
            c.ff_cfg, low_rank=self.weight_builder.low_rank,
        )
        self.state_manager = StateManager(c.state_cfg, activation=activation)
        self.plasticity_manager = PlasticityManager(
            geo=c.geo,
            stp_cfg=c.stp_cfg,
            hebb_cfg=c.hebb_cfg,
            W_rec_base=self.weight_builder.W_rec_base,
            stp_block_index=self.weight_builder.stp_block_specs,
        )
        self.dynamics = DynamicsCore(
            c.recurrent_cfg,
            stp_block_index=self.weight_builder.stp_block_specs,
            activation=activation,
        )

        self.sim = c.sim_cfg

        # Legacy compatibility
        self.rates_list: Tensor | None = None
        self.rec_input: Tensor | None = None
        self.u_list: Tensor | None = None
        self.x_list: Tensor | None = None
        self.readout: Tensor | None = None
        self.W_hebb_T: Tensor | None = None

    @property
    def cfg(self):
        return self.config

    def _init_ff_input(self, ff_input: Tensor | None) -> Tensor:
        sim = self.sim
        geo = sim.geo
        time = sim.time
        if ff_input is None:
            ff_input = self.ff_input_builder.build().to(geo.device)
        ff_input = ff_input.to(geo.device)
        if ff_input.dim() != 3:
            raise ValueError(f"ff_input must be 3D, got {tuple(ff_input.shape)}")
        if ff_input.shape[1] < time.N_STEPS:
            raise ValueError(
                f"ff_input has {ff_input.shape[1]} steps, need >= {time.N_STEPS}"
            )
        return ff_input

    def _apply_opto(self, weights: WeightOutputs) -> None:
        sim = self.sim
        if not sim.IF_OPTO or weights.W_stp_blocks is None or not weights.W_stp_blocks:
            return
        weights.W_stp_blocks[0] = weights.W_stp_blocks[0].clone()
        rand_idx = torch.randperm(
            weights.W_stp_blocks[0].size(0), device=weights.W_stp_blocks[0].device,
        )[: sim.N_OPTO]
        weights.W_stp_blocks[0][rand_idx] = 0

    def _update_legacy(self, final: ForwardOutputs, weights: WeightOutputs) -> None:
        self.rates_list = final.rates
        self.rec_input = final.rec_input
        self.u_list = final.stp_u
        self.x_list = final.stp_x
        self.readout = final.readout
        self.W_hebb_T = weights.W_rec if self.sim.IF_HEBB else None

    def forward(
        self,
        ff_input: Tensor | None = None,
        *,
        return_stp: bool = False,
        return_rec: bool = False,
        init_state: bool = True,
        return_outputs: bool = False,
    ) -> Tensor | ForwardOutputs:
        sim = self.sim
        geo = sim.geo
        time = sim.time
        ff_input = self._init_ff_input(ff_input)

        if init_state:
            state = self.state_manager.initialize(ff_input)
            if sim.TRAINING == 0:
                self.state_manager.init_last_state_buffers(state, ff_input.shape[0])
        else:
            state = self.state_manager.restore_last_state()

        batch_size = state.rates.shape[0]
        weights = self.weight_builder.build_all(
            batch_size=batch_size, expand_hebbian=sim.IF_HEBB,
        )
        self._apply_opto(weights)

        stp_modules = self.plasticity_manager.make_stp_modules(
            state, batch_size=batch_size, reset=init_state,
        )
        ff_stp_module = self.plasticity_manager.make_ff_stp_module(
            state, batch_size=batch_size, reset=init_state,
        )

        collector = OutputCollector(
            self.config.output_cfg, return_rec=return_rec, return_stp=return_stp,
        )
        hebb_start = time.N_STEADY

        for step in range(time.N_STEPS):
            if sim.IF_HEBB and step >= hebb_start:
                weights.W_rec, state.hebb_rates = self.plasticity_manager.apply_hebbian(
                    state.hebb_rates, state.rates, weights.W_rec,
                )

            if sim.RATE_NOISE:
                noise = torch.randn(
                    (batch_size, geo.N_NEURON), device=geo.device, dtype=state.rates.dtype,
                )
                state.rates = state.rates + noise * sim.VAR_RATE

            self.dynamics.step(
                state=state, ff_step=ff_input[:, step],
                W_rec=weights.W_rec, W_stp=weights.W_stp_blocks,
                stp_modules=stp_modules, ff_stp_module=ff_stp_module,
            )

            if sim.TRAINING == 0:
                self.state_manager.save_terminal_state(
                    step=step, state=state, end_indices=sim.end_indices,
                    stp_modules=stp_modules, ff_stp_module=ff_stp_module,
                )

            collector.update(step, state, stp_modules=stp_modules)

        readout: Tensor | None = None
        if sim.LR_TRAIN:
            ro = self.weight_builder.get_readout()
            if ro is None:
                raise RuntimeError("LR_TRAIN=True but no readout available")
            readout = collector.stacked_rates() @ ro / geo.Na[0]

        final = collector.finalize(readout=readout)
        self._update_legacy(final, weights)

        return final if return_outputs else final.rates
