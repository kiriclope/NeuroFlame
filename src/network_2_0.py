import warnings

import torch
from torch import nn

from src.configuration import Configuration
from src.ff_input import live_ff_input, init_ff_input
from src.utils import set_seed, clear_cache

from src.network_dynamics import DynamicsCore
from src.network_outputs import OutputCollector
from src.network_plasticity import PlasticityManager
from src.network_state_mgr import StateManager
from src.network_weights import WeightBuilder

warnings.filterwarnings("ignore")


class Network(nn.Module):
    """
    Thin orchestrator around state, weights, plasticity, dynamics, and outputs.
    """

    def __init__(self, conf_name, repo_root, **kwargs):
        super().__init__()

        self.config = Configuration(conf_name, repo_root)(**kwargs)
        self._load_config(self.config)

        self.weight_builder = WeightBuilder(self.config)
        self.state_manager = StateManager(self.config, activation=self._activation_fn())
        self.plasticity_manager = PlasticityManager(
            self.config,
            W_rec_base=self.weight_builder.W_rec_base,
            stp_block_index=self.weight_builder.stp_block_index,
        )
        self.dynamics = DynamicsCore(
            self.config,
            stp_block_index=self.weight_builder.stp_block_index,
        )

        set_seed(-1)
        clear_cache()

    def _load_config(self, config):
        for key, value in vars(config).items():
            if hasattr(self, key):
                raise ValueError(f"Config key collision: {key}")
            setattr(self, key, value)

    def _activation_fn(self):
        from src.activation import Activation
        return Activation()

    def init_ff_input(self):
        return init_ff_input(self).to(self.device)

    def _prepare_ff_input(self, ff_input):
        if ff_input is None:
            ff_input = init_ff_input(self)
        return ff_input.to(self.device)

    def forward(self, ff_input=None, RET_STP=0, RET_REC=0, IF_INIT=1):
        ff_input = self._prepare_ff_input(ff_input)

        if IF_INIT:
            state = self.state_manager.initialize(ff_input)
            self.dynamics.reset_buffers(
                batch_size=ff_input.shape[0],
                ff_shape=ff_input[:, 0].shape if not self.LIVE_FF_UPDATE else ff_input.shape,
            )
            if self.TRAINING == 0:
                self.state_manager.init_last_state_buffers(state, ff_input.shape[0])
        else:
            state = self.state_manager.restore()

        batch_size = state.rates.shape[0]
        weights = self.weight_builder.build_all(batch_size=batch_size)

        stp_modules = self.plasticity_manager.make_stp_modules(
            state, batch_size=batch_size, reset=bool(IF_INIT)
        )
        ff_stp_module = self.plasticity_manager.make_ff_stp_module(
            state, batch_size=batch_size, reset=bool(IF_INIT)
        )

        if self.IF_OPTO and weights.W_stp is not None and len(weights.W_stp) > 0:
            rand_idx = torch.randperm(
                weights.W_stp[0].size(0),
                device=weights.W_stp[0].device,
            )[: self.N_OPTO]
            weights.W_stp[0][rand_idx] = 0

        collector = OutputCollector(
            self.config,
            return_rec=bool(RET_REC),
            return_stp=bool(RET_STP),
        )

        for step in range(self.N_STEPS):
            if self.IF_HEBB and step >= self.N_HEBB:
                weights.W_rec, state.hebb_rates = self.plasticity_manager.apply_hebbian(
                    state.hebb_rates,
                    state.rates,
                    weights.W_rec,
                )

            if self.RATE_NOISE:
                rate_noise = torch.randn((batch_size, self.N_NEURON), device=self.device)
                state.rates = state.rates + rate_noise * self.VAR_RATE

            ff_step = (
                live_ff_input(self, step, ff_input)
                if self.LIVE_FF_UPDATE
                else ff_input[:, step]
            )

            state = self.dynamics.step(
                state=state,
                ff_step=ff_step,
                W_rec=weights.W_rec,
                W_stp=weights.W_stp,
                stp_modules=stp_modules,
                ff_stp_module=ff_stp_module,
            )

            if self.TRAINING == 0:
                self.state_manager.maybe_save(
                    step,
                    state,
                    self.end_indices,
                    stp_modules=stp_modules,
                    ff_stp_module=ff_stp_module,
                )

            collector.update(step, state, stp_modules=stp_modules)

        readout = None
        if self.LR_TRAIN:
            readout = torch.stack(collector.outputs["rates"], dim=1)
            readout = readout @ self.weight_builder.get_readout() / self.Na[0]

        final = collector.finalize(readout=readout)

        self.rates_list = final.rates
        if final.rec_input is not None:
            self.rec_input = final.rec_input
        if final.stp_u is not None:
            self.u_list = final.stp_u
        if final.stp_x is not None:
            self.x_list = final.stp_x
        if final.readout is not None:
            self.readout = final.readout

        if self.IF_HEBB:
            self.W_hebb_T = weights.W_rec

        return final.rates
