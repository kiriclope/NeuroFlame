import torch

from src.ff_input import init_ff_input
from src.network_state import NetworkState


class StateManager:
    def __init__(self, cfg, activation):
        self.cfg = cfg
        self.activation = activation

        for k, v in vars(cfg).items():
            setattr(self, k, v)

        self.rates_last = None
        self.rec_input_last = None
        self.thresh_last = None
        self.ff_prev_last = None
        self.hebb_rates_last = None
        self.end_mask = None

    def prepare_ff_input(self, ff_input):
        if ff_input is None:
            ff_input = init_ff_input(self)
        return ff_input.to(self.device)

    def initialize(self, ff_input):
        self.N_BATCH = ff_input.shape[0]

        thresh = self.thresh[: self.N_BATCH].clone()
        rec_input = torch.randn(
            (self.IF_NMDA + 1, self.N_BATCH, self.N_NEURON),
            device=self.device,
        )

        ff0 = ff_input if self.LIVE_FF_UPDATE else ff_input[:, 0]

        if self.IF_FF_ADAPT:
            self.thresh_ff = torch.zeros_like(ff_input[:, 0])

        rates = self.activation(ff0 + rec_input[0], func_name=self.TF_TYPE, thresh=thresh)
        hebb_rates = rates.clone() if self.IF_HEBB else None

        return NetworkState(
            rates=rates,
            rec_input=rec_input,
            thresh=thresh,
            ff_prev=torch.zeros_like(ff0),
            hebb_rates=hebb_rates,
        )

    def restore(self):
        return NetworkState(
            rates=self.rates_last,
            rec_input=self.rec_input_last,
            thresh=self.thresh_last,
            ff_prev=self.ff_prev_last,
            hebb_rates=self.hebb_rates_last,
        )

    def init_last_state_buffers(self, state, batch_size):
        self.rates_last = torch.zeros_like(state.rates.detach())
        self.rec_input_last = torch.zeros_like(state.rec_input.detach())
        self.thresh_last = torch.zeros_like(state.thresh.detach())
        self.ff_prev_last = torch.zeros_like(state.ff_prev.detach())
        self.end_mask = torch.ones((batch_size, 1), device=self.device)

        if self.IF_HEBB and state.hebb_rates is not None:
            self.hebb_rates_last = torch.zeros_like(state.hebb_rates.detach())

    def maybe_save(self, step, state, end_indices, stp_modules=None, ff_stp_module=None):
        if not torch.any(step == end_indices[-1]):
            return

        end_idx = torch.where(step == end_indices[-1])[0]

        self.end_mask[end_idx, 0] = float("nan")
        self.rates_last[end_idx] = state.rates[end_idx].detach()
        self.rec_input_last[:, end_idx] = state.rec_input[:, end_idx].detach()
        self.thresh_last[end_idx] = state.thresh[end_idx].detach()
        self.ff_prev_last[end_idx] = state.ff_prev[end_idx].detach()

        if self.IF_HEBB and state.hebb_rates is not None:
            self.hebb_rates_last[end_idx] = state.hebb_rates[end_idx].detach()

        if stp_modules is not None:
            if state.u_stp_last is None:
                state.u_stp_last = [torch.zeros_like(m.u_stp) for m in stp_modules]
                state.x_stp_last = [torch.zeros_like(m.x_stp) for m in stp_modules]
            for k, mod in enumerate(stp_modules):
                state.u_stp_last[k][end_idx] = mod.u_stp[end_idx].detach()
                state.x_stp_last[k][end_idx] = mod.x_stp[end_idx].detach()

        if ff_stp_module is not None:
            if state.u_ff_stp_last is None:
                state.u_ff_stp_last = torch.zeros_like(ff_stp_module.u_stp)
                state.x_ff_stp_last = torch.zeros_like(ff_stp_module.x_stp)
            state.u_ff_stp_last[end_idx] = ff_stp_module.u_stp[end_idx].detach()
            state.x_ff_stp_last[end_idx] = ff_stp_module.x_stp[end_idx].detach()
