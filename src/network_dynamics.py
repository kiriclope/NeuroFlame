import torch
from torch import nn

from src.activation import Activation
from src.network_weights import recurrent_matmul


class DynamicsCore(nn.Module):
    """
    One-step dynamics update.
    """

    def __init__(self, cfg, stp_block_index):
        super().__init__()
        self.cfg = cfg
        self.stp_block_index = stp_block_index
        self.activation = Activation()

        for k, v in vars(cfg).items():
            setattr(self, k, v)

        self.thresh_ff = None

    def reset_buffers(self, batch_size: int, ff_shape: torch.Size | None = None):
        if self.IF_FF_ADAPT and ff_shape is not None:
            self.thresh_ff = torch.zeros(ff_shape, device=self.device)

    def _compute_stp_hidden(self, rates, batch_size, stp_modules, W_stp):
        if not self.IF_STP or stp_modules is None or W_stp is None:
            return None

        hidden_stp_total = torch.zeros((batch_size, self.N_NEURON), device=self.device)

        for k, (pre, post, _) in enumerate(self.stp_block_index):
            aux = stp_modules[k](rates[:, self.slices[pre]])
            hidden_block = aux @ W_stp[k]
            hidden_stp_total[:, self.slices[post]] += hidden_block

        return hidden_stp_total

    def step(self, state, ff_step, W_rec, W_stp, stp_modules, ff_stp_module):
        rates = state.rates
        rec_input = state.rec_input
        thresh = state.thresh
        ff_prev = state.ff_prev

        batch_size = rates.shape[0]
        hidden = recurrent_matmul(rates, W_rec)

        hidden_stp_total = self._compute_stp_hidden(rates, batch_size, stp_modules, W_stp)
        if hidden_stp_total is not None:
            hidden = hidden + hidden_stp_total

        if self.IF_FF_STP and ff_stp_module is not None:
            hidden_ff_stp = torch.sign(ff_step) * (
                ff_stp_module(torch.abs(ff_step) / torch.sqrt(self.Ka[0])) * torch.sqrt(self.Ka[0])
            )
            ff_step = ff_step + hidden_ff_stp

        if self.IF_FF_DYN:
            ff_step = ff_prev * self.EXP_FF + ff_step * (1.0 - self.EXP_FF)
            ff_prev = ff_step

        if self.IF_BATCH_J:
            hidden[:, self.slices[0]].add_(
                self.Jab_batch * rates[:, self.slices[0]] @ self.W_batch_T
            )

        if self.SYN_DYN:
            rec_input[0] = rec_input[0] * self.EXP_DT_TAU_SYN + hidden * (1.0 - self.EXP_DT_TAU_SYN)
        else:
            rec_input[0] = hidden

        if self.IF_FF_ADAPT:
            if self.thresh_ff is None:
                self.thresh_ff = torch.zeros_like(ff_step)

            ff_step = torch.sign(ff_step) * nn.ReLU()(ff_step - self.thresh_ff)
            self.thresh_ff = (
                self.thresh_ff * self.EXP_FF_ADAPT
                + nn.ReLU()(ff_step) * self.A_FF_ADAPT * (1.0 - self.EXP_FF_ADAPT)
            )

        net_input = ff_step + rec_input[0]

        if self.IF_NMDA:
            if W_rec.dim() == 2:
                hidden_nmda = recurrent_matmul(rates[:, self.slices[0]], W_rec[self.slices[0]])
            else:
                hidden_nmda = recurrent_matmul(rates[:, self.slices[0]], W_rec[:, self.slices[0]])

            if hidden_stp_total is not None:
                hidden_nmda[:, self.slices[0]] += hidden_stp_total[:, self.slices[0]]

            rec_input[1] = (
                rec_input[1] * self.EXP_DT_TAU_NMDA
                + self.R_NMDA * hidden_nmda * (1.0 - self.EXP_DT_TAU_NMDA)
            )
            net_input = net_input + rec_input[1]

        non_linear = self.activation(net_input, func_name=self.TF_TYPE, thresh=thresh)

        if self.RATE_DYN:
            rates = rates * self.EXP_DT_TAU + non_linear * (1.0 - self.EXP_DT_TAU)
        else:
            rates = non_linear

        if self.IF_ADAPT:
            thresh[:, self.slices[0]] = (
                thresh[:, self.slices[0]] * self.EXP_ADAPT
                + rates[:, self.slices[0]].detach() * self.A_ADAPT * (1.0 - self.EXP_ADAPT)
            )

        state.rates = rates
        state.rec_input = rec_input
        state.thresh = thresh
        state.ff_prev = ff_prev
        return state
