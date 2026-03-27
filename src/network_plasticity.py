import torch
from torch import nn

from src.hebbian import Hebbian
from src.plasticity import Plasticity


class PlasticityManager(nn.Module):
    """
    Builds per-forward plasticity modules and applies Hebbian updates.
    """

    def __init__(self, cfg, W_rec_base: torch.Tensor, stp_block_index):
        super().__init__()
        self.cfg = cfg
        self.W_rec_base = W_rec_base
        self.stp_block_index = stp_block_index

        for k, v in vars(cfg).items():
            setattr(self, k, v)

        self.hebb = Hebbian(self.ETA, self.DT, self.HEBB_TYPE, self.HEBB_FRAC) if self.IF_HEBB else None

    def make_stp_modules(self, state, batch_size: int, reset: bool):
        if not self.IF_STP:
            return None

        stp_modules = []
        u_last = [] if reset else state.u_stp_last
        x_last = [] if reset else state.x_stp_last

        for pre, post, idx in self.stp_block_index:
            mod = Plasticity(
                self.USE[idx],
                self.TAU_FAC[idx],
                self.TAU_REC[idx],
                self.DT,
                (batch_size, self.Na[pre]),
                STP_TYPE=self.STP_TYPE,
                IF_INIT=reset,
                device=self.device,
            )

            if reset:
                u_last.append(torch.zeros_like(mod.u_stp.detach()))
                x_last.append(torch.zeros_like(mod.x_stp.detach()))
            else:
                mod.u_stp = u_last[len(stp_modules)].detach()
                mod.x_stp = x_last[len(stp_modules)].detach()

            stp_modules.append(mod)

        state.u_stp_last = u_last
        state.x_stp_last = x_last
        return stp_modules

    def make_ff_stp_module(self, state, batch_size: int, reset: bool):
        if not self.IF_FF_STP:
            return None

        mod = Plasticity(
            self.FF_USE,
            self.TAU_FF_FAC,
            self.TAU_FF_REC,
            self.DT,
            (batch_size, self.N_NEURON),
            STP_TYPE=self.STP_TYPE,
            IF_INIT=reset,
            device=self.device,
        )

        if reset:
            state.u_ff_stp_last = torch.zeros_like(mod.u_stp)
            state.x_ff_stp_last = torch.zeros_like(mod.x_stp)
        else:
            mod.u_stp = state.u_ff_stp_last
            mod.x_stp = state.x_ff_stp_last

        return mod

    def apply_hebbian(self, hebb_rates: torch.Tensor, rates: torch.Tensor, W_rec: torch.Tensor):
        if not self.IF_HEBB:
            return W_rec, hebb_rates

        for pre in range(self.N_POP):
            pre_rates = rates[:, self.slices[pre]]
            hebb_pre = hebb_rates[:, self.slices[pre]]

            for post in range(self.N_POP):
                idx = pre + post * self.N_POP
                if not self.IS_HEBB[idx]:
                    continue

                post_rates = rates[:, self.slices[post]]
                hebb_post = hebb_rates[:, self.slices[post]]

                W_hebb = self.hebb(pre_rates, post_rates, hebb_pre, hebb_post)
                W_hebb = W_hebb / torch.sqrt(self.Ka[0])

                W_rec[:, self.slices[pre], self.slices[post]] = (
                    self.W_rec_base[self.slices[pre], self.slices[post]].unsqueeze(0) + W_hebb
                )

        W_rec[:, self.slices[0]] = W_rec[:, self.slices[0]].clamp(min=0.0)
        if self.N_POP > 1:
            W_rec[:, self.slices[1]] = W_rec[:, self.slices[1]].clamp(max=0.0)

        if self.HEBB_TYPE == "bcm":
            hebb_rates = hebb_rates * self.EXP_HEBB + rates * (1.0 - self.EXP_HEBB)

        return W_rec, hebb_rates
