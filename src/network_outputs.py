import torch

from src.network_state import ForwardOutputs
from src.utils import print_activity


class OutputCollector:
    def __init__(self, cfg, return_rec: bool = False, return_stp: bool = False):
        self.cfg = cfg
        self.return_rec = return_rec
        self.return_stp = return_stp

        for k, v in vars(cfg).items():
            setattr(self, k, v)

        self.outputs = {
            "rates": [],
            "rec": [],
            "stp_u": [],
            "stp_x": [],
        }

        self.mv_rates = 0
        self.mv_rec = 0

    def update(self, step: int, state, stp_modules=None):
        if step >= (self.N_STEADY + self.N_HEBB - self.N_WINDOW - 1):
            self.mv_rates = self.mv_rates + state.rates
            if self.return_rec:
                self.mv_rec = self.mv_rec + state.rec_input

        emit = (step >= (self.N_STEADY + self.N_HEBB)) and (step % self.N_WINDOW == 0)
        if not emit:
            return

        if self.VERBOSE:
            print_activity(self, step, state.rates)

        self.outputs["rates"].append(self.mv_rates[:, self.slices[0]] / self.N_WINDOW)

        if self.return_rec:
            self.outputs["rec"].append(self.mv_rec[..., self.slices[0]] / self.N_WINDOW)

        if self.IF_STP and self.return_stp and stp_modules is not None:
            self.outputs["stp_u"].append(stp_modules[0].u_stp)
            self.outputs["stp_x"].append(stp_modules[0].x_stp)

        self.mv_rates = 0
        self.mv_rec = 0

    def finalize(self, readout=None):
        rates = torch.stack(self.outputs["rates"], dim=1)

        rec = None
        if self.return_rec:
            rec = torch.stack(self.outputs["rec"], dim=2)

        stp_u = None
        stp_x = None
        if self.IF_STP and self.return_stp:
            stp_u = torch.stack(self.outputs["stp_u"], dim=1)
            stp_x = torch.stack(self.outputs["stp_x"], dim=1)

        return ForwardOutputs(
            rates=rates,
            rec_input=rec,
            stp_u=stp_u,
            stp_x=stp_x,
            readout=readout,
        )
