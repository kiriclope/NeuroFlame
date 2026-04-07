import torch
from torch import nn


class Plasticity(nn.Module):
    def __init__(self, USE, TAU_FAC, TAU_REC, DT, size, device, STP_TYPE="markram", IF_INIT=1):
        super().__init__()
        N_BATCH = size[0]
        N_NEURON = size[1]

        self.DT = DT
        self.stp_type = STP_TYPE
        self.device = device

        # ── store raw time-constants (may be scalar or tensor) ────────
        self.TAU_FAC = TAU_FAC
        self.TAU_REC = TAU_REC

        # ── USE: always a (*, 1) tensor for broadcasting ─────────────
        self.USE = torch.as_tensor(USE, device=device, dtype=torch.float32)
        if self.USE.dim() == 0:
            self.USE = self.USE.unsqueeze(-1)
        elif self.USE.dim() == 1 and self.USE.shape[0] != 1:
            self.USE = self.USE.unsqueeze(-1)

        # ── DT / TAU ratios: handle scalar *and* tensor TAU ──────────
        TAU_FAC_t = torch.as_tensor(TAU_FAC, device=device, dtype=torch.float32)
        TAU_REC_t = torch.as_tensor(TAU_REC, device=device, dtype=torch.float32)

        # Safe division: where tau > 0 use DT/tau, else 0
        self.DT_TAU_FAC = torch.where(
            TAU_FAC_t > 0,
            torch.tensor(DT, device=device, dtype=torch.float32) / TAU_FAC_t,
            torch.tensor(0.0, device=device, dtype=torch.float32),
        )
        self.DT_TAU_REC = torch.where(
            TAU_REC_t > 0,
            torch.tensor(DT, device=device, dtype=torch.float32) / TAU_REC_t,
            torch.tensor(0.0, device=device, dtype=torch.float32),
        )

        # Add trailing dim for broadcasting against (N_BATCH, N_NEURON)
        if self.DT_TAU_FAC.dim() == 0:
            self.DT_TAU_FAC = self.DT_TAU_FAC.unsqueeze(-1)
        else:
            self.DT_TAU_FAC = self.DT_TAU_FAC.unsqueeze(-1)
        if self.DT_TAU_REC.dim() == 0:
            self.DT_TAU_REC = self.DT_TAU_REC.unsqueeze(-1)
        else:
            self.DT_TAU_REC = self.DT_TAU_REC.unsqueeze(-1)

        self.EXP_REC = torch.exp(-self.DT_TAU_REC)
        self.EXP_FAC = torch.exp(-self.DT_TAU_FAC)

        # ── boolean flags for zero-tau branches (element-wise safe) ───
        self._fac_is_zero = (TAU_FAC_t == 0).all().item()
        self._rec_is_zero = (TAU_REC_t == 0).all().item()

        if IF_INIT:
            self.u_stp = self.USE.expand(N_BATCH, N_NEURON).clone()
            self.x_stp = torch.ones((N_BATCH, N_NEURON), device=device)

    def markram_stp(self, rates):
        self.x_stp = (
            self.x_stp
            + (1.0 - self.x_stp) * self.DT_TAU_REC
            - self.DT * self.u_stp * self.x_stp * rates
        )
        self.u_stp = (
            self.u_stp
            + self.DT_TAU_FAC * (self.USE - self.u_stp)
            + self.DT * self.USE * (1.0 - self.u_stp) * rates
        )
        return (self.u_stp * self.x_stp) * rates

    def markram_stp_exp(self, rates):
        if self._fac_is_zero:
            self.u_stp = self.USE.expand_as(self.u_stp)
        if not self._rec_is_zero:
            self.x_stp = (
                1.0
                + (self.x_stp - 1.0) * self.EXP_REC
                - self.DT * self.u_stp * self.x_stp * rates
            )
        if not self._fac_is_zero:
            self.u_stp = (
                self.USE
                + (self.u_stp - self.USE) * self.EXP_FAC
                + self.DT * self.USE * (1.0 - self.u_stp) * rates
            )
        return (self.u_stp * self.x_stp) * rates

    def mato_stp(self, isi):
        self.u_stp = self.u_stp * torch.exp(-isi / self.TAU_FAC) + self.USE * (
            1.0 - self.u_stp * torch.exp(-isi / self.TAU_FAC)
        )
        self.x_stp = (
            self.x_stp * (1.0 - self.u_stp) * torch.exp(-isi / self.TAU_REC)
            + 1.0
            - torch.exp(-isi / self.TAU_REC)
        )
        return self.u_stp * self.x_stp

    def forward(self, rates):
        if self.stp_type == "mato":
            return self.mato_stp(rates)
        if self.stp_type == "markram_exp":
            return self.markram_stp_exp(rates)
        return self.markram_stp(rates)
