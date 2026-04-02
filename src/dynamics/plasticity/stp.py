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

        self.TAU_FAC = TAU_FAC
        self.TAU_REC = TAU_REC
        self.USE = torch.tensor(USE, device=device, dtype=torch.float32).unsqueeze(-1)

        if TAU_FAC > 0:
            self.DT_TAU_FAC = torch.tensor(DT / TAU_FAC, device=device).unsqueeze(-1)
        else:
            self.DT_TAU_FAC = torch.tensor(0.0, device=device).unsqueeze(-1)

        if TAU_REC > 0:
            self.DT_TAU_REC = torch.tensor(DT / TAU_REC, device=device).unsqueeze(-1)
        else:
            self.DT_TAU_REC = torch.tensor(0.0, device=device).unsqueeze(-1)

        self.EXP_REC = torch.exp(-self.DT_TAU_REC)
        self.EXP_FAC = torch.exp(-self.DT_TAU_FAC)

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
        if self.TAU_FAC == 0:
            self.u_stp = self.USE.expand_as(self.u_stp)
        if self.TAU_REC != 0:
            self.x_stp = (
                1.0
                + (self.x_stp - 1.0) * self.EXP_REC
                - self.DT * self.u_stp * self.x_stp * rates
            )
        if self.TAU_FAC != 0.0:
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
