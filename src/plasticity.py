import torch


class Plasticity:
    """
    A Short-Term Plasticity (STP) model class, that provides both the Markram and
    Hansel methods for Short-Term Plasticity in synapses.
    """

    def __init__(self, USE, TAU_FAC, TAU_REC, DT, size, STP_TYPE="markram", IF_INIT=1, device="cuda"):
        N_BATCH = size[0]
        N_NEURON = size[1]

        self.stp_type = STP_TYPE
        self.DT = DT

        self.USE = torch.tensor(USE, device=device).unsqueeze(-1)
        # print('USE', self.USE.shape)

        self.DT_TAU_FAC = torch.tensor(DT / TAU_FAC, device=device).unsqueeze(-1)
        # print('DT_TAU_FAC', self.DT_TAU_FAC.shape)

        self.DT_TAU_REC = torch.tensor(DT / TAU_REC, device=device).unsqueeze(-1)
        # print('DT_TAU_REC', self.DT_TAU_REC.shape)

        if IF_INIT:
            self.u_stp = self.USE * torch.ones((N_BATCH, N_NEURON), device=device)
            # print('u', self.u_stp.shape)

            self.x_stp = torch.ones((N_BATCH, N_NEURON), device=device)
            # print('x', self.x_stp.shape)

    def markram_stp(self, rates):
        u_plus = self.u_stp + self.USE * (1.0 - self.u_stp)

        self.x_stp = (
            self.x_stp
            + (1.0 - self.x_stp) * self.DT_TAU_REC
            - self.DT * u_plus * self.x_stp * rates
        )
        self.u_stp = (
            self.u_stp
            - self.DT_TAU_FAC * self.u_stp
            + self.DT * self.USE * (1.0 - self.u_stp) * rates
        )

        return (u_plus * self.x_stp) * rates

    def hansel_stp(self, rates):
        self.x_stp = (
            self.x_stp
            - (self.x_stp - 1.0) * self.DT_TAU_REC
            - self.DT * self.x_stp * self.u_stp * rates
        )
        self.u_stp = (
            self.u_stp
            - (self.u_stp - self.USE) * self.DT_TAU_FAC
            + self.DT * self.USE * (1.0 - self.u_stp) * rates
        )
        return self.u_stp * self.x_stp

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
        if self.stp_type == "hansel":
            return self.hansel_stp(rates)

        if self.stp_type == "mato":
            return self.mato_stp(rates)

        return self.markram_stp(rates)

    def __call__(self, rates):
        return self.forward(rates)
