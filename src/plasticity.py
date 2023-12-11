import numpy as np

class STP_Model():
    """
    A Short-Term Plasticity (STP) model class, that provides both the Markram and 
    Hansel methods for Short-Term Plasticity in synapses.
    """

    def __init__(self, N, DT):

        self.USE = 0.05
        self.TAU_REC = .10
        self.TAU_FAC = 1.0

        self.u_stp = np.ones(N).astype(np.float64) * self.USE
        self.x_stp = np.ones(N).astype(np.float64)

        self.A_u_x_stp = np.ones((N,), dtype=np.float64) * self.USE
        
        self.DT = DT
        self.DT_TAU_REC = DT / self.TAU_REC
        self.DT_TAU_FAC = DT / self.TAU_FAC

    def markram_stp(self, rates):
        
        u_plus = self.u_stp + self.USE * (1.0 - self.u_stp)
        
        self.x_stp = self.x_stp + (1.0 - self.x_stp) * self.DT_TAU_REC - self.DT * u_plus * self.x_stp * rates
        self.u_stp = self.u_stp - self.DT_TAU_FAC * self.u_stp + self.DT * self.USE * (1.0 - self.u_stp) * rates
        self.A_u_x_stp = u_plus * self.x_stp
        

    def hansel_stp(self, rates):

        self.x_stp = self.x_stp - self.DT_TAU_REC * (self.x_stp - 1.0) - self.DT * self.x_stp * self.u_stp * rates
        self.u_stp = self.u_stp - self.DT_TAU_FAC * (self.u_stp - self.USE) + self.DT * self.USE * rates * (1.0 - self.u_stp)
        self.A_u_x_stp = self.u_stp * self.x_stp

        # self.u_stp = self.u_stp - self.DT_TAU_FAC * (self.u_stp - self.USE) + self.DT * self.USE * rates
        # self.A_u_x_stp = self.u_stp
