import torch

class STP_Model():
    """
    A Short-Term Plasticity (STP) model class, that provides both the Markram and 
    Hansel methods for Short-Term Plasticity in synapses.
    """

    def __init__(self, N_NEURON, csumNa, DT, FLOAT, device):
         
        self.DT = DT
        
        USE = torch.tensor([0.03, 0.0], dtype=FLOAT, device=device)
        self.TAU_REC = torch.tensor([0.25, 0.10], dtype=FLOAT, device=device)
        self.TAU_FAC = torch.tensor([2.0, 1.0], dtype=FLOAT, device=device)
        
        self.USE = torch.ones(N_NEURON, dtype=FLOAT, device=device)
        self.DT_TAU_REC = torch.ones(N_NEURON, dtype=FLOAT, device=device)
        self.DT_TAU_FAC = torch.ones(N_NEURON, dtype=FLOAT, device=device)
        
        self.IS_STP = torch.zeros(N_NEURON, dtype=FLOAT, device=device)
        
        for i_pop in range(csumNa.shape[0]-1):
            self.USE[csumNa[i_pop] : csumNa[i_pop + 1]] = USE[i_pop]
            self.IS_STP[csumNa[i_pop] : csumNa[i_pop + 1]] = 1.0 * (USE[i_pop]>0)
            self.DT_TAU_REC[csumNa[i_pop] : csumNa[i_pop + 1]] = DT / self.TAU_REC[i_pop]
            self.DT_TAU_FAC[csumNa[i_pop] : csumNa[i_pop + 1]] = DT / self.TAU_FAC[i_pop]
            
        self.u_stp = torch.ones(N_NEURON, dtype=FLOAT, device=device) * self.USE
        self.x_stp = torch.ones(N_NEURON, dtype=FLOAT, device=device)

        self.A_u_x_stp = torch.ones(N_NEURON, dtype=FLOAT, device=device) * self.IS_STP
        
    def markram_stp(self, rates):
        
        u_plus = self.u_stp + self.USE * (1.0 - self.u_stp)
        
        self.x_stp = self.x_stp + (1.0 - self.x_stp) * self.DT_TAU_REC - self.DT * u_plus * self.x_stp * rates
        self.u_stp = self.u_stp - self.DT_TAU_FAC * self.u_stp + self.DT * self.USE * (1.0 - self.u_stp) * rates
        self.A_u_x_stp = u_plus * self.x_stp * self.IS_STP
        
    def hansel_stp(self, rates):

        self.x_stp = self.x_stp - self.DT_TAU_REC * (self.x_stp - 1.0) - self.DT * self.x_stp * self.u_stp * rates
        self.u_stp = self.u_stp - self.DT_TAU_FAC * (self.u_stp - self.USE) + self.DT * self.USE * rates * (1.0 - self.u_stp)
        self.A_u_x_stp = self.u_stp * self.x_stp * self.IS_STP

    def mato_stp(self, isi):
        
        self.u_stp = self.u_stp * torch.exp(-isi / self.TAU_FAC) + self.USE * (1.0 - self.u_stp * torch.exp(-isi / self.TAU_FAC))
        self.x_stp = self.x_stp * (1.0 - self.u_stp) * torch.exp(-isi / self.TAU_REC) + 1.0 - torch.exp(-isi / self.TAU_REC)
        self.A_u_x_stp = self.u_stp * self.x_stp
