import torch
from torch import nn

class Stimuli():
    def __init__(self, task, size, dtype=torch.float, device='cuda'):
        self.task = task
        self.size = size
        
        self.dtype = dtype
        self.device = device
        
    def odrStim(self, strength, footprint, phase):
        """
        Stimulus for the 8 target ODR (cosine shape)
        args:
        float: strength: strength of the stimulus
        float: footprint: footprint/tuning width of the stimulus
        float: phase: location of the stimulus
        """
        # Amp = self.I1[0]
        # if self.I1[1]>0:
        #     Amp = Amp + self.I1[1] * torch.randn((self.N_BATCH, 1), dtype=self.FLOAT, device=self.DEVICE)
        
        theta_list = torch.linspace(0, 2.0 * torch.pi, self.size[-1] + 1, dtype=self.dtype, device=self.device)[:-1]
        # print(theta_list.shape)
        return strength * (1.0 + footprint * torch.cos(theta_list - phase * torch.pi / 180.0))
    
    def dualStim(self, strength, footprint, phase):
        """
        Stimulus for the Dual Task
        args:
        float: strength: strength of the stimulus
        float: footprint: variance of the stimulus
        float: phase: stimulus gaussian vector
        """

        # dum = (2 * torch.rand(self.size)).to(torch.int).to(self.device)
        # return   pow(-1, dum) * strength * (footprint * phase)
        
        return strength * (footprint * phase)
    
    def forward(self, strength, footprint, phase):
        if self.task=='odr':
            return self.odrStim(strength, footprint, phase)
        elif self.task=='dual':
            return self.dualStim(strength, footprint, phase)
        else:
            return 0
        
    def __call__(self, strength, footprint, phase):
        # This method will be called when you do Conn()()
        return self.forward(strength, footprint, phase)
        
