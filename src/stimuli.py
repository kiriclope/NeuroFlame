import torch
from torch import nn

class Stimuli():
    def __init__(self, task, size, dtype=torch.float, device='cuda'):
        self.task = task
        self.size = size
        
        self.dtype = dtype
        self.device = device
        
    def odrStim(self, strength, footprint, phase, rnd_phase=0, theta=None):
        """
        Stimulus for the 8 target ODR (cosine shape)
        args:
        float: strength: strength of the stimulus
        float: footprint: footprint/tuning width of the stimulus
        float: phase: location of the stimulus
        """
        
        if rnd_phase:
            phase = torch.rand((self.size[0], 1), dtype=self.dtype, device=self.device) * 2.0 * torch.pi
        
        if theta is None:
            theta = torch.linspace(0, 2.0 * torch.pi, self.size[-1] + 1, dtype=self.dtype, device=self.device)[:-1]
            theta = theta.unsqueeze(0).expand((1, self.size[-1]))
        
        return strength * (1.0 + footprint * torch.cos(theta - phase))
    
    def dualStim(self, strength, footprint, phase):
        """
        Stimulus for the Dual Task
        args:
        float: strength: strength of the stimulus
        float: footprint: variance of the stimulus
        float: phase: stimulus gaussian vector
        """
        
        return strength * (footprint * phase)
    
    def forward(self, strength, footprint, phase, **kwargs):
        
        if 'odr' in self.task:
            return self.odrStim(strength, footprint, phase, **kwargs)
        if 'dual' in self.task:
            return self.dualStim(strength, footprint, phase)
        
        return 0
        
    def __call__(self, strength, footprint, phase, **kwargs):
        # This method will be called when you do Conn()()
        return self.forward(strength, footprint, phase, **kwargs)
        
