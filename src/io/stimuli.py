import torch
from torch import nn


def circular_gaussian(degrees, mu=0, sigma=30):
    delta = (degrees - mu + torch.pi) % (2.0 * torch.pi) - torch.pi
    return torch.exp(-0.5 * (delta / sigma) ** 2)


class Stimuli:
    def __init__(self, task, size, device):
        self.task = task
        self.size = size
        self.device = device

    def odrCosStim(self, strength, footprint, phase, rnd_phase=0, theta=None):
        if rnd_phase:
            phase = torch.rand((self.size[0], 1), device=self.device) * 2.0 * torch.pi
        if theta is None:
            theta = torch.linspace(0, 2.0 * torch.pi, self.size[-1] + 1, device=self.device)[:-1]
            theta = theta.unsqueeze(0).expand((1, self.size[-1]))
        return strength * nn.ReLU()(1.0 + footprint * torch.cos(theta - phase)) / (1.0 + footprint)

    def odrGaussStim(self, strength, footprint, phase, rnd_phase=0, theta=None):
        if rnd_phase:
            phase = torch.rand((self.size[0], 1), device=self.device) * 2.0 * torch.pi
        if theta is None:
            theta = torch.linspace(0, 2.0 * torch.pi, self.size[-1] + 1, device=self.device)[:-1]
            theta = theta.unsqueeze(0).expand((1, self.size[-1]))
        return strength * circular_gaussian(theta - phase, sigma=footprint)

    def dualStim(self, strength, footprint, phase):
        return strength * (footprint * phase)

    def forward(self, strength, footprint, phase, **kwargs):
        if "gauss" in self.task:
            return self.odrGaussStim(strength, footprint, phase, **kwargs)
        if "odr" in self.task:
            return self.odrCosStim(strength, footprint, phase, **kwargs)
        if "dual" in self.task:
            return self.dualStim(strength, footprint, phase)
        return 0

    def __call__(self, strength, footprint, phase, **kwargs):
        return self.forward(strength, footprint, phase, **kwargs)
