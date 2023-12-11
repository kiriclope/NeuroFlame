import torch
from torch import nn

class Activation(nn.Module):
    def __init__(self, func_name='relu', thresh=0.0):
        super().__init__()
        self.func_name = func_name
        self.thresh = thresh
        if func_name == 'relu':
            self.relu = nn.ReLU()
        
    def forward(self, x):
        if self.func_name == 'erf':
            return self.thresh * 0.5 * (1.0 - torch.erf(x / torch.sqrt(torch.tensor(2.0))))
        elif self.func_name == 'relu':
            return self.relu(x - self.thresh)
        else:
            raise ValueError('Invalid function name')
