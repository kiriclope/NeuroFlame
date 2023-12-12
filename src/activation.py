import torch
from torch import nn
import numpy as np

class Activation(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, THRESH=15):
        return nn.ReLU()(x)
        # return THRESH * 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))

# class Activation(nn.Module):
#     def __init__(self, func_name='relu', thresh=0.0):
#         super(Activation, self).__init__()

#         self.func_name = func_name
#         self.thresh = thresh

#     def forward(self, x):
#         if self.func_name == 'relu':
#             return nn.ReLU()(x - self.thresh)
#         else:
#             return self.custom_activation(x)
    
#     def custom_activation(self, x):
#         return self.thresh * 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))
