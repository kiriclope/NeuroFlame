import torch
from torch import nn


class Activation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, func_name="relu", thresh=15):

        if func_name == "relu":
            return nn.ReLU()(x - thresh)
        elif func_name == "erf":
            return torch.erf((x-thresh) / torch.sqrt(torch.tensor(2.0)))
        elif func_name == "sqrt":
            return (x >= 1.0) * torch.sqrt(torch.abs(4.0 * x - 3.0)) + x * x * (
                x >= 0
            ) * (x < 1.0)
        else:
            return thresh * 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))
