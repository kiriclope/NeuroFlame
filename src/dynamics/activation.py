import torch
from torch import nn

_SQRT2 = None


def _get_sqrt2(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    global _SQRT2
    if _SQRT2 is None or _SQRT2.device != device or _SQRT2.dtype != dtype:
        _SQRT2 = torch.rsqrt(torch.tensor(2.0, device=device, dtype=dtype))
    return _SQRT2


class Activation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, func_name="relu", thresh=0.0):
        if func_name == "relu":
            return torch.relu(x - thresh)
        elif func_name == "erf":
            rsqrt2 = _get_sqrt2(x.device, x.dtype)
            return torch.erf((x - thresh) * rsqrt2)
        elif func_name == "sqrt":
            return (x >= 1.0) * torch.sqrt(torch.abs(4.0 * x - 3.0)) + x * x * (
                x >= 0
            ) * (x < 1.0)
        else:
            rsqrt2 = _get_sqrt2(x.device, x.dtype)
            return thresh * 0.5 * (1.0 + torch.erf(x * rsqrt2))
