from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class NetworkState:
    rates: torch.Tensor
    rec_input: torch.Tensor
    thresh: torch.Tensor
    ff_prev: torch.Tensor

    hebb_rates: Optional[torch.Tensor] = None

    u_stp_last: Optional[list] = None
    x_stp_last: Optional[list] = None

    u_ff_stp_last: Optional[torch.Tensor] = None
    x_ff_stp_last: Optional[torch.Tensor] = None


@dataclass
class ForwardOutputs:
    rates: torch.Tensor
    rec_input: Optional[torch.Tensor] = None
    stp_u: Optional[torch.Tensor] = None
    stp_x: Optional[torch.Tensor] = None
    readout: Optional[torch.Tensor] = None
