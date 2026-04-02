"""Network state data containers."""
from __future__ import annotations

from dataclasses import dataclass

import torch

Tensor = torch.Tensor


@dataclass
class ForwardOutputs:
    rates: Tensor
    rec_input: Tensor | None = None
    stp_u: Tensor | None = None
    stp_x: Tensor | None = None
    readout: Tensor | None = None


@dataclass
class NetworkState:
    rates: Tensor
    rec_input: Tensor
    thresh: Tensor
    ff_state: Tensor
    hebb_rates: Tensor | None = None
    u_stp_last: list[Tensor] | None = None
    x_stp_last: list[Tensor] | None = None
    u_ff_stp_last: Tensor | None = None
    x_ff_stp_last: Tensor | None = None
    ff_adapt_thresh: Tensor | None = None
