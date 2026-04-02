"""Output collection during simulation."""
from __future__ import annotations

from collections.abc import Sequence

import torch

from src.config.types import OutputConfig, Tensor
from src.dynamics.plasticity.stp import Plasticity
from src.state.containers import ForwardOutputs, NetworkState
from src.utils import print_activity


class OutputCollector:
    def __init__(
        self, cfg: OutputConfig, return_rec: bool = False, return_stp: bool = False,
    ) -> None:
        self.cfg = cfg
        self.return_rec = return_rec
        self.return_stp = return_stp

        self._rates: list[Tensor] = []
        self._rec_traces: list[Tensor] = []
        self._stp_u: list[Tensor] = []
        self._stp_x: list[Tensor] = []

        self._mv_rates: Tensor | None = None
        self._mv_rec_input: Tensor | None = None

        time = cfg.time
        self._acc_start: int = time.N_STEADY + time.N_HEBB
        self._inv_window: float = 1.0 / time.N_WINDOW
        self._stacked_cache: Tensor | None = None

    def _ensure_accumulators(self, state: NetworkState) -> None:
        if self._mv_rates is None:
            self._mv_rates = torch.zeros_like(state.rates)
        if self.return_rec and self._mv_rec_input is None:
            self._mv_rec_input = torch.zeros_like(state.rec_input)

    def _emit(
        self, step: int, state: NetworkState,
        stp_modules: Sequence[Plasticity] | None,
    ) -> None:
        geo = self.cfg.geo
        assert self._mv_rates is not None

        if self.cfg.VERBOSE:
            print_activity(self.cfg, step, state.rates)

        self._rates.append(
            (self._mv_rates[:, geo.slices[0]] * self._inv_window).detach().clone()
        )

        if self.return_rec and self._mv_rec_input is not None:
            rec_trace = (
                self._mv_rec_input.permute(1, 0, 2)[:, :, geo.slices[0]]
                * self._inv_window
            )
            self._rec_traces.append(rec_trace.detach().clone())

        if self.cfg.IF_STP and self.return_stp and stp_modules and len(stp_modules) > 0:
            self._stp_u.append(stp_modules[0].u_stp.detach().clone())
            self._stp_x.append(stp_modules[0].x_stp.detach().clone())

        self._stacked_cache = None

    def update(
        self, step: int, state: NetworkState,
        stp_modules: Sequence[Plasticity] | None = None,
    ) -> None:
        if step < self._acc_start:
            return

        self._ensure_accumulators(state)
        assert self._mv_rates is not None

        self._mv_rates.add_(state.rates)
        if self.return_rec and self._mv_rec_input is not None:
            self._mv_rec_input.add_(state.rec_input)

        if (step - self._acc_start + 1) % self.cfg.time.N_WINDOW == 0:
            self._emit(step, state, stp_modules)
            self._mv_rates.zero_()
            if self.return_rec and self._mv_rec_input is not None:
                self._mv_rec_input.zero_()

    def stacked_rates(self) -> Tensor:
        if self._stacked_cache is not None:
            return self._stacked_cache
        if not self._rates:
            raise RuntimeError("No rate outputs collected")
        self._stacked_cache = torch.stack(self._rates, dim=1)
        return self._stacked_cache

    def finalize(self, readout: Tensor | None = None) -> ForwardOutputs:
        return ForwardOutputs(
            rates=self.stacked_rates(),
            rec_input=torch.stack(self._rec_traces, dim=1) if self._rec_traces else None,
            stp_u=torch.stack(self._stp_u, dim=1) if self._stp_u else None,
            stp_x=torch.stack(self._stp_x, dim=1) if self._stp_x else None,
            readout=readout,
        )
