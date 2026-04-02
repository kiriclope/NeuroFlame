from __future__ import annotations

from dataclasses import dataclass

import torch

from src.config.types import FFInputConfig, Tensor
from src.weights.lr import get_theta
from src.io.stimuli import Stimuli


@dataclass
class StimulusContext:
    phi0: Tensor
    theta: Tensor | None
    phase: Tensor | None
    grid_inputs: list[Tensor] | None


class TaskStimulusFactory:
    def __init__(self, cfg: FFInputConfig, low_rank=None) -> None:
        self.cfg = cfg
        self.low_rank = low_rank
        self.stimulus_fn = Stimuli(
            cfg.TASK,
            (cfg.N_BATCH, cfg.geo.Na[0]),
            device=cfg.geo.device,
        )

    def make_context(self, phi0: Tensor) -> StimulusContext:
        cfg = self.cfg
        theta = None
        if self.is_dual_task:
            if phi0.shape[0] < 3:
                raise ValueError(
                    f"Dual task requires phi0 with >= 3 rows, got {phi0.shape[0]}"
                )
            theta = get_theta(phi0[0], phi0[2]).unsqueeze(0)

        phase = None
        if self.is_rand_task:
            phase = torch.rand((cfg.N_BATCH, 1), device=cfg.geo.device) * 2.0 * torch.pi

        grid_inputs = self.build_grid_inputs() if self.is_flow_task else None

        return StimulusContext(
            phi0=phi0, theta=theta, phase=phase, grid_inputs=grid_inputs,
        )

    def build(self, index: int, context: StimulusContext) -> Tensor | None:
        if self.is_flow_task:
            return self._build_flow(index, context)
        if self.is_rand_task:
            return self._build_rand(index, context)
        if self.is_dual_task:
            return self._build_dual(index)
        return self._build_standard(index, context)

    def build_batch_dual_random_delay(self, batch_idx: int, index: int) -> Tensor:
        cfg = self.cfg
        strength = cfg.I0[index]
        odor_idx = index if strength > 0 else 5 + index
        return self.stimulus_fn(strength, cfg.SIGMA0[index], cfg.odors[odor_idx])

    @property
    def is_flow_task(self) -> bool:
        return "flow" in self.cfg.TASK

    @property
    def is_rand_task(self) -> bool:
        return "rand" in self.cfg.TASK

    @property
    def is_dual_task(self) -> bool:
        return "dual" in self.cfg.TASK

    def _build_standard(self, index: int, context: StimulusContext) -> Tensor:
        cfg = self.cfg
        return self.stimulus_fn(cfg.I0[index], cfg.SIGMA0[index], context.phi0[:, index])

    def _build_rand(self, index: int, context: StimulusContext) -> Tensor:
        cfg = self.cfg
        return self.stimulus_fn.odrCosStim(
            cfg.I0[index], cfg.SIGMA0[index], context.phase, theta=context.theta,
        )

    def _build_dual(self, index: int) -> Tensor | None:
        cfg = self.cfg
        if cfg.LR_TRAIN and cfg.RANDOM_DELAY == 0:
            if index == cfg.RWD and cfg.IF_RL != 0:
                return None
            odor_idx = index if cfg.I0[index] > 0 else 5 + index
            return self.stimulus_fn(cfg.I0[index], cfg.SIGMA0[index], cfg.odors[odor_idx])

        if cfg.RANDOM_DELAY == 0:
            stim_idx = 2 * index + 1
            if stim_idx >= cfg.PHI0.shape[-1]:
                raise IndexError(
                    f"PHI0 index {stim_idx} out of range for shape {cfg.PHI0.shape}"
                )
            return self.stimulus_fn(
                cfg.I0[index], cfg.SIGMA0[index], cfg.PHI0[2 * index + 1],
            )
        return None

    def _build_flow(self, index: int, context: StimulusContext) -> Tensor | None:
        cfg = self.cfg
        if index == cfg.GRID_INPUT:
            if context.grid_inputs is None:
                return None
            return torch.stack(context.grid_inputs).unsqueeze(1)
        if cfg.GRID_TEST is None:
            return None
        sign = (
            (-1) ** (cfg.GRID_TEST + 1) if cfg.GRID_TEST in (1, 6)
            else (-1) ** cfg.GRID_TEST
        )
        return self.stimulus_fn(
            sign * cfg.I0[index], cfg.SIGMA0[0], cfg.odors[cfg.GRID_TEST],
        )

    def build_grid_inputs(self) -> list[Tensor]:
        cfg = self.cfg
        if self.low_rank is None:
            raise RuntimeError("Flow task requires low_rank module to build grid inputs.")

        x = torch.linspace(
            cfg.GRID_X_RANGE[0], cfg.GRID_X_RANGE[1], cfg.GRID_SIZE,
            device=cfg.geo.device,
        )
        y = torch.linspace(
            cfg.GRID_Y_RANGE[0], cfg.GRID_Y_RANGE[1], cfg.GRID_SIZE,
            device=cfg.geo.device,
        )
        X, Y = torch.meshgrid(x, y, indexing="ij")

        V = self.low_rank.get_V()
        vec1 = V[:, 0]
        vec2 = V[:, 1]
        vec2 = vec2 - (vec2 @ vec1) * vec1 / (vec1 @ vec1)

        grid_inputs: list[Tensor] = []
        for i in range(cfg.GRID_SIZE):
            for j in range(cfg.GRID_SIZE):
                point = cfg.I0[cfg.GRID_INPUT] * (X[i, j] * vec1 + Y[i, j] * vec2)
                grid_inputs.append(point)
        return grid_inputs


class FFInputBuilder:
    def __init__(self, cfg: FFInputConfig, low_rank=None) -> None:
        self.cfg = cfg
        self.stimulus_factory = TaskStimulusFactory(cfg, low_rank=low_rank)

    def build(self) -> Tensor:
        phi0 = self._prepare_phi0()
        ff_input = self._build_background()
        if self.cfg.TASK != "None":
            context = self.stimulus_factory.make_context(phi0)
            self._add_task_stimuli(ff_input, context)
        return self._scale(ff_input)

    def _prepare_phi0(self) -> Tensor:
        phi0 = self.cfg.PHI0
        if "odr" in self.cfg.TASK and torch.any(phi0 > 2 * torch.pi):
            return torch.deg2rad(phi0)
        return phi0

    def _build_background(self) -> Tensor:
        cfg = self.cfg
        geo = cfg.geo
        time = cfg.time

        ff_input = torch.randn(
            (cfg.N_BATCH, time.N_STEPS, geo.N_NEURON), device=geo.device,
        )
        for i_pop in range(geo.N_POP):
            ff_input[..., geo.slices[i_pop]].mul_(cfg.VAR_FF[:, i_pop])

        for i_pop in range(geo.N_POP):
            baseline_pre = cfg.Ja0[:, i_pop]
            if cfg.BUMP_SWITCH[i_pop]:
                baseline_pre = baseline_pre / torch.sqrt(
                    torch.as_tensor(geo.Ka[0], device=geo.device)
                )
            ff_input[:, : time.N_STIM_ON[0], geo.slices[i_pop]].add_(baseline_pre)
            ff_input[:, time.N_STIM_ON[0] :, geo.slices[i_pop]].add_(cfg.Ja0[:, i_pop])
        return ff_input

    def _add_task_stimuli(self, ff_input: Tensor, context: StimulusContext) -> None:
        for index in range(len(self.cfg.time.N_STIM_ON)):
            stimulus = self.stimulus_factory.build(index, context)
            if stimulus is not None:
                self._apply_stimulus(ff_input, index, stimulus)

    def _apply_stimulus(self, ff_input: Tensor, index: int, stimulus: Tensor) -> None:
        cfg = self.cfg
        geo = cfg.geo
        time = cfg.time

        if cfg.ODR_TRAIN and stimulus.ndim != 3:
            stimulus = stimulus.unsqueeze(1)

        start = time.N_STIM_ON[index]
        if start >= time.N_STEPS:
            return

        if cfg.RANDOM_DELAY:
            self._apply_random_delay_stimulus(ff_input, index, stimulus)
            return

        stop = time.N_STIM_OFF[index]
        ff_input[:, start:stop, geo.slices[0]].add_(stimulus)

    def _apply_random_delay_stimulus(
        self, ff_input: Tensor, index: int, stimulus: Tensor,
    ) -> None:
        cfg = self.cfg
        geo = cfg.geo
        for batch_idx in range(cfg.N_BATCH):
            time_slice = slice(
                int(cfg.start_indices[index, batch_idx]),
                int(cfg.end_indices[index, batch_idx]),
            )
            if self.stimulus_factory.is_dual_task:
                stim_j = self.stimulus_factory.build_batch_dual_random_delay(batch_idx, index)
                ff_input[batch_idx, time_slice, geo.slices[0]].add_(stim_j)
            else:
                ff_input[batch_idx, time_slice, geo.slices[0]].add_(stimulus[batch_idx])

    def _scale(self, ff_input: Tensor) -> Tensor:
        cfg = self.cfg
        geo = cfg.geo
        scale = cfg.GAIN * torch.sqrt(torch.as_tensor(geo.Ka[0], device=geo.device)) * cfg.M0
        return ff_input * scale
