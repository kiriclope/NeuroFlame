"""Feedforward input construction."""
from __future__ import annotations

import torch

from src.io.stimuli import Stimuli
from src.weights.lr import get_theta

Tensor = torch.Tensor


class TaskStimulusFactory:
    def __init__(self, cfg, low_rank=None) -> None:
        # cfg is the live Configuration object — reads are always current
        self.cfg = cfg
        self.low_rank = low_rank

    @property
    def _stim(self):
        """Lazily rebuild Stimuli when N_BATCH changes."""
        cfg = self.cfg
        size = (cfg.N_BATCH, cfg.geo.Na[0])
        if not hasattr(self, '_cached_stim') or self._cached_size != size:
            self._cached_stim = Stimuli(cfg.TASK, size, device=cfg.geo.device)
            self._cached_size = size
        return self._cached_stim

    @property
    def is_dual_task(self) -> bool:
        return "dual" in self.cfg.TASK

    @property
    def is_rand_task(self) -> bool:
        return "rand" in self.cfg.TASK

    @property
    def is_flow_task(self) -> bool:
        return "flow" in self.cfg.TASK

    def make_context(self, phi0):
        cfg = self.cfg
        theta = None
        if self.is_dual_task and phi0.shape[0] >= 3:
            theta = get_theta(phi0[0], phi0[2]).unsqueeze(0)

        phase = None
        if self.is_rand_task:
            phase = torch.rand((cfg.N_BATCH, 1), device=cfg.geo.device) * 2.0 * torch.pi

        grid_inputs = self.build_grid_inputs() if self.is_flow_task else None

        return _StimulusContext(phi0=phi0, theta=theta, phase=phase, grid_inputs=grid_inputs)

    def build(self, index, context):
        if self.is_flow_task:
            return self._build_flow(index, context)
        if self.is_rand_task:
            return self._build_rand(index, context)
        if self.is_dual_task:
            return self._build_dual(index)
        return self._build_standard(index, context)

    def build_batch_dual_random_delay(self, batch_idx, index):
        cfg = self.cfg
        strength = cfg.I0[index]
        odor_idx = index if strength > 0 else 5 + index
        return self._stim(strength, cfg.SIGMA0[index], cfg.odors[odor_idx])

    def _build_standard(self, index, context):
        cfg = self.cfg
        return self._stim(cfg.I0[index], cfg.SIGMA0[index], context.phi0[:, index])

    def _build_rand(self, index, context):
        cfg = self.cfg
        return self._stim.odrCosStim(
            cfg.I0[index], cfg.SIGMA0[index], context.phase, theta=context.theta,
        )

    def _build_dual(self, index):
        cfg = self.cfg
        if cfg.LR_TRAIN and cfg.RANDOM_DELAY == 0:
            if index == cfg.RWD and cfg.IF_RL != 0:
                return None
            odor_idx = index if cfg.I0[index] > 0 else 5 + index
            return self._stim(cfg.I0[index], cfg.SIGMA0[index], cfg.odors[odor_idx])
        if cfg.RANDOM_DELAY == 0:
            return self._stim(cfg.I0[index], cfg.SIGMA0[index], cfg.PHI0[2 * index + 1])
        return None

    def _build_flow(self, index, context):
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
        return self._stim(sign * cfg.I0[index], cfg.SIGMA0[0], cfg.odors[cfg.GRID_TEST])

    def build_grid_inputs(self):
        cfg = self.cfg
        if self.low_rank is None:
            raise RuntimeError("Flow task requires low_rank module.")
        x = torch.linspace(cfg.GRID_X_RANGE[0], cfg.GRID_X_RANGE[1], cfg.GRID_SIZE, device=cfg.geo.device)
        y = torch.linspace(cfg.GRID_Y_RANGE[0], cfg.GRID_Y_RANGE[1], cfg.GRID_SIZE, device=cfg.geo.device)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        V = self.low_rank.get_V()
        vec1, vec2 = V[:, 0], V[:, 1]
        vec2 = vec2 - (vec2 @ vec1) * vec1 / (vec1 @ vec1)
        return [cfg.I0[cfg.GRID_INPUT] * (X[i, j] * vec1 + Y[i, j] * vec2)
                for i in range(cfg.GRID_SIZE) for j in range(cfg.GRID_SIZE)]


class _StimulusContext:
    __slots__ = ('phi0', 'theta', 'phase', 'grid_inputs')
    def __init__(self, phi0, theta, phase, grid_inputs):
        self.phi0 = phi0
        self.theta = theta
        self.phase = phase
        self.grid_inputs = grid_inputs


class FFInputBuilder:
    def __init__(self, cfg, low_rank=None) -> None:
        # cfg is the live Configuration object
        self.cfg = cfg
        self.stimulus_factory = TaskStimulusFactory(cfg, low_rank=low_rank)

    def build(self) -> Tensor:
        cfg = self.cfg
        phi0 = cfg.PHI0
        if "odr" in cfg.TASK and torch.any(phi0 > 2 * torch.pi):
            phi0 = torch.deg2rad(phi0)

        ff_input = self._build_background()
        if cfg.TASK != "None":
            context = self.stimulus_factory.make_context(phi0)
            self._add_task_stimuli(ff_input, context)
        return self._scale(ff_input)

    def _build_background(self) -> Tensor:
        cfg = self.cfg
        geo = cfg.geo
        ff_input = torch.randn(
            (cfg.N_BATCH, cfg.N_STEPS, geo.N_NEURON), device=geo.device,
        )
        for i_pop in range(geo.N_POP):
            ff_input[..., geo.slices[i_pop]].mul_(cfg.VAR_FF[:, i_pop])
        for i_pop in range(geo.N_POP):
            baseline_pre = cfg.Ja0[:, i_pop]
            if cfg.BUMP_SWITCH[i_pop]:
                baseline_pre = baseline_pre / torch.sqrt(
                    torch.as_tensor(geo.Ka[0], device=geo.device))
            ff_input[:, :cfg.N_STIM_ON[0], geo.slices[i_pop]].add_(baseline_pre)
            ff_input[:, cfg.N_STIM_ON[0]:, geo.slices[i_pop]].add_(cfg.Ja0[:, i_pop])
        return ff_input

    def _add_task_stimuli(self, ff_input, context):
        for index in range(len(self.cfg.N_STIM_ON)):
            stimulus = self.stimulus_factory.build(index, context)
            if stimulus is not None:
                self._apply_stimulus(ff_input, index, stimulus)

    def _apply_stimulus(self, ff_input, index, stimulus):
        cfg = self.cfg
        if cfg.ODR_TRAIN and stimulus.ndim != 3:
            stimulus = stimulus.unsqueeze(1)
        start = cfg.N_STIM_ON[index]
        if start >= cfg.N_STEPS:
            return
        if cfg.RANDOM_DELAY:
            self._apply_random_delay_stimulus(ff_input, index, stimulus)
            return
        stop = cfg.N_STIM_OFF[index]
        ff_input[:, start:stop, cfg.geo.slices[0]].add_(stimulus)

    def _apply_random_delay_stimulus(self, ff_input, index, stimulus):
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

    def _scale(self, ff_input):
        cfg = self.cfg
        geo = cfg.geo
        scale = cfg.GAIN * torch.sqrt(torch.as_tensor(geo.Ka[0], device=geo.device)) * cfg.M0
        return ff_input * scale
