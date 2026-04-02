"""Typed sub-configurations for each network module."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch

Tensor = torch.Tensor


class RawConfig(Protocol):
    """Structural type for the attributes Configuration exposes.

    This covers every attribute that at least one ``from_raw`` class-method
    reads.  Because ``Configuration`` sets attributes dynamically from YAML
    plus ``init_const`` / ``init_time_const``, the list is intentionally
    broad.
    """

    # ── geometry ──────────────────────────────────────────────────────
    N_POP: int
    N_NEURON: int
    Na: Tensor
    Ka: Tensor
    slices: list[slice]
    device: torch.device

    # ── time ──────────────────────────────────────────────────────────
    DT: float
    DURATION: float
    N_STEPS: int
    N_STEADY: int
    N_HEBB: int
    N_WINDOW: int
    N_STIM_ON: Tensor
    N_STIM_OFF: Tensor

    # ── connectivity ──────────────────────────────────────────────────
    CON_TYPE: str
    PROBA_TYPE: np.ndarray
    KAPPA: Tensor
    SIGMA: Tensor
    PHASE: Tensor
    LR_MEAN: Any
    LR_COV: Any
    PHI0: Tensor

    # ── trainable weights ─────────────────────────────────────────────
    LR_TRAIN: bool
    LR_NORM: int
    LR_INI: float
    LR_TYPE: str
    LR_MN: int
    LR_UeqV: int
    RANK: int
    LR_READOUT: int
    LR_READOUT_DIM: int
    LR_N_SUPPORTS: int
    LR_SUPPORT_WEIGHTS: Any
    LR_BASIS_DIM: Any
    LR_TRAIN_BIASES: bool
    LR_INIT_GAUSSIAN_BASIS: Any
    TRAIN_EI: bool
    IS_TRAIN: Tensor
    train_scale: Tensor
    GAIN: float
    CLAMP: bool

    # ── STP weight ────────────────────────────────────────────────────
    IF_STP: bool
    IS_STP: list[int]
    J_STP: Any

    # ── dynamics flags ────────────────────────────────────────────────
    TF_TYPE: str
    SYN_DYN: bool
    RATE_DYN: bool
    IF_NMDA: bool
    IF_FF_DYN: bool
    IF_FF_ADAPT: bool
    IF_ADAPT: bool
    IF_BATCH_J: bool
    IF_HEBB: bool
    IF_FF_STP: bool
    RATE_NOISE: bool
    IF_OPTO: bool

    # ── dynamics constants ────────────────────────────────────────────
    EXP_DT_TAU: Tensor
    EXP_DT_TAU_SYN: Tensor
    TAU_SYN: Tensor

    # ── Hebbian ───────────────────────────────────────────────────────
    IS_HEBB: list[int]
    ETA: float
    HEBB_TYPE: str
    HEBB_FRAC: float

    # ── STP module ────────────────────────────────────────────────────
    USE: list[float]
    TAU_FAC: list[float]
    TAU_REC: list[float]
    STP_TYPE: str

    # ── task ──────────────────────────────────────────────────────────
    TASK: str
    N_BATCH: int
    I0: list[float]
    SIGMA0: list[float]
    Ja0: Tensor
    M0: float
    VAR_FF: Tensor
    BUMP_SWITCH: list[int]
    RANDOM_DELAY: int
    DELAY_LIST: Any
    IF_RL: int
    RWD: int
    GRID_INPUT: int
    GRID_SIZE: int

    # ── simulation state ──────────────────────────────────────────────
    TRAINING: int
    thresh: Tensor
    VERBOSE: bool
    VAR_RATE: Tensor
    end_indices: Tensor
    start_indices: Tensor

    # ── base weight matrix ────────────────────────────────────────────
    Jab: Tensor


# ── leaf configs ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class NetworkGeometry:
    N_POP: int
    N_NEURON: int
    Na: Tensor
    Ka: Tensor
    slices: list[slice]
    device: torch.device

    @classmethod
    def from_raw(cls, m: RawConfig) -> NetworkGeometry:
        return cls(
            N_POP=m.N_POP, N_NEURON=m.N_NEURON, Na=m.Na, Ka=m.Ka,
            slices=m.slices, device=m.device,
        )


@dataclass(frozen=True)
class TimeConfig:
    DT: float
    DURATION: float
    N_STEPS: int
    N_STEADY: int
    N_HEBB: int
    N_WINDOW: int
    N_STIM_ON: Tensor
    N_STIM_OFF: Tensor

    @classmethod
    def from_raw(cls, m: RawConfig) -> TimeConfig:
        return cls(
            DT=m.DT, DURATION=m.DURATION,
            N_STEPS=m.N_STEPS, N_STEADY=m.N_STEADY,
            N_HEBB=m.N_HEBB, N_WINDOW=m.N_WINDOW,
            N_STIM_ON=m.N_STIM_ON, N_STIM_OFF=m.N_STIM_OFF,
        )


@dataclass(frozen=True)
class ConnectivityConfig:
    CON_TYPE: str
    PROBA_TYPE: np.ndarray
    KAPPA: Tensor
    SIGMA: Tensor
    PHASE: Tensor
    LR_MEAN: Any
    LR_COV: Any
    PHI0: Tensor

    @classmethod
    def from_raw(cls, m: RawConfig) -> ConnectivityConfig:
        return cls(
            CON_TYPE=m.CON_TYPE, PROBA_TYPE=m.PROBA_TYPE,
            KAPPA=m.KAPPA, SIGMA=m.SIGMA, PHASE=m.PHASE,
            LR_MEAN=m.LR_MEAN, LR_COV=m.LR_COV, PHI0=m.PHI0,
        )


@dataclass(frozen=True)
class TrainableWeightConfig:
    LR_TRAIN: bool
    LR_NORM: int
    LR_INI: float
    LR_TYPE: str
    LR_MN: int
    LR_UeqV: int
    RANK: int
    LR_READOUT: int
    LR_READOUT_DIM: int
    LR_N_SUPPORTS: int
    LR_SUPPORT_WEIGHTS: Any
    LR_BASIS_DIM: Any
    LR_TRAIN_BIASES: bool
    LR_INIT_GAUSSIAN_BASIS: Any
    TRAIN_EI: bool
    IS_TRAIN: Tensor
    train_scale: Tensor
    GAIN: float
    CLAMP: bool

    @classmethod
    def from_raw(cls, m: RawConfig) -> TrainableWeightConfig:
        return cls(
            LR_TRAIN=m.LR_TRAIN, LR_NORM=m.LR_NORM, LR_INI=m.LR_INI,
            LR_TYPE=m.LR_TYPE, LR_MN=m.LR_MN, LR_UeqV=m.LR_UeqV,
            RANK=m.RANK, LR_READOUT=m.LR_READOUT,
            LR_READOUT_DIM=m.LR_READOUT_DIM,
            LR_N_SUPPORTS=m.LR_N_SUPPORTS,
            LR_SUPPORT_WEIGHTS=m.LR_SUPPORT_WEIGHTS,
            LR_BASIS_DIM=m.LR_BASIS_DIM,
            LR_TRAIN_BIASES=m.LR_TRAIN_BIASES,
            LR_INIT_GAUSSIAN_BASIS=m.LR_INIT_GAUSSIAN_BASIS,
            TRAIN_EI=m.TRAIN_EI, IS_TRAIN=m.IS_TRAIN,
            train_scale=m.train_scale, GAIN=m.GAIN, CLAMP=m.CLAMP,
        )


@dataclass(frozen=True)
class STPWeightConfig:
    IF_STP: bool
    IS_STP: list[int]
    J_STP: Any
    W_STP: Any
    TRAIN_J_STP: bool

    @classmethod
    def from_raw(cls, m: RawConfig) -> STPWeightConfig:
        return cls(
            IF_STP=m.IF_STP, IS_STP=m.IS_STP, J_STP=m.J_STP,
            W_STP=getattr(m, "W_STP", None),
            TRAIN_J_STP=getattr(m, "TRAIN_J_STP", False),
        )


@dataclass(frozen=True)
class DynamicsFlags:
    TF_TYPE: str
    SYN_DYN: bool
    RATE_DYN: bool
    IF_NMDA: bool
    IF_FF_DYN: bool
    IF_FF_ADAPT: bool
    IF_ADAPT: bool
    IF_BATCH_J: bool
    IF_HEBB: bool
    IF_FF_STP: bool
    RATE_NOISE: bool
    IF_OPTO: bool

    @classmethod
    def from_raw(cls, m: RawConfig) -> DynamicsFlags:
        return cls(
            TF_TYPE=m.TF_TYPE, SYN_DYN=m.SYN_DYN, RATE_DYN=m.RATE_DYN,
            IF_NMDA=m.IF_NMDA, IF_FF_DYN=m.IF_FF_DYN,
            IF_FF_ADAPT=m.IF_FF_ADAPT, IF_ADAPT=m.IF_ADAPT,
            IF_BATCH_J=m.IF_BATCH_J, IF_HEBB=m.IF_HEBB,
            IF_FF_STP=m.IF_FF_STP, RATE_NOISE=m.RATE_NOISE,
            IF_OPTO=m.IF_OPTO,
        )


@dataclass(frozen=True)
class DynamicsConstants:
    EXP_DT_TAU: Tensor
    EXP_DT_TAU_SYN: Tensor
    EXP_DT_TAU_NMDA: Tensor | None
    R_NMDA: float
    EXP_ADAPT: Tensor | None
    A_ADAPT: float
    EXP_FF: Tensor | None
    EXP_FF_ADAPT: Tensor | None
    A_FF_ADAPT: float
    Jab_batch: Tensor | None
    W_batch_T: Tensor | None

    @classmethod
    def from_raw(cls, m: RawConfig) -> DynamicsConstants:
        return cls(
            EXP_DT_TAU=m.EXP_DT_TAU,
            EXP_DT_TAU_SYN=m.EXP_DT_TAU_SYN,
            EXP_DT_TAU_NMDA=getattr(m, "EXP_DT_TAU_NMDA", None),
            R_NMDA=getattr(m, "R_NMDA", 1.0),
            EXP_ADAPT=getattr(m, "EXP_ADAPT", None),
            A_ADAPT=getattr(m, "A_ADAPT", 0.0),
            EXP_FF=getattr(m, "EXP_FF", None),
            EXP_FF_ADAPT=getattr(m, "EXP_FF_ADAPT", None),
            A_FF_ADAPT=getattr(m, "A_FF_ADAPT", 0.0),
            Jab_batch=getattr(m, "Jab_batch", None),
            W_batch_T=getattr(m, "W_batch_T", None),
        )


@dataclass(frozen=True)
class HebbianConfig:
    IS_HEBB: list[int]
    ETA: float
    DT: float
    HEBB_TYPE: str
    HEBB_FRAC: float
    EXP_HEBB: Tensor | None
    IF_HEBB: bool

    @classmethod
    def from_raw(cls, m: RawConfig) -> HebbianConfig:
        return cls(
            IS_HEBB=m.IS_HEBB, ETA=m.ETA, DT=m.DT,
            HEBB_TYPE=m.HEBB_TYPE, HEBB_FRAC=m.HEBB_FRAC,
            EXP_HEBB=getattr(m, "EXP_HEBB", None),
            IF_HEBB=m.IF_HEBB,
        )


@dataclass(frozen=True)
class STPModuleConfig:
    IS_STP: list[int]
    IF_STP: bool
    IF_FF_STP: bool
    USE: list[float]
    TAU_FAC: list[float]
    TAU_REC: list[float]
    DT: float
    STP_TYPE: str
    FF_USE: float
    TAU_FF_FAC: float
    TAU_FF_REC: float

    @classmethod
    def from_raw(cls, m: RawConfig) -> STPModuleConfig:
        return cls(
            IS_STP=m.IS_STP, IF_STP=m.IF_STP,
            IF_FF_STP=m.IF_FF_STP,
            USE=m.USE, TAU_FAC=m.TAU_FAC, TAU_REC=m.TAU_REC,
            DT=m.DT, STP_TYPE=m.STP_TYPE,
            FF_USE=getattr(m, "FF_USE", 0.3),
            TAU_FF_FAC=getattr(m, "TAU_FF_FAC", 0.2),
            TAU_FF_REC=getattr(m, "TAU_FF_REC", 1.0),
        )


@dataclass(frozen=True)
class TaskConfig:
    TASK: str
    N_BATCH: int
    I0: list[float]
    SIGMA0: list[float]
    PHI0: Tensor
    Ja0: Tensor
    M0: float
    VAR_FF: Tensor
    BUMP_SWITCH: list[int]
    RANDOM_DELAY: int
    DELAY_LIST: Any
    ODR_TRAIN: bool
    IF_RL: int
    RWD: int
    odors: Tensor | None
    GRID_INPUT: int
    GRID_TEST: Any
    GRID_SIZE: int
    GRID_X_RANGE: tuple[float, float] | None
    GRID_Y_RANGE: tuple[float, float] | None

    @classmethod
    def from_raw(cls, m: RawConfig) -> TaskConfig:
        return cls(
            TASK=m.TASK, N_BATCH=m.N_BATCH,
            I0=m.I0, SIGMA0=m.SIGMA0, PHI0=m.PHI0,
            Ja0=m.Ja0, M0=m.M0, VAR_FF=m.VAR_FF,
            BUMP_SWITCH=m.BUMP_SWITCH, RANDOM_DELAY=m.RANDOM_DELAY,
            DELAY_LIST=m.DELAY_LIST,
            ODR_TRAIN=getattr(m, "ODR_TRAIN", False),
            IF_RL=m.IF_RL, RWD=m.RWD,
            odors=getattr(m, "odors", None),
            GRID_INPUT=m.GRID_INPUT,
            GRID_TEST=getattr(m, "GRID_TEST", None),
            GRID_SIZE=m.GRID_SIZE,
            GRID_X_RANGE=getattr(m, "GRID_X_RANGE", None),
            GRID_Y_RANGE=getattr(m, "GRID_Y_RANGE", None),
        )


@dataclass(frozen=True)
class SimulationState:
    TRAINING: int
    thresh: Tensor
    VERBOSE: bool
    VAR_RATE: Tensor
    N_OPTO: int
    end_indices: Tensor
    start_indices: Tensor

    @classmethod
    def from_raw(cls, m: RawConfig) -> SimulationState:
        return cls(
            TRAINING=m.TRAINING, thresh=m.thresh, VERBOSE=m.VERBOSE,
            VAR_RATE=m.VAR_RATE, N_OPTO=getattr(m, "N_OPTO", 0),
            end_indices=m.end_indices, start_indices=m.start_indices,
        )


# ── composite configs ─────────────────────────────────────────────────

@dataclass(frozen=True)
class WeightConfig:
    geo: NetworkGeometry
    conn: ConnectivityConfig
    trainable: TrainableWeightConfig
    stp: STPWeightConfig
    Jab: Tensor

    @property
    def GAIN(self) -> float:
        return self.trainable.GAIN

    @classmethod
    def from_raw(cls, m: RawConfig, geo: NetworkGeometry) -> WeightConfig:
        return cls(
            geo=geo,
            conn=ConnectivityConfig.from_raw(m),
            trainable=TrainableWeightConfig.from_raw(m),
            stp=STPWeightConfig.from_raw(m),
            Jab=m.Jab,
        )


@dataclass(frozen=True)
class RecurrentConfig:
    geo: NetworkGeometry
    flags: DynamicsFlags
    consts: DynamicsConstants
    stp_flag: bool = False

    @property
    def TF_TYPE(self) -> str: return self.flags.TF_TYPE
    @property
    def SYN_DYN(self) -> bool: return self.flags.SYN_DYN
    @property
    def RATE_DYN(self) -> bool: return self.flags.RATE_DYN
    @property
    def IF_NMDA(self) -> bool: return self.flags.IF_NMDA
    @property
    def IF_STP(self) -> bool: return self.stp_flag
    @property
    def IF_FF_STP(self) -> bool: return self.flags.IF_FF_STP
    @property
    def IF_FF_DYN(self) -> bool: return self.flags.IF_FF_DYN
    @property
    def IF_FF_ADAPT(self) -> bool: return self.flags.IF_FF_ADAPT
    @property
    def IF_ADAPT(self) -> bool: return self.flags.IF_ADAPT
    @property
    def IF_BATCH_J(self) -> bool: return self.flags.IF_BATCH_J
    @property
    def IF_HEBB(self) -> bool: return self.flags.IF_HEBB
    @property
    def EXP_DT_TAU(self) -> Tensor: return self.consts.EXP_DT_TAU
    @property
    def EXP_DT_TAU_SYN(self) -> Tensor: return self.consts.EXP_DT_TAU_SYN
    @property
    def EXP_DT_TAU_NMDA(self) -> Tensor | None: return self.consts.EXP_DT_TAU_NMDA
    @property
    def R_NMDA(self) -> float: return self.consts.R_NMDA
    @property
    def EXP_ADAPT(self) -> Tensor | None: return self.consts.EXP_ADAPT
    @property
    def A_ADAPT(self) -> float: return self.consts.A_ADAPT
    @property
    def EXP_FF(self) -> Tensor | None: return self.consts.EXP_FF
    @property
    def EXP_FF_ADAPT(self) -> Tensor | None: return self.consts.EXP_FF_ADAPT
    @property
    def A_FF_ADAPT(self) -> float: return self.consts.A_FF_ADAPT
    @property
    def Jab_batch(self) -> Tensor | None: return self.consts.Jab_batch
    @property
    def W_batch_T(self) -> Tensor | None: return self.consts.W_batch_T

    @classmethod
    def from_raw(
        cls, m: RawConfig, geo: NetworkGeometry,
        flags: DynamicsFlags, consts: DynamicsConstants,
    ) -> RecurrentConfig:
        return cls(geo=geo, flags=flags, consts=consts, stp_flag=m.IF_STP)


@dataclass(frozen=True)
class StateConfig:
    geo: NetworkGeometry
    time: TimeConfig
    flags: DynamicsFlags
    sim: SimulationState
    _if_stp: bool = False

    @property
    def IF_NMDA(self) -> bool: return self.flags.IF_NMDA
    @property
    def IF_HEBB(self) -> bool: return self.flags.IF_HEBB
    @property
    def IF_FF_ADAPT(self) -> bool: return self.flags.IF_FF_ADAPT
    @property
    def IF_STP(self) -> bool: return self._if_stp
    @property
    def IF_FF_STP(self) -> bool: return self.flags.IF_FF_STP
    @property
    def TF_TYPE(self) -> str: return self.flags.TF_TYPE
    @property
    def TRAINING(self) -> int: return self.sim.TRAINING
    @property
    def thresh(self) -> Tensor: return self.sim.thresh
    @property
    def VERBOSE(self) -> bool: return self.sim.VERBOSE

    @classmethod
    def from_raw(
        cls, m: RawConfig, geo: NetworkGeometry,
        time: TimeConfig, flags: DynamicsFlags, sim: SimulationState,
    ) -> StateConfig:
        return cls(
            geo=geo, time=time, flags=flags, sim=sim, _if_stp=m.IF_STP,
        )


@dataclass(frozen=True)
class OutputConfig:
    geo: NetworkGeometry
    time: TimeConfig
    sim: SimulationState
    _if_stp: bool = False

    @property
    def IF_STP(self) -> bool: return self._if_stp
    @property
    def VERBOSE(self) -> bool: return self.sim.VERBOSE
    @property
    def N_POP(self) -> int: return self.geo.N_POP
    @property
    def N_NEURON(self) -> int: return self.geo.N_NEURON
    @property
    def slices(self) -> list[slice]: return self.geo.slices
    @property
    def N_STEPS(self) -> int: return self.time.N_STEPS
    @property
    def N_STEADY(self) -> int: return self.time.N_STEADY
    @property
    def DURATION(self) -> float: return self.time.DURATION

    @classmethod
    def from_raw(
        cls, m: RawConfig, geo: NetworkGeometry,
        time: TimeConfig, sim: SimulationState,
    ) -> OutputConfig:
        return cls(geo=geo, time=time, sim=sim, _if_stp=m.IF_STP)


@dataclass(frozen=True)
class SimulationConfig:
    geo: NetworkGeometry
    time: TimeConfig
    flags: DynamicsFlags
    trainable: TrainableWeightConfig
    sim: SimulationState
    _if_stp: bool = False

    @property
    def TRAINING(self) -> int: return self.sim.TRAINING
    @property
    def RATE_NOISE(self) -> bool: return self.flags.RATE_NOISE
    @property
    def VAR_RATE(self) -> Tensor: return self.sim.VAR_RATE
    @property
    def LR_TRAIN(self) -> bool: return self.trainable.LR_TRAIN
    @property
    def LR_NORM(self) -> int: return self.trainable.LR_NORM
    @property
    def IF_HEBB(self) -> bool: return self.flags.IF_HEBB
    @property
    def IF_STP(self) -> bool: return self._if_stp
    @property
    def IF_OPTO(self) -> bool: return self.flags.IF_OPTO
    @property
    def N_OPTO(self) -> int: return self.sim.N_OPTO
    @property
    def end_indices(self) -> Tensor: return self.sim.end_indices

    @classmethod
    def from_raw(
        cls, m: RawConfig, geo: NetworkGeometry, time: TimeConfig,
        flags: DynamicsFlags, trainable: TrainableWeightConfig,
        sim: SimulationState,
    ) -> SimulationConfig:
        return cls(
            geo=geo, time=time, flags=flags,
            trainable=trainable, sim=sim, _if_stp=m.IF_STP,
        )


@dataclass(frozen=True)
class FFInputConfig:
    geo: NetworkGeometry
    time: TimeConfig
    task: TaskConfig
    trainable: TrainableWeightConfig
    _start_indices: Tensor | None = None
    _end_indices: Tensor | None = None

    @property
    def TASK(self) -> str: return self.task.TASK
    @property
    def N_BATCH(self) -> int: return self.task.N_BATCH
    @property
    def I0(self) -> list[float]: return self.task.I0
    @property
    def SIGMA0(self) -> list[float]: return self.task.SIGMA0
    @property
    def PHI0(self) -> Tensor: return self.task.PHI0
    @property
    def Ja0(self) -> Tensor: return self.task.Ja0
    @property
    def M0(self) -> float: return self.task.M0
    @property
    def VAR_FF(self) -> Tensor: return self.task.VAR_FF
    @property
    def GAIN(self) -> float: return self.trainable.GAIN
    @property
    def BUMP_SWITCH(self) -> list[int]: return self.task.BUMP_SWITCH
    @property
    def RANDOM_DELAY(self) -> int: return self.task.RANDOM_DELAY
    @property
    def DELAY_LIST(self) -> Any: return self.task.DELAY_LIST
    @property
    def ODR_TRAIN(self) -> bool: return self.task.ODR_TRAIN
    @property
    def IF_RL(self) -> int: return self.task.IF_RL
    @property
    def RWD(self) -> int: return self.task.RWD
    @property
    def LR_TRAIN(self) -> bool: return self.trainable.LR_TRAIN
    @property
    def odors(self) -> Tensor | None: return self.task.odors
    @property
    def GRID_INPUT(self) -> int: return self.task.GRID_INPUT
    @property
    def GRID_TEST(self) -> Any: return self.task.GRID_TEST
    @property
    def GRID_SIZE(self) -> int: return self.task.GRID_SIZE
    @property
    def GRID_X_RANGE(self): return self.task.GRID_X_RANGE
    @property
    def GRID_Y_RANGE(self): return self.task.GRID_Y_RANGE
    @property
    def start_indices(self) -> Tensor | None: return self._start_indices
    @property
    def end_indices(self) -> Tensor | None: return self._end_indices

    @classmethod
    def from_raw(
        cls, m: RawConfig, geo: NetworkGeometry, time: TimeConfig,
        task: TaskConfig, trainable: TrainableWeightConfig,
    ) -> FFInputConfig:
        return cls(
            geo=geo, time=time, task=task, trainable=trainable,
            _start_indices=m.start_indices, _end_indices=m.end_indices,
        )
