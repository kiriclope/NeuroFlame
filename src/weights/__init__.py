"""Weight construction and low-rank parametrizations."""
from src.weights.builder import WeightBuilder, WeightOutputs, recurrent_matmul
from src.weights.connectivity import Connectivity
from src.weights.lr import (
    LowRankWeights,
    SupportLowRankWeights,
    clamp_tensor,
    get_idx,
    get_overlap,
    get_theta,
    init_low_rank,
    masked_normalize,
    normalize_tensor,
)

__all__ = [
    "WeightBuilder",
    "WeightOutputs",
    "recurrent_matmul",
    "Connectivity",
    "LowRankWeights",
    "SupportLowRankWeights",
    "clamp_tensor",
    "get_idx",
    "get_overlap",
    "get_theta",
    "init_low_rank",
    "masked_normalize",
    "normalize_tensor",
]
