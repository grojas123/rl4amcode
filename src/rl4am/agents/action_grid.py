from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActionGrid:
    """Uniform grid of target risky weights."""

    weights: np.ndarray

    @classmethod
    def uniform(
        cls,
        bins: int,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ) -> ActionGrid:
        if bins < 2:
            raise ValueError("bins must be at least 2")
        if min_weight > max_weight:
            raise ValueError("min_weight must not exceed max_weight")
        weights = np.linspace(float(min_weight), float(max_weight), int(bins))
        return cls(weights=weights.astype(float))

    def __post_init__(self) -> None:
        array = np.asarray(self.weights, dtype=float)
        if array.ndim != 1:
            raise ValueError("weights must be one-dimensional")
        if array.size < 2:
            raise ValueError("weights must contain at least two entries")
        if not np.isfinite(array).all():
            raise ValueError("weights must contain only finite values")
        if np.any(np.diff(array) < 0.0):
            raise ValueError("weights must be sorted in ascending order")
        object.__setattr__(self, "weights", array)

    @property
    def size(self) -> int:
        return int(self.weights.shape[0])

    def weight_at(self, index: int) -> float:
        return float(self.weights[int(index)])

    def nearest_index(self, weight: float) -> int:
        distances = np.abs(self.weights - float(weight))
        return int(np.argmin(distances))

    def nearest_weight(self, weight: float) -> float:
        return self.weight_at(self.nearest_index(weight))
