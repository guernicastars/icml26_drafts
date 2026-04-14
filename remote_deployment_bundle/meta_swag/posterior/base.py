from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AggregatedAdapterResult:
    scheme: str
    mean_vector: np.ndarray
    weights: np.ndarray
    retained_count: int
    effective_sample_size: float
    beta: float | None
    threshold: float | None
    posterior_trace: float
    top_eigenvalues: tuple[float, ...]
    top_eigenvalue_ratio: float
    max_normalized_weight: float
    score_variance: float
    diagonal_variance: np.ndarray
    deviations: np.ndarray

    def sample(self, num_samples: int, rng: np.random.Generator) -> np.ndarray:
        if self.mean_vector.size == 0:
            return np.zeros((num_samples, 0), dtype=np.float32)

        diag_noise = rng.normal(size=(num_samples, self.mean_vector.size)).astype(np.float32)
        diag_term = np.sqrt(np.maximum(0.5 * self.diagonal_variance, 0.0))[None, :] * diag_noise

        if self.deviations.size == 0:
            low_rank_term = 0.0
        else:
            scale = np.sqrt(0.5 / max(self.deviations.shape[0] - 1, 1))
            coeffs = rng.normal(size=(num_samples, self.deviations.shape[0])).astype(np.float32)
            low_rank_term = scale * coeffs @ self.deviations

        return self.mean_vector[None, :] + diag_term + low_rank_term


def effective_sample_size(weights: np.ndarray) -> float:
    numerator = float(weights.sum() ** 2)
    denominator = float(np.square(weights).sum())
    return numerator / denominator if denominator > 0 else 0.0
