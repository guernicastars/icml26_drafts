from __future__ import annotations

import numpy as np

from ..posterior.base import AggregatedAdapterResult


def hm_am_ratio(variances: np.ndarray) -> float:
    variances = np.asarray(variances, dtype=np.float64)
    positive = variances[variances > 0]
    if len(positive) == 0:
        return 1.0
    hm = float(len(positive) / np.sum(1.0 / positive))
    am = float(np.mean(positive))
    return hm / am if am > 0 else 1.0


def posterior_spectrum_summary(result: AggregatedAdapterResult) -> dict[str, float]:
    eigenvalues = np.array(result.top_eigenvalues, dtype=np.float64)
    trace = result.posterior_trace
    return {
        "posterior_trace": trace,
        "top_eigenvalue": float(eigenvalues[0]) if len(eigenvalues) > 0 else 0.0,
        "top_eigenvalue_ratio": result.top_eigenvalue_ratio,
        "top5_eigenvalues": list(result.top_eigenvalues),
        "ess": result.effective_sample_size,
        "max_weight": result.max_normalized_weight,
        "score_variance": result.score_variance,
        "effective_rank": _effective_rank(eigenvalues) if len(eigenvalues) > 0 else 0.0,
    }


def _effective_rank(eigenvalues: np.ndarray) -> float:
    positive = eigenvalues[eigenvalues > 0]
    if len(positive) == 0:
        return 0.0
    normalized = positive / positive.sum()
    entropy = -float(np.sum(normalized * np.log(normalized + 1e-12)))
    return float(np.exp(entropy))
