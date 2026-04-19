"""Spectral analysis of hidden-state covariance through training.

Key finding (from data): the MP bulk analysis does NOT show signal_count ≈
d_eff.  Instead, signal_count ≈ d throughout training (~540/768 dirs always
above noise floor), while d_eff varies dramatically (6 → 129 → 49).

This reveals that formalisation is energy redistribution, not direction death:
the model concentrates energy into fewer dominant eigendirections while keeping
many weaker directions active.  d_eff (participation ratio) measures this
concentration; signal_count measures active directions and stays roughly constant.

The useful RMT metrics are therefore:
  - d_eff: energy concentration (participation ratio)
  - top1_fraction: fraction of variance in the top eigenvector
  - k_90pct: minimum number of eigenvectors capturing 90% of variance
  - spectral_entropy: -sum(p_i log p_i) where p_i = lambda_i / sum(lambda)
"""
from __future__ import annotations

import numpy as np

from uet.eigendecomp import effective_dimension


def analyse(eigenvalues: np.ndarray, d: int, n: int) -> dict:
    """Return spectral concentration metrics for a given eigenspectrum."""
    eigs = np.sort(eigenvalues)[::-1]
    positive = eigs[eigs > 0]

    # MP bulk noise floor: sigma^2 from lower-half median
    half = max(1, len(positive) // 2)
    sigma2 = float(np.median(positive[-half:]))
    ratio = d / max(n, 1)
    lambda_plus = sigma2 * (1.0 + np.sqrt(ratio)) ** 2
    signal_count = int(np.sum(positive > lambda_plus))

    d_eff = float(effective_dimension(positive))

    total = float(positive.sum())
    cumvar = np.cumsum(positive) / total
    k_90 = int(np.searchsorted(cumvar, 0.90)) + 1
    k_99 = int(np.searchsorted(cumvar, 0.99)) + 1
    top1_frac = float(positive[0] / total) if len(positive) > 0 else 0.0

    p = positive / total
    entropy = float(-np.sum(p * np.log(p + 1e-15)))

    return {
        "d_eff": round(d_eff, 4),
        "signal_count_mp": signal_count,
        "k_90pct": k_90,
        "k_99pct": k_99,
        "top1_fraction": round(top1_frac, 6),
        "spectral_entropy": round(entropy, 4),
        "sigma2": round(float(sigma2), 8),
        "lambda_plus": round(float(lambda_plus), 8),
        "d": d,
        "n": n,
    }
