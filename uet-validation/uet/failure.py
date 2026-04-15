from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uet.eigendecomp import (
    covariance,
    effective_dimension,
    eigenspectrum,
    pca_alignment_sin,
    spectral_gap_ratio,
    theorem_42_bound,
    top_eigenvectors,
)


@dataclass(frozen=True)
class FailureResult:
    d: int
    k: int
    gap_ratio: float
    sin_angle: float
    d_eff: float
    theorem_bound: float
    condition_violated: str


def _random_orthonormal(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(size=(d, k))
    q, _ = np.linalg.qr(raw)
    return q[:, :k]


def _build_covariance(
    d: int,
    k: int,
    signal_strength: float,
    gap_ratio: float,
    causal_basis: np.ndarray,
) -> np.ndarray:
    signal_evals = np.linspace(signal_strength, 0.6 * signal_strength, k)
    noise_level = signal_evals[-1] / max(gap_ratio, 1e-6)
    cov = causal_basis @ np.diag(signal_evals) @ causal_basis.T
    cov += noise_level * np.eye(d)
    return cov, signal_evals


def run_single_failure(
    d: int,
    k: int,
    gap_ratio: float,
    n_samples: int,
    signal_strength: float = 4.0,
    rng: np.random.Generator | None = None,
) -> FailureResult:
    rng = rng or np.random.default_rng()

    if k >= d:
        return FailureResult(
            d=d, k=k, gap_ratio=gap_ratio, sin_angle=1.0,
            d_eff=float(d), theorem_bound=float("inf"),
            condition_violated="k>=d",
        )

    causal_basis = _random_orthonormal(d, k, rng)
    cov, signal_evals = _build_covariance(d, k, signal_strength, gap_ratio, causal_basis)

    L = np.linalg.cholesky(cov + 1e-10 * np.eye(d))
    samples = (rng.normal(size=(n_samples, d)) @ L.T)

    sample_cov = covariance(samples)
    V_hat, all_evals = top_eigenvectors(sample_cov, k)
    evals = eigenspectrum(sample_cov)

    sin_angle = pca_alignment_sin(causal_basis, V_hat)
    d_eff_val = effective_dimension(evals)

    epsilon_f = 0.1
    grad_norm = 1.0
    lk = all_evals[k - 1] if k <= len(all_evals) else 0.0
    lk1 = all_evals[k] if k < len(all_evals) else 0.0
    bound = theorem_42_bound(epsilon_f, grad_norm, lk, lk1)

    violated = "none"
    if k / d > 0.5:
        violated = "k/d>0.5 (no sparsity)"
    elif gap_ratio < 2.0:
        violated = "gap<2 (no spectral gap)"
    elif d < 2 * k:
        violated = "d<2k (dimension too small)"

    return FailureResult(
        d=d, k=k, gap_ratio=gap_ratio, sin_angle=sin_angle,
        d_eff=d_eff_val, theorem_bound=bound,
        condition_violated=violated,
    )


def sweep_failure_modes(
    d_values: list[int],
    k_values: list[int],
    gap_values: list[float],
    n_samples: int = 2000,
    n_seeds: int = 5,
) -> list[FailureResult]:
    results = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        for d in d_values:
            for k in k_values:
                for gap in gap_values:
                    result = run_single_failure(d, k, gap, n_samples, rng=rng)
                    results.append(result)
    return results
