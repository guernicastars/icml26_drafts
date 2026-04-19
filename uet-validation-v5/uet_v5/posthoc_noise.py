"""Post-hoc noise-reservoir test (block-covariance formulation).

Key insight: we don't need to form H_aug = [H|Z] explicitly.
cov([H|Z]) = [[cov(H),  C_HZ ],
              [C_HZ^T,  cov(Z)]]
where C_HZ = H_c^T @ Z_c / n.

Each block is computed separately:
- cov(H): O(n*d^2) — computed ONCE from full n
- C_HZ:   O(n*d*m) per (m, seed) — the only per-cell cost
- cov(Z):  either identity (theoretical) or empirical O(n*m^2)

With n=99k, d=768, m=500: C_HZ costs ~3s.
Result: sin_theta computed with full n, bounded by O(sqrt((d+m)/n)).
"""
from __future__ import annotations

import logging

import numpy as np

from uet.eigendecomp import (
    effective_dimension,
    eigenspectrum,
    pca_alignment_sin,
    top_eigenvectors,
)

logger = logging.getLogger(__name__)


def run(
    H: np.ndarray,
    m_values: list[int],
    k: int | None = None,
    n_seeds: int = 5,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Return one row per (m, seed).

    Uses all n samples via block-covariance construction.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n, d = H.shape
    H64 = H.astype(np.float64)
    H_c = H64 - H64.mean(axis=0, keepdims=True)

    # Compute full-n baseline once
    cov_H = (H_c.T @ H_c) / (n - 1)
    eigs_H = eigenspectrum(cov_H)  # descending
    d_eff_base = effective_dimension(eigs_H)
    k_use = k if k is not None else max(1, round(d_eff_base))

    V_base, _ = top_eigenvectors(cov_H, k_use)   # (d, k)
    lambda_k = float(eigs_H[k_use - 1]) if k_use <= len(eigs_H) else 1.0

    col_rms = float(np.sqrt(np.mean(np.var(H_c, axis=0, ddof=1))))
    sigma_z2 = col_rms ** 2

    # Precompute eigenvalues for d_eff_aug formula
    tr_H = float(np.sum(eigs_H[eigs_H > 0]))
    tr_H2 = float(np.sum(eigs_H[eigs_H > 0] ** 2))

    rows: list[dict] = []

    for m in m_values:
        for seed in range(n_seeds):
            if m == 0:
                sin_th = 0.0
                d_eff_aug = d_eff_base
            else:
                gen = np.random.default_rng(seed * 100_000 + m)
                Z = gen.standard_normal((n, m)) * col_rms
                Z_c = Z - Z.mean(axis=0, keepdims=True)

                # Cross-covariance block (d × m)
                C_HZ = (H_c.T @ Z_c) / (n - 1)

                # Noise covariance block (m × m)
                cov_Z = (Z_c.T @ Z_c) / (n - 1)

                # Full augmented covariance (d+m) × (d+m)
                cov_aug = np.block([[cov_H, C_HZ], [C_HZ.T, cov_Z]])

                # Top-k eigenvectors of augmented covariance
                V_aug, _ = top_eigenvectors(cov_aug, k_use)   # (d+m, k)

                # Embed V_base into (d+m)-dim space and measure alignment
                V_base_pad = np.zeros((d + m, k_use))
                V_base_pad[:d, :] = V_base
                sin_th = pca_alignment_sin(V_base_pad, V_aug)

                # d_eff_aug from block formula
                tr_aug = tr_H + m * sigma_z2
                tr_aug2 = tr_H2 + m * sigma_z2 ** 2
                d_eff_aug = float(tr_aug ** 2 / max(tr_aug2, 1e-12))

            dk_bound = float(np.sqrt((d + m) / n)) if m > 0 else 0.0

            rows.append({
                "m": m,
                "seed": seed,
                "sin_theta": round(float(sin_th), 6),
                "dk_bound": round(dk_bound, 6),
                "d_eff_aug": round(float(d_eff_aug), 4),
                "d_eff_base": round(float(d_eff_base), 4),
                "k": k_use,
                "n": n,
                "d": d,
            })
            logger.info("m=%d seed=%d  sin=%.5f  dk_bound=%.5f  lambda_k=%.4f  sigma_z2=%.4f",
                        m, seed, float(sin_th), dk_bound, lambda_k, sigma_z2)

    return rows
