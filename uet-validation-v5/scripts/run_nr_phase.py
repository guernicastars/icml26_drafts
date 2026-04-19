"""Noise-reservoir phase transition.

For fixed model (Pythia-160M), fix m=200 and vary sigma_z_ratio
(noise std relative to col_rms(H)) across a wide range.

Prediction: sharp phase transition when sigma_z^2 crosses lambda_k.
Below lambda_k: sin_theta ≈ 0 (noise-reservoir regime).
Above lambda_k: sin_theta -> 1 (noise dominates).

The location of the transition directly verifies the spectral gap
hypothesis of Theorem 1 (Noise-Reservoir).
"""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import numpy as np

V5 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V5))
sys.path.insert(0, str(V5.parent / "uet-validation"))

from uet.eigendecomp import (
    covariance,
    effective_dimension,
    eigenspectrum,
    pca_alignment_sin,
    top_eigenvectors,
)
from uet.run_utils import setup_run_dir, setup_logging, dump_metadata

logger = logging.getLogger(__name__)

SIGMA_RATIOS = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
M_FIXED = 200
N_SEEDS = 3
K_VALUES = [5, 10, 25, 50]   # test several subspace dimensions


def _find_cached_H(v5_root: Path, name: str) -> Path | None:
    for d in sorted((v5_root / "results" / "nr_posthoc").glob("*"), reverse=True):
        p = d / f"{name}_H.npy"
        if p.exists():
            return p
    return None


def run_phase(H: np.ndarray, k: int, m: int, sigma_ratios: list[float],
              n_seeds: int) -> list[dict]:
    n, d = H.shape
    H64 = H.astype(np.float64)
    H_c = H64 - H64.mean(axis=0, keepdims=True)

    cov_H = (H_c.T @ H_c) / (n - 1)
    eigs_H = eigenspectrum(cov_H)
    V_base, _ = top_eigenvectors(cov_H, k)
    lambda_k = float(eigs_H[k - 1])
    col_rms = float(np.sqrt(np.mean(np.var(H_c, axis=0, ddof=1))))

    rows = []
    for ratio in sigma_ratios:
        sigma_z = col_rms * ratio
        sigma_z2 = sigma_z ** 2
        for seed in range(n_seeds):
            gen = np.random.default_rng(seed * 1000 + int(ratio * 1000))
            Z = gen.standard_normal((n, m)) * sigma_z
            Z_c = Z - Z.mean(axis=0, keepdims=True)

            C_HZ = (H_c.T @ Z_c) / (n - 1)
            cov_Z = (Z_c.T @ Z_c) / (n - 1)
            cov_aug = np.block([[cov_H, C_HZ], [C_HZ.T, cov_Z]])

            V_aug, _ = top_eigenvectors(cov_aug, k)
            V_base_pad = np.zeros((d + m, k))
            V_base_pad[:d, :] = V_base
            sin_th = pca_alignment_sin(V_base_pad, V_aug)

            rows.append({
                "k": k,
                "sigma_ratio": ratio,
                "sigma_z2": round(sigma_z2, 5),
                "lambda_k": round(lambda_k, 5),
                "ratio_sigma2_lambda_k": round(sigma_z2 / max(lambda_k, 1e-9), 5),
                "seed": seed,
                "m": m,
                "sin_theta": round(float(sin_th), 6),
                "n": n,
                "d": d,
            })
            logger.info("k=%d sigma_ratio=%.2f  sigma2/lambda_k=%.3f  sin=%.5f",
                        k, ratio, sigma_z2 / max(lambda_k, 1e-9), float(sin_th))
    return rows


def main() -> None:
    run_dir = setup_run_dir(V5 / "results", "nr_phase")
    setup_logging(run_dir)
    logger.info("NR phase transition test")

    H_path = _find_cached_H(V5, "pythia-160m")
    if H_path is None:
        logger.error("no cached pythia-160m H; run run_nr_posthoc.py first")
        return
    H = np.load(H_path).astype(np.float32)
    logger.info("loaded %s  shape=%s", H_path, H.shape)

    all_rows = []
    for k in K_VALUES:
        all_rows.extend(run_phase(H, k=k, m=M_FIXED,
                                  sigma_ratios=SIGMA_RATIOS, n_seeds=N_SEEDS))

    csv_path = run_dir / "phase.csv"
    fieldnames = ["k", "sigma_ratio", "sigma_z2", "lambda_k",
                  "ratio_sigma2_lambda_k", "seed", "m", "sin_theta", "n", "d"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    logger.info("wrote %s (%d rows)", csv_path, len(all_rows))

    summary = {}
    for k in K_VALUES:
        sub = [r for r in all_rows if r["k"] == k]
        # Find smallest sigma_ratio where sin_theta exceeds 0.5
        sub_sorted = sorted(sub, key=lambda r: r["sigma_ratio"])
        transition = None
        for r in sub_sorted:
            if r["sin_theta"] > 0.5:
                transition = r["sigma_ratio"]
                break
        summary[f"k{k}"] = {
            "transition_sigma_ratio": transition,
            "lambda_k": sub[0]["lambda_k"] if sub else None,
        }
    dump_metadata(run_dir, summary)
    print("\n=== NR Phase Transition Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
