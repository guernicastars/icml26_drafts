from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "uet-validation"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uet.eigendecomp import covariance, pca_alignment_sin, top_eigenvectors
from uet.run_utils import dump_config, dump_metadata, setup_logging, setup_run_dir

logger = logging.getLogger(__name__)

D_VALUES = [64, 128, 256]
K = 8
GAP_MULTIPLIERS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
N_VALUES = [200, 500, 2000, 5000, 20000]
N_SEEDS = 30


def run_one_cell(d: int, k: int, gap: float, n: int, seed: int) -> float:
    """
    Generate data X = U z + eps where signal eigenvalues are 1+gap, noise are 1.
    Returns sin_theta between planted U and PCA top-k.
    """
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((d, k)))

    # signal variance = 1+gap, noise variance = 1 → spectral gap = gap
    Z = rng.standard_normal((k, n)) * np.sqrt(1.0 + gap)
    noise = rng.standard_normal((d, n))
    X = (U @ Z + noise).T  # (n, d)

    cov = covariance(X)
    V_hat, _ = top_eigenvectors(cov, k)
    return pca_alignment_sin(U, V_hat)


def main():
    parser = argparse.ArgumentParser(description="L3: Synthetic PCA-causality alignment test")
    parser.add_argument("--d-values", type=int, nargs="+", default=D_VALUES)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--gap-multipliers", type=float, nargs="+", default=GAP_MULTIPLIERS)
    parser.add_argument("--n-values", type=int, nargs="+", default=N_VALUES)
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    run_dir = setup_run_dir(args.output_dir, "pca_alignment", args.run_name)
    setup_logging(run_dir)
    dump_config(run_dir, args)

    cells = [
        (d, args.k, gap, n)
        for d in args.d_values
        for gap in args.gap_multipliers
        for n in args.n_values
    ]
    logger.info("Running %d cells x %d seeds = %d total", len(cells), args.n_seeds, len(cells) * args.n_seeds)

    rows = []
    for d, k, gap, n in tqdm(cells, desc="cells"):
        sin_thetas = [run_one_cell(d, k, gap, n, seed=s) for s in range(args.n_seeds)]
        rows.append({
            "d": d,
            "k": k,
            "gap_multiplier": gap,
            "n_samples": n,
            "sin_theta_mean": float(np.mean(sin_thetas)),
            "sin_theta_std": float(np.std(sin_thetas)),
            "sin_theta_min": float(np.min(sin_thetas)),
            "sin_theta_max": float(np.max(sin_thetas)),
            "theorem_satisfied": int(gap >= 2.0 and k < d),
        })
        logger.info(
            "d=%3d k=%d gap=%.1f n=%5d  sin_theta=%.3f ± %.3f  theorem=%s",
            d, k, gap, n,
            rows[-1]["sin_theta_mean"], rows[-1]["sin_theta_std"],
            "OK" if rows[-1]["theorem_satisfied"] else "VIOLATED",
        )

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "alignment.csv", index=False)

    ok = df[df["theorem_satisfied"] == 1]["sin_theta_mean"].mean()
    violated = df[df["theorem_satisfied"] == 0]["sin_theta_mean"].mean()
    logger.info(
        "Summary: theorem satisfied sin_theta=%.3f, violated sin_theta=%.3f, ratio=%.2f",
        ok, violated, violated / max(ok, 1e-9),
    )

    dump_metadata(run_dir, {
        "n_cells": len(rows),
        "n_seeds": args.n_seeds,
        "sin_theta_theorem_ok": float(ok),
        "sin_theta_theorem_violated": float(violated),
        "ratio": float(violated / max(ok, 1e-9)),
    })


if __name__ == "__main__":
    main()
