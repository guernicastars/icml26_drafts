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

D = 256
K_VALUES = [4, 8, 16, 32]
N_SEEDS = 50
N_GRID_POINTS = 12


def n_theory(k: int, d: int) -> float:
    return k * np.log(d / k)


def run_one_cell(d: int, k: int, n: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((d, k)))
    # signal strength 3x noise so gap is clear; n is the variable
    Z = rng.standard_normal((k, n)) * 3.0
    noise = rng.standard_normal((d, n))
    X = (U @ Z + noise).T
    cov = covariance(X)
    V_hat, _ = top_eigenvectors(cov, k)
    return pca_alignment_sin(U, V_hat)


def main():
    parser = argparse.ArgumentParser(description="L4: O(k log(d/k)) sample complexity sweep")
    parser.add_argument("--d", type=int, default=D)
    parser.add_argument("--k-values", type=int, nargs="+", default=K_VALUES)
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--n-grid", type=int, default=N_GRID_POINTS)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    run_dir = setup_run_dir(args.output_dir, "sample_complexity", args.run_name)
    setup_logging(run_dir)
    dump_config(run_dir, args)

    rows = []
    for k in args.k_values:
        n_th = n_theory(k, args.d)
        n_values = np.unique(
            np.round(np.logspace(np.log10(n_th / 5), np.log10(100 * n_th), args.n_grid)).astype(int)
        )
        n_values = np.maximum(n_values, k + 1)
        logger.info("k=%d  n_theory=%.1f  n_grid=%s", k, n_th, n_values.tolist())

        for n in tqdm(n_values, desc=f"k={k}"):
            sin_thetas = [run_one_cell(args.d, k, int(n), seed=s) for s in range(args.n_seeds)]
            rows.append({
                "d": args.d,
                "k": k,
                "n_samples": int(n),
                "n_normalized": float(n) / n_th,
                "n_theory": n_th,
                "sin_theta_mean": float(np.mean(sin_thetas)),
                "sin_theta_std": float(np.std(sin_thetas)),
                "sin_theta_median": float(np.median(sin_thetas)),
            })
            logger.info(
                "k=%2d  n=%6d  n/n_th=%.2f  sin_theta=%.3f ± %.3f",
                k, n, float(n) / n_th,
                rows[-1]["sin_theta_mean"], rows[-1]["sin_theta_std"],
            )

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "sweep.csv", index=False)

    # Collapse check: compute dispersion of sin_theta at same n_normalized across k values
    pivot = df.pivot_table(
        index=pd.cut(df["n_normalized"], bins=5),
        columns="k",
        values="sin_theta_mean",
        aggfunc="mean",
    )
    logger.info("Collapse check (sin_theta by n_normalized bins and k):\n%s", pivot.to_string())

    dump_metadata(run_dir, {
        "d": args.d,
        "k_values": args.k_values,
        "n_seeds": args.n_seeds,
        "n_rows": int(len(df)),
    })


if __name__ == "__main__":
    main()
