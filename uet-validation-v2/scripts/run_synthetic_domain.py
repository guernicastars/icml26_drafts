from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "uet-validation"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uet.eigendecomp import covariance, effective_dimension, eigenspectrum, stable_rank
from uet.embedding_train import encode_dataset, train_autoencoder
from uet.run_utils import dump_config, dump_metadata, setup_logging, setup_run_dir
from uet_v2.synthetic_domain import generate_structured_data

logger = logging.getLogger(__name__)

D = 100
N_SAMPLES = 50_000
K_TRUE_VALUES = [3, 5, 10, 20, 50]
LATENT_DIMS = [2, 4, 8, 16, 32, 64, 100]
N_SEEDS = 2
SNR = 3.0


def main():
    parser = argparse.ArgumentParser(description="L7: Synthetic domain with known intrinsic rank")
    parser.add_argument("--d", type=int, default=D)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--k-true-values", type=int, nargs="+", default=K_TRUE_VALUES)
    parser.add_argument("--latent-dims", type=int, nargs="+", default=LATENT_DIMS)
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--snr", type=float, default=SNR)
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    run_dir = setup_run_dir(args.output_dir, "synthetic_domain", args.run_name)
    setup_logging(run_dir)
    dump_config(run_dir, args)

    all_rows = []
    for k_true in args.k_true_values:
        for seed in range(args.n_seeds):
            X, U_true = generate_structured_data(
                n=args.n_samples, d=args.d, k_true=k_true, snr=args.snr, seed=seed,
            )

            # Standardize
            mu = X.mean(axis=0)
            sigma = X.std(axis=0)
            sigma = np.where(sigma < 1e-8, 1.0, sigma)
            X_std = (X - mu) / sigma

            # Train/val split (80/20)
            n_train = int(0.8 * len(X_std))
            X_train = X_std[:n_train]
            X_val = X_std[n_train:]

            for latent_dim in args.latent_dims:
                logger.info("k_true=%d  seed=%d  latent_dim=%d", k_true, seed, latent_dim)
                try:
                    result = train_autoencoder(
                        X_train, X_val,
                        latent_dim=latent_dim,
                        hidden_dims=(256, 128),
                        n_epochs=args.n_epochs,
                        batch_size=args.batch_size,
                        lr=1e-3,
                        device=args.device,
                        dropout=0.0,
                        patience=20,
                    )
                    Z = encode_dataset(result.model, X_std, batch_size=1024, device=args.device)
                except Exception as e:
                    logger.warning("AE failed k_true=%d latent=%d: %s", k_true, latent_dim, e)
                    continue

                cov = covariance(Z)
                evals = eigenspectrum(cov)
                d_eff = effective_dimension(evals)
                sr = stable_rank(evals)

                all_rows.append({
                    "k_true": k_true,
                    "seed": seed,
                    "latent_dim": latent_dim,
                    "d_eff": d_eff,
                    "stable_rank": sr,
                    "val_recon_loss": float(result.val_losses[-1]) if result.val_losses else float("nan"),
                    "d": args.d,
                    "n_samples": args.n_samples,
                })
                logger.info(
                    "  k_true=%d  latent=%3d  d_eff=%.2f  (expected ~%d)",
                    k_true, latent_dim, d_eff, k_true,
                )

    df = pd.DataFrame(all_rows)
    df.to_csv(run_dir / "saturation.csv", index=False)

    if len(df) > 0:
        summary = (
            df.groupby(["k_true", "latent_dim"])["d_eff"]
            .mean()
            .reset_index()
            .pivot(index="latent_dim", columns="k_true", values="d_eff")
        )
        logger.info("d_eff saturation table (rows=latent_dim, cols=k_true):\n%s", summary.to_string())

    dump_metadata(run_dir, {
        "d": args.d,
        "n_samples": args.n_samples,
        "k_true_values": args.k_true_values,
        "latent_dims": args.latent_dims,
        "n_seeds": args.n_seeds,
        "n_rows": int(len(df)),
    })


if __name__ == "__main__":
    main()
