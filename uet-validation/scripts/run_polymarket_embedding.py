from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uet.clickhouse import ClickHouseConfig
from uet.eigendecomp import covariance, effective_dimension, eigenspectrum, stable_rank
from uet.embedding_train import encode_dataset, temporal_split, train_autoencoder
from uet.polymarket_data import build_features, fetch_resolved_markets, standardize

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Exp 2: Polymarket embedding + spectrum")
    parser.add_argument("--latent-dims", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--limit", type=int, default=None, help="Max markets to fetch")
    parser.add_argument("--min-volume", type=float, default=1000.0)
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("results/polymarket"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = ClickHouseConfig.from_env(database="polymarket")
    df = fetch_resolved_markets(config, min_volume=args.min_volume, limit=args.limit)
    feats = build_features(df)
    logger.info("Feature matrix: %s", feats.X.shape)

    X_std, mu, sigma = standardize(feats.X)
    train_idx, val_idx, _ = temporal_split(X_std, seed=42)
    X_train, X_val = X_std[train_idx], X_std[val_idx]

    results = []
    for latent in args.latent_dims:
        logger.info("Training AE with latent_dim=%d", latent)
        tr = train_autoencoder(
            X_train, X_val, latent_dim=latent, n_epochs=args.n_epochs,
            batch_size=args.batch_size, lr=args.lr, device=args.device,
        )
        Z = encode_dataset(tr.model, X_std, device=args.device)
        cov = covariance(Z)
        evals = eigenspectrum(cov)
        d_eff = effective_dimension(evals)
        sr = stable_rank(evals)

        results.append({
            "latent_dim": latent,
            "input_dim": X_std.shape[1],
            "n_samples": X_std.shape[0],
            "val_loss": tr.final_val_loss,
            "d_eff": d_eff,
            "stable_rank": sr,
            "d_eff_over_d": d_eff / latent,
        })
        np.save(args.output_dir / f"Z_latent{latent}.npy", Z)
        np.save(args.output_dir / f"eigenvalues_latent{latent}.npy", evals)
        logger.info("latent=%d val=%.5f d_eff=%.2f stable_rank=%.2f", latent, tr.final_val_loss, d_eff, sr)

    pd.DataFrame(results).to_csv(args.output_dir / "polymarket_embedding.csv", index=False)
    np.save(args.output_dir / "features_X.npy", feats.X)
    np.save(args.output_dir / "features_X_std.npy", X_std)
    with open(args.output_dir / "feature_names.txt", "w") as f:
        f.write("\n".join(feats.feature_names))
    logger.info("Done. Outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()
