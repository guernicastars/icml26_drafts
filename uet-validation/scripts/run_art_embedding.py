from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uet.art_data import build_features, fetch_all_sources, standardize
from uet.clickhouse import ClickHouseConfig
from uet.eigendecomp import covariance, effective_dimension, eigenspectrum, stable_rank
from uet.embedding_train import encode_dataset, temporal_split, train_autoencoder
from uet.run_utils import dump_config, dump_metadata, setup_logging, setup_run_dir

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Exp 4a: Art embedding + spectrum")
    parser.add_argument("--sources", nargs="+", default=["christies", "sothebys"])
    parser.add_argument("--per-source-limit", type=int, default=None)
    parser.add_argument("--latent-dims", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--n-epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    run_dir = setup_run_dir(args.output_dir, experiment="art", run_name=args.run_name)
    setup_logging(run_dir)
    dump_config(run_dir, args)

    config = ClickHouseConfig.from_env(database=args.sources[0])
    df = fetch_all_sources(sources=args.sources, per_source_limit=args.per_source_limit, config=config)
    feats = build_features(df)
    logger.info("Features: %s", feats.X.shape)

    X_std, mu, sigma = standardize(feats.X)
    train_idx, val_idx, test_idx = temporal_split(X_std, seed=42)
    X_train, X_val = X_std[train_idx], X_std[val_idx]

    data_dir = run_dir / "data"
    data_dir.mkdir(exist_ok=True)
    np.save(data_dir / "X_std.npy", X_std)
    np.save(data_dir / "mu.npy", mu)
    np.save(data_dir / "sigma.npy", sigma)
    np.save(data_dir / "train_idx.npy", train_idx)
    np.save(data_dir / "val_idx.npy", val_idx)
    np.save(data_dir / "test_idx.npy", test_idx)
    np.save(data_dir / "sources.npy", feats.sources)
    np.save(data_dir / "targets.npy", feats.targets)
    np.save(data_dir / "lot_ids.npy", feats.lot_ids)
    with open(data_dir / "feature_names.txt", "w") as f:
        f.write("\n".join(feats.feature_names))

    rows = []
    for latent in tqdm(args.latent_dims, desc="art latents", unit="dim"):
        logger.info("Training AE latent=%d", latent)
        tr = train_autoencoder(
            X_train, X_val, latent_dim=latent, n_epochs=args.n_epochs,
            batch_size=args.batch_size, lr=args.lr, device=args.device,
        )
        Z = encode_dataset(tr.model, X_std, device=args.device)
        evals = eigenspectrum(covariance(Z))
        d_eff = effective_dimension(evals)
        sr = stable_rank(evals)

        model_dir = run_dir / "models" / f"latent{latent}"
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(tr.model.state_dict(), model_dir / "autoencoder.pt")
        np.save(model_dir / "Z.npy", Z)
        np.save(model_dir / "eigenvalues.npy", evals)
        np.save(model_dir / "train_loss.npy", np.asarray(tr.train_losses))
        np.save(model_dir / "val_loss.npy", np.asarray(tr.val_losses))
        with open(model_dir / "spectrum.json", "w") as f:
            json.dump({
                "latent_dim": latent,
                "input_dim": X_std.shape[1],
                "final_val_loss": tr.final_val_loss,
                "d_eff": d_eff,
                "stable_rank": sr,
                "n_epochs_run": len(tr.train_losses),
            }, f, indent=2)

        rows.append({
            "latent_dim": latent,
            "input_dim": X_std.shape[1],
            "n_samples": X_std.shape[0],
            "val_loss": tr.final_val_loss,
            "d_eff": d_eff,
            "stable_rank": sr,
            "d_eff_over_d": d_eff / latent,
            "n_epochs_run": len(tr.train_losses),
        })
        logger.info("  latent=%d val=%.5f d_eff=%.2f sr=%.2f", latent, tr.final_val_loss, d_eff, sr)

    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "art_embedding.csv", index=False)

    dump_metadata(run_dir, {
        "n_lots": int(X_std.shape[0]),
        "n_features": int(X_std.shape[1]),
        "sources": args.sources,
        "latent_dims": args.latent_dims,
        "best_latent": int(summary.loc[summary["val_loss"].idxmin(), "latent_dim"]),
    })
    logger.info("Done. Run dir: %s", run_dir)


if __name__ == "__main__":
    main()
