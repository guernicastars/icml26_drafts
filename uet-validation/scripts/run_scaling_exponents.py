from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uet.plotting import plot_eigenspectrum, plot_scaling_comparison
from uet.pretrained import (
    A100_MODELS,
    PYTHIA_MODELS,
    RTX3060_MODELS,
    ModelSpectrum,
    compute_model_spectrum,
)
from uet.run_utils import dump_config, dump_metadata, setup_logging, setup_run_dir
from uet.scaling import (
    PYTHIA_TRAIN_TOKENS,
    fit_chinchilla,
    fit_uet_scaling,
)

logger = logging.getLogger(__name__)

PROFILES = {
    "rtx3060": RTX3060_MODELS,
    "a100": A100_MODELS,
    "tiny": PYTHIA_MODELS[:2],
}


def collect_spectra(
    models: list[str],
    device: str,
    max_tokens: int,
    batch_size: int,
    seq_len: int,
    run_dir: Path,
) -> list[ModelSpectrum]:
    spectra = []
    for name in tqdm(models, desc="models", unit="model"):
        logger.info("Processing %s", name)
        spec = compute_model_spectrum(
            name, device=device, max_tokens=max_tokens,
            batch_size=batch_size, seq_len=seq_len,
        )
        spectra.append(spec)

        model_dir = run_dir / "models" / name.split("/")[-1]
        model_dir.mkdir(parents=True, exist_ok=True)
        np.save(model_dir / "eigenvalues.npy", spec.eigenvalues)
        with open(model_dir / "spectrum.json", "w") as f:
            json.dump({
                "model_name": spec.model_name,
                "hidden_dim": spec.hidden_dim,
                "n_params": spec.n_params,
                "n_tokens_eval": spec.n_tokens_eval,
                "val_loss": spec.val_loss,
                "d_eff": spec.d_eff,
                "stable_rank": spec.stable_rank,
            }, f, indent=2)

        logger.info("  d_eff=%.1f stable_rank=%.1f val_loss=%.4f saved to %s",
                    spec.d_eff, spec.stable_rank, spec.val_loss, model_dir)
    return spectra


def spectra_to_dataframe(spectra: list[ModelSpectrum]) -> pd.DataFrame:
    rows = []
    for s in spectra:
        rows.append({
            "model": s.model_name,
            "model_short": s.model_name.split("/")[-1],
            "hidden_dim": s.hidden_dim,
            "n_params": s.n_params,
            "n_tokens_eval": s.n_tokens_eval,
            "val_loss": s.val_loss,
            "d_eff": s.d_eff,
            "stable_rank": s.stable_rank,
        })
    return pd.DataFrame(rows)


def fit_and_predict(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n_params = df["n_params"].values.astype(float)
    losses = df["val_loss"].values

    chinchilla = fit_chinchilla(n_params, losses)
    logger.info("Chinchilla: a=%.4f alpha=%.4f L_inf=%.4f",
                chinchilla["a"], chinchilla["alpha"], chinchilla["L_inf"])

    d_effs = df["d_eff"].values
    hidden_dims = df["hidden_dim"].values.astype(float)
    n_tokens = np.array([PYTHIA_TRAIN_TOKENS.get(m, 300e9) for m in df["model"]])

    uet = fit_uet_scaling(d_effs, hidden_dims, losses, n_tokens)
    logger.info("UET: c=%.4f L_inf=%.4f R2=%.4f",
                uet["c"], uet["L_inf"], uet.get("r_squared", float("nan")))

    df = df.copy()
    df["predicted_loss"] = uet["predicted"]
    df["uet_residual"] = uet.get("residuals", np.nan)
    df["chinchilla_residual"] = chinchilla["residuals"]

    fit_summary = {
        "chinchilla_a": float(chinchilla["a"]),
        "chinchilla_alpha": float(chinchilla["alpha"]),
        "chinchilla_L_inf": float(chinchilla["L_inf"]),
        "uet_c": float(uet["c"]),
        "uet_L_inf": float(uet["L_inf"]),
        "uet_r_squared": float(uet.get("r_squared", float("nan"))),
    }
    return df, fit_summary


def save_eigenspectrum_figure(spectra: list[ModelSpectrum], output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in spectra:
        plot_eigenspectrum(s.eigenvalues, s.model_name, ax)
    ax.legend()
    ax.set_title("Eigenspectrum across Pythia models")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Exp 1: Scaling exponents from d_eff")
    parser.add_argument("--profile", choices=list(PROFILES), default="rtx3060")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tokens", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    run_dir = setup_run_dir(args.output_dir, experiment="scaling", run_name=args.run_name)
    setup_logging(run_dir)
    dump_config(run_dir, args, extra={"models": PROFILES[args.profile]})

    spectra = collect_spectra(
        PROFILES[args.profile], args.device, args.max_tokens,
        args.batch_size, args.seq_len, run_dir,
    )
    df = spectra_to_dataframe(spectra)
    df, fit_summary = fit_and_predict(df)

    df.to_csv(run_dir / "scaling_exponents.csv", index=False)
    plot_scaling_comparison(df, run_dir / "scaling_exponents.png")
    save_eigenspectrum_figure(spectra, run_dir / "eigenspectrum.png")

    dump_metadata(run_dir, {
        "n_models": len(spectra),
        "fit": fit_summary,
        "d_eff_range": [float(df["d_eff"].min()), float(df["d_eff"].max())],
        "val_loss_range": [float(df["val_loss"].min()), float(df["val_loss"].max())],
    })
    logger.info("Done. Run dir: %s", run_dir)


if __name__ == "__main__":
    main()
