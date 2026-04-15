from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uet.pretrained import (
    A100_MODELS,
    PYTHIA_MODELS,
    RTX3060_MODELS,
    ModelSpectrum,
    compute_model_spectrum,
)
from uet.plotting import plot_eigenspectrum, plot_scaling_comparison
from uet.scaling import (
    PYTHIA_TRAIN_TOKENS,
    fit_chinchilla,
    fit_uet_scaling,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

PROFILES = {
    "rtx3060": RTX3060_MODELS,
    "a100": A100_MODELS,
    "tiny": PYTHIA_MODELS[:2],
}


def collect_spectra(
    models: list[str], device: str, max_tokens: int, batch_size: int,
) -> list[ModelSpectrum]:
    spectra = []
    for name in models:
        logger.info("Processing %s", name)
        spec = compute_model_spectrum(
            name, device=device, max_tokens=max_tokens, batch_size=batch_size,
        )
        spectra.append(spec)
        logger.info(
            "  d_eff=%.1f  stable_rank=%.1f  val_loss=%.4f",
            spec.d_eff, spec.stable_rank, spec.val_loss,
        )
    return spectra


def spectra_to_dataframe(spectra: list[ModelSpectrum]) -> pd.DataFrame:
    rows = []
    for s in spectra:
        rows.append({
            "model": s.model_name,
            "model_short": s.model_name.split("/")[-1],
            "hidden_dim": s.hidden_dim,
            "n_params": s.n_params,
            "val_loss": s.val_loss,
            "d_eff": s.d_eff,
            "stable_rank": s.stable_rank,
        })
    return pd.DataFrame(rows)


def fit_and_predict(df: pd.DataFrame) -> pd.DataFrame:
    n_params = df["n_params"].values.astype(float)
    losses = df["val_loss"].values

    chinchilla = fit_chinchilla(n_params, losses)
    logger.info(
        "Chinchilla fit: a=%.4f alpha=%.4f L_inf=%.4f",
        chinchilla["a"], chinchilla["alpha"], chinchilla["L_inf"],
    )

    d_effs = df["d_eff"].values
    hidden_dims = df["hidden_dim"].values.astype(float)
    n_tokens = np.array([
        PYTHIA_TRAIN_TOKENS.get(m, 300e9) for m in df["model"]
    ])

    uet = fit_uet_scaling(d_effs, hidden_dims, losses, n_tokens)
    logger.info("UET fit: c=%.4f L_inf=%.4f R2=%.4f", uet["c"], uet["L_inf"], uet.get("r_squared", float("nan")))

    df = df.copy()
    df["predicted_loss"] = uet["predicted"]
    df["uet_residual"] = uet.get("residuals", np.nan)
    df["chinchilla_residual"] = chinchilla["residuals"]
    return df


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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = PROFILES[args.profile]

    spectra = collect_spectra(models, args.device, args.max_tokens, args.batch_size)
    df = spectra_to_dataframe(spectra)
    df = fit_and_predict(df)

    csv_path = args.output_dir / "scaling_exponents.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved %s", csv_path)

    plot_scaling_comparison(df, args.output_dir / "scaling_exponents.png")
    save_eigenspectrum_figure(spectra, args.output_dir / "eigenspectrum.png")
    logger.info("Done")


if __name__ == "__main__":
    main()
