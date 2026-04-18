"""
UET fit across Pythia (GPT-NeoX) and OLMo (Llama-style) curricula.

Loads curriculum CSVs from any model. Each CSV must have columns:
  n_tokens, val_loss, d_eff, hidden_dim

Uses n_tokens directly (not step * tokens_per_step).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "uet-validation"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uet.run_utils import dump_config, dump_metadata, setup_logging, setup_run_dir
from uet.scaling_fit import fit_uet_curriculum, uet_predict

logger = logging.getLogger(__name__)
MIN_TOKENS = 20_000_000_000  # 20B tokens — safely past warmup for all models


def load_curriculum(csv_path: Path, label: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalise column names
    if "n_tokens" not in df.columns and "step" in df.columns:
        from uet.scaling_fit import pythia_step_to_tokens
        df["n_tokens"] = pythia_step_to_tokens(df["step"].values)
    required = {"n_tokens", "val_loss", "d_eff", "hidden_dim"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")
    df["label"] = label or csv_path.parent.name
    return df[df["n_tokens"] >= MIN_TOKENS].reset_index(drop=True)


def fit_and_log(df: pd.DataFrame, label: str) -> dict | None:
    try:
        fit = fit_uet_curriculum(
            d_eff=df["d_eff"].values,
            d=df["hidden_dim"].values,
            n=df["n_tokens"].values,
            L=df["val_loss"].values,
        )
    except Exception as e:
        logger.warning("Fit failed for %s: %s", label, e)
        return None
    logger.info(
        "%-35s  c=%.4g  L_inf=%.4f  R2=%.4f  RMSE=%.4f  n=%d  converged=%s",
        label, fit.c, fit.L_inf, fit.r_squared, fit.rmse, fit.n_points, fit.converged,
    )
    return {"label": label, "c": fit.c, "L_inf": fit.L_inf, "r_squared": fit.r_squared,
            "rmse": fit.rmse, "n_points": fit.n_points, "converged": fit.converged, "_fit": fit}


def plot_fits(fits: list[dict], df_all: pd.DataFrame, output: Path) -> None:
    colors = ["C0", "C1", "C2", "C3", "C4"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, info in enumerate(fits):
        label = info["label"]
        g = df_all[df_all["label"] == label].sort_values("n_tokens")
        c = colors[i % len(colors)]
        ax.scatter(g["n_tokens"], g["val_loss"], s=25, color=c, zorder=3)
        fit = info["_fit"]
        ns = np.logspace(np.log10(g["n_tokens"].min()), np.log10(g["n_tokens"].max()), 200)
        d_eff_interp = np.interp(np.log(ns), np.log(g["n_tokens"].values), g["d_eff"].values)
        d_val = float(g["hidden_dim"].iloc[0])
        pred = uet_predict(fit.c, fit.L_inf, d_eff_interp, np.full_like(ns, d_val), ns)
        ax.plot(ns, pred, color=c, lw=1.8,
                label=f"{label}  c={fit.c:.2g}  R²={fit.r_squared:.3f}")
        ax.axhline(fit.L_inf, ls="--", color=c, alpha=0.45, lw=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("tokens seen (n)")
    ax.set_ylabel("val loss")
    ax.legend(fontsize=8)
    ax.set_title("UET fit: cross-family c comparison")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", output)


def main():
    parser = argparse.ArgumentParser(description="UET fit: cross-family (Pythia + OLMo)")
    parser.add_argument("--curriculum-csvs", type=Path, nargs="+", required=True,
                        help="Path(s) to curriculum.csv files.")
    parser.add_argument("--labels", type=str, nargs="+", default=None)
    parser.add_argument("--min-tokens", type=float, default=MIN_TOKENS,
                        help="Drop rows with n_tokens below this value.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    run_dir = setup_run_dir(args.output_dir, "uet_fit_cross_family", args.run_name)
    setup_logging(run_dir)
    dump_config(run_dir, args)

    labels = args.labels or [p.parent.name for p in args.curriculum_csvs]
    frames = []
    for csv_path, label in zip(args.curriculum_csvs, labels):
        try:
            df = load_curriculum(csv_path, label)
            df = df[df["n_tokens"] >= args.min_tokens]
            frames.append(df)
            logger.info("Loaded %s: %d points", label, len(df))
        except Exception as e:
            logger.warning("Skipping %s: %s", csv_path, e)

    if not frames:
        logger.error("No curricula loaded.")
        return

    df_all = pd.concat(frames, ignore_index=True)

    fits = []
    for label, g in df_all.groupby("label"):
        info = fit_and_log(g, label)
        if info:
            fits.append(info)

    if len(fits) >= 2:
        c_vals = np.array([f["c"] for f in fits])
        spread = float(c_vals.max() / c_vals.min())
        logger.info("c spread across families: max/min=%.2fx  values=%s",
                    spread, [f"{f['label']}:{f['c']:.3g}" for f in fits])

    fits_df = pd.DataFrame([{k: v for k, v in f.items() if k != "_fit"} for f in fits])
    fits_df.to_csv(run_dir / "cross_family_fits.csv", index=False)

    if fits:
        plot_fits(fits, df_all, run_dir / "cross_family_uet.png")

    dump_metadata(run_dir, {
        "min_tokens": args.min_tokens,
        "n_models": len(fits),
        "fits": [{k: v for k, v in f.items() if k != "_fit"} for f in fits],
        "c_spread": float(np.array([f["c"] for f in fits]).max() /
                          np.array([f["c"] for f in fits]).min()) if fits else None,
    })


if __name__ == "__main__":
    main()
