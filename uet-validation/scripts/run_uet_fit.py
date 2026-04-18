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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uet.run_utils import dump_config, dump_metadata, setup_logging, setup_run_dir
from uet.scaling_fit import (
    PYTHIA_TOKENS_PER_STEP,
    fit_uet_curriculum,
    pythia_step_to_tokens,
    uet_predict,
)

logger = logging.getLogger(__name__)


def load_curriculum_csv(curriculum_dir: Path) -> pd.DataFrame:
    csv_path = curriculum_dir / "curriculum.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No curriculum.csv in {curriculum_dir}")
    df = pd.read_csv(csv_path)
    df["model"] = curriculum_dir.name
    df["n_tokens"] = pythia_step_to_tokens(df["step"].values)
    return df


def fit_one_model(df: pd.DataFrame, label: str) -> dict | None:
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
        "%s: c=%.4g L_inf=%.4f R2=%.4f RMSE=%.4f n=%d converged=%s",
        label, fit.c, fit.L_inf, fit.r_squared, fit.rmse, fit.n_points, fit.converged,
    )
    return {
        "label": label,
        "c": fit.c,
        "L_inf": fit.L_inf,
        "r_squared": fit.r_squared,
        "rmse": fit.rmse,
        "n_points": fit.n_points,
        "converged": fit.converged,
        "fit": fit,
    }


def plot_per_model(df_all: pd.DataFrame, fits: list[dict], output: Path) -> None:
    per_model = [f for f in fits if f["label"] != "JOINT"]
    if not per_model:
        return
    n = len(per_model)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.2), squeeze=False)
    for ax, fit_info in zip(axes[0], per_model):
        model = fit_info["label"]
        g = df_all[df_all["model"] == model].sort_values("n_tokens")
        g = g[g["n_tokens"] > 0]
        if g.empty:
            continue
        ax.scatter(g["n_tokens"], g["val_loss"], s=28, color="C0", label="val loss")
        fit = fit_info["fit"]
        pred_sorted_idx = np.argsort(g["n_tokens"].values)
        n_sorted = g["n_tokens"].values[pred_sorted_idx]
        pred_sorted = uet_predict(
            fit.c, fit.L_inf,
            g["d_eff"].values[pred_sorted_idx],
            g["hidden_dim"].values[pred_sorted_idx],
            n_sorted,
        )
        ax.plot(n_sorted, pred_sorted, "r-", lw=1.5,
                label=f"UET fit R²={fit.r_squared:.3f}")
        ax.axhline(fit.L_inf, ls="--", color="gray", alpha=0.6,
                   label=f"L∞={fit.L_inf:.3f}")
        ax.set_xscale("log")
        ax.set_xlabel("tokens seen (n)")
        ax.set_ylabel("val loss")
        ax.set_title(f"{model}\nc={fit.c:.3g}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Refit UET scaling law on curriculum data (varying n).",
    )
    parser.add_argument(
        "--curriculum-dirs", type=Path, nargs="+", required=True,
        help="Directories containing curriculum.csv (one per model).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--drop-init-steps", type=int, default=1,
        help="Drop checkpoints with step < this value (init point unreliable).",
    )
    parser.add_argument(
        "--min-step", type=int, default=1000,
        help="UET scaling law applies asymptotically. Drop warmup steps below this.",
    )
    args = parser.parse_args()

    run_dir = setup_run_dir(args.output_dir, experiment="uet_fit", run_name=args.run_name)
    setup_logging(run_dir)
    dump_config(run_dir, args, extra={
        "curriculum_dirs": [str(p) for p in args.curriculum_dirs],
    })

    frames = [load_curriculum_csv(p) for p in args.curriculum_dirs]
    df_raw = pd.concat(frames, ignore_index=True)
    df_raw = df_raw[df_raw["step"] >= args.drop_init_steps].reset_index(drop=True)
    df_raw.to_csv(run_dir / "all_curriculum_points.csv", index=False)

    df_all = df_raw[df_raw["step"] >= args.min_step].reset_index(drop=True)
    logger.info("Loaded %d checkpoints after min_step=%d filter (was %d)",
                len(df_all), args.min_step, len(df_raw))

    fits: list[dict] = []
    for model, g in df_all.groupby("model"):
        info = fit_one_model(g, model)
        if info is not None:
            fits.append(info)

    joint_info = fit_one_model(df_all, "JOINT")
    if joint_info is not None:
        fits.append(joint_info)

    fits_df = pd.DataFrame([{k: v for k, v in f.items() if k != "fit"} for f in fits])
    fits_df.to_csv(run_dir / "uet_fits.csv", index=False)

    plot_per_model(df_all, fits, run_dir / "uet_fit_per_model.png")

    dump_metadata(run_dir, {
        "n_curriculum_dirs": len(args.curriculum_dirs),
        "n_total_points": int(len(df_all)),
        "pythia_tokens_per_step": int(PYTHIA_TOKENS_PER_STEP),
        "fits": [{k: v for k, v in f.items() if k != "fit"} for f in fits],
    })
    logger.info("Done. Run dir: %s", run_dir)


if __name__ == "__main__":
    main()
