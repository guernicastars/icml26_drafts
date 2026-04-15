from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STYLE = {
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

plt.rcParams.update(STYLE)


def plot_eigenspectrum(eigenvalues: np.ndarray, model_name: str, ax: plt.Axes):
    idx = np.arange(1, len(eigenvalues) + 1)
    ax.semilogy(idx, eigenvalues, "-", linewidth=1.5, label=model_name.split("/")[-1])
    ax.set_xlabel("Component index")
    ax.set_ylabel("Eigenvalue")


def plot_scaling_comparison(df: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    ax.loglog(df["n_params"], df["val_loss"], "ko-", markersize=6)
    for _, row in df.iterrows():
        ax.annotate(
            row["model_short"], (row["n_params"], row["val_loss"]),
            textcoords="offset points", xytext=(5, 5), fontsize=8,
        )
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Validation loss")
    ax.set_title("(a) Empirical scaling")

    ax = axes[1]
    ax.semilogx(df["n_params"], df["d_eff"], "s-", color="C1", markersize=6)
    ax.set_xlabel("Parameters")
    ax.set_ylabel("d_eff")
    ax.set_title("(b) Effective dimension")

    ax = axes[2]
    if "predicted_loss" in df.columns:
        ax.plot(df["val_loss"], df["predicted_loss"], "o", color="C2", markersize=6)
        lo = min(df["val_loss"].min(), df["predicted_loss"].min()) * 0.95
        hi = max(df["val_loss"].max(), df["predicted_loss"].max()) * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xlabel("Empirical loss")
    ax.set_ylabel("UET-predicted loss")
    ax.set_title("(c) Prediction vs. empirical")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_failure_heatmap(df: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax_idx, gap_val in enumerate(sorted(df["gap_ratio"].unique())[:3]):
        sub = df[df["gap_ratio"] == gap_val]
        pivot = sub.pivot_table(
            values="sin_angle", index="k", columns="d", aggfunc="mean",
        )
        im = axes[ax_idx].imshow(
            pivot.values, aspect="auto", cmap="RdYlGn_r",
            vmin=0, vmax=1, origin="lower",
        )
        axes[ax_idx].set_xticks(range(len(pivot.columns)))
        axes[ax_idx].set_xticklabels(pivot.columns, fontsize=8)
        axes[ax_idx].set_yticks(range(len(pivot.index)))
        axes[ax_idx].set_yticklabels(pivot.index, fontsize=8)
        axes[ax_idx].set_xlabel("d")
        axes[ax_idx].set_ylabel("k")
        axes[ax_idx].set_title(f"gap_ratio={gap_val}")

    fig.colorbar(im, ax=axes, label="sin angle (PCA misalignment)", shrink=0.8)
    fig.savefig(output_path)
    plt.close(fig)


def plot_curriculum(df: pd.DataFrame, model_name: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["step"], df["d_eff"], "o-", markersize=4, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("d_eff")
    ax.set_title(f"Effective dimension during training: {model_name.split('/')[-1]}")
    ax.axhline(df["d_eff"].iloc[-1], color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
