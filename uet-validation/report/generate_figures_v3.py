"""Generate .dat files for the v3 extended-validation section."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

V3_RESULTS = Path(__file__).resolve().parent.parent.parent / "uet-validation-v3" / "results"
DATA = Path(__file__).resolve().parent / "data"
DATA.mkdir(exist_ok=True)


def _latest(subdir: str) -> Path | None:
    base = V3_RESULTS / subdir
    if not base.exists():
        return None
    dirs = sorted(base.iterdir(), reverse=True)
    return dirs[0] if dirs else None


def _write(name: str, df: pd.DataFrame) -> None:
    path = DATA / name
    df.to_csv(path, index=False, sep=" ")
    print(f"  wrote {name} ({len(df)} rows)")


def dump_predictive_scaling() -> None:
    run = _latest("predictive_scaling")
    if run is None:
        print("  SKIP predictive_scaling: no run dir")
        return
    df = pd.read_csv(run / "summary.csv")
    _write("predictive_scaling_summary.dat", df)


def dump_grokking() -> None:
    dirs = sorted((V3_RESULTS / "grokking_deff").iterdir(), reverse=True) if (V3_RESULTS / "grokking_deff").exists() else []
    if not dirs:
        print("  SKIP grokking: no run dir")
        return
    run = dirs[0]
    traj = pd.read_csv(run / "trajectory.csv")
    # downsample to every DEFF_EVERY point (every 500 steps)
    deff_rows = traj[traj["step"] % 500 == 0].copy()
    _write("grokking_trajectory.dat", deff_rows[["step", "train_acc", "test_acc", "d_eff", "stable_rank"]])

    # also write per-LOG_EVERY for loss/acc curve
    _write("grokking_trajectory_full.dat", traj[["step", "train_loss", "train_acc", "test_acc"]])


def dump_noise_dim() -> None:
    run = _latest("noise_dim_test")
    if run is None:
        print("  SKIP noise_dim: no run dir")
        return
    df = pd.read_csv(run / "noise_dim.csv")
    _write("noise_dim.dat", df[["n_noise_dims", "total_input_dim", "recon_loss_mean",
                                "recon_loss_std", "d_eff_mean", "d_eff_std"]])


def dump_arch_diversity() -> None:
    run = _latest("arch_diversity")
    if run is None:
        print("  SKIP arch_diversity: no run dir")
        return
    df = pd.read_csv(run / "arch_diversity.csv")
    for arch in ("MLP", "CNN"):
        sub = df[df["arch"] == arch].copy()
        _write(f"arch_diversity_{arch.lower()}.dat", sub[["epoch", "d_eff", "stable_rank", "train_acc", "test_acc"]])


def dump_double_descent() -> None:
    run = _latest("double_descent")
    if run is None:
        print("  SKIP double_descent: no run dir")
        return
    df = pd.read_csv(run / "double_descent.csv")
    _write("double_descent.dat", df[["width", "n_params", "interp_ratio", "d_eff"]])


def main() -> None:
    print("Generating v3 .dat files...")
    dump_predictive_scaling()
    dump_grokking()
    dump_noise_dim()
    dump_arch_diversity()
    dump_double_descent()
    print("Done.")


if __name__ == "__main__":
    main()
