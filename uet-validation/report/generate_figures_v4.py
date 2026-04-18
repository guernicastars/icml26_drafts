"""Generate .dat files for the v4 extended-validation section (real-data experiments)."""
from __future__ import annotations

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


def dump_double_descent_v2() -> None:
    run = _latest("double_descent_v2")
    if run is None or not (run / "sweep.csv").exists():
        print("  SKIP double_descent_v2: no results yet")
        return
    df = pd.read_csv(run / "sweep.csv")
    # export summary: mean over seeds for each (n_train, noise_rate, width)
    grp = df.groupby(["n_train", "noise_rate", "width"], as_index=False).agg(
        n_params=("n_params", "first"),
        interp_ratio=("interp_ratio", "first"),
        test_acc_mean=("test_acc", "mean"),
        d_eff_mean=("d_eff", "mean"),
        d_eff_std=("d_eff", "std"),
    )
    # write one file per (n_train, noise_rate) combination for plotting
    for n_train in df["n_train"].unique():
        for noise in df["noise_rate"].unique():
            sub = grp[(grp["n_train"] == n_train) & (grp["noise_rate"] == noise)].copy()
            sub = sub.sort_values("width")
            fname = f"dd_v2_n{n_train}_noise{int(noise*100)}.dat"
            _write(fname, sub[["width", "n_params", "interp_ratio", "test_acc_mean", "d_eff_mean", "d_eff_std"]])


def dump_arch_diversity_v2() -> None:
    run = _latest("arch_diversity_v2")
    if run is None or not (run / "trajectories.csv").exists():
        print("  SKIP arch_diversity_v2: no results yet")
        return
    df = pd.read_csv(run / "trajectories.csv")
    for dataset in df["dataset"].unique():
        for arch in df["arch"].unique():
            sub = df[(df["dataset"] == dataset) & (df["arch"] == arch)].copy()
            grp = sub.groupby("epoch", as_index=False).agg(
                d_eff_mean=("d_eff", "mean"),
                d_eff_std=("d_eff", "std"),
                test_acc_mean=("test_acc", "mean"),
            )
            dname = dataset.replace("-", "").lower()
            fname = f"arch_v2_{dname}_{arch.lower()}.dat"
            _write(fname, grp)


def dump_grokking_lambda() -> None:
    run = _latest("grokking_lambda")
    if run is None or not (run / "trajectories.csv").exists():
        print("  SKIP grokking_lambda: no results yet")
        return
    df = pd.read_csv(run / "trajectories.csv")
    df = df.dropna(subset=["d_eff"])
    lambdas = sorted(df["lambda"].unique())
    for lam in lambdas:
        sub = df[df["lambda"] == lam].copy()
        grp = sub.groupby("step", as_index=False).agg(
            train_acc_mean=("train_acc", "mean"),
            test_acc_mean=("test_acc", "mean"),
            d_eff_mean=("d_eff", "mean"),
        )
        lam_str = str(lam).replace(".", "p")
        _write(f"grokking_lam{lam_str}.dat", grp)


def dump_deff_intervention() -> None:
    run = _latest("deff_intervention")
    if run is None or not (run / "sweep.csv").exists():
        print("  SKIP deff_intervention: no results yet")
        return
    df = pd.read_csv(run / "sweep.csv")
    grp = df.groupby("k", as_index=False).agg(
        d_eff_mean=("d_eff_empirical", "mean"),
        d_eff_std=("d_eff_empirical", "std"),
        test_loss_mean=("test_loss", "mean"),
        test_loss_std=("test_loss", "std"),
        test_acc_mean=("test_acc", "mean"),
    )
    _write("deff_intervention.dat", grp)


def dump_neural_collapse() -> None:
    run = _latest("neural_collapse")
    if run is None or not (run / "trajectory.csv").exists():
        print("  SKIP neural_collapse: no results yet")
        return
    df = pd.read_csv(run / "trajectory.csv")
    for dataset in df["dataset"].unique():
        sub = df[df["dataset"] == dataset].copy()
        grp = sub.groupby("epoch", as_index=False).agg(
            d_eff_mean=("d_eff", "mean"),
            d_eff_std=("d_eff", "std"),
            nc1_mean=("nc1", "mean"),
            nc2_mean=("nc2_mean_cos", "mean"),
            test_acc_mean=("test_acc", "mean"),
        )
        dname = dataset.replace("-", "").lower()
        _write(f"neural_collapse_{dname}.dat", grp)


def dump_fm_probe() -> None:
    run = _latest("fm_deff_probe")
    if run is None or not (run / "saturation.csv").exists():
        print("  SKIP fm_deff_probe: no results yet")
        return
    df = pd.read_csv(run / "saturation.csv")
    for model in df["model"].unique():
        sub = df[df["model"] == model].copy()
        mname = model.replace("-", "").replace("/", "").replace(" ", "_").lower()
        _write(f"fm_probe_{mname}.dat", sub[["proj_dim", "d_eff", "full_d_eff"]])


def main() -> None:
    print("Generating v4 .dat files...")
    dump_double_descent_v2()
    dump_arch_diversity_v2()
    dump_grokking_lambda()
    dump_deff_intervention()
    dump_neural_collapse()
    dump_fm_probe()
    print("Done.")


if __name__ == "__main__":
    main()
