"""Generate .dat files for the v5 HiLD-targeted experiments."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_ICML = Path(__file__).resolve().parent.parent.parent  # /home/vlad/development/ICML26
V5_RESULTS = _ICML / "uet-validation-v5" / "results"
V1_RESULTS = _ICML / "uet-validation" / "results"
V2_RESULTS = _ICML / "uet-validation-v2" / "results"
DATA = Path(__file__).resolve().parent / "data"
DATA.mkdir(exist_ok=True)


def _latest(base: Path) -> Path | None:
    if not base.exists():
        return None
    dirs = sorted(base.iterdir(), reverse=True)
    return dirs[0] if dirs else None


def _write(name: str, df: pd.DataFrame) -> None:
    path = DATA / name
    df.to_csv(path, index=False, sep=" ")
    print(f"  wrote {name} ({len(df)} rows)")


def dump_nr_posthoc() -> None:
    run = _latest(V5_RESULTS / "nr_posthoc")
    if run is None or not (run / "posthoc.csv").exists():
        print("  SKIP nr_posthoc: no results yet")
        return
    df = pd.read_csv(run / "posthoc.csv")
    if df.empty:
        print("  SKIP nr_posthoc: empty CSV")
        return
    for model in df["model"].unique():
        sub = df[df["model"] == model].copy()
        grp = sub.groupby("m", as_index=False).agg(
            sin_theta_mean=("sin_theta", "mean"),
            sin_theta_std=("sin_theta", "std"),
            dk_bound=("dk_bound", "first"),
            d_eff_aug_mean=("d_eff_aug", "mean"),
        )
        mname = model.replace("-", "").replace("_", "")
        _write(f"nr_posthoc_{mname}.dat", grp)
    # Also write combined across models
    grp_all = df.groupby("m", as_index=False).agg(
        sin_theta_mean=("sin_theta", "mean"),
        sin_theta_std=("sin_theta", "std"),
        dk_bound=("dk_bound", "first"),
    )
    _write("nr_posthoc_all.dat", grp_all)


def dump_rmt_bulk() -> None:
    run = _latest(V5_RESULTS / "rmt_bulk")
    if run is None or not (run / "bulk.csv").exists():
        print("  SKIP rmt_bulk: no results yet")
        return
    df = pd.read_csv(run / "bulk.csv").sort_values("step")
    for model in df["model"].unique():
        sub = df[df["model"] == model].copy()
        mname = model.replace("-", "")
        _write(f"rmt_bulk_{mname}.dat",
               sub[["step", "d_eff", "k_90pct", "top1_fraction", "spectral_entropy"]])


def dump_changepoint() -> None:
    run = _latest(V5_RESULTS / "changepoint")
    if run is None or not (run / "cp.csv").exists():
        print("  SKIP changepoint: no results yet")
        return
    df = pd.read_csv(run / "cp.csv")
    _write("changepoint.dat", df)


def dump_distill_rank() -> None:
    run = _latest(V5_RESULTS / "distill_rank")
    if run is None or not (run / "rank.csv").exists():
        print("  SKIP distill_rank: no results yet")
        return
    df = pd.read_csv(run / "rank.csv")
    grp = df.groupby("k", as_index=False).agg(
        rel_mse_mean=("rel_mse", "mean"),
        rel_mse_std=("rel_mse", "std"),
        z_deff_mean=("z_deff", "mean"),
    )
    _write("distill_rank.dat", grp)


def dump_rmt_trajectory() -> None:
    """Export d_eff + k_90 + top1 vs step for the key Pythia-160m trajectory."""
    run = _latest(V5_RESULTS / "rmt_bulk")
    if run is None or not (run / "bulk.csv").exists():
        return
    df = pd.read_csv(run / "bulk.csv")
    sub = df[df["model"] == "pythia-160m"].sort_values("step").copy()
    sub["log_step"] = np.log10(sub["step"].clip(lower=1))
    _write("rmt_trajectory_pythia160m.dat",
           sub[["step", "log_step", "d_eff", "k_90pct", "top1_fraction", "spectral_entropy"]])


def dump_form_ablation() -> None:
    run = _latest(V2_RESULTS / "form_ablation")
    if run is None or not (run / "ablation.csv").exists():
        print("  SKIP form_ablation: no results")
        return
    df = pd.read_csv(run / "ablation.csv")
    _write("form_ablation.dat", df[["model", "fit", "n_params", "r_squared", "rmse", "aic", "bic"]])


def dump_curriculum_all() -> None:
    for d, short in [
        (V1_RESULTS / "curriculum" / "20260418_020501_pythia-160m", "pythia160m"),
        (V1_RESULTS / "curriculum" / "20260418_020501_pythia-410m", "pythia410m"),
        (V2_RESULTS / "curriculum" / "20260418_053410_pythia-70m-deduped", "pythia70m"),
        (V2_RESULTS / "curriculum" / "20260418_140612_OLMo-1B", "olmo1b"),
    ]:
        p = d / "curriculum.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p).dropna(subset=["d_eff"])
        _write(f"curriculum_{short}.dat", df[["step", "val_loss", "d_eff"]])


def main() -> None:
    print("Generating v5 .dat files...")
    dump_curriculum_all()
    dump_form_ablation()
    dump_rmt_bulk()
    dump_rmt_trajectory()
    dump_changepoint()
    dump_nr_posthoc()
    dump_distill_rank()
    print("Done.")


if __name__ == "__main__":
    main()
