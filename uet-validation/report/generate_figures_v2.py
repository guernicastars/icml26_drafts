"""Generate .dat files for the v2 extended-validation section."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

V2_RESULTS = Path(__file__).resolve().parent.parent.parent / "uet-validation-v2" / "results"
DATA = Path(__file__).resolve().parent / "data"
DATA.mkdir(exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────

def _latest(subdir: str) -> Path | None:
    base = V2_RESULTS / subdir
    if not base.exists():
        return None
    dirs = sorted(base.iterdir(), reverse=True)
    return dirs[0] if dirs else None


def _write(name: str, df: pd.DataFrame) -> None:
    path = DATA / name
    df.to_csv(path, index=False, sep=" ")
    print(f"  wrote {name} ({len(df)} rows)")


# ── L1: 70M curriculum ────────────────────────────────────────────────────

def dump_curriculum_70m() -> None:
    run = _latest("curriculum")
    if run is None:
        print("  SKIP 70m curriculum: no run dir")
        return
    # find the pythia-70m run specifically
    dirs = [d for d in (V2_RESULTS / "curriculum").iterdir() if "70m" in d.name]
    if not dirs:
        print("  SKIP 70m curriculum: no 70m dir")
        return
    csv = sorted(dirs)[-1] / "curriculum.csv"
    if not csv.exists():
        print(f"  SKIP 70m curriculum: {csv} missing")
        return
    df = pd.read_csv(csv)
    df["n_tokens"] = df["step"] * 2_097_152
    _write("curriculum_pythia-70m.dat", df[["step", "n_tokens", "val_loss", "d_eff", "stable_rank", "hidden_dim"]])


# ── L1: 70M UET fit predicted curve ───────────────────────────────────────

def dump_uet_fit_70m() -> None:
    run = _latest("uet_fit")
    if run is None:
        print("  SKIP uet_fit_70m: no run dir")
        return
    fits_csv = run / "uet_fits.csv"
    if not fits_csv.exists():
        print(f"  SKIP uet_fit_70m: {fits_csv} missing")
        return
    fits = pd.read_csv(fits_csv)
    row_70m = fits[fits["label"].str.contains("70m")]
    if row_70m.empty:
        print("  SKIP uet_fit_70m: no 70m row")
        return
    c = float(row_70m["c"].iloc[0])
    L_inf = float(row_70m["L_inf"].iloc[0])

    # load 70m curriculum
    dirs70 = [d for d in (V2_RESULTS / "curriculum").iterdir() if "70m" in d.name]
    if not dirs70:
        return
    df = pd.read_csv(sorted(dirs70)[-1] / "curriculum.csv")
    df = df[df["step"] >= 1000].copy()
    df["n_tokens"] = df["step"] * 2_097_152

    safe_deff = np.maximum(df["d_eff"].values, 1e-6)
    d = float(df["hidden_dim"].iloc[0])
    safe_ratio = np.maximum(d / safe_deff, 1.0 + 1e-9)
    feature = safe_deff * np.log(safe_ratio)
    df["predicted"] = c * feature / np.maximum(df["n_tokens"].values, 1.0) + L_inf
    df["L_inf"] = L_inf
    _write("uet_fit_pythia-70m.dat", df[["n_tokens", "val_loss", "predicted", "L_inf"]])


# ── L3: PCA alignment ─────────────────────────────────────────────────────

def dump_pca_alignment() -> None:
    run = _latest("pca_alignment")
    if run is None:
        print("  SKIP pca_alignment: no run dir")
        return
    df = pd.read_csv(run / "alignment.csv")

    # For the main figure: pool over d values, keep n_samples=5000 and n_samples=500
    for n in [500, 5000, 20000]:
        sub = df[df["n_samples"] == n].groupby("gap_multiplier")["sin_theta_mean"].mean().reset_index()
        sub.columns = ["gap_multiplier", "sin_theta_mean"]
        _write(f"pca_alignment_n{n}.dat", sub)

    # Summary table: all-n means by gap and theorem status
    summary = df.groupby(["gap_multiplier", "theorem_satisfied"])["sin_theta_mean"].mean().reset_index()
    _write("pca_alignment_summary.dat", summary)


# ── L4: sample complexity collapse ────────────────────────────────────────

def dump_sample_complexity() -> None:
    run = _latest("sample_complexity")
    if run is None:
        print("  SKIP sample_complexity: no run dir")
        return
    df = pd.read_csv(run / "sweep.csv")
    _write("sample_complexity.dat", df[["k", "n_normalized", "sin_theta_mean", "sin_theta_std"]])


# ── L5: per-layer d_eff ───────────────────────────────────────────────────

def dump_layer_deff() -> None:
    run = _latest("layer_deff")
    if run is None:
        print("  SKIP layer_deff: no run dir")
        return
    df = pd.read_csv(run / "layer_deff.csv")
    for model_short in ["pythia-160m-deduped", "pythia-70m-deduped"]:
        sub = df[df["model"] == model_short]
        if sub.empty:
            continue
        pivot = sub.pivot_table(index="layer", columns="step", values="d_eff").reset_index()
        pivot.columns = ["layer"] + [f"step_{int(c)}" for c in pivot.columns[1:]]
        pivot = pivot.fillna("NaN")
        safe_name = model_short.replace("-deduped", "").replace("-", "_")
        _write(f"layer_deff_{safe_name}.dat", pivot)


# ── L7: synthetic saturation ──────────────────────────────────────────────

def dump_synthetic_saturation() -> None:
    run = _latest("synthetic_domain")
    if run is None:
        print("  SKIP synthetic_domain: no run dir")
        return
    df = pd.read_csv(run / "saturation.csv")
    pivot = (
        df.groupby(["k_true", "latent_dim"])["d_eff"]
        .mean()
        .reset_index()
        .pivot(index="latent_dim", columns="k_true", values="d_eff")
        .reset_index()
    )
    pivot.columns = ["latent_dim"] + [f"k{int(c)}" for c in pivot.columns[1:]]
    _write("synthetic_saturation.dat", pivot)


# ── main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating v2 .dat files...")
    dump_curriculum_70m()
    dump_uet_fit_70m()
    dump_pca_alignment()
    dump_sample_complexity()
    dump_layer_deff()
    dump_synthetic_saturation()
    print("Done.")
