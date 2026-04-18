from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from uet.scaling_fit import PYTHIA_TOKENS_PER_STEP, uet_predict

RUN = "20260418_020501"
RESULTS = REPO / "results"
REPORT = Path(__file__).resolve().parent
DATA = REPORT / "data"
DATA.mkdir(exist_ok=True)


def write_dat(path: Path, df: pd.DataFrame, columns: list[str]) -> None:
    with path.open("w") as f:
        f.write(" ".join(columns) + "\n")
        for row in df[columns].itertuples(index=False, name=None):
            f.write(" ".join(f"{v:.6g}" for v in row) + "\n")


def dump_curriculum():
    rows = []
    for tag, label in [("pythia-160m", "pythia-160m"), ("pythia-410m", "pythia-410m")]:
        df = pd.read_csv(RESULTS / "curriculum" / f"{RUN}_{tag}" / "curriculum.csv")
        df["model"] = label
        df["n_tokens"] = df["step"].astype(float) * PYTHIA_TOKENS_PER_STEP
        rows.append(df)
    full = pd.concat(rows, ignore_index=True)
    for label, g in full.groupby("model"):
        g = g.sort_values("step")
        write_dat(DATA / f"curriculum_{label}.dat", g,
                  ["step", "val_loss", "d_eff", "stable_rank", "hidden_dim", "n_tokens"])
    return full


def dump_uet_fit(full: pd.DataFrame):
    fits = pd.read_csv(RESULTS / "uet_fit" / RUN / "uet_fits.csv")
    fits_per_model = fits[fits["label"] != "JOINT"].copy()
    fits_per_model["model"] = fits_per_model["label"].str.replace(f"{RUN}_", "", regex=False)
    fits_per_model[["model", "c", "L_inf", "r_squared", "rmse", "n_points"]].to_csv(
        DATA / "uet_fit_summary.csv", index=False,
    )

    joint = fits[fits["label"] == "JOINT"].iloc[0]
    summary = {
        "joint_c": float(joint["c"]),
        "joint_L_inf": float(joint["L_inf"]),
        "joint_r2": float(joint["r_squared"]),
        "per_model": fits_per_model.to_dict(orient="records"),
    }
    (DATA / "uet_fit_summary.json").write_text(json.dumps(summary, indent=2))

    for _, row in fits_per_model.iterrows():
        label = row["model"]
        g = full[full["model"] == label].sort_values("step")
        g = g[(g["step"] >= 1000) & (g["n_tokens"] > 0)]
        if g.empty:
            continue
        pred = uet_predict(
            row["c"], row["L_inf"],
            g["d_eff"].values.astype(float),
            g["hidden_dim"].values.astype(float),
            g["n_tokens"].values.astype(float),
        )
        out = g[["n_tokens", "val_loss"]].copy()
        out["predicted"] = pred
        out["L_inf"] = row["L_inf"]
        write_dat(DATA / f"uet_fit_{label}.dat", out,
                  ["n_tokens", "val_loss", "predicted", "L_inf"])


def dump_failure():
    df = pd.read_csv(RESULTS / "failure" / RUN / "failure_sweep.csv")
    summary = (
        df.groupby(["d", "gap_ratio"])
          .agg(sin_angle=("sin_angle", "mean"),
               theorem_bound=("theorem_bound", "mean"),
               violated=("condition_violated", lambda s: (s != "none").mean()))
          .reset_index()
    )
    write_dat(DATA / "failure_sweep_mean.dat", summary,
              ["d", "gap_ratio", "sin_angle", "theorem_bound", "violated"])

    stats = {
        "n_rows": int(len(df)),
        "frac_violated": float((df["condition_violated"] != "none").mean()),
        "sin_angle_mean": float(df["sin_angle"].mean()),
        "sin_angle_median": float(df["sin_angle"].median()),
        "d_values": sorted(df["d"].unique().tolist()),
        "k_values": sorted(df["k"].unique().tolist()),
        "gap_values": sorted(df["gap_ratio"].unique().tolist()),
    }

    violated = df[df["condition_violated"] != "none"]
    ok = df[df["condition_violated"] == "none"]
    stats["sin_angle_ok_mean"] = float(ok["sin_angle"].mean())
    stats["sin_angle_violated_mean"] = float(violated["sin_angle"].mean())

    (DATA / "failure_summary.json").write_text(json.dumps(stats, indent=2))

    by_d = df.groupby("d").agg(
        violated=("condition_violated", lambda s: (s != "none").mean()),
        sin_angle=("sin_angle", "mean"),
    ).reset_index()
    write_dat(DATA / "failure_by_d.dat", by_d,
              ["d", "violated", "sin_angle"])

    fixed_k = df[(df["k"] == 8) & (~df["theorem_bound"].isin([np.inf, -np.inf]))]
    fixed_k = fixed_k.groupby(["d", "gap_ratio"]).agg(
        sin_angle=("sin_angle", "mean"),
        theorem_bound=("theorem_bound", "mean"),
    ).reset_index()
    write_dat(DATA / "failure_k8.dat", fixed_k,
              ["d", "gap_ratio", "sin_angle", "theorem_bound"])
    for d_val in sorted(fixed_k["d"].unique()):
        sub = fixed_k[fixed_k["d"] == d_val].sort_values("gap_ratio")
        write_dat(DATA / f"failure_k8_d{int(d_val)}.dat", sub,
                  ["gap_ratio", "sin_angle", "theorem_bound"])


def dump_scaling():
    df = pd.read_csv(RESULTS / "scaling" / RUN / "scaling_exponents.csv")
    df["model_tag"] = df["model_short"].str.replace("-deduped", "", regex=False)
    write_dat(DATA / "scaling_final.dat", df,
              ["n_params", "hidden_dim", "d_eff", "stable_rank", "val_loss"])
    df[["model_tag", "hidden_dim", "n_params", "val_loss", "d_eff", "stable_rank"]].to_csv(
        DATA / "scaling_final_table.csv", index=False,
    )


def dump_cross_domain():
    art = pd.read_csv(RESULTS / "art" / RUN / "art_embedding.csv")
    poly = pd.read_csv(RESULTS / "polymarket" / RUN / "polymarket_embedding.csv")
    art["domain"] = "art"
    poly["domain"] = "polymarket"

    for name, df in [("art", art), ("polymarket", poly)]:
        df_sorted = df.sort_values("latent_dim")
        write_dat(DATA / f"latent_sweep_{name}.dat", df_sorted,
                  ["latent_dim", "val_loss", "d_eff", "stable_rank", "d_eff_over_d"])

    combined = pd.concat([art, poly], ignore_index=True)
    combined.to_csv(DATA / "latent_sweep_combined.csv", index=False)

    spectra = pd.read_csv(RESULTS / "cross_domain" / RUN / "cross_domain_spectra.csv")
    spectra.to_csv(DATA / "cross_domain_summary.csv", index=False)


def dump_theoretical_curve():
    c = 2.0e7
    L_inf = 3.06
    d = 1024.0
    d_eff = 80.0
    n = np.logspace(9, 11.5, 60)
    L = uet_predict(c, L_inf, np.full_like(n, d_eff), np.full_like(n, d), n)
    df = pd.DataFrame({"n_tokens": n, "loss": L})
    write_dat(DATA / "uet_theoretical.dat", df, ["n_tokens", "loss"])


def main():
    full = dump_curriculum()
    dump_uet_fit(full)
    dump_failure()
    dump_scaling()
    dump_cross_domain()
    dump_theoretical_curve()
    print("wrote data files to", DATA)


if __name__ == "__main__":
    main()
