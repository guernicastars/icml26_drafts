"""Empirical Davis-Kahan constant fit.

Theorem 1 predicts sin_theta <= C * sqrt((d+m)/n) for the noise-reservoir.
We fit the constant C from the (m, n, sin_theta) measurements of NR-posthoc
across all models.

A good fit validates the theorem's functional form; a poor fit signals
either a non-trivial constant hidden in the derivation or a violated
assumption.
"""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V5 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V5))
sys.path.insert(0, str(V5.parent / "uet-validation"))

from uet.run_utils import setup_run_dir, setup_logging, dump_metadata

logger = logging.getLogger(__name__)


def _latest(base: Path) -> Path | None:
    if not base.exists():
        return None
    dirs = sorted(base.iterdir(), reverse=True)
    return dirs[0] if dirs else None


def main() -> None:
    run_dir = setup_run_dir(V5 / "results", "dk_constant")
    setup_logging(run_dir)

    nr_run = _latest(V5 / "results" / "nr_posthoc")
    if nr_run is None or not (nr_run / "posthoc.csv").exists():
        logger.error("run run_nr_posthoc.py first")
        return

    df = pd.read_csv(nr_run / "posthoc.csv")
    df = df[df["m"] > 0].copy()
    df["x"] = np.sqrt((df["d"] + df["m"]) / df["n"])

    # Fit sin_theta = C * x (no intercept, single parameter)
    x = df["x"].to_numpy()
    y = df["sin_theta"].to_numpy()
    C = float((x * y).sum() / (x ** 2).sum())
    y_pred = C * x
    r2 = 1.0 - float(((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    residual_max = float(np.abs(y - y_pred).max())

    logger.info("fit: sin_theta = %.4f * sqrt((d+m)/n),  R^2 = %.4f,  max resid = %.5f",
                C, r2, residual_max)

    # Per-model constant
    per_model = {}
    for m_name in df["model"].unique():
        sub = df[df["model"] == m_name]
        xx = sub["x"].to_numpy()
        yy = sub["sin_theta"].to_numpy()
        C_m = float((xx * yy).sum() / (xx ** 2).sum())
        per_model[m_name] = round(C_m, 5)
        logger.info("  %s:  C = %.5f", m_name, C_m)

    out_df = df.copy()
    out_df["sin_theta_pred"] = out_df["x"] * C
    out_df[["model", "m", "seed", "x", "sin_theta", "sin_theta_pred"]].to_csv(
        run_dir / "dk_fit.csv", index=False
    )

    dump_metadata(run_dir, {
        "C_global": round(C, 5),
        "R2": round(r2, 5),
        "residual_max": round(residual_max, 5),
        "per_model_C": per_model,
        "n_points": int(len(df)),
    })
    print(f"\n=== DK Constant Fit ===")
    print(f"  sin_theta = {C:.4f} * sqrt((d+m)/n)   R^2 = {r2:.4f}")
    print(f"  per-model C: {per_model}")


if __name__ == "__main__":
    main()
