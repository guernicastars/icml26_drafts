"""Changepoint detection on d_eff(t) trajectories.

Reads existing curriculum and grokking CSVs, runs PELT on each d_eff
series, outputs a table of (model/condition, tau_step, d_eff_before,
d_eff_after, drop_fraction).
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
from uet_v5.changepoint import detect

logger = logging.getLogger(__name__)

ROOT = V5.parent


def _latest(base: Path) -> Path | None:
    if not base.exists():
        return None
    dirs = sorted(base.iterdir(), reverse=True)
    return dirs[0] if dirs else None


def _load_curriculum(csv_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    df = pd.read_csv(csv_path)
    if "d_eff" not in df.columns:
        return None
    df = df.dropna(subset=["d_eff"]).sort_values("step")
    if len(df) < 6:
        return None
    return df["step"].to_numpy(), df["d_eff"].to_numpy()


def process_curriculum(rows: list[dict]) -> None:
    for model_dir in [
        ROOT / "uet-validation/results/curriculum" / d
        for d in ["20260418_020501_pythia-160m", "20260418_020501_pythia-410m"]
    ]:
        csv_p = model_dir / "curriculum.csv"
        if not csv_p.exists():
            logger.warning("missing %s", csv_p)
            continue
        pair = _load_curriculum(csv_p)
        if pair is None:
            continue
        steps, deff = pair
        model_name = model_dir.name.split("_")[-1]
        cp = detect(steps, deff)
        row = {"source": f"curriculum_{model_name}", **cp}
        rows.append(row)
        logger.info("curriculum %s  tau=%d  peak=%.1f  final=%.1f  drop=%.2f",
                    model_name, cp["tau_step"], cp["d_eff_peak"], cp["d_eff_final"], cp["drop_fraction"])

    for model_dir in [
        ROOT / "uet-validation-v2/results/curriculum" / d
        for d in ["20260418_053410_pythia-70m-deduped", "20260418_140612_OLMo-1B"]
    ]:
        csv_p = model_dir / "curriculum.csv"
        if not csv_p.exists():
            logger.warning("missing %s", csv_p)
            continue
        pair = _load_curriculum(csv_p)
        if pair is None:
            continue
        steps, deff = pair
        model_name = model_dir.name.split("_")[-1]
        cp = detect(steps, deff)
        row = {"source": f"curriculum_{model_name}", **cp}
        rows.append(row)
        logger.info("curriculum %s  tau=%d  peak=%.1f  final=%.1f  drop=%.2f",
                    model_name, cp["tau_step"], cp["d_eff_peak"], cp["d_eff_final"], cp["drop_fraction"])


def process_grokking(rows: list[dict]) -> None:
    run = _latest(ROOT / "uet-validation-v3/results/grokking_lambda")
    if run is None:
        logger.warning("no grokking_lambda results")
        return
    csv_p = run / "trajectories.csv"
    if not csv_p.exists():
        return
    df = pd.read_csv(csv_p).dropna(subset=["d_eff"])
    for lam in sorted(df["lambda"].unique()):
        sub = df[df["lambda"] == lam].sort_values("step")
        # average over seeds
        grp = sub.groupby("step", as_index=False)["d_eff"].mean()
        if len(grp) < 6:
            continue
        steps = grp["step"].to_numpy()
        deff = grp["d_eff"].to_numpy()
        cp = detect(steps, deff)
        lam_str = f"{lam:.3g}"
        row = {"source": f"grokking_lam{lam_str}", **cp}
        rows.append(row)
        logger.info("grokking lam=%s  tau=%d  peak=%.1f  final=%.1f  drop=%.2f",
                    lam_str, cp["tau_step"], cp["d_eff_peak"], cp["d_eff_final"], cp["drop_fraction"])


def main() -> None:
    run_dir = setup_run_dir(V5 / "results", "changepoint")
    setup_logging(run_dir)
    logger.info("changepoint detection")

    rows: list[dict] = []
    process_curriculum(rows)
    process_grokking(rows)

    csv_path = run_dir / "cp.csv"
    if rows:
        fieldnames = ["source", "tau_step", "d_eff_peak", "d_eff_final",
                      "drop_fraction", "has_interior_peak"]
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        logger.info("wrote %s (%d rows)", csv_path, len(rows))

    dump_metadata(run_dir, {"n_series": len(rows)})
    print("\n=== Changepoint Summary ===")
    for r in rows:
        print(f"  {r['source']}: tau={r['tau_step']}  "
              f"peak={r['d_eff_peak']:.1f}  final={r['d_eff_final']:.1f}  "
              f"drop={r['drop_fraction']:.2f}  interior={r['has_interior_peak']}")


if __name__ == "__main__":
    main()
