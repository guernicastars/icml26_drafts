"""Distillation-rank test.

Trains k-bottleneck MLP students on cached Pythia-160M hidden states H.
Expects H to have been cached by run_nr_posthoc.py at:
  uet-validation-v5/results/nr_posthoc/*/pythia-160m_H.npy

Sweeps k in [2, 4, 8, 16, 32, 64, 128, 256, 512] with 3 seeds each.
Reports MSE and relative MSE; expects sharp knee near k = d_eff(H) ≈ 49.
"""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch

V5 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V5))
sys.path.insert(0, str(V5.parent / "uet-validation"))

from uet.eigendecomp import covariance, effective_dimension, eigenspectrum
from uet.run_utils import setup_run_dir, setup_logging, dump_metadata
from uet_v5.distill_rank import train_student

logger = logging.getLogger(__name__)

K_VALUES = [2, 4, 8, 16, 32, 64, 128, 256, 512]
SEEDS    = [0, 1, 2]
MODEL_NAME = "pythia-160m"
N_TRAIN    = 20_000   # subsample so training is fast; d_eff stable from ~5k


def _find_cached_H(results_root: Path) -> Path | None:
    nr = results_root / "nr_posthoc"
    if not nr.exists():
        return None
    for run_dir in sorted(nr.iterdir(), reverse=True):
        candidate = run_dir / f"{MODEL_NAME}_H.npy"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    run_dir = setup_run_dir(V5 / "results", "distill_rank")
    setup_logging(run_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("distill_rank device=%s", device)

    H_path = _find_cached_H(V5 / "results")
    if H_path is None:
        logger.error("no cached H found — run run_nr_posthoc.py first")
        return
    logger.info("loading H from %s", H_path)

    H = np.load(H_path).astype(np.float64)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(H), min(N_TRAIN, len(H)), replace=False)
    H_sub = H[idx]

    eigs = eigenspectrum(covariance(H_sub))
    d_eff_base = effective_dimension(eigs)
    logger.info("H shape=%s  d_eff=%.2f", H_sub.shape, d_eff_base)

    rows: list[dict] = []
    for k in K_VALUES:
        for seed in SEEDS:
            row = train_student(H_sub, k=k, hidden=384, epochs=80, device=device, seed=seed)
            row["d_eff_base"] = round(float(d_eff_base), 3)
            rows.append(row)

    csv_path = run_dir / "rank.csv"
    fieldnames = ["k", "seed", "mse", "rel_mse", "z_deff", "d_eff_base"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    logger.info("wrote %s (%d rows)", csv_path, len(rows))

    summary: dict = {}
    for k in K_VALUES:
        sub = [r for r in rows if r["k"] == k]
        mses = [r["rel_mse"] for r in sub]
        summary[f"k{k}"] = {"rel_mse_mean": round(float(np.mean(mses)), 5)}
    summary["d_eff_base"] = round(float(d_eff_base), 2)
    dump_metadata(run_dir, summary)

    print("\n=== Distill Rank Summary ===")
    print(f"  d_eff_base = {d_eff_base:.2f}")
    for k in K_VALUES:
        sub = [r for r in rows if r["k"] == k]
        print(f"  k={k:4d}  rel_mse={np.mean([r['rel_mse'] for r in sub]):.4f}  "
              f"z_deff={np.mean([r['z_deff'] for r in sub]):.2f}")


if __name__ == "__main__":
    main()
