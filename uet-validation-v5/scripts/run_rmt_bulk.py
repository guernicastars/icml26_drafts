"""Marchenko-Pastur bulk analysis on cached eigenspectra.

Reads eigenvalues.npy from each curriculum checkpoint directory and
computes: MP bulk edge, signal eigenvalue count, d_eff.  Validates that
signal_count ≈ d_eff through the discovery→formalisation trajectory.
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

V5 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V5))
sys.path.insert(0, str(V5.parent / "uet-validation"))

from uet.run_utils import setup_run_dir, setup_logging, dump_metadata
from uet_v5.mp_bulk import analyse

logger = logging.getLogger(__name__)

ROOT = V5.parent
N_HARVEST = 500_000   # max_tokens used in harvest_activations

CURRICULUM_DIRS = [
    (ROOT / "uet-validation/results/curriculum/20260418_020501_pythia-160m",  768, "pythia-160m"),
    (ROOT / "uet-validation/results/curriculum/20260418_020501_pythia-410m", 1024, "pythia-410m"),
]


def main() -> None:
    run_dir = setup_run_dir(V5 / "results", "rmt_bulk")
    setup_logging(run_dir)
    logger.info("RMT bulk analysis")

    rows: list[dict] = []

    for model_dir, d, model_name in CURRICULUM_DIRS:
        ckpt_root = model_dir / "checkpoints"
        if not ckpt_root.exists():
            logger.warning("missing %s", ckpt_root)
            continue

        for ckpt_dir in sorted(ckpt_root.iterdir()):
            eig_path = ckpt_dir / "eigenvalues.npy"
            spec_path = ckpt_dir / "spectrum.json"
            if not eig_path.exists():
                continue

            eigs = np.load(eig_path)
            step = int(ckpt_dir.name.replace("step", ""))

            val_loss = None
            if spec_path.exists():
                with spec_path.open() as f:
                    spec = json.load(f)
                    val_loss = spec.get("val_loss")

            stats = analyse(eigs, d=d, n=N_HARVEST)
            row = {"model": model_name, "step": step, "val_loss": val_loss, **stats}
            rows.append(row)
            logger.info("%s step=%d  d_eff=%.1f  k90=%d  mp_s=%d  entropy=%.2f",
                        model_name, step, stats["d_eff"], stats["k_90pct"],
                        stats["signal_count_mp"], stats["spectral_entropy"])

    csv_path = run_dir / "bulk.csv"
    if rows:
        fieldnames = ["model", "step", "val_loss", "d_eff", "signal_count_mp",
                      "k_90pct", "k_99pct", "top1_fraction", "spectral_entropy",
                      "sigma2", "lambda_plus", "d", "n"]
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        logger.info("wrote %s (%d rows)", csv_path, len(rows))

    summary = {}
    if rows:
        final_rows = [r for r in rows if r["step"] in (143000, 738020)]
        summary = {
            "n_checkpoints": len(rows),
            "final_rows": [
                {"model": r["model"], "step": r["step"],
                 "d_eff": r["d_eff"], "k_90pct": r["k_90pct"],
                 "signal_count_mp": r["signal_count_mp"],
                 "spectral_entropy": r["spectral_entropy"]}
                for r in final_rows
            ],
        }
    dump_metadata(run_dir, summary)

    print("\n=== RMT Bulk Summary ===")
    print(f"  n_checkpoints: {len(rows)}")
    for r in rows:
        if r["step"] in (143000, 8000, 64, 738020):
            print(f"  {r['model']} step={r['step']:>8d}  "
                  f"d_eff={r['d_eff']:6.1f}  k_90={r['k_90pct']:4d}  "
                  f"mp_signal={r['signal_count_mp']:4d}  "
                  f"top1={r['top1_fraction']:.3f}")


if __name__ == "__main__":
    main()
