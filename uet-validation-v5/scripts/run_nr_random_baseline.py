"""Pure Gaussian baseline for noise-reservoir.

Three conditions:
  (a) Pure random Gaussian H (no network): d_eff ≈ d (no spectral gap)
  (b) Random-init Pythia forward pass: d_eff << d (architectural structure)
  (c) Trained Pythia (final): d_eff << d (learned structure)

Predictions:
  (a) sin_theta ≈ 1 (DK condition violated, no gap)
  (b) sin_theta small (architecture creates gap)
  (c) sin_theta very small (learned gap tighter than arch-only gap)

This isolates the spectral-gap requirement from the specific cause.
"""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import numpy as np

V5 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V5))
sys.path.insert(0, str(V5.parent / "uet-validation"))

from uet.run_utils import setup_run_dir, setup_logging, dump_metadata
from uet_v5.posthoc_noise import run as posthoc_run

logger = logging.getLogger(__name__)

N = 50_000
D = 512      # match Pythia-70M
M_VALUES = [0, 50, 100, 200, 500]
N_SEEDS = 3
K_TEST = 10


def _find_cached(v5_root: Path, name: str, condition: str) -> Path | None:
    base = v5_root / "results"
    if condition == "trained":
        for d in sorted((base / "nr_posthoc").glob("*"), reverse=True):
            p = d / f"{name}_H.npy"
            if p.exists():
                return p
    elif condition == "untrained":
        for d in sorted((base / "nr_untrained").glob("*"), reverse=True):
            p = d / f"{name}-untrained_H.npy"
            if p.exists():
                return p
    return None


def main() -> None:
    run_dir = setup_run_dir(V5 / "results", "nr_random")
    setup_logging(run_dir)

    all_rows: list[dict] = []

    # (a) Pure Gaussian
    logger.info("=== pure Gaussian baseline (d=%d, n=%d) ===", D, N)
    rng = np.random.default_rng(42)
    H_gauss = rng.standard_normal((N, D)).astype(np.float32)
    rows = posthoc_run(H_gauss, m_values=M_VALUES, n_seeds=N_SEEDS, k=K_TEST)
    for r in rows:
        r["condition"] = "pure_gaussian"
    all_rows.extend(rows)

    # (b) Untrained Pythia
    for name, short in [("pythia-70m", "pythia-70m")]:
        p = _find_cached(V5, short, "untrained")
        if p is None:
            logger.warning("no untrained %s", short)
            continue
        H = np.load(p).astype(np.float32)
        if len(H) > N:
            idx = rng.choice(len(H), N, replace=False)
            H = H[idx]
        logger.info("=== untrained %s (shape=%s) ===", short, H.shape)
        rows = posthoc_run(H, m_values=M_VALUES, n_seeds=N_SEEDS, k=K_TEST)
        for r in rows:
            r["condition"] = f"untrained_{short}"
        all_rows.extend(rows)

    # (c) Trained Pythia
    for name, short in [("pythia-70m", "pythia-70m")]:
        p = _find_cached(V5, short, "trained")
        if p is None:
            logger.warning("no trained %s", short)
            continue
        H = np.load(p).astype(np.float32)
        if len(H) > N:
            idx = rng.choice(len(H), N, replace=False)
            H = H[idx]
        logger.info("=== trained %s (shape=%s) ===", short, H.shape)
        rows = posthoc_run(H, m_values=M_VALUES, n_seeds=N_SEEDS, k=K_TEST)
        for r in rows:
            r["condition"] = f"trained_{short}"
        all_rows.extend(rows)

    csv_path = run_dir / "random.csv"
    fieldnames = ["condition", "m", "seed", "sin_theta", "dk_bound",
                  "d_eff_aug", "d_eff_base", "k", "n", "d"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    logger.info("wrote %s", csv_path)

    summary = {}
    for cond in sorted({r["condition"] for r in all_rows}):
        sub = [r for r in all_rows if r["condition"] == cond]
        m_max_rows = [r for r in sub if r["m"] == max(M_VALUES)]
        summary[cond] = {
            "d_eff": sub[0]["d_eff_base"],
            "sin_at_m_max": round(float(np.mean([r["sin_theta"] for r in m_max_rows])), 5),
        }
    dump_metadata(run_dir, summary)
    print("\n=== NR Random Baseline Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
