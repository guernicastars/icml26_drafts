from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uet.failure import sweep_failure_modes
from uet.plotting import plot_failure_heatmap

logger = logging.getLogger(__name__)

DEFAULT_D = [16, 32, 64, 128, 256]
DEFAULT_K = [2, 4, 8, 16, 32, 64]
DEFAULT_GAP = [0.5, 1.0, 2.0, 5.0, 10.0]


def main():
    parser = argparse.ArgumentParser(description="Exp 3: Failure mode sweep")
    parser.add_argument("--d-values", type=int, nargs="+", default=DEFAULT_D)
    parser.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K)
    parser.add_argument("--gap-values", type=float, nargs="+", default=DEFAULT_GAP)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_configs = len(args.d_values) * len(args.k_values) * len(args.gap_values) * args.n_seeds
    logger.info("Sweeping %d configurations (%d seeds)", n_configs // args.n_seeds, args.n_seeds)

    results = sweep_failure_modes(
        args.d_values, args.k_values, args.gap_values,
        n_samples=args.n_samples, n_seeds=args.n_seeds,
    )

    df = pd.DataFrame([
        {
            "d": r.d, "k": r.k, "gap_ratio": r.gap_ratio,
            "sin_angle": r.sin_angle, "d_eff": r.d_eff,
            "theorem_bound": r.theorem_bound,
            "condition_violated": r.condition_violated,
        }
        for r in results
    ])

    csv_path = args.output_dir / "failure_sweep.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved %s (%d rows)", csv_path, len(df))

    plot_failure_heatmap(df, args.output_dir / "failure_sweep.png")
    logger.info("Done")


if __name__ == "__main__":
    main()
