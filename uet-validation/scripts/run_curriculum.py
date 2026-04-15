from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uet.pretrained import PYTHIA_CHECKPOINTS, compute_model_spectrum
from uet.plotting import plot_curriculum

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Exp 7: d_eff(t) curriculum tracking")
    parser.add_argument("--model", default="EleutherAI/pythia-160m-deduped")
    parser.add_argument("--checkpoints", type=int, nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tokens", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = args.checkpoints or PYTHIA_CHECKPOINTS

    rows = []
    for step in checkpoints:
        logger.info("Checkpoint step=%d", step)
        spec = compute_model_spectrum(
            args.model, device=args.device, max_tokens=args.max_tokens,
            batch_size=args.batch_size, revision=str(step),
        )
        rows.append({
            "step": step,
            "val_loss": spec.val_loss,
            "d_eff": spec.d_eff,
            "stable_rank": spec.stable_rank,
            "hidden_dim": spec.hidden_dim,
        })
        logger.info("  step=%d d_eff=%.1f val_loss=%.4f", step, spec.d_eff, spec.val_loss)

    df = pd.DataFrame(rows)
    model_short = args.model.split("/")[-1]

    csv_path = args.output_dir / f"curriculum_{model_short}.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved %s", csv_path)

    plot_curriculum(df, args.model, args.output_dir / f"curriculum_{model_short}.png")
    logger.info("Done")


if __name__ == "__main__":
    main()
