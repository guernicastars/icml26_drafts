from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uet.plotting import plot_curriculum
from uet.pretrained import PYTHIA_CHECKPOINTS, compute_model_spectrum
from uet.run_utils import dump_config, dump_metadata, setup_logging, setup_run_dir

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Exp 7: d_eff(t) curriculum tracking")
    parser.add_argument("--model", default="EleutherAI/pythia-160m-deduped")
    parser.add_argument("--checkpoints", type=int, nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tokens", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    model_short = args.model.split("/")[-1]
    run_dir = setup_run_dir(
        args.output_dir, experiment="curriculum",
        run_name=args.run_name or model_short,
    )
    setup_logging(run_dir)
    dump_config(run_dir, args)

    checkpoints = args.checkpoints or PYTHIA_CHECKPOINTS
    logger.info("Running %d checkpoints for %s", len(checkpoints), args.model)

    rows = []
    for step in tqdm(checkpoints, desc=f"{model_short} ckpts", unit="ckpt"):
        logger.info("Checkpoint step=%d", step)
        spec = compute_model_spectrum(
            args.model, device=args.device, max_tokens=args.max_tokens,
            batch_size=args.batch_size, seq_len=args.seq_len, revision=str(step),
        )

        step_dir = run_dir / "checkpoints" / f"step{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        np.save(step_dir / "eigenvalues.npy", spec.eigenvalues)
        with open(step_dir / "spectrum.json", "w") as f:
            json.dump({
                "step": step,
                "val_loss": spec.val_loss,
                "d_eff": spec.d_eff,
                "stable_rank": spec.stable_rank,
                "hidden_dim": spec.hidden_dim,
            }, f, indent=2)

        rows.append({
            "step": step,
            "val_loss": spec.val_loss,
            "d_eff": spec.d_eff,
            "stable_rank": spec.stable_rank,
            "hidden_dim": spec.hidden_dim,
        })
        logger.info("  step=%d d_eff=%.1f val_loss=%.4f", step, spec.d_eff, spec.val_loss)

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "curriculum.csv", index=False)
    plot_curriculum(df, args.model, run_dir / "curriculum.png")

    dump_metadata(run_dir, {
        "model": args.model,
        "n_checkpoints": len(df),
        "d_eff_final": float(df["d_eff"].iloc[-1]),
        "d_eff_initial": float(df["d_eff"].iloc[0]),
        "val_loss_final": float(df["val_loss"].iloc[-1]),
    })
    logger.info("Done. Run dir: %s", run_dir)


if __name__ == "__main__":
    main()
