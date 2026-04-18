from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "uet-validation"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uet.eigendecomp import covariance, effective_dimension, eigenspectrum, stable_rank
from uet.run_utils import dump_config, dump_metadata, setup_logging, setup_run_dir
from uet_v2.layer_analysis import harvest_all_layer_activations

logger = logging.getLogger(__name__)

MODELS = [
    "EleutherAI/pythia-70m-deduped",
    "EleutherAI/pythia-160m-deduped",
]
# step 64 = discovery collapse; step 1000 = formalisation onset; step 143000 = final
CHECKPOINTS = [64, 1000, 143000]


def analyse_layers(
    model_name: str,
    revision: str,
    device: str,
    max_tokens: int,
    seq_len: int,
    batch_size: int,
) -> list[dict]:
    step = int(revision)
    layer_embeds, val_loss = harvest_all_layer_activations(
        model_name, device=device, max_tokens=max_tokens,
        seq_len=seq_len, batch_size=batch_size, revision=revision,
    )
    rows = []
    for layer_idx, embeddings in sorted(layer_embeds.items()):
        cov = covariance(embeddings)
        evals = eigenspectrum(cov)
        rows.append({
            "model": model_name.split("/")[-1],
            "step": step,
            "layer": layer_idx,
            "d_eff": effective_dimension(evals),
            "stable_rank": stable_rank(evals),
            "hidden_dim": embeddings.shape[1],
            "val_loss": val_loss,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="L5: Per-layer d_eff analysis")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=CHECKPOINTS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tokens", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    run_dir = setup_run_dir(args.output_dir, "layer_deff", args.run_name)
    setup_logging(run_dir)
    dump_config(run_dir, args)

    all_rows = []
    for model_name in args.models:
        for step in args.checkpoints:
            logger.info("Model %s checkpoint step=%d", model_name.split("/")[-1], step)
            try:
                rows = analyse_layers(
                    model_name, revision=str(step),
                    device=args.device, max_tokens=args.max_tokens,
                    seq_len=args.seq_len, batch_size=args.batch_size,
                )
                all_rows.extend(rows)
            except Exception as e:
                logger.warning("Failed %s step=%d: %s", model_name, step, e)

    df = pd.DataFrame(all_rows)
    df.to_csv(run_dir / "layer_deff.csv", index=False)

    if len(df) > 0:
        summary = df.groupby(["model", "step"])["d_eff"].agg(["min", "max", "mean"]).reset_index()
        logger.info("\n%s", summary.to_string(index=False))

    dump_metadata(run_dir, {
        "n_models": len(args.models),
        "n_checkpoints": len(args.checkpoints),
        "n_rows": int(len(df)),
    })


if __name__ == "__main__":
    main()
