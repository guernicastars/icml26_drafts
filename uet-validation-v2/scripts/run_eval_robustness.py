from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "uet-validation"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uet.eigendecomp import covariance, effective_dimension, eigenspectrum, stable_rank
from uet.pretrained import PYTHIA_CHECKPOINTS, _load_model_and_tokenizer
from uet.run_utils import dump_config, dump_metadata, setup_logging, setup_run_dir
from uet.scaling_fit import fit_uet_curriculum, pythia_step_to_tokens

logger = logging.getLogger(__name__)

POST_WARMUP_STEPS = [s for s in PYTHIA_CHECKPOINTS if s >= 1000]

EVAL_DATASETS = [
    ("wikitext", "wikitext-103-v1", "test"),
    ("c4", "en", "validation"),
    ("ptb_text_only", "penn_treebank", "test"),
]


@torch.no_grad()
def _harvest_on_dataset(
    model,
    tokenizer,
    dataset_name: str,
    dataset_config: str,
    split: str,
    max_tokens: int,
    seq_len: int,
    batch_size: int,
    device: str,
) -> tuple[float, float]:
    """Returns (val_loss, d_eff) for one dataset."""
    from datasets import load_dataset
    from uet.pretrained import _tokenize_dataset

    try:
        input_ids = _tokenize_dataset(
            tokenizer,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split,
            max_tokens=max_tokens,
            seq_len=seq_len,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load {dataset_name}/{dataset_config}/{split}: {e}") from e

    all_hidden = []
    total_loss = 0.0
    total_tokens = 0

    for i in range(0, len(input_ids), batch_size):
        batch = input_ids[i : i + batch_size].to(device)
        outputs = model(batch, output_hidden_states=True, labels=batch)
        hidden = outputs.hidden_states[-1]
        all_hidden.append(hidden.float().cpu().numpy().reshape(-1, hidden.shape[-1]))
        total_loss += outputs.loss.item() * batch.numel()
        total_tokens += batch.numel()

    embeddings = np.concatenate(all_hidden, axis=0)
    cov = covariance(embeddings)
    evals = eigenspectrum(cov)
    d_eff = effective_dimension(evals)
    val_loss = total_loss / max(total_tokens, 1)
    return val_loss, d_eff


def main():
    parser = argparse.ArgumentParser(description="L6: Eval-set robustness of fitted c constant")
    parser.add_argument("--model", default="EleutherAI/pythia-160m-deduped")
    parser.add_argument("--checkpoints", type=int, nargs="+", default=POST_WARMUP_STEPS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tokens", type=int, default=150_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    run_dir = setup_run_dir(args.output_dir, "eval_robustness", args.run_name)
    setup_logging(run_dir)
    dump_config(run_dir, args)

    all_rows = []
    for step in args.checkpoints:
        logger.info("Loading %s step=%d", args.model.split("/")[-1], step)
        try:
            model, tokenizer = _load_model_and_tokenizer(
                args.model, args.device, revision=str(step)
            )
        except Exception as e:
            logger.warning("Failed to load step=%d: %s", step, e)
            continue

        n_tokens = int(pythia_step_to_tokens(step))

        for ds_name, ds_config, split in EVAL_DATASETS:
            try:
                val_loss, d_eff = _harvest_on_dataset(
                    model, tokenizer,
                    dataset_name=ds_name, dataset_config=ds_config, split=split,
                    max_tokens=args.max_tokens, seq_len=args.seq_len,
                    batch_size=args.batch_size, device=args.device,
                )
                all_rows.append({
                    "model": args.model.split("/")[-1],
                    "step": step,
                    "n_tokens": n_tokens,
                    "dataset": f"{ds_name}/{ds_config}",
                    "val_loss": val_loss,
                    "d_eff": d_eff,
                    "hidden_dim": model.config.hidden_size,
                })
                logger.info(
                    "step=%6d %-30s  val_loss=%.4f  d_eff=%.1f",
                    step, f"{ds_name}/{ds_config}", val_loss, d_eff,
                )
            except Exception as e:
                logger.warning("step=%d dataset=%s failed: %s", step, ds_name, e)

        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)
    df.to_csv(run_dir / "robustness.csv", index=False)

    fit_rows = []
    for dataset, g in df.groupby("dataset"):
        if len(g) < 3:
            continue
        try:
            fit = fit_uet_curriculum(
                d_eff=g["d_eff"].values,
                d=g["hidden_dim"].values,
                n=g["n_tokens"].values,
                L=g["val_loss"].values,
            )
            fit_rows.append({
                "dataset": dataset,
                "c": fit.c, "L_inf": fit.L_inf,
                "r_squared": fit.r_squared, "rmse": fit.rmse,
                "n_points": fit.n_points,
            })
            logger.info("Fit %-30s  c=%.4g  L_inf=%.4f  R2=%.4f", dataset, fit.c, fit.L_inf, fit.r_squared)
        except Exception as e:
            logger.warning("Fit failed for %s: %s", dataset, e)

    fits_df = pd.DataFrame(fit_rows)
    fits_df.to_csv(run_dir / "c_by_dataset.csv", index=False)

    if len(fits_df) >= 2:
        c_vals = fits_df["c"].values
        spread = float(c_vals.max() / max(c_vals.min(), 1e-30))
        logger.info("c spread across datasets: max/min=%.2f", spread)

    dump_metadata(run_dir, {
        "model": args.model,
        "n_checkpoints": len(args.checkpoints),
        "n_datasets": len(df["dataset"].unique()) if len(df) else 0,
        "fits": fit_rows,
    })


if __name__ == "__main__":
    main()
