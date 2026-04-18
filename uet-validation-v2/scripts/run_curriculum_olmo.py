"""
OLMo-1B curriculum: harvests d_eff and val_loss at ~12 log-spaced training checkpoints.

OLMo-1B-hf has 351 step-based branches on HF: step{N}-tokens{M}B
Branch format encodes both training step and tokens seen directly.
Tokens/step ≈ 4.19M (slightly higher than Pythia's 2.097M).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "uet-validation"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uet.eigendecomp import covariance, effective_dimension, eigenspectrum, stable_rank
from uet.pretrained import _tokenize_dataset
from uet.run_utils import dump_config, dump_metadata, setup_logging, setup_run_dir

logger = logging.getLogger(__name__)

MODEL = "allenai/OLMo-1B-hf"
N_CHECKPOINTS = 14  # log-spaced across full training run


def _parse_branch(name: str) -> tuple[int, int] | None:
    m = re.match(r"step(\d+)-tokens(\d+\.?\d*)B", name)
    if not m:
        return None
    step = int(m.group(1))
    tokens = int(float(m.group(2)) * 1_000_000_000)
    return step, tokens


def select_checkpoints(n: int = N_CHECKPOINTS) -> list[tuple[int, int, str]]:
    api = HfApi()
    refs = api.list_repo_refs(MODEL)
    parsed = []
    for b in refs.branches:
        result = _parse_branch(b.name)
        if result:
            step, tokens = result
            parsed.append((step, tokens, b.name))
    parsed.sort()

    steps = np.array([s for s, _, _ in parsed])
    # log-space over full range, keep discovery-phase anchor at step 1000
    targets = np.unique(np.round(
        np.logspace(np.log10(max(steps.min(), 1)), np.log10(steps.max()), n)
    ).astype(int))

    selected = []
    seen_steps = set()
    for target in targets:
        idx = int(np.argmin(np.abs(steps - target)))
        s, t, b = parsed[idx]
        if s not in seen_steps:
            selected.append((s, t, b))
            seen_steps.add(s)
    return selected


@torch.no_grad()
def harvest_one(
    branch: str,
    device: str,
    max_tokens: int,
    seq_len: int,
    batch_size: int,
) -> tuple[float, float, int]:
    """Returns (val_loss, d_eff, hidden_dim)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=branch, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=branch, dtype=torch.float16,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
    )
    model.eval()
    input_ids = _tokenize_dataset(tokenizer, max_tokens=max_tokens, seq_len=seq_len)

    all_hidden, total_loss, total_tokens = [], 0.0, 0
    n_batches = (len(input_ids) + batch_size - 1) // batch_size
    for i in range(0, len(input_ids), batch_size):
        batch = input_ids[i : i + batch_size].to(model.device)
        out = model(batch, output_hidden_states=True, labels=batch)
        h = out.hidden_states[-1]
        all_hidden.append(h.float().cpu().numpy().reshape(-1, h.shape[-1]))
        total_loss += out.loss.item() * batch.numel()
        total_tokens += batch.numel()

    del model
    torch.cuda.empty_cache()

    embeddings = np.concatenate(all_hidden, axis=0)
    cov = covariance(embeddings)
    evals = eigenspectrum(cov)
    d_eff = effective_dimension(evals)
    val_loss = total_loss / max(total_tokens, 1)
    hidden_dim = embeddings.shape[1]
    logger.info("branch=%-40s  val_loss=%.4f  d_eff=%.1f  hidden=%d",
                branch, val_loss, d_eff, hidden_dim)
    return val_loss, d_eff, hidden_dim


def main():
    parser = argparse.ArgumentParser(description="OLMo-1B curriculum: d_eff(n) sweep")
    parser.add_argument("--n-checkpoints", type=int, default=N_CHECKPOINTS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tokens", type=int, default=150_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    run_dir = setup_run_dir(args.output_dir, "curriculum", args.run_name or "OLMo-1B")
    setup_logging(run_dir)
    dump_config(run_dir, args)

    checkpoints = select_checkpoints(args.n_checkpoints)
    logger.info("Selected %d checkpoints:", len(checkpoints))
    for step, tokens, branch in checkpoints:
        logger.info("  step=%7d  n_tokens=%.0fB  branch=%s", step, tokens / 1e9, branch)

    rows = []
    for step, n_tokens, branch in tqdm(checkpoints, desc="OLMo-1B ckpts", unit="ckpt"):
        try:
            val_loss, d_eff, hidden_dim = harvest_one(
                branch, args.device, args.max_tokens, args.seq_len, args.batch_size,
            )
        except Exception as e:
            logger.warning("Failed step=%d branch=%s: %s", step, branch, e)
            continue

        step_dir = run_dir / "checkpoints" / f"step{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        with open(step_dir / "spectrum.json", "w") as f:
            json.dump({"step": step, "n_tokens": n_tokens, "val_loss": val_loss,
                       "d_eff": d_eff, "hidden_dim": hidden_dim}, f, indent=2)

        rows.append({"step": step, "n_tokens": n_tokens, "val_loss": val_loss,
                     "d_eff": d_eff, "hidden_dim": hidden_dim, "branch": branch})

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "curriculum.csv", index=False)
    logger.info("Saved %d rows to curriculum.csv", len(df))

    dump_metadata(run_dir, {
        "model": MODEL,
        "n_checkpoints": len(df),
        "d_eff_final": float(df["d_eff"].iloc[-1]) if len(df) else None,
        "val_loss_final": float(df["val_loss"].iloc[-1]) if len(df) else None,
    })


if __name__ == "__main__":
    main()
