"""Noise-reservoir test: trained vs untrained (random-init) models.

Key prediction (Theorem 1): the noise-reservoir property requires a
SIGNAL SUBSPACE with spectral gap above the noise floor.

Trained model: d_eff << d, spectral gap exists -> sin_theta stays small.
Untrained model: d_eff ≈ d (flat spectrum), no gap -> sin_theta large.

This is the cleanest falsification test: if UET's frame were just "math of
block-diagonal covariance", untrained models should also show the effect.
If it's about LEARNING creating structured representations, only trained
models should.
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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from uet.run_utils import setup_run_dir, setup_logging, dump_metadata
from uet_v5.posthoc_noise import run as posthoc_run

logger = logging.getLogger(__name__)

M_VALUES = [0, 50, 100, 200, 500]
N_SEEDS  = 3
K_TEST   = 10
MAX_TOKENS = 50_000


@torch.no_grad()
def harvest_from_untrained(model_name: str, device: str,
                           max_tokens: int = 50_000, seq_len: int = 1024,
                           batch_size: int = 4) -> np.ndarray:
    """Load model at step=0 (random init) and harvest hidden states."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map=device,
        revision="step0",
    )
    model.eval()

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-v1", split="test",
                      trust_remote_code=True)
    text = "\n\n".join(row["text"] for row in ds if row["text"].strip())
    tokens = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
    n_seqs = len(tokens) // seq_len
    input_ids = torch.tensor(tokens[:n_seqs * seq_len]).reshape(n_seqs, seq_len)

    all_h = []
    for i in tqdm(range(0, len(input_ids), batch_size), desc="untrained harvest"):
        b = input_ids[i:i + batch_size].to(device)
        out = model(b, output_hidden_states=True)
        h = out.hidden_states[-1].float().cpu().numpy().reshape(-1, out.hidden_states[-1].shape[-1])
        all_h.append(h)

    del model
    torch.cuda.empty_cache()
    return np.concatenate(all_h, axis=0)


def main() -> None:
    run_dir = setup_run_dir(V5 / "results", "nr_untrained")
    setup_logging(run_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_rows: list[dict] = []

    # Load cached trained Pythia-70M H
    for cached_run in sorted((V5 / "results" / "nr_posthoc").glob("*"), reverse=True):
        p = cached_run / "pythia-70m_H.npy"
        if p.exists():
            H_trained = np.load(p).astype(np.float32)
            logger.info("trained H: loaded %s, shape=%s", p, H_trained.shape)
            break
    else:
        logger.error("no trained Pythia-70M H found")
        return

    # Subsample for comparison
    n_sub = min(50_000, len(H_trained))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(H_trained), n_sub, replace=False)
    H_trained_sub = H_trained[idx]

    logger.info("=== trained Pythia-70M (final checkpoint) ===")
    rows = posthoc_run(H_trained_sub, m_values=M_VALUES, n_seeds=N_SEEDS, k=K_TEST)
    for r in rows:
        r["condition"] = "trained"
    all_rows.extend(rows)

    # Harvest from step=0 (random-init) Pythia-70M
    logger.info("=== untrained Pythia-70M (step=0, random init) ===")
    try:
        H_untrained = harvest_from_untrained("EleutherAI/pythia-70m-deduped",
                                              device=device, max_tokens=MAX_TOKENS)
        np.save(run_dir / "pythia-70m-untrained_H.npy", H_untrained.astype(np.float32))
        rows = posthoc_run(H_untrained, m_values=M_VALUES, n_seeds=N_SEEDS, k=K_TEST)
        for r in rows:
            r["condition"] = "untrained"
        all_rows.extend(rows)
    except Exception as exc:
        logger.error("untrained harvest failed: %s", exc)

    csv_path = run_dir / "untrained.csv"
    fieldnames = ["condition", "m", "seed", "sin_theta", "dk_bound",
                  "d_eff_aug", "d_eff_base", "k", "n", "d"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    logger.info("wrote %s (%d rows)", csv_path, len(all_rows))

    summary = {}
    for cond in ["trained", "untrained"]:
        sub = [r for r in all_rows if r["condition"] == cond]
        if not sub:
            continue
        m_max_rows = [r for r in sub if r["m"] == max(M_VALUES)]
        summary[cond] = {
            "d_eff": sub[0]["d_eff_base"],
            "sin_at_m_max": round(float(np.mean([r["sin_theta"] for r in m_max_rows])), 5),
        }
    dump_metadata(run_dir, summary)
    print("\n=== NR Trained vs Untrained ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
