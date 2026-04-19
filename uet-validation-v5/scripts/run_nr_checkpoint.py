"""Noise-reservoir evolution across training.

For Pythia-160M, harvest H at multiple curriculum checkpoints and
measure NR-posthoc sin_theta at each.

Prediction: as formalisation progresses, the spectral gap grows and
NR robustness improves (sin_theta at fixed m decreases).
"""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

V5 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V5))
sys.path.insert(0, str(V5.parent / "uet-validation"))

from uet.run_utils import setup_run_dir, setup_logging, dump_metadata
from uet_v5.posthoc_noise import run as posthoc_run

logger = logging.getLogger(__name__)

MODEL = "EleutherAI/pythia-160m-deduped"
CHECKPOINTS = [64, 128, 512, 1000, 4000, 8000, 32000, 143000]
M_FIXED = 200
N_SEEDS = 3
K_TEST = 10
MAX_TOKENS = 50_000
SEQ_LEN = 1024
BATCH_SIZE = 4


@torch.no_grad()
def harvest_at_step(model_name: str, step: int, device: str) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map=device,
        revision=f"step{step}",
    )
    model.eval()

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-v1", split="test",
                      trust_remote_code=True)
    text = "\n\n".join(row["text"] for row in ds if row["text"].strip())
    tokens = tokenizer.encode(text, add_special_tokens=False)[:MAX_TOKENS]
    n_seqs = len(tokens) // SEQ_LEN
    input_ids = torch.tensor(tokens[:n_seqs * SEQ_LEN]).reshape(n_seqs, SEQ_LEN)

    all_h = []
    for i in range(0, len(input_ids), BATCH_SIZE):
        b = input_ids[i:i + BATCH_SIZE].to(device)
        out = model(b, output_hidden_states=True)
        h = out.hidden_states[-1].float().cpu().numpy().reshape(-1, out.hidden_states[-1].shape[-1])
        all_h.append(h)

    del model
    torch.cuda.empty_cache()
    return np.concatenate(all_h, axis=0)


def main() -> None:
    run_dir = setup_run_dir(V5 / "results", "nr_checkpoint")
    setup_logging(run_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("NR evolution across checkpoints, device=%s", device)

    all_rows: list[dict] = []

    for step in CHECKPOINTS:
        logger.info("=== step=%d ===", step)
        try:
            H = harvest_at_step(MODEL, step, device=device)
        except Exception as exc:
            logger.error("harvest failed at step=%d: %s", step, exc)
            continue
        logger.info("H shape=%s", H.shape)

        rows = posthoc_run(H, m_values=[0, M_FIXED], n_seeds=N_SEEDS, k=K_TEST)
        for r in rows:
            r["step"] = step
        all_rows.extend(rows)

        m_fixed_rows = [r for r in rows if r["m"] == M_FIXED]
        sin_mean = float(np.mean([r["sin_theta"] for r in m_fixed_rows]))
        logger.info("step=%d  d_eff=%.2f  k=%d  sin@m=%d: %.5f",
                    step, m_fixed_rows[0]["d_eff_base"], K_TEST, M_FIXED, sin_mean)

    csv_path = run_dir / "checkpoint.csv"
    fieldnames = ["step", "m", "seed", "sin_theta", "dk_bound",
                  "d_eff_aug", "d_eff_base", "k", "n", "d"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    logger.info("wrote %s", csv_path)

    summary = {}
    for step in CHECKPOINTS:
        sub = [r for r in all_rows if r["step"] == step and r["m"] == M_FIXED]
        if not sub:
            continue
        summary[f"step_{step}"] = {
            "d_eff": sub[0]["d_eff_base"],
            "sin_theta_at_m200_mean": round(float(np.mean([r["sin_theta"] for r in sub])), 5),
        }
    dump_metadata(run_dir, summary)
    print("\n=== NR Checkpoint Evolution ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
