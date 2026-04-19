"""Per-layer noise-reservoir test on Pythia-160M.

Test NR-posthoc across all 12 transformer layers.
Prediction: deeper layers have more structured representations
(lower d_eff, larger spectral gap) and stronger NR robustness.
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
M_FIXED = 200
N_SEEDS = 3
K_TEST = 10
MAX_TOKENS = 50_000
SEQ_LEN = 1024
BATCH_SIZE = 4


@torch.no_grad()
def harvest_all_layers(model_name: str, device: str) -> tuple[list[np.ndarray], int]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map=device,
    )
    model.eval()

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-v1", split="test",
                      trust_remote_code=True)
    text = "\n\n".join(row["text"] for row in ds if row["text"].strip())
    tokens = tokenizer.encode(text, add_special_tokens=False)[:MAX_TOKENS]
    n_seqs = len(tokens) // SEQ_LEN
    input_ids = torch.tensor(tokens[:n_seqs * SEQ_LEN]).reshape(n_seqs, SEQ_LEN)

    n_layers = model.config.num_hidden_layers
    logger.info("harvesting %d layers", n_layers)
    layer_H = [[] for _ in range(n_layers + 1)]  # layer 0 = embedding, 1..L = transformer

    for i in tqdm(range(0, len(input_ids), BATCH_SIZE), desc="harvest all layers"):
        b = input_ids[i:i + BATCH_SIZE].to(device)
        out = model(b, output_hidden_states=True)
        for L in range(n_layers + 1):
            h = out.hidden_states[L].float().cpu().numpy().reshape(-1, out.hidden_states[L].shape[-1])
            layer_H[L].append(h)

    layer_H = [np.concatenate(lst, axis=0) for lst in layer_H]
    del model
    torch.cuda.empty_cache()
    return layer_H, n_layers


def main() -> None:
    run_dir = setup_run_dir(V5 / "results", "nr_layer")
    setup_logging(run_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    layer_H, n_layers = harvest_all_layers(MODEL, device=device)
    logger.info("harvested %d layers, each shape=%s", len(layer_H), layer_H[0].shape)

    all_rows: list[dict] = []

    for L, H in enumerate(layer_H):
        logger.info("=== layer=%d ===", L)
        # downsample to reduce memory per layer
        if len(H) > 50_000:
            idx = np.random.default_rng(0).choice(len(H), 50_000, replace=False)
            H = H[idx]
        rows = posthoc_run(H, m_values=[0, M_FIXED], n_seeds=N_SEEDS, k=K_TEST)
        for r in rows:
            r["layer"] = L
        all_rows.extend(rows)

        m_fixed = [r for r in rows if r["m"] == M_FIXED]
        sin_mean = float(np.mean([r["sin_theta"] for r in m_fixed]))
        logger.info("layer=%d  d_eff=%.1f  k=%d  sin@m=%d: %.5f",
                    L, m_fixed[0]["d_eff_base"], K_TEST, M_FIXED, sin_mean)

    csv_path = run_dir / "layer.csv"
    fieldnames = ["layer", "m", "seed", "sin_theta", "dk_bound",
                  "d_eff_aug", "d_eff_base", "k", "n", "d"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    logger.info("wrote %s", csv_path)

    summary = {}
    for L in range(n_layers + 1):
        sub = [r for r in all_rows if r["layer"] == L and r["m"] == M_FIXED]
        if not sub:
            continue
        summary[f"layer_{L}"] = {
            "d_eff": sub[0]["d_eff_base"],
            "sin_theta_mean": round(float(np.mean([r["sin_theta"] for r in sub])), 5),
        }
    dump_metadata(run_dir, summary)
    print("\n=== Per-Layer NR Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
