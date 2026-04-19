"""Post-hoc noise-reservoir experiment on real Pythia/OLMo embeddings.

For each model at its final checkpoint:
  1. Harvest n_harvest hidden states on WikiText-103.
  2. Cache H to disk (reused by run_distill_rank.py).
  3. For each noise budget m, append m Gaussian columns scaled to col_rms(H).
  4. Measure sin_theta(top_k H, top_k H_aug) and d_eff of H_aug.

UET prediction: sin_theta < 0.05 for all tested m; d_eff_aug ≈ d_eff_base.
"""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch

V5 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V5))
sys.path.insert(0, str(V5.parent / "uet-validation"))

from uet.run_utils import setup_run_dir, setup_logging, dump_metadata
from uet_v5.harvest import harvest as harvest_activations
from uet_v5.posthoc_noise import run as posthoc_run

logger = logging.getLogger(__name__)

MODELS = [
    ("EleutherAI/pythia-70m-deduped",  "pythia-70m"),
    ("EleutherAI/pythia-160m-deduped", "pythia-160m"),
    ("EleutherAI/pythia-410m-deduped", "pythia-410m"),
    ("allenai/OLMo-1B-hf",             "olmo-1b"),
]

M_VALUES   = [0, 50, 100, 200, 500]   # d+m_max = d+500, manageable covariance
N_SEEDS    = 3
K_TEST     = 10      # top-10 dominant directions (well above noise floor)
MAX_TOKENS = 100_000  # ~100 seqs × 1024 tokens
CACHE_DIR  = None


def main() -> None:
    run_dir = setup_run_dir(V5 / "results", "nr_posthoc")
    setup_logging(run_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("NR-posthoc, device=%s", device)

    all_rows: list[dict] = []

    for model_id, short_name in MODELS:
        logger.info("=== %s ===", short_name)

        # Check for existing cached H to skip re-harvesting
        cached = None
        if CACHE_DIR:
            p = Path(CACHE_DIR) / f"{short_name}_H.npy"
            if p.exists():
                cached = p
        if cached is None:
            for prev_run in sorted((V5 / "results" / "nr_posthoc").glob("*"), reverse=True):
                p = prev_run / f"{short_name}_H.npy"
                if p.exists():
                    cached = p
                    break

        if cached is not None:
            logger.info("reusing cached H from %s", cached)
            H = np.load(cached).astype(np.float64)
            val_loss = float("nan")
        else:
            try:
                H, val_loss = harvest_activations(
                    model_id, device=device, max_tokens=MAX_TOKENS, batch_size=4,
                )
            except Exception as exc:
                logger.error("harvest failed for %s: %s", short_name, exc)
                continue

        # Cache H for distill_rank reuse
        cache_path = run_dir / f"{short_name}_H.npy"
        np.save(cache_path, H.astype(np.float32))
        logger.info("cached H shape=%s val_loss=%.4f -> %s", H.shape, val_loss, cache_path)

        rows = posthoc_run(H, m_values=M_VALUES, n_seeds=N_SEEDS, k=K_TEST)
        for r in rows:
            r["model"] = short_name
            r["val_loss"] = round(float(val_loss), 5)
        all_rows.extend(rows)

        # Print per-model summary
        m0 = [r for r in rows if r["m"] == 0]
        m_max_rows = [r for r in rows if r["m"] == max(M_VALUES)]
        if m0 and m_max_rows:
            logger.info(
                "%s  d_eff=%.1f  k=%d  sin@m_max=%.4f  dk_bound@m_max=%.4f",
                short_name,
                m0[0]["d_eff_base"],
                K_TEST,
                float(np.mean([r["sin_theta"] for r in m_max_rows])),
                float(np.mean([r["dk_bound"] for r in m_max_rows])),
            )

    csv_path = run_dir / "posthoc.csv"
    fieldnames = ["model", "m", "seed", "sin_theta", "dk_bound",
                  "d_eff_aug", "d_eff_base", "k", "n", "d", "val_loss"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    logger.info("wrote %s (%d rows)", csv_path, len(all_rows))

    summary: dict = {}
    for model_id, short_name in MODELS:
        sub = [r for r in all_rows if r["model"] == short_name]
        if not sub:
            continue
        m_max = max(r["m"] for r in sub)
        worst = [r for r in sub if r["m"] == m_max]
        mean_sin = float(np.mean([r["sin_theta"] for r in worst]))
        mean_dk = float(np.mean([r["dk_bound"] for r in worst]))
        summary[short_name] = {
            "d_eff_base": sub[0]["d_eff_base"],
            "k_test": K_TEST,
            "sin_theta_at_m_max_mean": round(mean_sin, 5),
            "dk_bound_at_m_max": round(mean_dk, 5),
            "sin_below_dk_bound": mean_sin < mean_dk,
        }
    dump_metadata(run_dir, summary)

    print("\n=== NR-posthoc Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
