"""Grokking lambda ablation: vary weight decay on modular addition (a+b) mod 97.

Separates two hypotheses:
  H1: d_eff collapse is caused by weight decay (artefact)
  H2: d_eff collapse is caused by task structure discovery (formalisation)

If H1: lambda=0 shows no grokking AND no d_eff collapse.
If H2: lambda=0 shows grokking AND d_eff collapse.
If mixed: grokking slows/stops at lambda=0 but d_eff still collapses ->
          weight decay mediates grokking but not d_eff compression.
"""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

V3 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3))
sys.path.insert(0, str(V3.parent / "uet-validation"))

from uet.run_utils import setup_run_dir, setup_logging, dump_metadata
from uet.eigendecomp import covariance, eigenspectrum, effective_dimension, stable_rank

logger = logging.getLogger(__name__)

P          = 97
TRAIN_FRAC = 0.4
EMBED_DIM  = 128
HIDDEN     = 256
N_STEPS    = 80_000   # extended for lambda=0 which grokks later
LOG_EVERY  = 200
DEFF_EVERY = 500
LR         = 1e-3
BATCH_SIZE = 512
SEEDS      = [0, 1, 2]
LAMBDAS    = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0]


class ModAdder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(P, EMBED_DIM)
        self.fc1 = nn.Linear(2 * EMBED_DIM, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.fc3 = nn.Linear(HIDDEN, P)

    def forward(self, a, b, return_hidden=False):
        x = torch.cat([self.embed(a), self.embed(b)], dim=-1)
        h1 = F.gelu(self.fc1(x))
        h2 = F.gelu(self.fc2(h1))
        if return_hidden:
            return self.fc3(h2), h2
        return self.fc3(h2)


def build_dataset(device: torch.device) -> tuple[TensorDataset, TensorDataset]:
    pairs = [(a, b, (a + b) % P) for a in range(P) for b in range(P)]
    rng = np.random.default_rng(42)
    rng.shuffle(pairs)
    n_train = int(len(pairs) * TRAIN_FRAC)
    def to_ds(ps):
        a = torch.tensor([x[0] for x in ps], dtype=torch.long, device=device)
        b = torch.tensor([x[1] for x in ps], dtype=torch.long, device=device)
        c = torch.tensor([x[2] for x in ps], dtype=torch.long, device=device)
        return TensorDataset(a, b, c)
    return to_ds(pairs[:n_train]), to_ds(pairs[n_train:])


@torch.no_grad()
def eval_acc(model: ModAdder, ds: TensorDataset, device: torch.device) -> float:
    a, b, c = ds.tensors
    return float((model(a, b).argmax(1) == c).float().mean())


@torch.no_grad()
def get_deff(model: ModAdder, ds: TensorDataset) -> tuple[float, float]:
    a, b, _ = ds.tensors
    _, h = model(a, b, return_hidden=True)
    H = h.detach().cpu().numpy().astype(np.float64)
    eigs = eigenspectrum(covariance(H))
    return effective_dimension(eigs), stable_rank(eigs)


def run_one(lam: float, seed: int, train_ds: TensorDataset, test_ds: TensorDataset,
            device: torch.device) -> list[dict]:
    torch.manual_seed(seed)
    model = ModAdder().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=lam)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    rows: list[dict] = []
    step = 0
    memorized_step = generalized_step = None

    while step < N_STEPS:
        for a, b, c in loader:
            opt.zero_grad()
            F.cross_entropy(model(a, b), c).backward()
            opt.step()
            step += 1

            if step % LOG_EVERY == 0:
                tr_acc = eval_acc(model, train_ds, device)
                te_acc = eval_acc(model, test_ds, device)
                deff, srank = (get_deff(model, train_ds) if step % DEFF_EVERY == 0
                               else (float("nan"), float("nan")))
                rows.append({
                    "lambda": lam, "seed": seed, "step": step,
                    "train_acc": round(tr_acc, 4), "test_acc": round(te_acc, 4),
                    "d_eff": round(deff, 3) if not np.isnan(deff) else None,
                    "stable_rank": round(srank, 3) if not np.isnan(srank) else None,
                })
                if memorized_step is None and tr_acc >= 0.99:
                    memorized_step = step
                    logger.info("lam=%.3f seed=%d memorized at step=%d deff=%.1f", lam, seed, step, deff)
                if generalized_step is None and te_acc >= 0.99:
                    generalized_step = step
                    logger.info("lam=%.3f seed=%d generalized at step=%d deff=%.1f", lam, seed, step, deff)

            if step >= N_STEPS:
                break

    logger.info("lam=%.3f seed=%d done: memorized=%s generalized=%s",
                lam, seed, memorized_step, generalized_step)
    return rows


def main() -> None:
    run_dir = setup_run_dir(V3 / "results", "grokking_lambda")
    setup_logging(run_dir)
    logger.info("grokking lambda sweep: lambdas=%s seeds=%s", LAMBDAS, SEEDS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)

    train_ds, test_ds = build_dataset(device)

    all_rows: list[dict] = []
    for lam in LAMBDAS:
        for seed in SEEDS:
            logger.info("starting lam=%.3f seed=%d", lam, seed)
            all_rows += run_one(lam, seed, train_ds, test_ds, device)

    csv_path = run_dir / "trajectories.csv"
    fieldnames = ["lambda", "seed", "step", "train_acc", "test_acc", "d_eff", "stable_rank"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    logger.info("wrote %s (%d rows)", csv_path, len(all_rows))

    # summarise per lambda (mean over seeds)
    summary: dict = {}
    for lam in LAMBDAS:
        sub = [r for r in all_rows if r["lambda"] == lam and r["d_eff"] is not None]
        by_seed: dict[int, list] = {}
        for r in sub:
            by_seed.setdefault(r["seed"], []).append(r)

        grokked = 0
        mem_steps, gen_steps, d_drops = [], [], []
        for seed_rows in by_seed.values():
            sorted_rows = sorted(seed_rows, key=lambda x: x["step"])
            accs = [(r["step"], r["train_acc"], r["test_acc"], r["d_eff"]) for r in sorted_rows]
            mem = next((s for s, ta, _, _ in accs if ta >= 0.99), None)
            gen = next((s for s, _, va, _ in accs if va >= 0.99), None)
            if gen is not None:
                grokked += 1
                gen_steps.append(gen)
                if mem is not None:
                    mem_steps.append(mem)
                deffs = [d for _, _, _, d in accs if d is not None]
                if deffs:
                    d_drops.append(deffs[0] - deffs[-1])

        summary[f"lambda_{lam}"] = {
            "grokked_frac": grokked / len(SEEDS),
            "mean_memorization_step": round(float(np.mean(mem_steps)), 0) if mem_steps else None,
            "mean_generalization_step": round(float(np.mean(gen_steps)), 0) if gen_steps else None,
            "mean_d_eff_drop": round(float(np.mean(d_drops)), 2) if d_drops else None,
        }

    dump_metadata(run_dir, summary)
    print("\n=== Grokking Lambda Sweep Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
