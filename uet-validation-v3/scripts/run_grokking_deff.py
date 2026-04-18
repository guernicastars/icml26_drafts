"""Grokking + deff: test if deff collapse precedes generalization.

Task: modular addition (a + b) mod p, p=97.
Model: 2-layer MLP with embedding, weight decay (standard grokking recipe).
Track: train_acc, test_acc, deff(penultimate activations) over steps.
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

V3 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3))
sys.path.insert(0, str(V3.parent / "uet-validation"))

from uet.run_utils import setup_run_dir, setup_logging, dump_metadata
from uet.eigendecomp import covariance, eigenspectrum, effective_dimension, stable_rank

logger = logging.getLogger(__name__)

P = 97
HIDDEN = 128
EMBED = 128
TRAIN_FRAC = 0.4
STEPS = 60_000
LR = 1e-3
WD = 1.0
BATCH_SIZE = 512
LOG_EVERY = 200
DEFF_EVERY = 500
SEED = 0


class ModAdder(nn.Module):
    def __init__(self, p: int, embed_dim: int, hidden: int):
        super().__init__()
        self.emb = nn.Embedding(p, embed_dim)
        self.fc1 = nn.Linear(2 * embed_dim, hidden)
        self.fc2 = nn.Linear(hidden, p)

    def forward(self, a: torch.Tensor, b: torch.Tensor, return_hidden: bool = False):
        ea = self.emb(a)
        eb = self.emb(b)
        x = torch.cat([ea, eb], dim=-1)
        h = F.gelu(self.fc1(x))
        logits = self.fc2(h)
        if return_hidden:
            return logits, h
        return logits


def make_data(p: int, device: torch.device) -> tuple:
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    a = a.flatten()
    b = b.flatten()
    y = (a + b) % p
    return a.to(device), b.to(device), y.to(device)


@torch.no_grad()
def accuracy(model, a, b, y):
    model.eval()
    logits = model(a, b)
    pred = logits.argmax(dim=-1)
    return float((pred == y).float().mean())


@torch.no_grad()
def compute_deff(model, a, b) -> tuple[float, float]:
    model.eval()
    _, h = model(a, b, return_hidden=True)
    H = h.detach().cpu().numpy().astype(np.float64)
    cov = covariance(H)
    eigs = eigenspectrum(cov)
    return effective_dimension(eigs), stable_rank(eigs)


def main() -> None:
    run_dir = setup_run_dir(V3 / "results", "grokking_deff")
    setup_logging(run_dir)
    logger.info("grokking + deff, p=%d, hidden=%d, wd=%.2f", P, HIDDEN, WD)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)

    a_all, b_all, y_all = make_data(P, device)
    n = len(a_all)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))
    n_train = int(n * TRAIN_FRAC)
    tr_idx, te_idx = perm[:n_train].to(device), perm[n_train:].to(device)
    a_tr, b_tr, y_tr = a_all[tr_idx], b_all[tr_idx], y_all[tr_idx]
    a_te, b_te, y_te = a_all[te_idx], b_all[te_idx], y_all[te_idx]
    logger.info("train=%d test=%d", len(tr_idx), len(te_idx))

    model = ModAdder(P, EMBED, HIDDEN).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.98))

    rows = []
    for step in range(STEPS + 1):
        model.train()
        batch_idx = torch.randint(0, len(tr_idx), (BATCH_SIZE,), device=device)
        a_b, b_b, y_b = a_tr[batch_idx], b_tr[batch_idx], y_tr[batch_idx]
        logits = model(a_b, b_b)
        loss = F.cross_entropy(logits, y_b)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % LOG_EVERY == 0 or step == STEPS:
            tr_acc = accuracy(model, a_tr, b_tr, y_tr)
            te_acc = accuracy(model, a_te, b_te, y_te)
            if step % DEFF_EVERY == 0 or step == STEPS:
                deff, srank = compute_deff(model, a_all, b_all)
            row = {
                "step": step, "train_loss": float(loss.item()),
                "train_acc": tr_acc, "test_acc": te_acc,
                "d_eff": deff, "stable_rank": srank,
                "hidden_dim": HIDDEN,
            }
            rows.append(row)
            if step % DEFF_EVERY == 0:
                logger.info("step=%d tr_acc=%.3f te_acc=%.3f deff=%.1f",
                            step, tr_acc, te_acc, deff)

    csv_path = run_dir / "trajectory.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "train_loss", "train_acc",
                                          "test_acc", "d_eff", "stable_rank", "hidden_dim"])
        w.writeheader()
        w.writerows(rows)
    logger.info("wrote %s", csv_path)

    arr = np.array([(r["step"], r["train_acc"], r["test_acc"], r["d_eff"]) for r in rows])
    steps, tr_acc, te_acc, deff = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    mem_step = int(steps[np.argmax(tr_acc >= 0.99)]) if (tr_acc >= 0.99).any() else -1
    gen_step = int(steps[np.argmax(te_acc >= 0.99)]) if (te_acc >= 0.99).any() else -1
    if mem_step > 0 and gen_step > 0:
        deff_at_mem = float(deff[np.argmin(np.abs(steps - mem_step))])
        deff_at_gen = float(deff[np.argmin(np.abs(steps - gen_step))])
    else:
        deff_at_mem = deff_at_gen = float("nan")

    deff_peak_idx = int(np.argmax(deff))
    deff_final = float(deff[-1])
    grokking_gap = gen_step - mem_step if (mem_step > 0 and gen_step > 0) else -1

    summary = {
        "p": P, "hidden": HIDDEN, "wd": WD, "train_frac": TRAIN_FRAC,
        "memorization_step": mem_step, "generalization_step": gen_step,
        "grokking_gap_steps": grokking_gap,
        "d_eff_at_memorization": deff_at_mem,
        "d_eff_at_generalization": deff_at_gen,
        "d_eff_peak_step": int(steps[deff_peak_idx]),
        "d_eff_peak_value": float(deff[deff_peak_idx]),
        "d_eff_final": deff_final,
        "d_eff_drop_mem_to_gen": deff_at_mem - deff_at_gen,
    }
    dump_metadata(run_dir, summary)
    logger.info("summary=%s", summary)
    print(f"\n=== Grokking summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
