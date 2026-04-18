"""Double descent on real MNIST with label noise, multiple n_train and seeds.

UET prediction: d_eff peaks or is unstable at the DD interpolation threshold,
then decreases as the model becomes overparameterized.
Kaplan has no prediction on d_eff; this discriminates the frameworks.
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset

V3 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3))
sys.path.insert(0, str(V3.parent / "uet-validation"))

from uet.run_utils import setup_run_dir, setup_logging, dump_metadata
from uet.eigendecomp import covariance, eigenspectrum, effective_dimension, stable_rank
from uet_v3.real_data import load_mnist

logger = logging.getLogger(__name__)

WIDTHS     = [4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
N_TRAINS   = [1000, 4000]
NOISE_RATES = [0.0, 0.15, 0.30]
SEEDS      = [0, 1, 2]
N_TEST     = 2000
N_CLASSES  = 10
INPUT_DIM  = 784
EPOCHS     = 500
LR         = 1e-3
BATCH_SIZE = 256
DEFF_PROBE_N = 1000


class MLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, width)
        self.fc2 = nn.Linear(width, N_CLASSES)

    def forward(self, x, return_hidden=False):
        h = F.relu(self.fc1(x))
        if return_hidden:
            return self.fc2(h), h
        return self.fc2(h)


def n_params(width: int) -> int:
    return INPUT_DIM * width + width + width * N_CLASSES + N_CLASSES


def make_subset(dataset, n: int, noise_rate: float, seed: int, device: torch.device):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), n, replace=False)
    X = torch.stack([dataset[i][0] for i in idx]).to(device)
    y = torch.tensor([dataset[i][1].item() for i in idx], device=device)
    if noise_rate > 0:
        flip_mask = rng.random(n) < noise_rate
        y[flip_mask] = torch.from_numpy(rng.integers(0, N_CLASSES, flip_mask.sum())).to(device)
    return X, y


@torch.no_grad()
def compute_deff(model: MLP, X: torch.Tensor) -> tuple[float, float]:
    model.eval()
    _, h = model(X, return_hidden=True)
    H = h.detach().cpu().numpy().astype(np.float64)
    eigs = eigenspectrum(covariance(H))
    return effective_dimension(eigs), stable_rank(eigs)


def train(model: MLP, X_tr: torch.Tensor, y_tr: torch.Tensor) -> None:
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()


@torch.no_grad()
def accuracy(model: MLP, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    return float((model(X).argmax(1) == y).float().mean())


def main() -> None:
    run_dir = setup_run_dir(V3 / "results", "double_descent_v2")
    setup_logging(run_dir)
    logger.info("double descent v2 on real MNIST: %d width configs, %d n_train, %d noise rates, %d seeds",
                len(WIDTHS), len(N_TRAINS), len(NOISE_RATES), len(SEEDS))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)

    train_ds, test_ds = load_mnist(flat=True)
    rng0 = np.random.default_rng(999)
    test_idx = rng0.choice(len(test_ds), N_TEST, replace=False)
    X_te = torch.stack([test_ds[i][0] for i in test_idx]).to(device)
    y_te = torch.tensor([test_ds[i][1].item() for i in test_idx], device=device)

    fieldnames = ["n_train", "noise_rate", "seed", "width", "n_params",
                  "interp_ratio", "train_acc", "test_acc", "d_eff", "stable_rank"]
    csv_path = run_dir / "sweep.csv"
    rows = []

    for n_train in N_TRAINS:
        for noise_rate in NOISE_RATES:
            for seed in SEEDS:
                X_tr, y_tr = make_subset(train_ds, n_train, noise_rate, seed, device)
                probe_n = min(DEFF_PROBE_N, n_train)
                X_probe = X_tr[:probe_n]

                for width in WIDTHS:
                    np_count = n_params(width)
                    model = MLP(width).to(device)
                    torch.manual_seed(seed * 1000 + width)
                    train(model, X_tr, y_tr)
                    tr_acc = accuracy(model, X_tr, y_tr)
                    te_acc = accuracy(model, X_te, y_te)
                    deff, srank = compute_deff(model, X_probe)
                    row = {
                        "n_train": n_train, "noise_rate": noise_rate, "seed": seed,
                        "width": width, "n_params": np_count,
                        "interp_ratio": round(np_count / n_train, 4),
                        "train_acc": round(tr_acc, 4), "test_acc": round(te_acc, 4),
                        "d_eff": round(deff, 3), "stable_rank": round(srank, 3),
                    }
                    rows.append(row)
                    logger.info("n=%d noise=%.2f seed=%d width=%d np=%d tr=%.3f te=%.3f deff=%.1f",
                                n_train, noise_rate, seed, width, np_count, tr_acc, te_acc, deff)

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    logger.info("wrote %s (%d rows)", csv_path, len(rows))

    # summary stats: DD peak location and d_eff at peak per (n_train, noise_rate)
    summary: dict = {"runs": len(rows)}
    for n_train in N_TRAINS:
        for noise_rate in NOISE_RATES:
            key = f"n{n_train}_noise{int(noise_rate*100)}"
            sub = [r for r in rows if r["n_train"] == n_train and r["noise_rate"] == noise_rate]
            by_width: dict[int, list] = {}
            for r in sub:
                by_width.setdefault(r["width"], []).append(r)
            mean_te = {w: float(np.mean([r["test_acc"] for r in rs])) for w, rs in by_width.items()}
            mean_deff = {w: float(np.mean([r["d_eff"] for r in rs])) for w, rs in by_width.items()}
            peak_w = min(mean_te, key=mean_te.get)
            summary[key] = {
                "dd_peak_width": peak_w,
                "dd_peak_n_params": n_params(peak_w),
                "d_eff_at_peak": round(mean_deff[peak_w], 3),
                "test_acc_at_peak": round(mean_te[peak_w], 4),
                "test_acc_best": round(max(mean_te.values()), 4),
            }

    dump_metadata(run_dir, summary)
    print("\n=== Double Descent v2 Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
