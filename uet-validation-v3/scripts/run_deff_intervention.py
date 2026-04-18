"""d_eff intervention experiment: force d_eff via bottleneck projection.

UET prediction: L(n) ~ c * k * log(d/k) / n + L_inf where k = forced bottleneck dim.
Kaplan/Chinchilla prediction: loss insensitive to k at fixed n (they have no d_eff term).

We fit both predictions on the (k, test_loss) sweep and compare R^2.
This is the cleanest test distinguishing UET from other scaling frameworks.
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
from scipy.optimize import curve_fit
from torch.utils.data import DataLoader, TensorDataset

V3 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3))
sys.path.insert(0, str(V3.parent / "uet-validation"))

from uet.run_utils import setup_run_dir, setup_logging, dump_metadata
from uet.eigendecomp import covariance, eigenspectrum, effective_dimension, stable_rank
from uet_v3.real_data import load_mnist

logger = logging.getLogger(__name__)

N_TRAIN    = 5000
N_TEST     = 2000
INPUT_DIM  = 784
D_AMBIENT  = 256  # d in UET formula (hidden dim before bottleneck)
K_VALUES   = [2, 4, 8, 16, 32, 64, 128, 256]
EPOCHS     = 200
LR         = 1e-3
WD         = 1e-4
BATCH_SIZE = 256
SEEDS      = [0, 1, 2, 3, 4]
N_CLASSES  = 10


class BottleneckMLP(nn.Module):
    """784 -> D_AMBIENT -> k (bottleneck, linear) -> D_AMBIENT -> 10."""
    def __init__(self, k: int):
        super().__init__()
        self.enc1    = nn.Linear(INPUT_DIM, D_AMBIENT)
        self.bottle  = nn.Linear(D_AMBIENT, k)
        self.up      = nn.Linear(k, D_AMBIENT)
        self.head    = nn.Linear(D_AMBIENT, N_CLASSES)

    def forward(self, x, return_hidden=False):
        h = F.gelu(self.enc1(x))
        b = self.bottle(h)       # forced bottleneck
        h2 = F.gelu(self.up(b))
        if return_hidden:
            return self.head(h2), b
        return self.head(h2)


@torch.no_grad()
def compute_deff(model: BottleneckMLP, X: torch.Tensor) -> tuple[float, float]:
    model.eval()
    _, h = model(X, return_hidden=True)
    H = h.detach().cpu().numpy().astype(np.float64)
    eigs = eigenspectrum(covariance(H))
    return effective_dimension(eigs), stable_rank(eigs)


@torch.no_grad()
def test_loss(model: BottleneckMLP, X_te: torch.Tensor, y_te: torch.Tensor) -> float:
    model.eval()
    return float(F.cross_entropy(model(X_te), y_te).item())


def train(model: BottleneckMLP, X_tr: torch.Tensor, y_tr: torch.Tensor) -> None:
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()


def uet_curve(k, c, l_inf):
    """UET: L = c * k * log(D/k) / N + L_inf"""
    k = np.asarray(k, dtype=float)
    return c * k * np.log(D_AMBIENT / np.maximum(k, 1e-8)) / N_TRAIN + l_inf


def kaplan_curve(k, a):
    """Kaplan has no k dependence -- flat line."""
    return np.full_like(np.asarray(k, dtype=float), a)


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main() -> None:
    run_dir = setup_run_dir(V3 / "results", "deff_intervention")
    setup_logging(run_dir)
    logger.info("d_eff intervention: k=%s n_train=%d seeds=%d", K_VALUES, N_TRAIN, len(SEEDS))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)

    train_ds, test_ds = load_mnist(flat=True)
    rng = np.random.default_rng(42)

    tr_idx = rng.choice(len(train_ds), N_TRAIN, replace=False)
    te_idx = rng.choice(len(test_ds),  N_TEST,  replace=False)
    X_tr = torch.stack([train_ds[i][0] for i in tr_idx]).to(device)
    y_tr = torch.tensor([train_ds[i][1].item() for i in tr_idx], device=device)
    X_te = torch.stack([test_ds[i][0] for i in te_idx]).to(device)
    y_te = torch.tensor([test_ds[i][1].item() for i in te_idx], device=device)

    fieldnames = ["k", "seed", "d_eff_empirical", "stable_rank", "test_loss", "test_acc"]
    csv_path = run_dir / "sweep.csv"
    rows = []

    for k in K_VALUES:
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = BottleneckMLP(k).to(device)
            train(model, X_tr, y_tr)

            deff, srank = compute_deff(model, X_tr)
            tl = test_loss(model, X_te, y_te)
            with torch.no_grad():
                te_acc = float((model(X_te).argmax(1) == y_te).float().mean())

            row = {
                "k": k, "seed": seed,
                "d_eff_empirical": round(deff, 3), "stable_rank": round(srank, 3),
                "test_loss": round(tl, 5), "test_acc": round(te_acc, 4),
            }
            rows.append(row)
            logger.info("k=%d seed=%d deff=%.2f test_loss=%.4f te_acc=%.3f",
                        k, seed, deff, tl, te_acc)

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    logger.info("wrote %s", csv_path)

    # fit UET and Kaplan to mean test_loss per k
    k_arr = np.array(K_VALUES, dtype=float)
    mean_loss = np.array([np.mean([r["test_loss"] for r in rows if r["k"] == k]) for k in K_VALUES])

    try:
        popt_uet, _ = curve_fit(uet_curve, k_arr, mean_loss, p0=[1e-4, 0.5], maxfev=5000)
        uet_pred = uet_curve(k_arr, *popt_uet)
        uet_r2   = r_squared(mean_loss, uet_pred)
    except Exception:
        popt_uet, uet_r2 = [float("nan"), float("nan")], float("nan")

    try:
        popt_kap, _ = curve_fit(kaplan_curve, k_arr, mean_loss, p0=[mean_loss.mean()], maxfev=1000)
        kaplan_pred = kaplan_curve(k_arr, *popt_kap)
        kaplan_r2   = r_squared(mean_loss, kaplan_pred)
    except Exception:
        popt_kap, kaplan_r2 = [float("nan")], float("nan")

    summary = {
        "n_train": N_TRAIN, "d_ambient": D_AMBIENT, "k_values": K_VALUES,
        "uet_c":    round(float(popt_uet[0]), 6),
        "uet_l_inf": round(float(popt_uet[1]), 5),
        "uet_r2":   round(uet_r2, 4),
        "kaplan_r2": round(kaplan_r2, 4),
        "uet_advantage_r2": round(uet_r2 - kaplan_r2, 4),
    }
    dump_metadata(run_dir, summary)
    print("\n=== d_eff Intervention Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
