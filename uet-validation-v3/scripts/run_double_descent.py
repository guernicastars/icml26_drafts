"""Double descent + deff: vary MLP width on MNIST subset with label noise.

Tests if deff instability coincides with interpolation threshold (double descent peak).
UET predicts: deff collapses to stable value in overparameterized regime.
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

N_TRAIN = 1000
N_TEST = 2000
NOISE_RATE = 0.15
WIDTHS = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
N_CLASSES = 10
INPUT_DIM = 28 * 28
TRAIN_EPOCHS = 200
LR = 1e-3
SEED = 42
BATCH_SIZE = 256


class MLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, width),
            nn.ReLU(),
            nn.Linear(width, N_CLASSES),
        )

    def forward(self, x, return_hidden: bool = False):
        h = F.relu(self.net[0](x))
        out = self.net[2](h)
        if return_hidden:
            return out, h
        return out


def load_mnist_subset(n_train: int, n_test: int, noise_rate: float, seed: int, device: torch.device):
    try:
        from torchvision.datasets import MNIST
        import torchvision.transforms as T
        ds_tr = MNIST("/tmp/mnist", train=True, download=True,
                      transform=T.Compose([T.ToTensor(), T.Lambda(lambda x: x.view(-1))]))
        ds_te = MNIST("/tmp/mnist", train=False, download=True,
                      transform=T.Compose([T.ToTensor(), T.Lambda(lambda x: x.view(-1))]))
        rng = np.random.default_rng(seed)
        tr_idx = rng.choice(len(ds_tr), n_train, replace=False)
        te_idx = rng.choice(len(ds_te), n_test, replace=False)
        X_tr = torch.stack([ds_tr[i][0] for i in tr_idx]).to(device)
        y_tr = torch.tensor([ds_tr[i][1] for i in tr_idx], dtype=torch.long, device=device)
        X_te = torch.stack([ds_te[i][0] for i in te_idx]).to(device)
        y_te = torch.tensor([ds_te[i][1] for i in te_idx], dtype=torch.long, device=device)
    except Exception as e:
        logger.warning("MNIST failed (%s), using random data", e)
        rng = np.random.default_rng(seed)
        X_tr = torch.from_numpy(rng.random((n_train, INPUT_DIM), dtype=np.float32)).to(device)
        y_tr = torch.from_numpy(rng.integers(0, N_CLASSES, n_train)).to(device)
        X_te = torch.from_numpy(rng.random((n_test, INPUT_DIM), dtype=np.float32)).to(device)
        y_te = torch.from_numpy(rng.integers(0, N_CLASSES, n_test)).to(device)

    # inject label noise
    rng2 = np.random.default_rng(seed + 1)
    noise_mask = rng2.random(len(y_tr)) < noise_rate
    y_noisy = y_tr.clone()
    y_noisy[noise_mask] = torch.from_numpy(
        rng2.integers(0, N_CLASSES, int(noise_mask.sum()))
    ).to(device)
    return X_tr, y_noisy, X_te, y_te


def train_model(model, X_tr, y_tr, device):
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for _ in range(TRAIN_EPOCHS):
        for xb, yb in loader:
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()


@torch.no_grad()
def evaluate(model, X_tr, y_tr, X_te, y_te) -> tuple[float, float]:
    model.eval()
    tr_acc = float((model(X_tr).argmax(1) == y_tr).float().mean())
    te_acc = float((model(X_te).argmax(1) == y_te).float().mean())
    return tr_acc, te_acc


@torch.no_grad()
def get_deff(model, X: torch.Tensor) -> tuple[float, float]:
    model.eval()
    _, h = model(X, return_hidden=True)
    H = h.detach().cpu().numpy().astype(np.float64)
    cov = covariance(H)
    eigs = eigenspectrum(cov)
    return effective_dimension(eigs), stable_rank(eigs)


def n_params(width: int) -> int:
    return INPUT_DIM * width + width + width * N_CLASSES + N_CLASSES


def main() -> None:
    run_dir = setup_run_dir(V3 / "results", "double_descent")
    setup_logging(run_dir)
    logger.info("double descent + deff, n_train=%d noise=%.2f", N_TRAIN, NOISE_RATE)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)

    X_tr, y_tr, X_te, y_te = load_mnist_subset(N_TRAIN, N_TEST, NOISE_RATE, SEED, device)
    logger.info("interpolation threshold: ~%d params (= n_train=%d)", N_TRAIN, N_TRAIN)

    rows = []
    for width in WIDTHS:
        np_count = n_params(width)
        model = MLP(width).to(device)
        train_model(model, X_tr, y_tr, device)
        tr_acc, te_acc = evaluate(model, X_tr, y_tr, X_te, y_te)
        deff, srank = get_deff(model, X_tr)
        row = {
            "width": width, "n_params": np_count,
            "interp_ratio": np_count / N_TRAIN,
            "train_acc": tr_acc, "test_acc": te_acc,
            "d_eff": deff, "stable_rank": srank,
        }
        rows.append(row)
        logger.info("width=%d n_params=%d tr=%.3f te=%.3f deff=%.1f ratio=%.2f",
                    width, np_count, tr_acc, te_acc, deff, np_count / N_TRAIN)

    csv_path = run_dir / "double_descent.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["width", "n_params", "interp_ratio",
                                          "train_acc", "test_acc", "d_eff", "stable_rank"])
        w.writeheader()
        w.writerows(rows)
    logger.info("wrote %s", csv_path)

    te_accs = np.array([r["test_acc"] for r in rows])
    deff_vals = np.array([r["d_eff"] for r in rows])
    widths = np.array([r["width"] for r in rows])
    peak_idx = int(np.argmin(te_accs))
    deff_at_peak = float(deff_vals[peak_idx])
    deff_over_threshold = float(deff_vals[widths > 32].mean()) if (widths > 32).any() else float("nan")
    deff_under_threshold = float(deff_vals[widths <= 32].mean()) if (widths <= 32).any() else float("nan")
    summary = {
        "interpolation_threshold_params": N_TRAIN,
        "double_descent_peak_width": int(widths[peak_idx]),
        "double_descent_peak_n_params": rows[peak_idx]["n_params"],
        "d_eff_at_peak": deff_at_peak,
        "d_eff_mean_overparameterized": deff_over_threshold,
        "d_eff_mean_underparameterized": deff_under_threshold,
        "test_acc_min": float(te_accs.min()),
        "test_acc_max": float(te_accs.max()),
    }
    dump_metadata(run_dir, summary)
    print(f"\n=== Double Descent Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
