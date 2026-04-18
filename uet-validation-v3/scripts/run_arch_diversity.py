"""Architecture diversity: does two-phase d_eff dynamic exist in MLP and CNN?

If discovery→formalisation is transformer-specific, UET universality claim fails.
Test on CIFAR-10: MLP (flattened) + tiny CNN. Track deff every N steps.
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
from torch.utils.data import DataLoader

V3 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3))
sys.path.insert(0, str(V3.parent / "uet-validation"))

from uet.run_utils import setup_run_dir, setup_logging, dump_metadata
from uet.eigendecomp import covariance, eigenspectrum, effective_dimension, stable_rank

logger = logging.getLogger(__name__)

N_CLASSES = 10
N_TRAIN = 5000
N_TEST = 1000
IMG_SIZE = 32
FLAT_DIM = 3 * IMG_SIZE * IMG_SIZE
TRAIN_EPOCHS = 60
LOG_EVERY = 5
LR = 1e-3
BATCH = 128
SEED = 0
DEFF_PROBE_SIZE = 512


class FlatMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(FLAT_DIM, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, N_CLASSES)

    def forward(self, x, return_hidden: bool = False):
        x = x.view(x.size(0), -1)
        h1 = F.gelu(self.fc1(x))
        h2 = F.gelu(self.fc2(h1))
        out = self.fc3(h2)
        if return_hidden:
            return out, h2
        return out


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, N_CLASSES)

    def forward(self, x, return_hidden: bool = False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        h = F.gelu(self.fc1(x))
        out = self.fc2(h)
        if return_hidden:
            return out, h
        return out


def load_cifar10(n_train: int, n_test: int, seed: int, device: torch.device):
    try:
        from torchvision.datasets import CIFAR10
        import torchvision.transforms as T
        transform = T.Compose([T.ToTensor(),
                               T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        ds_tr = CIFAR10("/tmp/cifar10", train=True, download=True, transform=transform)
        ds_te = CIFAR10("/tmp/cifar10", train=False, download=True, transform=transform)
        rng = np.random.default_rng(seed)
        tr_idx = rng.choice(len(ds_tr), n_train, replace=False)
        te_idx = rng.choice(len(ds_te), n_test, replace=False)
        dl_tr = DataLoader([ds_tr[i] for i in tr_idx], batch_size=len(tr_idx))
        dl_te = DataLoader([ds_te[i] for i in te_idx], batch_size=len(te_idx))
        X_tr, y_tr = next(iter(dl_tr))
        X_te, y_te = next(iter(dl_te))
    except Exception as e:
        logger.warning("CIFAR-10 failed (%s), synthetic fallback", e)
        rng = np.random.default_rng(seed)
        X_tr = torch.from_numpy(rng.random((n_train, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32))
        y_tr = torch.from_numpy(rng.integers(0, N_CLASSES, n_train))
        X_te = torch.from_numpy(rng.random((n_test, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32))
        y_te = torch.from_numpy(rng.integers(0, N_CLASSES, n_test))
    return X_tr.to(device), y_tr.to(device), X_te.to(device), y_te.to(device)


@torch.no_grad()
def compute_metrics(model, X_tr, y_tr, X_te, y_te) -> tuple[float, float, float, float]:
    model.eval()
    tr_acc = float((model(X_tr).argmax(1) == y_tr).float().mean())
    te_acc = float((model(X_te).argmax(1) == y_te).float().mean())
    probe = X_tr[:DEFF_PROBE_SIZE]
    _, h = model(probe, return_hidden=True)
    H = h.detach().cpu().numpy().astype(np.float64)
    cov = covariance(H)
    eigs = eigenspectrum(cov)
    return tr_acc, te_acc, effective_dimension(eigs), stable_rank(eigs)


def train_and_track(model, X_tr, y_tr, X_te, y_te, model_name: str, device: torch.device):
    loader = DataLoader(list(zip(X_tr.cpu(), y_tr.cpu())), batch_size=BATCH, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rows = []
    for epoch in range(TRAIN_EPOCHS + 1):
        if epoch > 0:
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                F.cross_entropy(model(xb), yb).backward()
                opt.step()
        if epoch % LOG_EVERY == 0:
            tr_acc, te_acc, deff, srank = compute_metrics(model, X_tr, y_tr, X_te, y_te)
            rows.append({
                "arch": model_name, "epoch": epoch,
                "train_acc": tr_acc, "test_acc": te_acc,
                "d_eff": deff, "stable_rank": srank,
            })
            logger.info("[%s] epoch=%d tr=%.3f te=%.3f deff=%.1f", model_name, epoch, tr_acc, te_acc, deff)
    return rows


def main() -> None:
    run_dir = setup_run_dir(V3 / "results", "arch_diversity")
    setup_logging(run_dir)
    logger.info("arch diversity: MLP + CNN on CIFAR-10 subset n=%d", N_TRAIN)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)

    X_tr, y_tr, X_te, y_te = load_cifar10(N_TRAIN, N_TEST, SEED, device)

    all_rows = []
    for name, model in [("MLP", FlatMLP()), ("CNN", TinyCNN())]:
        model = model.to(device)
        rows = train_and_track(model, X_tr, y_tr, X_te, y_te, name, device)
        all_rows.extend(rows)

    csv_path = run_dir / "arch_diversity.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arch", "epoch", "train_acc",
                                          "test_acc", "d_eff", "stable_rank"])
        w.writeheader()
        w.writerows(all_rows)
    logger.info("wrote %s", csv_path)

    summary = {}
    for arch in ("MLP", "CNN"):
        arch_rows = [r for r in all_rows if r["arch"] == arch]
        deffs = np.array([r["d_eff"] for r in arch_rows])
        epochs = np.array([r["epoch"] for r in arch_rows])
        peak_idx = int(np.argmax(deffs))
        two_phase = bool(deffs[peak_idx] > deffs[-1] * 1.1)
        summary[arch] = {
            "d_eff_initial": float(deffs[0]),
            "d_eff_peak": float(deffs[peak_idx]),
            "d_eff_peak_epoch": int(epochs[peak_idx]),
            "d_eff_final": float(deffs[-1]),
            "two_phase_observed": two_phase,
            "final_test_acc": arch_rows[-1]["test_acc"],
        }
    dump_metadata(run_dir, summary)
    print("\n=== Architecture Diversity Summary ===")
    for arch, s in summary.items():
        print(f"\n  {arch}:")
        for k, v in s.items():
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
