"""Architecture diversity on real MNIST and CIFAR-10.

Tests whether two-phase d_eff dynamics (discovery -> formalisation) appear with
real task signal, across MLP, deep MLP, and CNN architectures.
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
from uet_v3.real_data import load_mnist, load_cifar10

logger = logging.getLogger(__name__)

EPOCHS     = 60
LOG_EVERY  = 2
LR         = 1e-3
WD         = 1e-4
BATCH_SIZE = 256
SEEDS      = [0, 1]
DEFF_N     = 2000  # samples to estimate d_eff from penultimate layer


class FlatMLP2(nn.Module):
    """2-hidden-layer MLP."""
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_classes)

    def forward(self, x, return_hidden=False):
        h1 = F.gelu(self.fc1(x))
        h2 = F.gelu(self.fc2(h1))
        if return_hidden:
            return self.fc3(h2), h2
        return self.fc3(h2)


class FlatMLP3(nn.Module):
    """3-hidden-layer MLP."""
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_classes)

    def forward(self, x, return_hidden=False):
        h = F.gelu(self.fc1(x))
        h = F.gelu(self.fc2(h))
        h3 = F.gelu(self.fc3(h))
        if return_hidden:
            return self.fc4(h3), h3
        return self.fc4(h3)


class TinyCNN(nn.Module):
    """2 conv blocks + global avg pool + FC."""
    def __init__(self, in_channels: int, n_classes: int, img_size: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.AdaptiveAvgPool2d(4)
        self.fc1   = nn.Linear(64 * 16, 128)
        self.fc2   = nn.Linear(128, n_classes)

    def forward(self, x, return_hidden=False):
        h = F.relu(self.conv1(x))
        h = F.relu(F.max_pool2d(self.conv2(h), 2))
        h = self.pool(h).flatten(1)
        h1 = F.relu(self.fc1(h))
        if return_hidden:
            return self.fc2(h1), h1
        return self.fc2(h1)


ARCHS = {
    "MLP2": lambda n_classes, in_dim, **kw: FlatMLP2(in_dim, n_classes),
    "MLP3": lambda n_classes, in_dim, **kw: FlatMLP3(in_dim, n_classes),
    "CNN":  lambda n_classes, in_channels, img_size, **kw: TinyCNN(in_channels, n_classes, img_size),
}

DATASETS = {
    "MNIST":    {"n_classes": 10, "in_dim": 784,  "in_channels": 1, "img_size": 28},
    "CIFAR-10": {"n_classes": 10, "in_dim": 3072, "in_channels": 3, "img_size": 32},
}


@torch.no_grad()
def compute_deff(model, X_probe: torch.Tensor) -> tuple[float, float]:
    model.eval()
    _, h = model(X_probe, return_hidden=True)
    H = h.detach().cpu().numpy().astype(np.float64)
    eigs = eigenspectrum(covariance(H))
    return effective_dimension(eigs), stable_rank(eigs)


@torch.no_grad()
def accuracy(model, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    return float((model(X).argmax(1) == y).float().mean())


def train_one_epoch(model, loader, opt):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        opt.zero_grad()
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)


def _build_probe(ds, n: int, img: bool, cfg: dict, device: torch.device):
    """Build a fixed probe set on GPU for d_eff computation."""
    n = min(n, len(ds))
    if img:
        X = torch.stack([ds[i][0].reshape(cfg["in_channels"], cfg["img_size"], cfg["img_size"])
                         for i in range(n)]).to(device)
    else:
        X = torch.stack([ds[i][0].flatten() for i in range(n)]).to(device)
    y = torch.tensor([ds[i][1].item() for i in range(n)], device=device)
    return X, y


def _build_loader(ds, img: bool, cfg: dict) -> DataLoader:
    """CPU DataLoader — batches moved to GPU inside training loop."""
    if img:
        X = torch.stack([ds[i][0].reshape(cfg["in_channels"], cfg["img_size"], cfg["img_size"])
                         for i in range(len(ds))])
    else:
        X = torch.stack([ds[i][0].flatten() for i in range(len(ds))])
    y = torch.tensor([ds[i][1].item() for i in range(len(ds))]).long()
    return DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)


def run_dataset(dataset_name: str, train_ds, test_ds, run_dir: Path, device: torch.device) -> list[dict]:
    cfg = DATASETS[dataset_name]

    # small probe sets stay on GPU; full train stays on CPU and is batched
    flat_probe, _ = _build_probe(test_ds, DEFF_N, img=False, cfg=cfg, device=device)
    img_probe,  _ = _build_probe(test_ds, DEFF_N, img=True,  cfg=cfg, device=device)
    flat_eval, y_eval = _build_probe(test_ds, len(test_ds), img=False, cfg=cfg, device=device)
    img_eval,  _      = _build_probe(test_ds, len(test_ds), img=True,  cfg=cfg, device=device)

    rows = []

    for arch_name, arch_fn in ARCHS.items():
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)

            is_cnn = arch_name == "CNN"
            model  = arch_fn(**cfg).to(device)
            loader = _build_loader(train_ds, img=is_cnn, cfg=cfg)
            X_probe = img_probe if is_cnn else flat_probe
            X_eval  = img_eval  if is_cnn else flat_eval
            opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

            logger.info("dataset=%s arch=%s seed=%d", dataset_name, arch_name, seed)
            for epoch in range(1, EPOCHS + 1):
                model.train()
                total_loss = 0.0
                for xb, yb in loader:
                    xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                    opt.zero_grad()
                    loss = F.cross_entropy(model(xb), yb)
                    loss.backward()
                    opt.step()
                    total_loss += loss.item() * len(xb)
                train_loss = total_loss / len(loader.dataset)

                if epoch % LOG_EVERY == 0:
                    deff, srank = compute_deff(model, X_probe)
                    te_acc = accuracy(model, X_eval, y_eval)
                    rows.append({
                        "dataset": dataset_name, "arch": arch_name, "seed": seed,
                        "epoch": epoch, "train_loss": round(train_loss, 5),
                        "test_acc": round(te_acc, 4),
                        "d_eff": round(deff, 3), "stable_rank": round(srank, 3),
                    })
                    logger.info("  epoch=%d te_acc=%.3f deff=%.2f", epoch, te_acc, deff)

    return rows


def main() -> None:
    run_dir = setup_run_dir(V3 / "results", "arch_diversity_v2")
    setup_logging(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("arch diversity v2, device=%s", device)

    mnist_tr, mnist_te = load_mnist(flat=True)
    cifar_tr, cifar_te = load_cifar10(flat=True)

    all_rows: list[dict] = []
    all_rows += run_dataset("MNIST",    mnist_tr, mnist_te, run_dir, device)
    all_rows += run_dataset("CIFAR-10", cifar_tr, cifar_te, run_dir, device)

    csv_path = run_dir / "trajectories.csv"
    fieldnames = ["dataset", "arch", "seed", "epoch", "train_loss", "test_acc", "d_eff", "stable_rank"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    logger.info("wrote %s (%d rows)", csv_path, len(all_rows))

    summary: dict = {}
    for dataset_name in DATASETS:
        for arch_name in ARCHS:
            sub = [r for r in all_rows if r["dataset"] == dataset_name and r["arch"] == arch_name]
            if not sub:
                continue
            seeds_data = {}
            for r in sub:
                seeds_data.setdefault(r["seed"], []).append(r)
            final_deffs = [sorted(rs, key=lambda x: x["epoch"])[-1]["d_eff"] for rs in seeds_data.values()]
            init_deffs  = [sorted(rs, key=lambda x: x["epoch"])[0]["d_eff"]  for rs in seeds_data.values()]
            key = f"{dataset_name}_{arch_name}"
            summary[key] = {
                "d_eff_initial_mean": round(float(np.mean(init_deffs)), 2),
                "d_eff_final_mean":   round(float(np.mean(final_deffs)), 2),
                "compression_ratio":  round(float(np.mean(init_deffs)) / max(float(np.mean(final_deffs)), 0.1), 2),
                "two_phase_observed": float(np.mean(init_deffs)) > float(np.mean(final_deffs)) * 1.5,
            }

    dump_metadata(run_dir, summary)
    print("\n=== Arch Diversity v2 Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
