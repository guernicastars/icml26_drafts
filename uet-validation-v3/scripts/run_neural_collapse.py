"""Neural Collapse on CIFAR-10 and CIFAR-100.

UET prediction: at end of training, d_eff(penultimate) -> N_classes - 1.
This is the formalisation endpoint: representations collapse to the ETF simplex dimension.

NC1: within-class variability collapses (tr(S_W) / tr(S_B) -> 0).
NC2: class means converge to ETF (mean |cos(mu_i, mu_j)| -> -1/(N_classes - 1)).
d_eff tracks NC1/NC2 saturation.
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
from uet_v3.real_data import load_cifar10, load_cifar100

logger = logging.getLogger(__name__)

EPOCHS     = 80
LOG_EVERY  = 5
LR         = 0.1
MOMENTUM   = 0.9
WD         = 5e-4
BATCH_SIZE = 128
SEEDS      = [0, 1]

DATASETS = {
    "CIFAR-10":  {"n_classes": 10,  "loader": load_cifar10},
    "CIFAR-100": {"n_classes": 100, "loader": load_cifar100},
}


class SmallResNet(nn.Module):
    """3 conv blocks (32, 64, 128 ch) + GAP + FC(512) + head."""
    def __init__(self, n_classes: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),
        )
        self.fc1   = nn.Linear(128 * 4, 512)
        self.fc_out = nn.Linear(512, n_classes)

    def forward(self, x, return_hidden=False):
        h = self.block3(self.block2(self.block1(x))).flatten(1)
        h1 = F.relu(self.fc1(h))
        if return_hidden:
            return self.fc_out(h1), h1
        return self.fc_out(h1)


@torch.no_grad()
def compute_nc_metrics(model: SmallResNet, X: torch.Tensor, y: torch.Tensor,
                        n_classes: int) -> dict:
    model.eval()
    _, h = model(X, return_hidden=True)
    H = h.detach().cpu().numpy().astype(np.float64)
    labels = y.cpu().numpy()

    # d_eff of full penultimate layer
    eigs = eigenspectrum(covariance(H))
    deff  = effective_dimension(eigs)
    srank = stable_rank(eigs)

    # NC1: within-class scatter / between-class scatter (trace ratio)
    global_mean = H.mean(0)
    S_W_trace = 0.0
    class_means = []
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() == 0:
            class_means.append(global_mean)
            continue
        Hc = H[mask]
        mu_c = Hc.mean(0)
        class_means.append(mu_c)
        S_W_trace += np.sum((Hc - mu_c) ** 2) / H.shape[0]
    class_means = np.stack(class_means)  # (n_classes, d)
    centered = class_means - global_mean
    S_B_trace = float(np.sum(centered ** 2)) / n_classes
    nc1 = S_W_trace / max(S_B_trace, 1e-12)

    # NC2: mean off-diagonal cosine similarity
    norms = np.linalg.norm(centered, axis=1, keepdims=True) + 1e-12
    normed = centered / norms
    cos_mat = normed @ normed.T
    upper = cos_mat[np.triu_indices(n_classes, k=1)]
    nc2 = float(np.abs(upper).mean()) if len(upper) > 0 else float("nan")
    nc2_target = 1.0 / (n_classes - 1)  # ETF prediction

    return {"d_eff": round(deff, 3), "stable_rank": round(srank, 3),
            "nc1": round(float(nc1), 5), "nc2_mean_cos": round(nc2, 5),
            "nc2_etf_target": round(nc2_target, 5)}


def run_dataset(dataset_name: str, n_classes: int, loader_fn, run_dir: Path,
                device: torch.device) -> list[dict]:
    train_ds, test_ds = loader_fn(flat=False)

    # probe and eval stay on GPU (small); full train batched from CPU to avoid OOM
    probe_n = 4000
    probe_idx = np.random.default_rng(0).choice(len(train_ds), probe_n, replace=False)
    X_probe = torch.stack([train_ds[i][0] for i in probe_idx]).to(device)
    y_probe = torch.tensor([train_ds[i][1].item() for i in probe_idx], device=device)

    X_test = torch.stack([test_ds[i][0] for i in range(min(4000, len(test_ds)))]).to(device)
    y_test = torch.tensor([test_ds[i][1].item() for i in range(min(4000, len(test_ds)))], device=device)

    # full train stays on CPU
    X_tr_cpu = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    y_tr_cpu = torch.tensor([train_ds[i][1].item() for i in range(len(train_ds))]).long()
    loader = DataLoader(TensorDataset(X_tr_cpu, y_tr_cpu), batch_size=BATCH_SIZE,
                        shuffle=True, pin_memory=True)
    rows = []

    for seed in SEEDS:
        torch.manual_seed(seed)
        model = SmallResNet(n_classes).to(device)
        opt   = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

        logger.info("dataset=%s seed=%d starting", dataset_name, seed)
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                opt.zero_grad()
                F.cross_entropy(model(xb), yb).backward()
                opt.step()
            sched.step()

            if epoch % LOG_EVERY == 0:
                nc = compute_nc_metrics(model, X_probe, y_probe, n_classes)
                with torch.no_grad():
                    model.eval()
                    te_acc = float((model(X_test).argmax(1) == y_test).float().mean())
                row = {"dataset": dataset_name, "seed": seed, "epoch": epoch,
                       "test_acc": round(te_acc, 4), **nc}
                rows.append(row)
                logger.info("  epoch=%d te_acc=%.3f deff=%.2f nc1=%.4f nc2=%.4f",
                            epoch, te_acc, nc["d_eff"], nc["nc1"], nc["nc2_mean_cos"])

    return rows


def main() -> None:
    run_dir = setup_run_dir(V3 / "results", "neural_collapse")
    setup_logging(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("neural collapse on CIFAR-10/100, device=%s", device)

    all_rows: list[dict] = []
    for dataset_name, cfg in DATASETS.items():
        all_rows += run_dataset(dataset_name, cfg["n_classes"], cfg["loader"], run_dir, device)

    csv_path = run_dir / "trajectory.csv"
    fieldnames = ["dataset", "seed", "epoch", "test_acc", "d_eff", "stable_rank",
                  "nc1", "nc2_mean_cos", "nc2_etf_target"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    logger.info("wrote %s (%d rows)", csv_path, len(all_rows))

    summary: dict = {}
    for dataset_name, cfg in DATASETS.items():
        n_classes = cfg["n_classes"]
        etf_dim = n_classes - 1
        sub = [r for r in all_rows if r["dataset"] == dataset_name]
        if not sub:
            continue
        final_rows = [max([r for r in sub if r["seed"] == s], key=lambda x: x["epoch"])
                      for s in SEEDS]
        final_deffs = [r["d_eff"] for r in final_rows]
        final_nc1s  = [r["nc1"]  for r in final_rows]
        summary[dataset_name] = {
            "etf_dim_prediction": etf_dim,
            "d_eff_final_mean": round(float(np.mean(final_deffs)), 2),
            "d_eff_final_std":  round(float(np.std(final_deffs)),  2),
            "d_eff_close_to_etf": abs(float(np.mean(final_deffs)) - etf_dim) < etf_dim * 0.2,
            "nc1_final_mean":   round(float(np.mean(final_nc1s)), 5),
        }

    dump_metadata(run_dir, summary)
    print("\n=== Neural Collapse Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
