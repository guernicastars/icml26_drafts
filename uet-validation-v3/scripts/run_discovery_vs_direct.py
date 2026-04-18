"""Discovery vs Direct low-d training.

Three regimes on MNIST classification:
  (1) direct_low: MLP with latent_dim=k_target throughout training
  (2) high_compress: wide MLP (d=256), then re-init bottleneck layer to k_target and fine-tune
  (3) high_only: MLP with d=256 throughout (no compression)

If (2) >> (1): discovery-formalisation pattern is real.
If (1) ≈ (2): UET discovery claim weakened.
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
from uet.eigendecomp import covariance, eigenspectrum, effective_dimension

logger = logging.getLogger(__name__)

N_TRAIN = 2000
N_TEST = 2000
N_CLASSES = 10
INPUT_DIM = 28 * 28
D_HIGH = 256
K_TARGETS = [4, 8, 16, 32]
EPOCHS_PHASE1 = 60
EPOCHS_PHASE2 = 60
LR = 1e-3
BATCH = 256
SEED = 0


class BottleneckMLP(nn.Module):
    def __init__(self, bottleneck: int, d_high: int = D_HIGH):
        super().__init__()
        self.encoder = nn.Linear(INPUT_DIM, d_high)
        self.bottleneck = nn.Linear(d_high, bottleneck)
        self.decoder = nn.Linear(bottleneck, N_CLASSES)

    def forward(self, x, return_hidden: bool = False):
        h_enc = F.gelu(self.encoder(x))
        h_bot = F.gelu(self.bottleneck(h_enc))
        out = self.decoder(h_bot)
        if return_hidden:
            return out, h_bot
        return out


class DirectMLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, width)
        self.fc2 = nn.Linear(width, N_CLASSES)

    def forward(self, x, return_hidden: bool = False):
        h = F.gelu(self.fc1(x))
        out = self.fc2(h)
        if return_hidden:
            return out, h
        return out


def load_mnist(n_tr: int, n_te: int, seed: int, device):
    try:
        from torchvision.datasets import MNIST
        import torchvision.transforms as T
        ds_tr = MNIST("/tmp/mnist", train=True, download=True,
                      transform=T.Compose([T.ToTensor(), T.Lambda(lambda x: x.view(-1))]))
        ds_te = MNIST("/tmp/mnist", train=False, download=True,
                      transform=T.Compose([T.ToTensor(), T.Lambda(lambda x: x.view(-1))]))
        rng = np.random.default_rng(seed)
        X_tr = torch.stack([ds_tr[i][0] for i in rng.choice(len(ds_tr), n_tr, replace=False)]).to(device)
        y_tr = torch.tensor([ds_tr[i][1] for i in rng.choice(len(ds_tr), n_tr, replace=False)],
                            dtype=torch.long, device=device)
        X_te = torch.stack([ds_te[i][0] for i in rng.choice(len(ds_te), n_te, replace=False)]).to(device)
        y_te = torch.tensor([ds_te[i][1] for i in rng.choice(len(ds_te), n_te, replace=False)],
                            dtype=torch.long, device=device)
    except Exception as e:
        logger.warning("MNIST failed: %s, using random", e)
        rng = np.random.default_rng(seed)
        X_tr = torch.from_numpy(rng.random((n_tr, INPUT_DIM), dtype=np.float32)).to(device)
        y_tr = torch.from_numpy(rng.integers(0, N_CLASSES, n_tr)).to(device)
        X_te = torch.from_numpy(rng.random((n_te, INPUT_DIM), dtype=np.float32)).to(device)
        y_te = torch.from_numpy(rng.integers(0, N_CLASSES, n_te)).to(device)
    return X_tr, y_tr, X_te, y_te


def run_epochs(model, X_tr, y_tr, n_epochs, device):
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(n_epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()


@torch.no_grad()
def test_acc_and_deff(model, X_te, y_te) -> tuple[float, float]:
    model.eval()
    logits, h = model(X_te, return_hidden=True)
    acc = float((logits.argmax(1) == y_te).float().mean())
    H = h.detach().cpu().numpy().astype(np.float64)
    cov = covariance(H)
    eigs = eigenspectrum(cov)
    return acc, effective_dimension(eigs)


def main() -> None:
    run_dir = setup_run_dir(V3 / "results", "discovery_vs_direct")
    setup_logging(run_dir)
    logger.info("discovery vs direct, d_high=%d, k_targets=%s", D_HIGH, K_TARGETS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)

    X_tr, y_tr, X_te, y_te = load_mnist(N_TRAIN, N_TEST, SEED, device)

    rows = []
    for k in K_TARGETS:
        torch.manual_seed(SEED)

        # Regime 1: direct low-d
        direct = DirectMLP(k).to(device)
        run_epochs(direct, X_tr, y_tr, EPOCHS_PHASE1 + EPOCHS_PHASE2, device)
        acc_direct, deff_direct = test_acc_and_deff(direct, X_te, y_te)
        logger.info("k=%d DIRECT: acc=%.3f deff=%.1f", k, acc_direct, deff_direct)

        # Regime 2: high-d then compress (bottleneck)
        bottleneck = BottleneckMLP(k, D_HIGH).to(device)
        run_epochs(bottleneck, X_tr, y_tr, EPOCHS_PHASE1, device)
        run_epochs(bottleneck, X_tr, y_tr, EPOCHS_PHASE2, device)
        acc_bottle, deff_bottle = test_acc_and_deff(bottleneck, X_te, y_te)
        logger.info("k=%d BOTTLE: acc=%.3f deff=%.1f", k, acc_bottle, deff_bottle)

        # Regime 3: high-d only
        high = DirectMLP(D_HIGH).to(device)
        run_epochs(high, X_tr, y_tr, EPOCHS_PHASE1 + EPOCHS_PHASE2, device)
        acc_high, deff_high = test_acc_and_deff(high, X_te, y_te)
        logger.info("k=%d HIGH:   acc=%.3f deff=%.1f", k, acc_high, deff_high)

        rows.append({
            "k_target": k,
            "direct_test_acc": acc_direct, "direct_deff": deff_direct,
            "bottleneck_test_acc": acc_bottle, "bottleneck_deff": deff_bottle,
            "high_test_acc": acc_high, "high_deff": deff_high,
            "bottle_vs_direct_acc": acc_bottle - acc_direct,
        })

    csv_path = run_dir / "discovery_vs_direct.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = ["k_target", "direct_test_acc", "direct_deff",
                      "bottleneck_test_acc", "bottleneck_deff",
                      "high_test_acc", "high_deff", "bottle_vs_direct_acc"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    logger.info("wrote %s", csv_path)

    discovery_supported = all(r["bottle_vs_direct_acc"] > 0.02 for r in rows)
    summary = {
        "discovery_formalisation_supported": discovery_supported,
        "bottle_vs_direct_acc_mean": float(np.mean([r["bottle_vs_direct_acc"] for r in rows])),
        "k_targets": K_TARGETS,
    }
    dump_metadata(run_dir, summary)
    print("\n=== Discovery vs Direct ===")
    for r in rows:
        print(f"  k={r['k_target']}: direct={r['direct_test_acc']:.3f}  bottle={r['bottleneck_test_acc']:.3f}  "
              f"high={r['high_test_acc']:.3f}  Δ={r['bottle_vs_direct_acc']:+.3f}")
    print(f"\n  discovery-formalisation supported: {discovery_supported}")


if __name__ == "__main__":
    main()
