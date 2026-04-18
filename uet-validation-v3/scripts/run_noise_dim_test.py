"""Noise dimension test: does adding random input dims hurt? UET unique prediction.

Setup: structured data X = U @ Z + eps, d=64, k_true=10.
Pad input with m random noise dims, m in {0, 8, 16, 32, 64, 128, 256}.
Train AE, measure reconstruction loss + deff of latent code.
UET predicts: noise dims dilute signal → should not hurt (possibly help via
spectral separation), deff remains near k_true.
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
sys.path.insert(0, str(V3.parent / "uet-validation-v2"))

from uet.run_utils import setup_run_dir, setup_logging, dump_metadata
from uet.eigendecomp import covariance, eigenspectrum, effective_dimension, stable_rank
from uet_v2.synthetic_domain import generate_structured_data

logger = logging.getLogger(__name__)

D_SIGNAL = 64
K_TRUE = 10
LATENT_DIM = 16
NOISE_PADS = [0, 8, 16, 32, 64, 128, 256, 512]
N_SAMPLES = 8000
SNR = 2.0
N_EPOCHS = 200
LR = 1e-3
BATCH = 256
N_SEEDS = 3
HIDDEN_AE = 128


class AE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def train_ae(model, X_tensor, device):
    dl = DataLoader(TensorDataset(X_tensor), batch_size=BATCH, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(N_EPOCHS):
        for (xb,) in dl:
            xb = xb.to(device)
            recon, _ = model(xb)
            loss = F.mse_loss(recon, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()


@torch.no_grad()
def eval_ae(model, X_tensor, device) -> tuple[float, float, float]:
    model.eval()
    X = X_tensor.to(device)
    recon, z = model(X)
    recon_loss = float(F.mse_loss(recon, X))
    Z = z.detach().cpu().numpy().astype(np.float64)
    cov = covariance(Z)
    eigs = eigenspectrum(cov)
    return recon_loss, effective_dimension(eigs), stable_rank(eigs)


def main() -> None:
    run_dir = setup_run_dir(V3 / "results", "noise_dim_test")
    setup_logging(run_dir)
    logger.info("noise dim test: d_signal=%d k_true=%d latent=%d", D_SIGNAL, K_TRUE, LATENT_DIM)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)

    rows = []
    for m in NOISE_PADS:
        recon_list, deff_list, srank_list = [], [], []
        for seed in range(N_SEEDS):
            X_signal, _ = generate_structured_data(N_SAMPLES, D_SIGNAL, K_TRUE, SNR, seed)
            rng = np.random.default_rng(seed + 1000)
            X_noise = rng.standard_normal((N_SAMPLES, m)).astype(np.float32)
            X = np.concatenate([X_signal, X_noise], axis=1) if m > 0 else X_signal
            X_t = torch.from_numpy(X).float()
            input_dim = X.shape[1]
            model = AE(input_dim, LATENT_DIM, HIDDEN_AE).to(device)
            train_ae(model, X_t, device)
            recon, deff, srank = eval_ae(model, X_t, device)
            recon_list.append(recon)
            deff_list.append(deff)
            srank_list.append(srank)
        row = {
            "n_noise_dims": m, "total_input_dim": D_SIGNAL + m,
            "signal_fraction": D_SIGNAL / (D_SIGNAL + m),
            "recon_loss_mean": float(np.mean(recon_list)),
            "recon_loss_std": float(np.std(recon_list)),
            "d_eff_mean": float(np.mean(deff_list)),
            "d_eff_std": float(np.std(deff_list)),
            "stable_rank_mean": float(np.mean(srank_list)),
        }
        rows.append(row)
        logger.info("m=%d total_d=%d recon=%.4f deff=%.2f",
                    m, D_SIGNAL + m, row["recon_loss_mean"], row["d_eff_mean"])

    csv_path = run_dir / "noise_dim.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = ["n_noise_dims", "total_input_dim", "signal_fraction",
                      "recon_loss_mean", "recon_loss_std", "d_eff_mean", "d_eff_std", "stable_rank_mean"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    logger.info("wrote %s", csv_path)

    recon_baseline = rows[0]["recon_loss_mean"]
    recon_max = max(r["recon_loss_mean"] for r in rows)
    recon_hurt = recon_max / recon_baseline if recon_baseline > 0 else float("nan")
    summary = {
        "d_signal": D_SIGNAL, "k_true": K_TRUE, "latent_dim": LATENT_DIM,
        "recon_loss_baseline_no_noise": recon_baseline,
        "recon_loss_max_with_noise": recon_max,
        "recon_degradation_ratio": recon_hurt,
        "uet_prediction_supported": recon_hurt < 1.5,
        "d_eff_no_noise": rows[0]["d_eff_mean"],
        "d_eff_512_noise": rows[-1]["d_eff_mean"],
    }
    dump_metadata(run_dir, summary)
    print("\n=== Noise Dimension Test ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
