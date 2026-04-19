"""Distillation-rank test.

Train k-bottleneck MLP students to reconstruct teacher hidden states H.
Measures whether MSE has a sharp transition at k = d_eff(H).

UET prediction: for k < d_eff, information loss is sharp; for k >= d_eff
the reconstruction MSE plateaus — the formalisation endpoint is the
minimal-rank sufficient bottleneck.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class BottleneckMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, k: int):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, k))
        self.dec = nn.Sequential(nn.Linear(k, hidden), nn.GELU(), nn.Linear(hidden, in_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.enc(x)
        return self.dec(z), z


def train_student(
    H: np.ndarray,
    k: int,
    hidden: int = 384,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 512,
    device: str = "cuda",
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    H_t = torch.from_numpy(H.astype(np.float32))
    loader = DataLoader(TensorDataset(H_t), batch_size=batch_size, shuffle=True)

    model = BottleneckMLP(H.shape[1], hidden, k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for (xb,) in loader:
            xb = xb.to(device)
            recon, _ = model(xb)
            opt.zero_grad()
            F.mse_loss(recon, xb).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        H_gpu = H_t.to(device)
        recon_all, z_all = model(H_gpu)
        mse = float(F.mse_loss(recon_all, H_gpu).cpu())
        rel_mse = mse / float(H_gpu.var())
        z_np = z_all.cpu().numpy().astype(np.float64)

    from uet.eigendecomp import covariance, effective_dimension, eigenspectrum
    z_eigs = eigenspectrum(covariance(z_np))
    z_deff = effective_dimension(z_eigs)

    logger.info("k=%d  mse=%.5f  rel_mse=%.4f  z_deff=%.2f", k, mse, rel_mse, z_deff)
    return {"k": k, "seed": seed, "mse": round(mse, 6), "rel_mse": round(rel_mse, 5),
            "z_deff": round(z_deff, 3)}
