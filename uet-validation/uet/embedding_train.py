from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layers = []
        prev = input_dim
        for h in hidden_dims:
            encoder_layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)])
            prev = h
        encoder_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)])
            prev = h
        decoder_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)


@dataclass
class TrainingResult:
    model: Autoencoder
    train_losses: list[float]
    val_losses: list[float]
    final_val_loss: float


def train_autoencoder(
    X_train: np.ndarray,
    X_val: np.ndarray,
    latent_dim: int,
    hidden_dims: tuple[int, ...] = (128, 64),
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    n_epochs: int = 100,
    device: str = "cuda",
    dropout: float = 0.1,
    patience: int = 10,
) -> TrainingResult:
    X_train_t = torch.from_numpy(X_train.astype(np.float32))
    X_val_t = torch.from_numpy(X_val.astype(np.float32)).to(device)

    loader = DataLoader(
        TensorDataset(X_train_t), batch_size=batch_size, shuffle=True, drop_last=False,
    )

    model = Autoencoder(
        input_dim=X_train.shape[1],
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        running = 0.0
        n = 0
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            _, x_hat = model(batch)
            loss = loss_fn(x_hat, batch)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            running += loss.item() * batch.size(0)
            n += batch.size(0)
        train_loss = running / max(n, 1)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            _, x_hat = model(X_val_t)
            val_loss = loss_fn(x_hat, X_val_t).item()
        val_losses.append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            logger.info("epoch %d train=%.5f val=%.5f", epoch, train_loss, val_loss)

        if epochs_no_improve >= patience:
            logger.info("Early stop at epoch %d (no improvement for %d epochs)", epoch, patience)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainingResult(
        model=model,
        train_losses=train_losses,
        val_losses=val_losses,
        final_val_loss=best_val,
    )


@torch.no_grad()
def encode_dataset(
    model: Autoencoder, X: np.ndarray, batch_size: int = 1024, device: str = "cuda",
) -> np.ndarray:
    model.eval()
    model = model.to(device)
    X_t = torch.from_numpy(X.astype(np.float32))
    out = []
    for i in range(0, len(X_t), batch_size):
        batch = X_t[i : i + batch_size].to(device, non_blocking=True)
        out.append(model.encode(batch).cpu().numpy())
    return np.concatenate(out, axis=0)


def temporal_split(
    X: np.ndarray,
    timestamps: np.ndarray | None = None,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    if timestamps is not None:
        order = np.argsort(timestamps)
    else:
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)

    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]
    return train_idx, val_idx, test_idx
