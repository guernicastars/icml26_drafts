from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from ..adapters.state import (
    AdapterStateManifest,
    flatten_adapter_state,
    iter_trainable_parameters,
    restore_adapter_state,
)
from .base import AggregatedAdapterResult, effective_sample_size


def compute_diagonal_fisher(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.nn.Module], torch.Tensor],
    manifest: AdapterStateManifest,
    num_batches: int,
    dataloader,
) -> np.ndarray:
    """Diagonal empirical Fisher in LoRA-space via per-sample gradients."""
    fisher_diag = None
    total_samples = 0

    model.eval()
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        model.zero_grad()
        loss = loss_fn(model, batch)
        loss.backward()

        grad_vector = []
        for spec in manifest.parameters:
            param = dict(iter_trainable_parameters(model))[spec.name]
            if param.grad is not None:
                grad_vector.append(param.grad.detach().cpu().reshape(-1).float())
            else:
                grad_vector.append(torch.zeros(spec.numel, dtype=torch.float32))

        grad_flat = torch.cat(grad_vector).numpy().astype(np.float64)
        sq = grad_flat ** 2

        if fisher_diag is None:
            fisher_diag = sq
        else:
            fisher_diag += sq
        total_samples += 1

    model.zero_grad()
    if fisher_diag is None or total_samples == 0:
        return np.zeros(manifest.total_params, dtype=np.float32)

    fisher_diag /= total_samples
    return fisher_diag.astype(np.float32)


def laplace_posterior(
    model: torch.nn.Module,
    manifest: AdapterStateManifest,
    fisher_diag: np.ndarray,
    prior_precision: float = 1.0,
) -> AggregatedAdapterResult:
    """Build a Laplace posterior from the diagonal Fisher at the current model state."""
    map_vector, _ = flatten_adapter_state(model, manifest)

    posterior_precision = fisher_diag + prior_precision
    posterior_variance = 1.0 / np.maximum(posterior_precision, 1e-8)
    posterior_variance = posterior_variance.astype(np.float32)

    posterior_trace = float(posterior_variance.sum())
    sorted_vars = np.sort(posterior_variance)[::-1]
    top_eigenvalues = tuple(float(v) for v in sorted_vars[:5])
    top_eigenvalue_ratio = float(top_eigenvalues[0] / posterior_trace) if posterior_trace > 0 else 0.0

    weights = np.ones(1, dtype=np.float32)

    return AggregatedAdapterResult(
        scheme="laplace",
        mean_vector=map_vector,
        weights=weights,
        retained_count=1,
        effective_sample_size=1.0,
        beta=None,
        threshold=None,
        posterior_trace=posterior_trace,
        top_eigenvalues=top_eigenvalues,
        top_eigenvalue_ratio=top_eigenvalue_ratio,
        max_normalized_weight=1.0,
        score_variance=0.0,
        diagonal_variance=posterior_variance,
        deviations=np.zeros((0, map_vector.size), dtype=np.float32),
    )


def tune_prior_precision(
    model: torch.nn.Module,
    manifest: AdapterStateManifest,
    fisher_diag: np.ndarray,
    candidates: list[float] | None = None,
    loss_fn: Callable[[torch.nn.Module], torch.Tensor] | None = None,
    val_dataloader=None,
    num_val_batches: int = 10,
) -> float:
    """Select prior precision via approximate marginal likelihood on a validation set.

    Falls back to 1.0 if no validation data is provided.
    """
    if loss_fn is None or val_dataloader is None:
        return 1.0

    if candidates is None:
        candidates = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]

    map_vector, _ = flatten_adapter_state(model, manifest)
    map_norm_sq = float(np.sum(map_vector ** 2))

    d = manifest.total_params
    best_precision = 1.0
    best_score = -float("inf")

    for lam in candidates:
        precision = fisher_diag + lam
        log_det_posterior = float(np.sum(np.log(np.maximum(precision, 1e-12))))
        log_det_prior = d * np.log(lam) if lam > 0 else 0.0
        log_det_fisher = float(np.sum(np.log(np.maximum(fisher_diag, 1e-12))))

        # log p(D|theta_MAP) approximated by negative validation loss
        val_loss = 0.0
        n_batches = 0
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                if batch_idx >= num_val_batches:
                    break
                val_loss += float(loss_fn(model, batch).item())
                n_batches += 1
        if n_batches > 0:
            val_loss /= n_batches

        log_ml = (
            -val_loss
            - 0.5 * lam * map_norm_sq
            + 0.5 * log_det_prior
            - 0.5 * log_det_posterior
        )

        if log_ml > best_score:
            best_score = log_ml
            best_precision = lam

    return best_precision
