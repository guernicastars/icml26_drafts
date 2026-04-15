from __future__ import annotations

import numpy as np
from numpy.linalg import eigh, eigvalsh, svd


def covariance(X: np.ndarray) -> np.ndarray:
    centered = X - X.mean(axis=0, keepdims=True)
    n = max(X.shape[0] - 1, 1)
    return centered.T @ centered / n


def eigenspectrum(cov: np.ndarray) -> np.ndarray:
    vals = eigvalsh(cov)
    return vals[::-1].copy()


def effective_dimension(eigenvalues: np.ndarray) -> float:
    positive = eigenvalues[eigenvalues > 0]
    if len(positive) == 0:
        return 0.0
    trace = positive.sum()
    trace_sq = (positive**2).sum()
    return float(trace**2 / trace_sq)


def stable_rank(eigenvalues: np.ndarray) -> float:
    positive = eigenvalues[eigenvalues > 0]
    if len(positive) == 0:
        return 0.0
    return float(positive.sum() / positive[0])


def participation_ratio(eigenvalues: np.ndarray) -> float:
    return effective_dimension(eigenvalues)


def spectral_gap(eigenvalues: np.ndarray, k: int) -> float:
    if k < 1 or k >= len(eigenvalues):
        raise ValueError(f"k must be in [1, {len(eigenvalues) - 1}]")
    return float(eigenvalues[k - 1] - eigenvalues[k])


def spectral_gap_ratio(eigenvalues: np.ndarray, k: int) -> float:
    if k < 1 or k >= len(eigenvalues):
        raise ValueError(f"k must be in [1, {len(eigenvalues) - 1}]")
    denom = eigenvalues[k]
    if denom <= 0:
        return float("inf")
    return float(eigenvalues[k - 1] / denom)


def pca_alignment_sin(V_true: np.ndarray, V_hat: np.ndarray) -> float:
    svals = svd(V_true.T @ V_hat, compute_uv=False)
    smallest = float(np.clip(np.min(svals), 0.0, 1.0))
    return float(np.sqrt(max(0.0, 1.0 - smallest**2)))


def top_eigenvectors(cov: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = eigh(cov)
    order = np.argsort(vals)[::-1]
    return vecs[:, order[:k]], vals[order]


def theorem_42_bound(
    epsilon_f: float,
    grad_op_norm: float,
    lambda_k: float,
    lambda_k_plus_1: float,
) -> float:
    gap = lambda_k - lambda_k_plus_1
    if gap <= 0:
        return float("inf")
    C2 = 1.0
    return C2 * epsilon_f * grad_op_norm / gap
