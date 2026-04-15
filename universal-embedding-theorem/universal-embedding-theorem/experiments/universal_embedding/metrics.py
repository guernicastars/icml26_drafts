from __future__ import annotations

import numpy as np


def empirical_covariance(samples: np.ndarray) -> np.ndarray:
    centered = samples - samples.mean(axis=0, keepdims=True)
    denominator = max(samples.shape[0] - 1, 1)
    return centered.T @ centered / denominator


def top_eigenpairs(covariance: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    sorted_values = eigenvalues[order]
    sorted_vectors = eigenvectors[:, order]
    return sorted_vectors[:, :rank], sorted_values


def estimate_top_pca_basis(samples: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    covariance = empirical_covariance(samples)
    return top_eigenpairs(covariance, rank)


def effective_dimension(covariance: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(covariance)
    largest = float(np.max(eigenvalues))
    if largest <= 0.0:
        return 0.0
    return float(np.sum(eigenvalues) / largest)


def largest_principal_angle_sine(true_basis: np.ndarray, estimated_basis: np.ndarray) -> float:
    singular_values = np.linalg.svd(true_basis.T @ estimated_basis, compute_uv=False)
    smallest = float(np.clip(np.min(singular_values), 0.0, 1.0))
    return float(np.sqrt(max(0.0, 1.0 - smallest**2)))


def spectral_gap(covariance: np.ndarray, rank: int) -> float:
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    if rank <= 0 or rank > len(eigenvalues):
        raise ValueError(f"rank must be between 1 and {len(eigenvalues)}")
    if rank == len(eigenvalues):
        return float(eigenvalues[rank - 1])
    return float(eigenvalues[rank - 1] - eigenvalues[rank])
