from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianEmbeddingProblem:
    samples: np.ndarray
    causal_basis: np.ndarray
    nuisance_basis: np.ndarray
    covariance: np.ndarray
    signal_eigenvalues: np.ndarray
    nuisance_eigenvalues: np.ndarray
    isotropic_noise: float


@dataclass(frozen=True)
class SparseRegressionProblem:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    true_weights: np.ndarray
    causal_basis: np.ndarray
    covariance: np.ndarray


def _full_orthonormal_basis(dimension: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(size=(dimension, dimension))
    q_matrix, r_matrix = np.linalg.qr(raw)
    signs = np.sign(np.diag(r_matrix))
    signs[signs == 0.0] = 1.0
    return q_matrix * signs


def _sample_from_basis(
    num_samples: int,
    basis: np.ndarray,
    eigenvalues: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    latent = rng.normal(size=(num_samples, basis.shape[1]))
    scaled = latent * np.sqrt(np.clip(eigenvalues, 0.0, None))
    return scaled @ basis.T


def generate_gaussian_embedding_problem(
    num_samples: int,
    ambient_dim: int,
    causal_dim: int,
    signal_strength: float = 4.0,
    nuisance_strength: float = 0.5,
    nuisance_rank: int | None = None,
    isotropic_noise: float = 0.05,
    rng: np.random.Generator | None = None,
) -> GaussianEmbeddingProblem:
    if causal_dim >= ambient_dim:
        raise ValueError("causal_dim must be smaller than ambient_dim")

    rng = rng or np.random.default_rng()
    basis = _full_orthonormal_basis(ambient_dim, rng)
    nuisance_rank = ambient_dim - causal_dim if nuisance_rank is None else min(nuisance_rank, ambient_dim - causal_dim)

    causal_basis = basis[:, :causal_dim]
    nuisance_basis = basis[:, causal_dim : causal_dim + nuisance_rank]

    signal_eigenvalues = np.linspace(signal_strength, 0.6 * signal_strength, causal_dim)
    nuisance_eigenvalues = (
        np.linspace(nuisance_strength, 0.5 * nuisance_strength, nuisance_rank)
        if nuisance_rank > 0
        else np.zeros(0, dtype=np.float64)
    )

    samples = np.zeros((num_samples, ambient_dim), dtype=np.float64)
    if causal_dim > 0:
        samples = samples + _sample_from_basis(num_samples, causal_basis, signal_eigenvalues, rng)
    if nuisance_rank > 0:
        samples = samples + _sample_from_basis(num_samples, nuisance_basis, nuisance_eigenvalues, rng)
    if isotropic_noise > 0.0:
        samples = samples + rng.normal(scale=np.sqrt(isotropic_noise), size=(num_samples, ambient_dim))

    covariance = causal_basis @ np.diag(signal_eigenvalues) @ causal_basis.T
    if nuisance_rank > 0:
        covariance = covariance + nuisance_basis @ np.diag(nuisance_eigenvalues) @ nuisance_basis.T
    covariance = covariance + isotropic_noise * np.eye(ambient_dim)

    return GaussianEmbeddingProblem(
        samples=samples,
        causal_basis=causal_basis,
        nuisance_basis=nuisance_basis,
        covariance=covariance,
        signal_eigenvalues=signal_eigenvalues,
        nuisance_eigenvalues=nuisance_eigenvalues,
        isotropic_noise=isotropic_noise,
    )


def generate_sparse_regression_problem(
    num_train: int,
    num_test: int,
    ambient_dim: int,
    causal_dim: int,
    signal_strength: float = 1.0,
    nuisance_strength: float = 0.02,
    response_noise: float = 0.1,
    rng: np.random.Generator | None = None,
) -> SparseRegressionProblem:
    if causal_dim >= ambient_dim:
        raise ValueError("causal_dim must be smaller than ambient_dim")

    rng = rng or np.random.default_rng()
    basis = _full_orthonormal_basis(ambient_dim, rng)
    causal_basis = basis[:, :causal_dim]

    eigenvalues = np.concatenate(
        [
            np.full(causal_dim, signal_strength, dtype=np.float64),
            np.full(ambient_dim - causal_dim, nuisance_strength, dtype=np.float64),
        ]
    )
    covariance = basis @ np.diag(eigenvalues) @ basis.T

    x_train = _sample_from_basis(num_train, basis, eigenvalues, rng)
    x_test = _sample_from_basis(num_test, basis, eigenvalues, rng)

    coefficients = rng.normal(size=causal_dim)
    coefficient_norm = float(np.linalg.norm(coefficients))
    if coefficient_norm == 0.0:
        coefficients[0] = 1.0
        coefficient_norm = 1.0
    coefficients = coefficients / coefficient_norm
    true_weights = causal_basis @ coefficients

    y_train = x_train @ true_weights + rng.normal(scale=response_noise, size=num_train)
    y_test = x_test @ true_weights + rng.normal(scale=response_noise, size=num_test)

    return SparseRegressionProblem(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        true_weights=true_weights,
        causal_basis=causal_basis,
        covariance=covariance,
    )


def append_noise_dimensions(
    samples: np.ndarray,
    extra_dims: int,
    noise_std: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if extra_dims <= 0:
        return samples.copy()
    rng = rng or np.random.default_rng()
    noise_block = rng.normal(scale=noise_std, size=(samples.shape[0], extra_dims))
    return np.concatenate([samples, noise_block], axis=1)
