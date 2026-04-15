from __future__ import annotations

import math

import numpy as np


def feature_count_for_alpha(embedding_dim: int, alpha: float, max_features: int) -> int:
    return int(min(max_features, max(2 * embedding_dim, round(math.exp(alpha * embedding_dim)))))


def sample_feature_dictionary(
    embedding_dim: int,
    num_features: int,
    rng: np.random.Generator,
) -> np.ndarray:
    dictionary = rng.normal(size=(num_features, embedding_dim))
    norms = np.linalg.norm(dictionary, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return dictionary / norms


def sample_sparse_coefficients(
    num_features: int,
    sparsity: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if sparsity > num_features:
        raise ValueError("sparsity cannot exceed num_features")
    support = np.sort(rng.choice(num_features, size=sparsity, replace=False))
    coefficients = np.zeros(num_features, dtype=np.float64)
    coefficients[support] = rng.choice(np.asarray([-1.0, 1.0]), size=sparsity)
    return coefficients, support


def encode_superposition(dictionary: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return coefficients @ dictionary


def decode_superposition(dictionary: np.ndarray, activation: np.ndarray) -> np.ndarray:
    return dictionary @ activation


def max_pairwise_coherence(dictionary: np.ndarray) -> float:
    gram = dictionary @ dictionary.T
    np.fill_diagonal(gram, 0.0)
    return float(np.max(np.abs(gram)))


def evaluate_superposition(
    dictionary: np.ndarray,
    sparsity: int,
    trials: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    num_features, embedding_dim = dictionary.shape
    active_errors: list[float] = []
    inactive_interference: list[float] = []
    total_rmse: list[float] = []

    for _ in range(trials):
        coefficients, support = sample_sparse_coefficients(num_features, sparsity, rng)
        activation = encode_superposition(dictionary, coefficients)
        decoded = decode_superposition(dictionary, activation)

        active_error = float(np.mean(np.abs(decoded[support] - coefficients[support])))
        mask = np.ones(num_features, dtype=bool)
        mask[support] = False
        inactive_error = float(np.max(np.abs(decoded[mask]))) if np.any(mask) else 0.0
        rmse = float(np.sqrt(np.mean((decoded - coefficients) ** 2)))

        active_errors.append(active_error)
        inactive_interference.append(inactive_error)
        total_rmse.append(rmse)

    coherence = max_pairwise_coherence(dictionary)
    theoretical_scale = sparsity * math.sqrt(max(math.log(max(num_features, 2)), 1e-12) / embedding_dim)
    return {
        "num_features": float(num_features),
        "feature_to_dimension_ratio": float(num_features / embedding_dim),
        "pairwise_coherence": coherence,
        "active_mae": float(np.mean(active_errors)),
        "inactive_max_abs": float(np.mean(inactive_interference)),
        "decoder_rmse": float(np.mean(total_rmse)),
        "theoretical_scale": float(theoretical_scale),
        "active_mae_to_scale": float(np.mean(active_errors) / max(theoretical_scale, 1e-12)),
        "inactive_to_scale": float(np.mean(inactive_interference) / max(theoretical_scale, 1e-12)),
    }
