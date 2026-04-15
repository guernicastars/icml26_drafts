from __future__ import annotations

import numpy as np


def sample_sparse_signal(ambient_dim: int, sparsity: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if sparsity > ambient_dim:
        raise ValueError("sparsity cannot exceed ambient_dim")
    signal = np.zeros(ambient_dim, dtype=np.float64)
    support = np.sort(rng.choice(ambient_dim, size=sparsity, replace=False))
    values = rng.normal(size=sparsity)
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        values[0] = 1.0
        norm = 1.0
    signal[support] = values / norm
    return signal, support


def gaussian_measurements(
    signal: np.ndarray,
    num_measurements: int,
    noise_std: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    sensing = rng.normal(scale=1.0 / np.sqrt(max(num_measurements, 1)), size=(num_measurements, signal.size))
    targets = sensing @ signal
    if noise_std > 0.0:
        targets = targets + rng.normal(scale=noise_std, size=num_measurements)
    return sensing, targets


def orthogonal_matching_pursuit(
    sensing: np.ndarray,
    targets: np.ndarray,
    sparsity: int,
    tol: float = 1e-10,
) -> tuple[np.ndarray, list[int]]:
    _, ambient_dim = sensing.shape
    residual = targets.copy()
    support: list[int] = []
    coefficients = np.zeros(0, dtype=np.float64)

    for _ in range(min(sparsity, ambient_dim)):
        correlations = np.abs(sensing.T @ residual)
        if support:
            correlations[np.asarray(support)] = -np.inf
        new_index = int(np.argmax(correlations))
        if new_index in support:
            break
        support.append(new_index)
        submatrix = sensing[:, support]
        coefficients, *_ = np.linalg.lstsq(submatrix, targets, rcond=None)
        residual = targets - submatrix @ coefficients
        if np.linalg.norm(residual) <= tol:
            break

    estimate = np.zeros(ambient_dim, dtype=np.float64)
    if support:
        estimate[np.asarray(support)] = coefficients
    return estimate, support


def relative_l2_error(truth: np.ndarray, estimate: np.ndarray) -> float:
    denominator = float(np.linalg.norm(truth))
    if denominator == 0.0:
        return float(np.linalg.norm(estimate))
    return float(np.linalg.norm(truth - estimate) / denominator)


def support_recall(true_support: np.ndarray, estimated_support: list[int]) -> float:
    true_set = set(int(index) for index in true_support.tolist())
    estimated_set = set(estimated_support)
    if not true_set:
        return 1.0
    return float(len(true_set & estimated_set) / len(true_set))
