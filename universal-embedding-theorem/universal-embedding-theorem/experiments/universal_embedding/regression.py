from __future__ import annotations

import numpy as np

from .synthetic_data import SparseRegressionProblem


def minimum_norm_weights(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(x_train, rcond=1e-10) @ y_train


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = y_true - y_pred
    return float(np.mean(residual**2))


def evaluate_minimum_norm_regression(problem: SparseRegressionProblem) -> dict[str, float]:
    weights = minimum_norm_weights(problem.x_train, problem.y_train)
    train_predictions = problem.x_train @ weights
    test_predictions = problem.x_test @ weights
    return {
        "weight_norm": float(np.linalg.norm(weights)),
        "train_mse": mean_squared_error(problem.y_train, train_predictions),
        "test_mse": mean_squared_error(problem.y_test, test_predictions),
    }
