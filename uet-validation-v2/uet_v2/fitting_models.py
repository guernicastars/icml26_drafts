from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelFit:
    name: str
    n_params: int
    r_squared: float
    rmse: float
    aic: float
    bic: float
    params: dict
    predicted: np.ndarray
    residuals: np.ndarray


def compute_aic_bic(residuals: np.ndarray, n_params: int) -> tuple[float, float]:
    n = len(residuals)
    ss_res = float(np.sum(residuals ** 2))
    sigma2 = max(ss_res / n, 1e-30)
    log_lik = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1.0)
    aic = 2.0 * n_params - 2.0 * log_lik
    bic = n_params * np.log(n) - 2.0 * log_lik
    return aic, bic


def _r2_rmse(predicted: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    resids = predicted - actual
    ss_res = float(np.sum(resids ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(ss_res / len(actual)))
    return r2, rmse


def fit_kaplan(n: np.ndarray, L: np.ndarray) -> ModelFit:
    """L = A * n^{-alpha} + L_inf  (Kaplan 2020, 3 free params; ignores d_eff)."""
    n = np.asarray(n, dtype=float)
    L = np.asarray(L, dtype=float)

    L_min = float(L.min())
    L_inf_hi = max(0.0, L_min * 0.999)

    def residuals(params):
        A, alpha, L_inf = params
        return A * np.maximum(n, 1.0) ** (-alpha) + L_inf - L

    best: ModelFit | None = None
    for L_inf_init in np.linspace(0.0, L_inf_hi, 9):
        try:
            result = least_squares(
                residuals,
                x0=[1.0, 0.3, L_inf_init],
                bounds=([0.0, 1e-3, 0.0], [np.inf, 2.0, L_inf_hi]),
                method="trf",
                max_nfev=10_000,
            )
        except Exception as e:
            logger.debug("kaplan fit attempt failed: %s", e)
            continue
        A, alpha, L_inf = result.x
        pred = A * np.maximum(n, 1.0) ** (-alpha) + L_inf
        resids = pred - L
        r2, rmse = _r2_rmse(pred, L)
        aic, bic = compute_aic_bic(resids, n_params=3)
        fit = ModelFit(
            name="kaplan", n_params=3, r_squared=r2, rmse=rmse, aic=aic, bic=bic,
            params={"A": float(A), "alpha": float(alpha), "L_inf": float(L_inf)},
            predicted=pred, residuals=resids,
        )
        if best is None or fit.rmse < best.rmse:
            best = fit

    if best is None:
        raise RuntimeError("Kaplan fit: all initialisations failed")
    return best


def fit_free_uet(
    d_eff: np.ndarray, d: np.ndarray, n: np.ndarray, L: np.ndarray
) -> ModelFit:
    """L = c * d_eff^alpha * log(d/d_eff)^beta / n^gamma + L_inf  (5 free params)."""
    d_eff = np.asarray(d_eff, dtype=float)
    d = np.asarray(d, dtype=float)
    n = np.asarray(n, dtype=float)
    L = np.asarray(L, dtype=float)

    safe_deff = np.maximum(d_eff, 1e-6)
    safe_ratio = np.maximum(d / safe_deff, 1.0 + 1e-9)
    log_ratio = np.log(safe_ratio)

    L_min = float(L.min())
    L_inf_hi = max(0.0, L_min * 0.999)

    def residuals(params):
        c, alpha, beta, gamma, L_inf = params
        feature = c * (safe_deff ** alpha) * (log_ratio ** beta) / (np.maximum(n, 1.0) ** gamma)
        return feature + L_inf - L

    best: ModelFit | None = None
    for L_inf_init in np.linspace(0.0, L_inf_hi, 5):
        try:
            result = least_squares(
                residuals,
                x0=[1e7, 1.0, 1.0, 1.0, L_inf_init],
                bounds=([0.0, 0.1, 0.1, 0.1, 0.0], [np.inf, 3.0, 3.0, 3.0, L_inf_hi]),
                method="trf",
                max_nfev=20_000,
            )
        except Exception as e:
            logger.debug("free-uet fit attempt failed: %s", e)
            continue
        c, alpha, beta, gamma, L_inf = result.x
        pred = (
            c * (safe_deff ** alpha) * (log_ratio ** beta)
            / np.maximum(n, 1.0) ** gamma
            + L_inf
        )
        resids = pred - L
        r2, rmse = _r2_rmse(pred, L)
        aic, bic = compute_aic_bic(resids, n_params=5)
        fit = ModelFit(
            name="free_uet", n_params=5, r_squared=r2, rmse=rmse, aic=aic, bic=bic,
            params={
                "c": float(c), "alpha": float(alpha), "beta": float(beta),
                "gamma": float(gamma), "L_inf": float(L_inf),
            },
            predicted=pred, residuals=resids,
        )
        if best is None or fit.rmse < best.rmse:
            best = fit

    if best is None:
        raise RuntimeError("Free-UET fit: all initialisations failed")
    return best
