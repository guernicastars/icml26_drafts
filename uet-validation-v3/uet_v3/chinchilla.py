from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChinchillaFit:
    A: float
    B: float
    alpha: float
    beta: float
    E: float
    r_squared: float
    rmse: float
    predicted: np.ndarray
    residuals: np.ndarray
    converged: bool


def chinchilla_predict(
    A: float, B: float, alpha: float, beta: float, E: float,
    N: np.ndarray, D: np.ndarray,
) -> np.ndarray:
    return A / np.maximum(N, 1.0) ** alpha + B / np.maximum(D, 1.0) ** beta + E


def fit_chinchilla(
    N: np.ndarray, D: np.ndarray, L: np.ndarray,
) -> ChinchillaFit:
    """Hoffmann 2022: L = A/N^alpha + B/D^beta + E. 5 free params.
    N = parameter count, D = training tokens.
    """
    N = np.asarray(N, dtype=float)
    D = np.asarray(D, dtype=float)
    L = np.asarray(L, dtype=float)

    L_min = float(L.min())
    E_hi = max(0.0, L_min * 0.999)

    def residuals(params):
        A, B, alpha, beta, E = params
        return chinchilla_predict(A, B, alpha, beta, E, N, D) - L

    best: ChinchillaFit | None = None
    for E_init in np.linspace(0.0, E_hi, 5):
        for alpha_init in (0.2, 0.35, 0.5):
            for beta_init in (0.2, 0.35, 0.5):
                try:
                    result = least_squares(
                        residuals,
                        x0=[10.0, 10.0, alpha_init, beta_init, E_init],
                        bounds=([0.0, 0.0, 1e-3, 1e-3, 0.0],
                                [np.inf, np.inf, 2.0, 2.0, E_hi]),
                        method="trf",
                        max_nfev=10_000,
                    )
                except Exception as e:
                    logger.debug("chinchilla fit failed: %s", e)
                    continue
                A, B, alpha, beta, E = result.x
                pred = chinchilla_predict(A, B, alpha, beta, E, N, D)
                resids = pred - L
                ss_res = float(np.sum(resids ** 2))
                ss_tot = float(np.sum((L - L.mean()) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                rmse = float(np.sqrt(ss_res / len(L)))
                fit = ChinchillaFit(
                    A=float(A), B=float(B), alpha=float(alpha), beta=float(beta),
                    E=float(E), r_squared=r2, rmse=rmse,
                    predicted=pred, residuals=resids,
                    converged=bool(result.success),
                )
                if best is None or fit.rmse < best.rmse:
                    best = fit

    if best is None:
        raise RuntimeError("Chinchilla fit: all initialisations failed")
    return best
