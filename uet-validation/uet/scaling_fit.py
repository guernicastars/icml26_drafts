from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)

PYTHIA_BATCH_SIZE = 1024
PYTHIA_SEQ_LEN = 2048
PYTHIA_TOKENS_PER_STEP = PYTHIA_BATCH_SIZE * PYTHIA_SEQ_LEN


@dataclass(frozen=True)
class UETFit:
    c: float
    L_inf: float
    r_squared: float
    rmse: float
    n_points: int
    predicted: np.ndarray
    residuals: np.ndarray
    converged: bool


def uet_predict(
    c: float, L_inf: float,
    d_eff: np.ndarray, d: np.ndarray, n: np.ndarray,
) -> np.ndarray:
    safe_d_eff = np.maximum(d_eff, 1e-6)
    safe_ratio = np.maximum(d / safe_d_eff, 1.0 + 1e-9)
    feature = safe_d_eff * np.log(safe_ratio)
    return c * feature / np.maximum(n, 1.0) + L_inf


def fit_uet_curriculum(
    d_eff: np.ndarray,
    d: np.ndarray,
    n: np.ndarray,
    L: np.ndarray,
) -> UETFit:
    d_eff = np.asarray(d_eff, dtype=float)
    d = np.asarray(d, dtype=float)
    n = np.asarray(n, dtype=float)
    L = np.asarray(L, dtype=float)

    mask = (n > 0) & (d_eff > 0) & (d > d_eff) & np.isfinite(L)
    d_eff, d, n, L = d_eff[mask], d[mask], n[mask], L[mask]
    if len(L) < 3:
        raise ValueError(f"Need at least 3 valid points, got {len(L)}")

    feature = d_eff * np.log(d / d_eff)
    x = feature / n
    L_min = float(L.min())
    L_inf_hi = max(0.0, L_min * 0.999)

    best: UETFit | None = None
    for L_inf_init in np.linspace(0.0, L_inf_hi, 9):
        y = L - L_inf_init
        xx = float(np.dot(x, x))
        c_init = float(np.dot(x, y) / xx) if xx > 0 else 1.0
        c_init = max(c_init, 1e-6)

        def residuals(params):
            c, L_inf = params
            return uet_predict(c, L_inf, d_eff, d, n) - L

        try:
            result = least_squares(
                residuals,
                x0=[c_init, L_inf_init],
                bounds=([0.0, 0.0], [np.inf, L_inf_hi]),
                method="trf",
                max_nfev=10_000,
            )
        except Exception as e:
            logger.debug("least_squares failed at L_inf_init=%.3f: %s", L_inf_init, e)
            continue

        c_hat, L_inf_hat = float(result.x[0]), float(result.x[1])
        pred = uet_predict(c_hat, L_inf_hat, d_eff, d, n)
        resids = pred - L
        ss_res = float(np.sum(resids ** 2))
        ss_tot = float(np.sum((L - L.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmse = float(np.sqrt(ss_res / len(L)))

        fit = UETFit(
            c=c_hat, L_inf=L_inf_hat, r_squared=r2, rmse=rmse,
            n_points=int(len(L)), predicted=pred, residuals=resids,
            converged=bool(result.success),
        )
        if best is None or fit.rmse < best.rmse:
            best = fit

    if best is None:
        raise RuntimeError("All fits failed")
    return best


def pythia_step_to_tokens(step: int | np.ndarray) -> np.ndarray:
    return np.asarray(step) * PYTHIA_TOKENS_PER_STEP
