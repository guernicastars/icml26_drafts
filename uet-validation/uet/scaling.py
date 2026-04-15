from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def theoretical_excess_risk(k: float, d: int, n: int) -> float:
    if k <= 0 or d <= k or n <= 0:
        return float("inf")
    return k * np.log(d / k) / n


def chinchilla_power_law(N: np.ndarray, a: float, alpha: float, L_inf: float) -> np.ndarray:
    return a * np.power(N, -alpha) + L_inf


def fit_chinchilla(n_params: np.ndarray, losses: np.ndarray) -> dict:
    try:
        popt, pcov = curve_fit(
            chinchilla_power_law, n_params, losses,
            p0=[10.0, 0.5, 2.0],
            bounds=([0, 0, 0], [1e6, 5.0, 10.0]),
            maxfev=10000,
        )
    except RuntimeError:
        return {"a": np.nan, "alpha": np.nan, "L_inf": np.nan, "residuals": np.full(len(losses), np.nan)}

    predicted = chinchilla_power_law(n_params, *popt)
    return {
        "a": popt[0],
        "alpha": popt[1],
        "L_inf": popt[2],
        "residuals": losses - predicted,
    }


def uet_predicted_exponent(d_effs: np.ndarray, hidden_dims: np.ndarray) -> np.ndarray:
    return d_effs * np.log(hidden_dims / d_effs)


def fit_uet_scaling(
    d_effs: np.ndarray,
    hidden_dims: np.ndarray,
    losses: np.ndarray,
    n_tokens_train: np.ndarray,
) -> dict:
    def model(X, c, L_inf):
        d_eff, d, n = X
        return c * d_eff * np.log(d / d_eff) / n + L_inf

    X_data = np.array([d_effs, hidden_dims, n_tokens_train])

    try:
        popt, pcov = curve_fit(
            model, X_data, losses,
            p0=[1.0, 2.0],
            bounds=([0, 0], [1e6, 10.0]),
            maxfev=10000,
        )
    except RuntimeError:
        return {"c": np.nan, "L_inf": np.nan, "predicted": np.full(len(losses), np.nan)}

    predicted = model(X_data, *popt)
    return {
        "c": popt[0],
        "L_inf": popt[1],
        "predicted": predicted,
        "residuals": losses - predicted,
        "r_squared": 1.0 - np.sum((losses - predicted) ** 2) / np.sum((losses - losses.mean()) ** 2),
    }


PYTHIA_TRAIN_TOKENS = {
    "EleutherAI/pythia-70m-deduped": 299_892_736_000,
    "EleutherAI/pythia-160m-deduped": 299_892_736_000,
    "EleutherAI/pythia-410m-deduped": 299_892_736_000,
    "EleutherAI/pythia-1b-deduped": 299_892_736_000,
    "EleutherAI/pythia-2.8b-deduped": 299_892_736_000,
    "EleutherAI/pythia-6.9b-deduped": 299_892_736_000,
    "EleutherAI/pythia-12b-deduped": 299_892_736_000,
}
