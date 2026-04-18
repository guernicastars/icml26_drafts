import numpy as np
import pytest

from uet.scaling_fit import (
    PYTHIA_TOKENS_PER_STEP,
    fit_uet_curriculum,
    pythia_step_to_tokens,
    uet_predict,
)


def _synthetic_curriculum(c: float, L_inf: float, d: int, seed: int = 0, noise: float = 0.02):
    rng = np.random.default_rng(seed)
    n = np.logspace(7, 11, 12)
    d_eff = np.linspace(30, 90, 12)
    d_arr = np.full_like(n, d, dtype=float)
    L_true = uet_predict(c, L_inf, d_eff, d_arr, n)
    return d_eff, d_arr, n.astype(float), L_true + rng.normal(scale=noise, size=L_true.shape)


def test_pythia_tokens_per_step():
    assert PYTHIA_TOKENS_PER_STEP == 1024 * 2048


def test_uet_predict_monotone_in_n():
    d_eff = np.array([50.0, 50.0, 50.0])
    d = np.array([768.0, 768.0, 768.0])
    n = np.array([1e8, 1e9, 1e10])
    L = uet_predict(c=100.0, L_inf=2.0, d_eff=d_eff, d=d, n=n)
    assert np.all(np.diff(L) < 0)


def test_fit_recovers_synthetic_params():
    c_true, L_inf_true, d = 1e9, 2.5, 768
    d_eff, d_arr, n, L = _synthetic_curriculum(c_true, L_inf_true, d, seed=42, noise=0.02)
    fit = fit_uet_curriculum(d_eff=d_eff, d=d_arr, n=n, L=L)
    assert fit.r_squared > 0.99
    assert abs(fit.L_inf - L_inf_true) < 0.1
    assert abs(fit.c / c_true - 1.0) < 0.1
    assert fit.converged


def test_fit_rejects_too_few_points():
    with pytest.raises(ValueError):
        fit_uet_curriculum(
            d_eff=np.array([10.0, 20.0]),
            d=np.array([100.0, 100.0]),
            n=np.array([1e9, 2e9]),
            L=np.array([3.0, 2.5]),
        )


def test_fit_drops_invalid_points():
    d_eff = np.array([50.0, 50.0, 50.0, 200.0, 50.0])
    d = np.array([768.0, 768.0, 768.0, 100.0, 768.0])
    n = np.array([1e8, 1e9, 1e10, 1e9, 0.0])
    L = np.array([3.5, 3.0, 2.6, 4.0, 5.0])
    fit = fit_uet_curriculum(d_eff=d_eff, d=d, n=n, L=L)
    assert fit.n_points == 3
