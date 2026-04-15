from __future__ import annotations

import numpy as np
import pytest

from uet.scaling import (
    chinchilla_power_law,
    fit_chinchilla,
    fit_uet_scaling,
    theoretical_excess_risk,
    uet_predicted_exponent,
)


class TestTheoreticalExcessRisk:
    def test_basic(self):
        risk = theoretical_excess_risk(k=10, d=100, n=1000)
        assert risk == pytest.approx(10 * np.log(10) / 1000)

    def test_invalid_k_zero(self):
        assert theoretical_excess_risk(0, 100, 1000) == float("inf")

    def test_invalid_d_le_k(self):
        assert theoretical_excess_risk(10, 10, 1000) == float("inf")


class TestChinchillaPowerLaw:
    def test_shape(self):
        N = np.array([1e6, 1e7, 1e8])
        result = chinchilla_power_law(N, a=10.0, alpha=0.5, L_inf=2.0)
        assert result.shape == (3,)

    def test_decreasing(self):
        N = np.logspace(6, 10, 20)
        result = chinchilla_power_law(N, a=10.0, alpha=0.5, L_inf=2.0)
        assert np.all(np.diff(result) < 0)

    def test_converges_to_linf(self):
        N = np.array([1e20])
        result = chinchilla_power_law(N, a=10.0, alpha=0.5, L_inf=2.0)
        assert result[0] == pytest.approx(2.0, abs=0.01)


class TestFitChinchilla:
    def test_recovers_params(self):
        N = np.logspace(6, 10, 20)
        true_a, true_alpha, true_linf = 5.0, 0.3, 2.5
        losses = chinchilla_power_law(N, true_a, true_alpha, true_linf)
        result = fit_chinchilla(N, losses)
        assert result["a"] == pytest.approx(true_a, rel=0.05)
        assert result["alpha"] == pytest.approx(true_alpha, rel=0.05)
        assert result["L_inf"] == pytest.approx(true_linf, rel=0.05)


class TestUETPredictedExponent:
    def test_shape(self):
        d_effs = np.array([10.0, 20.0, 50.0])
        hidden = np.array([768.0, 1024.0, 2048.0])
        result = uet_predicted_exponent(d_effs, hidden)
        assert result.shape == (3,)


class TestFitUETScaling:
    def test_with_synthetic(self):
        d_effs = np.array([10.0, 20.0, 50.0, 100.0, 200.0])
        hidden = np.array([512.0, 768.0, 1024.0, 2048.0, 4096.0])
        n_tokens = np.array([1e3, 2e3, 5e3, 1e4, 2e4])
        true_c, true_linf = 100.0, 2.5
        losses = true_c * d_effs * np.log(hidden / d_effs) / n_tokens + true_linf

        result = fit_uet_scaling(d_effs, hidden, losses, n_tokens)
        assert not np.isnan(result["c"])
        assert result["r_squared"] > 0.99
