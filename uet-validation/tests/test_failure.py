from __future__ import annotations

import numpy as np
import pytest

from uet.failure import FailureResult, run_single_failure, sweep_failure_modes


class TestRunSingleFailure:
    def test_returns_failure_result(self):
        result = run_single_failure(d=32, k=4, gap_ratio=5.0, n_samples=500, rng=np.random.default_rng(0))
        assert isinstance(result, FailureResult)
        assert 0.0 <= result.sin_angle <= 1.0
        assert result.d_eff > 0

    def test_k_ge_d(self):
        result = run_single_failure(d=10, k=10, gap_ratio=5.0, n_samples=500)
        assert result.sin_angle == 1.0
        assert result.condition_violated == "k>=d"

    def test_good_conditions_low_sin(self):
        result = run_single_failure(
            d=128, k=4, gap_ratio=10.0, n_samples=5000,
            signal_strength=10.0, rng=np.random.default_rng(42),
        )
        assert result.sin_angle < 0.5
        assert result.condition_violated == "none"

    def test_bad_gap_high_sin(self):
        result = run_single_failure(
            d=64, k=4, gap_ratio=0.5, n_samples=2000,
            rng=np.random.default_rng(42),
        )
        assert result.condition_violated != "none"


class TestSweepFailureModes:
    def test_correct_count(self):
        results = sweep_failure_modes(
            d_values=[16, 32], k_values=[2, 4], gap_values=[1.0, 5.0],
            n_samples=200, n_seeds=2,
        )
        assert len(results) == 2 * 2 * 2 * 2  # seeds * d * k * gap

    def test_all_failure_results(self):
        results = sweep_failure_modes(
            d_values=[32], k_values=[4], gap_values=[5.0],
            n_samples=200, n_seeds=1,
        )
        assert all(isinstance(r, FailureResult) for r in results)
