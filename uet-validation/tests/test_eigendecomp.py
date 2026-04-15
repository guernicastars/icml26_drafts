from __future__ import annotations

import numpy as np
import pytest

from uet.eigendecomp import (
    covariance,
    effective_dimension,
    eigenspectrum,
    pca_alignment_sin,
    participation_ratio,
    spectral_gap,
    spectral_gap_ratio,
    stable_rank,
    theorem_42_bound,
    top_eigenvectors,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sample_data(rng):
    n, d = 500, 10
    return rng.normal(size=(n, d))


@pytest.fixture
def known_cov():
    evals = np.array([10.0, 5.0, 2.0, 1.0, 0.1])
    return np.diag(evals), evals


class TestCovariance:
    def test_shape(self, sample_data):
        cov = covariance(sample_data)
        assert cov.shape == (10, 10)

    def test_symmetric(self, sample_data):
        cov = covariance(sample_data)
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)

    def test_positive_semidefinite(self, sample_data):
        cov = covariance(sample_data)
        evals = np.linalg.eigvalsh(cov)
        assert np.all(evals >= -1e-10)


class TestEigenspectrum:
    def test_sorted_descending(self, sample_data):
        cov = covariance(sample_data)
        evals = eigenspectrum(cov)
        assert np.all(np.diff(evals) <= 1e-10)

    def test_matches_known(self, known_cov):
        diag_cov, expected = known_cov
        evals = eigenspectrum(diag_cov)
        np.testing.assert_allclose(evals, expected, atol=1e-10)


class TestEffectiveDimension:
    def test_identity_gives_d(self):
        evals = np.ones(20)
        assert effective_dimension(evals) == pytest.approx(20.0)

    def test_single_spike(self):
        evals = np.array([100.0, 0.0, 0.0])
        assert effective_dimension(evals) == pytest.approx(1.0)

    def test_empty(self):
        assert effective_dimension(np.array([])) == 0.0

    def test_alias(self):
        evals = np.array([5.0, 3.0, 1.0])
        assert participation_ratio(evals) == effective_dimension(evals)


class TestStableRank:
    def test_identity(self):
        evals = np.ones(10)
        assert stable_rank(evals) == pytest.approx(10.0)

    def test_single_spike(self):
        evals = np.array([100.0, 1.0, 1.0])
        assert stable_rank(evals) == pytest.approx(102.0 / 100.0)


class TestSpectralGap:
    def test_known_gap(self):
        evals = np.array([10.0, 5.0, 2.0])
        assert spectral_gap(evals, 1) == pytest.approx(5.0)
        assert spectral_gap(evals, 2) == pytest.approx(3.0)

    def test_ratio(self):
        evals = np.array([10.0, 2.0, 1.0])
        assert spectral_gap_ratio(evals, 1) == pytest.approx(5.0)

    def test_out_of_bounds(self):
        evals = np.array([10.0, 5.0])
        with pytest.raises(ValueError):
            spectral_gap(evals, 0)
        with pytest.raises(ValueError):
            spectral_gap(evals, 2)


class TestPCAAlignment:
    def test_identical_subspaces(self):
        V = np.eye(5, 2)
        assert pca_alignment_sin(V, V) == pytest.approx(0.0, abs=1e-10)

    def test_orthogonal_subspaces(self):
        V1 = np.eye(4, 2)
        V2 = np.eye(4, 2, k=2)
        assert pca_alignment_sin(V1, V2) == pytest.approx(1.0, abs=1e-10)

    def test_bounded_01(self, rng):
        V1 = np.linalg.qr(rng.normal(size=(10, 3)))[0][:, :3]
        V2 = np.linalg.qr(rng.normal(size=(10, 3)))[0][:, :3]
        sin_val = pca_alignment_sin(V1, V2)
        assert 0.0 <= sin_val <= 1.0


class TestTopEigenvectors:
    def test_shape_and_order(self, known_cov):
        diag_cov, expected = known_cov
        vecs, vals = top_eigenvectors(diag_cov, 3)
        assert vecs.shape == (5, 3)
        assert vals[0] >= vals[1] >= vals[2]

    def test_top_k_eigenvalues(self, known_cov):
        diag_cov, expected = known_cov
        _, vals = top_eigenvectors(diag_cov, 2)
        np.testing.assert_allclose(vals[:2], expected[:2], atol=1e-10)


class TestTheorem42Bound:
    def test_positive_gap(self):
        bound = theorem_42_bound(0.1, 1.0, 10.0, 2.0)
        assert bound == pytest.approx(0.1 / 8.0)

    def test_zero_gap(self):
        assert theorem_42_bound(0.1, 1.0, 5.0, 5.0) == float("inf")

    def test_negative_gap(self):
        assert theorem_42_bound(0.1, 1.0, 3.0, 5.0) == float("inf")
