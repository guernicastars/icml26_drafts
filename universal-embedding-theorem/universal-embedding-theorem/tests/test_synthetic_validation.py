from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import numpy.testing as npt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from universal_embedding.metrics import effective_dimension, largest_principal_angle_sine  # noqa: E402
from universal_embedding.recovery import (  # noqa: E402
    gaussian_measurements,
    orthogonal_matching_pursuit,
    relative_l2_error,
    sample_sparse_signal,
    support_recall,
)
from universal_embedding.regression import mean_squared_error, minimum_norm_weights  # noqa: E402
from universal_embedding.superposition import (  # noqa: E402
    decode_superposition,
    encode_superposition,
    feature_count_for_alpha,
    max_pairwise_coherence,
    sample_feature_dictionary,
    sample_sparse_coefficients,
)
from universal_embedding.synthetic_data import (  # noqa: E402
    append_noise_dimensions,
    generate_gaussian_embedding_problem,
    generate_sparse_regression_problem,
)


def test_generated_embedding_problem_has_expected_shapes() -> None:
    rng = np.random.default_rng(0)
    problem = generate_gaussian_embedding_problem(
        num_samples=128,
        ambient_dim=32,
        causal_dim=4,
        nuisance_rank=8,
        rng=rng,
    )
    assert problem.samples.shape == (128, 32)
    assert problem.causal_basis.shape == (32, 4)
    assert problem.nuisance_basis.shape == (32, 8)
    assert problem.covariance.shape == (32, 32)
    assert effective_dimension(problem.covariance) > 1.0


def test_largest_principal_angle_is_zero_for_identical_subspaces() -> None:
    basis = np.eye(6, dtype=np.float64)[:, :3]
    assert np.isclose(largest_principal_angle_sine(basis, basis), 0.0)


def test_omp_recovers_noiseless_sparse_signal() -> None:
    rng = np.random.default_rng(1)
    truth, support = sample_sparse_signal(ambient_dim=64, sparsity=4, rng=rng)
    sensing, targets = gaussian_measurements(truth, num_measurements=48, noise_std=0.0, rng=rng)
    estimate, estimated_support = orthogonal_matching_pursuit(sensing, targets, sparsity=4)
    assert relative_l2_error(truth, estimate) < 1e-8
    assert support_recall(support, estimated_support) == 1.0


def test_append_noise_dimensions_keeps_original_block() -> None:
    rng = np.random.default_rng(2)
    samples = rng.normal(size=(10, 5))
    augmented = append_noise_dimensions(samples, extra_dims=3, noise_std=0.1, rng=rng)
    assert augmented.shape == (10, 8)
    npt.assert_allclose(augmented[:, :5], samples)


def test_sparse_regression_problem_has_low_effective_dimension() -> None:
    rng = np.random.default_rng(3)
    problem = generate_sparse_regression_problem(
        num_train=64,
        num_test=64,
        ambient_dim=32,
        causal_dim=4,
        signal_strength=1.0,
        nuisance_strength=0.01,
        rng=rng,
    )
    assert effective_dimension(problem.covariance) < 5.0


def test_minimum_norm_regression_fits_synthetic_problem() -> None:
    rng = np.random.default_rng(4)
    problem = generate_sparse_regression_problem(
        num_train=256,
        num_test=512,
        ambient_dim=32,
        causal_dim=4,
        response_noise=0.01,
        rng=rng,
    )
    weights = minimum_norm_weights(problem.x_train, problem.y_train)
    train_mse = mean_squared_error(problem.y_train, problem.x_train @ weights)
    test_mse = mean_squared_error(problem.y_test, problem.x_test @ weights)
    assert train_mse < 0.02
    assert test_mse < 0.2


def test_superposition_dictionary_rows_are_unit_norm() -> None:
    rng = np.random.default_rng(5)
    dictionary = sample_feature_dictionary(embedding_dim=16, num_features=64, rng=rng)
    norms = np.linalg.norm(dictionary, axis=1)
    npt.assert_allclose(norms, np.ones_like(norms), atol=1e-8)


def test_superposition_exact_recovery_for_orthonormal_dictionary() -> None:
    dictionary = np.eye(8, dtype=np.float64)
    coefficients = np.zeros(8, dtype=np.float64)
    coefficients[[1, 5]] = [1.0, -1.0]
    activation = encode_superposition(dictionary, coefficients)
    decoded = decode_superposition(dictionary, activation)
    npt.assert_allclose(decoded, coefficients)
    assert np.isclose(max_pairwise_coherence(dictionary), 0.0)


def test_feature_count_for_alpha_respects_lower_and_upper_bounds() -> None:
    count = feature_count_for_alpha(embedding_dim=32, alpha=0.2, max_features=200)
    assert count >= 64
    assert count <= 200


def test_sample_sparse_coefficients_has_requested_support_size() -> None:
    rng = np.random.default_rng(6)
    coefficients, support = sample_sparse_coefficients(num_features=32, sparsity=5, rng=rng)
    assert len(support) == 5
    assert np.count_nonzero(coefficients) == 5
