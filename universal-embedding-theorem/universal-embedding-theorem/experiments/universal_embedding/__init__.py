from .metrics import (
    effective_dimension,
    empirical_covariance,
    estimate_top_pca_basis,
    largest_principal_angle_sine,
    spectral_gap,
    top_eigenpairs,
)
from .recovery import (
    gaussian_measurements,
    orthogonal_matching_pursuit,
    relative_l2_error,
    sample_sparse_signal,
    support_recall,
)
from .regression import evaluate_minimum_norm_regression, mean_squared_error, minimum_norm_weights
from .superposition import (
    decode_superposition,
    encode_superposition,
    evaluate_superposition,
    feature_count_for_alpha,
    max_pairwise_coherence,
    sample_feature_dictionary,
    sample_sparse_coefficients,
)
from .synthetic_data import (
    GaussianEmbeddingProblem,
    SparseRegressionProblem,
    append_noise_dimensions,
    generate_gaussian_embedding_problem,
    generate_sparse_regression_problem,
)

__all__ = [
    "GaussianEmbeddingProblem",
    "SparseRegressionProblem",
    "append_noise_dimensions",
    "effective_dimension",
    "empirical_covariance",
    "estimate_top_pca_basis",
    "evaluate_minimum_norm_regression",
    "evaluate_superposition",
    "feature_count_for_alpha",
    "gaussian_measurements",
    "generate_gaussian_embedding_problem",
    "generate_sparse_regression_problem",
    "largest_principal_angle_sine",
    "max_pairwise_coherence",
    "mean_squared_error",
    "minimum_norm_weights",
    "decode_superposition",
    "encode_superposition",
    "orthogonal_matching_pursuit",
    "relative_l2_error",
    "sample_feature_dictionary",
    "sample_sparse_signal",
    "sample_sparse_coefficients",
    "spectral_gap",
    "support_recall",
    "top_eigenpairs",
]
