from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from universal_embedding.metrics import effective_dimension, empirical_covariance, estimate_top_pca_basis, largest_principal_angle_sine, spectral_gap
from universal_embedding.recovery import (
    gaussian_measurements,
    orthogonal_matching_pursuit,
    relative_l2_error,
    sample_sparse_signal,
    support_recall,
)
from universal_embedding.regression import evaluate_minimum_norm_regression
from universal_embedding.superposition import evaluate_superposition, feature_count_for_alpha, sample_feature_dictionary
from universal_embedding.synthetic_data import append_noise_dimensions, generate_gaussian_embedding_problem, generate_sparse_regression_problem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic validation experiments for the Universal Embedding Theorem.")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/artifacts"))
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--dims", nargs="+", type=int, default=[32, 128, 512])
    parser.add_argument("--causal-dims", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[64, 256, 1024])
    parser.add_argument("--measurement-factors", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--noise-augment-dims", nargs="+", type=int, default=[0, 16, 64, 256])
    parser.add_argument("--signal-strength", type=float, default=4.0)
    parser.add_argument("--nuisance-strength", type=float, default=0.5)
    parser.add_argument("--isotropic-noise", type=float, default=0.05)
    parser.add_argument("--measurement-noise", type=float, default=0.01)
    parser.add_argument("--response-noise", type=float, default=0.1)
    parser.add_argument("--test-size", type=int, default=2048)
    parser.add_argument("--recovery-tolerance", type=float, default=0.1)
    parser.add_argument("--superposition-alphas", nargs="+", type=float, default=[0.10, 0.15, 0.20])
    parser.add_argument("--superposition-sparsities", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--superposition-max-features", type=int, default=2048)
    parser.add_argument("--superposition-trials", type=int, default=128)
    return parser.parse_args()


def _seed(*values: int) -> int:
    seed = 17
    for value in values:
        seed = (seed * 1315423911 + value) % (2**32 - 1)
    return seed


def _measurement_budget(ambient_dim: int, causal_dim: int, factor: float) -> int:
    baseline = causal_dim * math.log(max(ambient_dim / max(causal_dim, 1), 2.0))
    return max(causal_dim, math.ceil(factor * baseline))


def run_pca_alignment(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for ambient_dim in args.dims:
        for causal_dim in args.causal_dims:
            if causal_dim >= ambient_dim:
                continue
            for num_samples in args.sample_sizes:
                for seed in range(args.seeds):
                    rng = np.random.default_rng(_seed(seed, ambient_dim, causal_dim, num_samples, 1))
                    problem = generate_gaussian_embedding_problem(
                        num_samples=num_samples,
                        ambient_dim=ambient_dim,
                        causal_dim=causal_dim,
                        signal_strength=args.signal_strength,
                        nuisance_strength=args.nuisance_strength,
                        isotropic_noise=args.isotropic_noise,
                        rng=rng,
                    )
                    estimated_basis, _ = estimate_top_pca_basis(problem.samples, causal_dim)
                    sample_covariance = empirical_covariance(problem.samples)
                    rows.append(
                        {
                            "ambient_dim": ambient_dim,
                            "causal_dim": causal_dim,
                            "num_samples": num_samples,
                            "seed": seed,
                            "alignment_error": largest_principal_angle_sine(problem.causal_basis, estimated_basis),
                            "effective_dimension": effective_dimension(sample_covariance),
                            "spectral_gap": spectral_gap(sample_covariance, causal_dim),
                        }
                    )
    return pd.DataFrame(rows)


def run_sparse_recovery(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for ambient_dim in args.dims:
        for causal_dim in args.causal_dims:
            if causal_dim >= ambient_dim:
                continue
            baseline = causal_dim * math.log(max(ambient_dim / causal_dim, 2.0))
            for factor in args.measurement_factors:
                num_measurements = _measurement_budget(ambient_dim, causal_dim, factor)
                for seed in range(args.seeds):
                    rng = np.random.default_rng(_seed(seed, ambient_dim, causal_dim, num_measurements, 2))
                    truth, support = sample_sparse_signal(ambient_dim, causal_dim, rng)
                    sensing, targets = gaussian_measurements(truth, num_measurements, args.measurement_noise, rng)
                    estimate, estimated_support = orthogonal_matching_pursuit(sensing, targets, causal_dim)
                    error = relative_l2_error(truth, estimate)
                    rows.append(
                        {
                            "ambient_dim": ambient_dim,
                            "causal_dim": causal_dim,
                            "measurement_factor": factor,
                            "num_measurements": num_measurements,
                            "normalized_measurements": num_measurements / max(baseline, 1.0),
                            "seed": seed,
                            "relative_error": error,
                            "support_recall": support_recall(support, estimated_support),
                            "recovered": float(error <= args.recovery_tolerance),
                        }
                    )
    return pd.DataFrame(rows)


def run_double_descent(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for ambient_dim in args.dims:
        for causal_dim in args.causal_dims:
            if causal_dim >= ambient_dim:
                continue
            for num_samples in args.sample_sizes:
                for seed in range(args.seeds):
                    rng = np.random.default_rng(_seed(seed, ambient_dim, causal_dim, num_samples, 3))
                    problem = generate_sparse_regression_problem(
                        num_train=num_samples,
                        num_test=args.test_size,
                        ambient_dim=ambient_dim,
                        causal_dim=causal_dim,
                        signal_strength=1.0,
                        nuisance_strength=0.02,
                        response_noise=args.response_noise,
                        rng=rng,
                    )
                    metrics = evaluate_minimum_norm_regression(problem)
                    rows.append(
                        {
                            "ambient_dim": ambient_dim,
                            "causal_dim": causal_dim,
                            "num_samples": num_samples,
                            "model_to_sample_ratio": ambient_dim / num_samples,
                            "seed": seed,
                            "effective_dimension": effective_dimension(problem.covariance),
                            **metrics,
                        }
                    )
    return pd.DataFrame(rows)


def run_noise_augmentation(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    reference_samples = max(args.sample_sizes)
    for ambient_dim in args.dims:
        for causal_dim in args.causal_dims:
            if causal_dim >= ambient_dim:
                continue
            for seed in range(args.seeds):
                base_rng = np.random.default_rng(_seed(seed, ambient_dim, causal_dim, reference_samples, 4))
                problem = generate_gaussian_embedding_problem(
                    num_samples=reference_samples,
                    ambient_dim=ambient_dim,
                    causal_dim=causal_dim,
                    signal_strength=args.signal_strength,
                    nuisance_strength=args.nuisance_strength,
                    isotropic_noise=args.isotropic_noise,
                    rng=base_rng,
                )
                base_basis, _ = estimate_top_pca_basis(problem.samples, causal_dim)
                base_covariance = empirical_covariance(problem.samples)
                base_error = largest_principal_angle_sine(problem.causal_basis, base_basis)
                base_effective_dimension = effective_dimension(base_covariance)
                for extra_dims in args.noise_augment_dims:
                    noise_rng = np.random.default_rng(_seed(seed, ambient_dim, causal_dim, extra_dims, 5))
                    augmented_samples = append_noise_dimensions(
                        problem.samples,
                        extra_dims=extra_dims,
                        noise_std=math.sqrt(args.isotropic_noise),
                        rng=noise_rng,
                    )
                    augmented_basis, _ = estimate_top_pca_basis(augmented_samples, causal_dim)
                    augmented_causal_basis = np.vstack(
                        [
                            problem.causal_basis,
                            np.zeros((extra_dims, causal_dim), dtype=np.float64),
                        ]
                    )
                    augmented_covariance = empirical_covariance(augmented_samples)
                    augmented_error = largest_principal_angle_sine(augmented_causal_basis, augmented_basis)
                    rows.append(
                        {
                            "ambient_dim": ambient_dim,
                            "causal_dim": causal_dim,
                            "extra_dims": extra_dims,
                            "seed": seed,
                            "base_alignment_error": base_error,
                            "augmented_alignment_error": augmented_error,
                            "alignment_delta": augmented_error - base_error,
                            "base_effective_dimension": base_effective_dimension,
                            "augmented_effective_dimension": effective_dimension(augmented_covariance),
                        }
                    )
    return pd.DataFrame(rows)


def run_superposition(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for embedding_dim in args.dims:
        for sparsity in args.superposition_sparsities:
            if sparsity > embedding_dim:
                continue
            for alpha in args.superposition_alphas:
                num_features = feature_count_for_alpha(
                    embedding_dim=embedding_dim,
                    alpha=alpha,
                    max_features=args.superposition_max_features,
                )
                if num_features <= embedding_dim:
                    continue
                for seed in range(args.seeds):
                    rng = np.random.default_rng(_seed(seed, embedding_dim, sparsity, int(alpha * 1000), 6))
                    dictionary = sample_feature_dictionary(
                        embedding_dim=embedding_dim,
                        num_features=num_features,
                        rng=rng,
                    )
                    metrics = evaluate_superposition(
                        dictionary=dictionary,
                        sparsity=sparsity,
                        trials=args.superposition_trials,
                        rng=rng,
                    )
                    rows.append(
                        {
                            "embedding_dim": embedding_dim,
                            "sparsity": sparsity,
                            "alpha": alpha,
                            "seed": seed,
                            **metrics,
                        }
                    )
    return pd.DataFrame(rows)


def make_plots(
    pca_df: pd.DataFrame,
    recovery_df: pd.DataFrame,
    double_df: pd.DataFrame,
    noise_df: pd.DataFrame,
    superposition_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 13))

    if not pca_df.empty:
        focus_causal_dim = int(pca_df["causal_dim"].min())
        grouped = (
            pca_df[pca_df["causal_dim"] == focus_causal_dim]
            .groupby(["ambient_dim", "num_samples"], as_index=False)
            .mean(numeric_only=True)
            .sort_values(["ambient_dim", "num_samples"])
        )
        for ambient_dim, subset in grouped.groupby("ambient_dim"):
            axes[0, 0].plot(subset["num_samples"], subset["alignment_error"], marker="o", label=f"d={ambient_dim}")
        axes[0, 0].set_xscale("log")
        axes[0, 0].set_xlabel("Samples")
        axes[0, 0].set_ylabel("sin largest principal angle")
        axes[0, 0].set_title(f"PCA recovery (k={focus_causal_dim})")
        axes[0, 0].legend(fontsize=8)

    if not recovery_df.empty:
        grouped = (
            recovery_df.groupby("measurement_factor", as_index=False)
            .mean(numeric_only=True)
            .sort_values("measurement_factor")
        )
        axes[0, 1].plot(grouped["measurement_factor"], grouped["recovered"], marker="o", label="success rate")
        axes[0, 1].plot(grouped["measurement_factor"], grouped["support_recall"], marker="s", label="support recall")
        axes[0, 1].set_xlabel("Measurement factor vs k log(d / k)")
        axes[0, 1].set_ylabel("Mean score")
        axes[0, 1].set_title("Sparse recovery scaling")
        axes[0, 1].legend(fontsize=8)

    if not double_df.empty:
        focus_causal_dim = int(double_df["causal_dim"].min())
        grouped = (
            double_df[double_df["causal_dim"] == focus_causal_dim]
            .groupby("model_to_sample_ratio", as_index=False)
            .mean(numeric_only=True)
            .sort_values("model_to_sample_ratio")
        )
        axes[1, 0].plot(grouped["model_to_sample_ratio"], grouped["test_mse"], marker="o", label="test")
        axes[1, 0].plot(grouped["model_to_sample_ratio"], grouped["train_mse"], marker="s", label="train")
        axes[1, 0].axvline(1.0, linestyle="--", color="gray", linewidth=1)
        axes[1, 0].set_xlabel("d / n")
        axes[1, 0].set_ylabel("MSE")
        axes[1, 0].set_title(f"Double descent proxy (k={focus_causal_dim})")
        axes[1, 0].legend(fontsize=8)

    if not noise_df.empty:
        focus_causal_dim = int(noise_df["causal_dim"].min())
        grouped = (
            noise_df[noise_df["causal_dim"] == focus_causal_dim]
            .groupby("extra_dims", as_index=False)
            .mean(numeric_only=True)
            .sort_values("extra_dims")
        )
        axes[1, 1].plot(grouped["extra_dims"], grouped["alignment_delta"], marker="o")
        axes[1, 1].axhline(0.0, linestyle="--", color="gray", linewidth=1)
        axes[1, 1].set_xlabel("Extra appended noise dimensions")
        axes[1, 1].set_ylabel("Augmented minus base alignment error")
        axes[1, 1].set_title(f"Noise-dimension augmentation (k={focus_causal_dim})")

    if not superposition_df.empty:
        focus_sparsity = int(superposition_df["sparsity"].min())
        grouped = (
            superposition_df[superposition_df["sparsity"] == focus_sparsity]
            .groupby(["embedding_dim", "num_features"], as_index=False)
            .mean(numeric_only=True)
            .sort_values(["embedding_dim", "num_features"])
        )
        for embedding_dim, subset in grouped.groupby("embedding_dim"):
            axes[2, 0].plot(
                subset["feature_to_dimension_ratio"],
                subset["active_mae"],
                marker="o",
                label=f"d={embedding_dim}",
            )
        axes[2, 0].set_xlabel("m / d")
        axes[2, 0].set_ylabel("Active coefficient MAE")
        axes[2, 0].set_title(f"Superposition decoding (s={focus_sparsity})")
        axes[2, 0].legend(fontsize=8)

        grouped_bound = (
            superposition_df.groupby(["embedding_dim", "sparsity"], as_index=False)
            .mean(numeric_only=True)
            .sort_values(["embedding_dim", "sparsity"])
        )
        for sparsity, subset in grouped_bound.groupby("sparsity"):
            axes[2, 1].plot(
                subset["embedding_dim"],
                subset["active_mae_to_scale"],
                marker="o",
                label=f"s={sparsity}",
            )
        axes[2, 1].axhline(1.0, linestyle="--", color="gray", linewidth=1)
        axes[2, 1].set_xscale("log")
        axes[2, 1].set_xlabel("Embedding dimension d")
        axes[2, 1].set_ylabel("Active MAE / [s sqrt(log m / d)]")
        axes[2, 1].set_title("Superposition scaling vs theorem rate")
        axes[2, 1].legend(fontsize=8)
    else:
        axes[2, 0].axis("off")
        axes[2, 1].axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "synthetic_validation_overview.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pca_df = run_pca_alignment(args)
    recovery_df = run_sparse_recovery(args)
    double_df = run_double_descent(args)
    noise_df = run_noise_augmentation(args)
    superposition_df = run_superposition(args)

    pca_df.to_csv(args.output_dir / "pca_alignment.csv", index=False)
    recovery_df.to_csv(args.output_dir / "sparse_recovery.csv", index=False)
    double_df.to_csv(args.output_dir / "double_descent.csv", index=False)
    noise_df.to_csv(args.output_dir / "noise_augmentation.csv", index=False)
    superposition_df.to_csv(args.output_dir / "superposition.csv", index=False)

    pca_summary = pca_df.groupby(["ambient_dim", "causal_dim", "num_samples"], as_index=False).mean(numeric_only=True)
    recovery_summary = recovery_df.groupby(["ambient_dim", "causal_dim", "measurement_factor"], as_index=False).mean(numeric_only=True)
    double_summary = double_df.groupby(["ambient_dim", "causal_dim", "num_samples"], as_index=False).mean(numeric_only=True)
    noise_summary = noise_df.groupby(["ambient_dim", "causal_dim", "extra_dims"], as_index=False).mean(numeric_only=True)
    superposition_summary = (
        superposition_df.groupby(["embedding_dim", "sparsity", "alpha", "num_features"], as_index=False).mean(numeric_only=True)
        if not superposition_df.empty
        else pd.DataFrame()
    )

    pca_summary.to_csv(args.output_dir / "pca_alignment_summary.csv", index=False)
    recovery_summary.to_csv(args.output_dir / "sparse_recovery_summary.csv", index=False)
    double_summary.to_csv(args.output_dir / "double_descent_summary.csv", index=False)
    noise_summary.to_csv(args.output_dir / "noise_augmentation_summary.csv", index=False)
    superposition_summary.to_csv(args.output_dir / "superposition_summary.csv", index=False)

    make_plots(pca_df, recovery_df, double_df, noise_df, superposition_df, args.output_dir)

    print("PCA alignment summary:")
    print(pca_summary.to_string(index=False))
    print("\nSparse recovery summary:")
    print(recovery_summary.to_string(index=False))
    print("\nDouble descent summary:")
    print(double_summary.to_string(index=False))
    print("\nNoise augmentation summary:")
    print(noise_summary.to_string(index=False))
    print("\nSuperposition summary:")
    if superposition_summary.empty:
        print("No superposition configurations were valid for this run.")
    else:
        print(superposition_summary.to_string(index=False))
    print(f"\nSaved artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
