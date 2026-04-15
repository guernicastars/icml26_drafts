from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a LaTeX empirical note from synthetic validation artifacts.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("experiments/artifacts"))
    parser.add_argument("--output-tex", type=Path, default=Path("paper/generated_results.tex"))
    parser.add_argument("--figure-path", type=str, default="../experiments/artifacts/synthetic_validation_overview.png")
    return parser.parse_args()


def _format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _top_rows(df: pd.DataFrame, columns: list[str], limit: int = 6) -> list[dict[str, object]]:
    if df.empty:
        return []
    return df.loc[:, columns].head(limit).to_dict(orient="records")


def _latex_table(rows: list[dict[str, object]], headers: list[tuple[str, str]], caption: str, label: str) -> str:
    if not rows:
        return "\\paragraph{Missing artifact.} The requested summary table was not available.\n"

    column_spec = "l" * len(headers)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        " & ".join(header for _, header in headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        formatted = []
        for key, _ in headers:
            value = row[key]
            if isinstance(value, float):
                formatted.append(_format_float(value))
            else:
                formatted.append(str(value))
        lines.append(" & ".join(formatted) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def best_sparse_recovery_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    if df.empty:
        return []
    rows: list[dict[str, object]] = []
    grouped = df.sort_values(["ambient_dim", "causal_dim", "measurement_factor"]).groupby(["ambient_dim", "causal_dim"])
    for (ambient_dim, causal_dim), sub in grouped:
        success = sub[sub["recovered"] >= 0.95]
        chosen = success.iloc[0] if not success.empty else sub.iloc[-1]
        rows.append(
            {
                "ambient_dim": ambient_dim,
                "causal_dim": causal_dim,
                "measurement_factor": float(chosen["measurement_factor"]),
                "num_measurements": float(chosen["num_measurements"]),
                "relative_error": float(chosen["relative_error"]),
                "support_recall": float(chosen["support_recall"]),
            }
        )
    return rows[:6]


def summarize_double_descent(df: pd.DataFrame) -> list[dict[str, object]]:
    if df.empty:
        return []
    rows: list[dict[str, object]] = []
    grouped = df.groupby(["ambient_dim", "causal_dim"])
    for (ambient_dim, causal_dim), sub in grouped:
        peak = sub.loc[sub["test_mse"].idxmax()]
        best = sub.loc[sub["test_mse"].idxmin()]
        rows.append(
            {
                "ambient_dim": ambient_dim,
                "causal_dim": causal_dim,
                "peak_ratio": float(peak["model_to_sample_ratio"]),
                "peak_test_mse": float(peak["test_mse"]),
                "best_ratio": float(best["model_to_sample_ratio"]),
                "best_test_mse": float(best["test_mse"]),
            }
        )
    return rows[:6]


def render_note(args: argparse.Namespace) -> str:
    pca = pd.read_csv(args.artifacts_dir / "pca_alignment_summary.csv")
    recovery = pd.read_csv(args.artifacts_dir / "sparse_recovery_summary.csv")
    double = pd.read_csv(args.artifacts_dir / "double_descent_summary.csv")
    noise = pd.read_csv(args.artifacts_dir / "noise_augmentation_summary.csv")
    superposition = pd.read_csv(args.artifacts_dir / "superposition_summary.csv")

    best_pca = pca.sort_values(["alignment_error", "ambient_dim", "causal_dim", "num_samples"])
    best_noise = noise.sort_values(["alignment_delta", "ambient_dim", "causal_dim", "extra_dims"])
    best_superposition = superposition.sort_values(
        ["active_mae_to_scale", "embedding_dim", "sparsity", "num_features"]
    )

    mean_alignment = float(pca["alignment_error"].mean())
    mean_superposition_ratio = float(superposition["active_mae_to_scale"].mean())
    mean_noise_delta = float(noise["alignment_delta"].mean())
    recovery_by_group = recovery.groupby(["ambient_dim", "causal_dim"])
    first_success_factors = []
    for _, sub in recovery_by_group:
        success = sub[sub["recovered"] >= 0.95].sort_values("measurement_factor")
        if not success.empty:
            first_success_factors.append(float(success.iloc[0]["measurement_factor"]))
    recovery_coverage = float(len(first_success_factors) / max(len(recovery_by_group), 1))
    median_success_factor = float(pd.Series(first_success_factors).median()) if first_success_factors else float("nan")
    double_group_rows = summarize_double_descent(double)
    inflation_ratios = [
        float(row["peak_test_mse"] / max(float(row["best_test_mse"]), 1e-12))
        for row in double_group_rows
    ]
    median_peak_inflation = float(pd.Series(inflation_ratios).median()) if inflation_ratios else float("nan")

    pca_table = _latex_table(
        _top_rows(best_pca, ["ambient_dim", "causal_dim", "num_samples", "alignment_error", "effective_dimension", "spectral_gap"]),
        [
            ("ambient_dim", "$d$"),
            ("causal_dim", "$k$"),
            ("num_samples", "$n$"),
            ("alignment_error", "alignment"),
            ("effective_dimension", "$d_{\\mathrm{eff}}$"),
            ("spectral_gap", "gap"),
        ],
        "Best PCA alignment configurations from the current synthetic sweep.",
        "tab:pca_alignment",
    )
    recovery_table = _latex_table(
        best_sparse_recovery_rows(recovery),
        [
            ("ambient_dim", "$d$"),
            ("causal_dim", "$k$"),
            ("measurement_factor", "factor"),
            ("num_measurements", "$M$"),
            ("relative_error", "rel. error"),
            ("support_recall", "support"),
        ],
        "First sparse-recovery regime reaching near-perfect recovery for each $(d, k)$ pair.",
        "tab:sparse_recovery",
    )
    double_table = _latex_table(
        summarize_double_descent(double),
        [
            ("ambient_dim", "$d$"),
            ("causal_dim", "$k$"),
            ("peak_ratio", "peak $d/n$"),
            ("peak_test_mse", "peak MSE"),
            ("best_ratio", "best $d/n$"),
            ("best_test_mse", "best MSE"),
        ],
        "Double-descent proxy summary under minimum-norm regression.",
        "tab:double_descent",
    )
    superposition_table = _latex_table(
        _top_rows(
            best_superposition,
            ["embedding_dim", "sparsity", "alpha", "num_features", "feature_to_dimension_ratio", "active_mae", "active_mae_to_scale"],
        ),
        [
            ("embedding_dim", "$d$"),
            ("sparsity", "$s$"),
            ("alpha", "$\\alpha$"),
            ("num_features", "$m$"),
            ("feature_to_dimension_ratio", "$m/d$"),
            ("active_mae", "MAE"),
            ("active_mae_to_scale", "MAE / scale"),
        ],
        "Representative superposition configurations and their decoding error relative to the theorem rate.",
        "tab:superposition",
    )
    noise_table = _latex_table(
        _top_rows(
            best_noise[best_noise["extra_dims"] > 0],
            ["ambient_dim", "causal_dim", "extra_dims", "alignment_delta", "augmented_effective_dimension"],
        ),
        [
            ("ambient_dim", "$d$"),
            ("causal_dim", "$k$"),
            ("extra_dims", "$r$"),
            ("alignment_delta", "$\\Delta$ align."),
            ("augmented_effective_dimension", "$d_{\\mathrm{eff}}^{+}$"),
        ],
        "Noise-augmentation summary sorted by the change in alignment error.",
        "tab:noise_augmentation",
    )

    lines = [
        "% This file is generated by experiments/render_empirical_note.py.",
        "\\section{Current Synthetic Empirics}",
        "The current implementation operationalizes five claims from the draft: PCA-based causal subspace recovery, sparse gradient recovery, a minimum-norm double-descent proxy, appended-noise augmentation, and a random-feature superposition study.",
        "",
        "\\paragraph{Headline metrics.}",
        f"Across the current sweep, mean PCA alignment error is {_format_float(mean_alignment)}, {int(round(recovery_coverage * 100))}\\% of $(d, k)$ recovery groups reach near-perfect recovery within the tested factor range with median first-success factor {_format_float(median_success_factor)}, median peak-to-best test-loss inflation in the double-descent proxy is {_format_float(median_peak_inflation)}, mean superposition MAE ratio to the theorem scale is {_format_float(mean_superposition_ratio)}, and mean noise-augmentation alignment delta is {_format_float(mean_noise_delta)}.",
        "",
        "\\paragraph{Interpretation.}",
        "The strongest signal at this stage is that sparse recovery turns on at a small constant multiple of $k \\log(d / k)$, while the minimum-norm regression proxy exhibits the expected interpolation spike near $d \\approx n$. The superposition study is now in place and already measures error relative to the predicted $s\\sqrt{\\log m / d}$ rate, and the current runs sit comfortably below that scale. The appended-noise study currently behaves conservatively in finite samples: we see tiny positive alignment deltas rather than the population-level improvement conjectured in the draft, which is exactly the sort of gap worth tightening in the next round.",
        "",
        pca_table,
        recovery_table,
        double_table,
        superposition_table,
        noise_table,
        "\\begin{figure}[t]",
        "\\centering",
        f"\\IfFileExists{{{args.figure_path}}}{{\\includegraphics[width=0.98\\linewidth]{{{args.figure_path}}}}}{{\\fbox{{Figure not found: {args.figure_path}}}}}",
        "\\caption{Overview plot emitted by the synthetic validation runner.}",
        "\\label{fig:synthetic_overview}",
        "\\end{figure}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text(render_note(args))
    print(f"Wrote {args.output_tex}")


if __name__ == "__main__":
    main()
