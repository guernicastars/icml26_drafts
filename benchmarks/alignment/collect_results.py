"""Aggregate alignment experiment results across models and seeds."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from meta_swag.statistics.tests import paired_wilcoxon, cluster_bootstrap_ci


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/alignment")
    parser.add_argument("--output-dir", default="aggregates/alignment")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bon_frames = []
    summary_frames = []

    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        bon_path = exp_dir / "best_of_n.csv"
        summary_path = exp_dir / "summary.csv"

        if bon_path.exists():
            df = pd.read_csv(bon_path)
            df["experiment"] = exp_dir.name
            bon_frames.append(df)
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            df["experiment"] = exp_dir.name
            summary_frames.append(df)

    if not bon_frames:
        print("No results found.")
        return

    bon_df = pd.concat(bon_frames, ignore_index=True)
    summary_df = pd.concat(summary_frames, ignore_index=True)

    bon_agg = (
        bon_df.groupby(["experiment", "scheme", "n"])
        .agg(
            gold_reward_mean=("gold_reward_mean", "mean"),
            gold_reward_std=("gold_reward_mean", "std"),
            proxy_reward_mean=("proxy_reward_mean", "mean"),
            gap_mean=("gap", "mean"),
            gap_std=("gap", "std"),
        )
        .reset_index()
    )
    bon_agg.to_csv(output_dir / "bon_aggregated.csv", index=False)

    summary_agg = (
        summary_df.groupby(["experiment", "scheme"])
        .agg(
            ess_mean=("ess", "mean"),
            max_weight_mean=("max_weight", "mean"),
            posterior_trace_mean=("posterior_trace", "mean"),
            top_eigenvalue_ratio_mean=("top_eigenvalue_ratio", "mean"),
            gold_n1_mean=("gold_reward_n1", "mean"),
            gold_n256_mean=("gold_reward_n256", "mean"),
            overopt_gap_mean=("overopt_gap", "mean"),
            overopt_gap_std=("overopt_gap", "std"),
        )
        .reset_index()
    )
    summary_agg.to_csv(output_dir / "summary_aggregated.csv", index=False)

    test_rows = []
    for experiment in summary_df["experiment"].unique():
        exp_df = summary_df[summary_df["experiment"] == experiment]
        baseline_scheme = "last_iterate"
        baseline = exp_df[exp_df["scheme"] == baseline_scheme]["overopt_gap"].values
        for scheme in exp_df["scheme"].unique():
            if scheme == baseline_scheme:
                continue
            scheme_vals = exp_df[exp_df["scheme"] == scheme]["overopt_gap"].values
            min_len = min(len(baseline), len(scheme_vals))
            if min_len < 3:
                continue
            result = paired_wilcoxon(scheme_vals[:min_len], baseline[:min_len])
            test_rows.append({
                "experiment": experiment,
                "scheme": scheme,
                "baseline": baseline_scheme,
                "metric": "overopt_gap",
                **result,
            })

    if test_rows:
        pd.DataFrame(test_rows).to_csv(output_dir / "paired_tests.csv", index=False)

    print(f"Aggregated results saved to {output_dir}")
    print(summary_agg.to_string(index=False))


if __name__ == "__main__":
    main()
