"""Aggregate MMLU results across models."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/mmlu")
    parser.add_argument("--output-dir", default="aggregates/mmlu")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        results_path = model_dir / "mmlu_results.csv"
        if results_path.exists():
            df = pd.read_csv(results_path)
            df["model"] = model_dir.name
            frames.append(df)

    if not frames:
        print("No MMLU results found.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_dir / "mmlu_all.csv", index=False)

    summary = (
        combined.groupby(["model", "scheme"])
        .agg(
            overall_accuracy=("accuracy", "mean"),
            n_subjects=("subject", "count"),
            total_correct=("correct", "sum"),
            total_questions=("n_questions", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "mmlu_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
