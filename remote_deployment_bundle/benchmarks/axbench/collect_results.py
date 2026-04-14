"""Aggregate per-experiment CSVs into paper-ready comparison tables.

Reads final_summary.csv from every results/<experiment>/ directory and
produces:

  aggregates/per_experiment_by_scheme.csv
      mean+-std of test_composite, delta_over_unsteered, ESS, max_weight,
      top_eigenvalue_ratio grouped by (experiment, scheme).

  aggregates/leaderboard_composite.csv
      pivot: rows = scheme, columns = experiment, values = mean test_composite.

  aggregates/leaderboard_delta.csv
      pivot: rows = scheme, columns = experiment, values = mean delta_over_unsteered.

  aggregates/posterior_health.csv
      mean ESS, max_normalized_weight, top_eigenvalue_ratio per (experiment, scheme).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BENCHMARK_ROOT = Path(__file__).resolve().parent

METRIC_COLS = [
    "test_composite",
    "delta_over_unsteered",
    "concept_relevance",
    "instruction_relevance",
    "fluency",
    "ess",
    "max_normalized_weight",
    "posterior_trace",
    "top_eigenvalue_ratio",
    "selected_factor",
]


def parse_experiment_name(name: str) -> dict[str, str]:
    parts = name.rsplit("_", 2)
    if len(parts) < 3:
        return {"experiment": name, "model_tag": name, "layer": "", "kind": ""}
    model_tag, layer_tag, kind = parts
    if kind == "lora":
        pass
    elif name.endswith("_preference_lora"):
        model_tag, layer_tag = name[: -len("_preference_lora")].rsplit("_", 1)
        kind = "preference_lora"
    layer = layer_tag[1:] if layer_tag.startswith("L") else layer_tag
    return {
        "experiment": name,
        "model_tag": model_tag,
        "layer": layer,
        "kind": kind,
    }


def load_experiment_summaries(results_dir: Path) -> pd.DataFrame:
    frames = []
    for sub in sorted(results_dir.iterdir()):
        if not sub.is_dir():
            continue
        summary_path = sub / "final_summary.csv"
        if not summary_path.exists():
            print(f"  skipping {sub.name}: no final_summary.csv")
            continue
        df = pd.read_csv(summary_path)
        if df.empty:
            print(f"  skipping {sub.name}: empty summary")
            continue
        meta = parse_experiment_name(sub.name)
        for k, v in meta.items():
            df[k] = v
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def aggregate_by_scheme(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in METRIC_COLS if c in df.columns]
    grouped = df.groupby(["experiment", "model_tag", "layer", "kind", "scheme"])[present]
    agg = grouped.agg(["mean", "std", "count"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    return agg.reset_index()


def pivot_metric(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    present = [c for c in METRIC_COLS if c in df.columns]
    if value_col not in present:
        return pd.DataFrame()
    summary = (
        df.groupby(["experiment", "scheme"])[value_col]
        .mean()
        .reset_index()
    )
    return summary.pivot(index="scheme", columns="experiment", values=value_col)


def posterior_health(df: pd.DataFrame) -> pd.DataFrame:
    diag_cols = [c for c in ("ess", "max_normalized_weight", "top_eigenvalue_ratio",
                             "posterior_trace", "retained_count") if c in df.columns]
    if not diag_cols:
        return pd.DataFrame()
    return (
        df.groupby(["experiment", "model_tag", "layer", "kind", "scheme"])[diag_cols]
        .mean()
        .reset_index()
    )


def print_paper_table(pivot: pd.DataFrame, title: str) -> None:
    if pivot.empty:
        return
    print(f"\n=== {title} ===")
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.max_columns", None,
                           "display.width", 200):
        print(pivot.to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate Meta-SWAG AxBench results.")
    ap.add_argument("--results-dir", default=str(BENCHMARK_ROOT / "results"))
    ap.add_argument("--out-dir", default=str(BENCHMARK_ROOT / "aggregates"))
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results dir not found: {results_dir}")

    print(f"Loading summaries from {results_dir}")
    df = load_experiment_summaries(results_dir)
    if df.empty:
        print("No experiment results found.")
        return

    print(f"Loaded {len(df)} rows from {df['experiment'].nunique()} experiments")

    df.to_csv(out_dir / "all_summaries.csv", index=False)

    per_scheme = aggregate_by_scheme(df)
    per_scheme.to_csv(out_dir / "per_experiment_by_scheme.csv", index=False)

    composite = pivot_metric(df, "test_composite")
    delta = pivot_metric(df, "delta_over_unsteered")
    composite.to_csv(out_dir / "leaderboard_composite.csv")
    delta.to_csv(out_dir / "leaderboard_delta.csv")

    health = posterior_health(df)
    health.to_csv(out_dir / "posterior_health.csv", index=False)

    print_paper_table(composite, "Leaderboard: mean test composite")
    print_paper_table(delta, "Leaderboard: mean delta over unsteered")

    print(f"\nAggregates written to {out_dir}")


if __name__ == "__main__":
    main()
