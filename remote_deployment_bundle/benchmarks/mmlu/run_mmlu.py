"""Evaluate MMLU on aligned adapters produced by alignment or axbench benchmarks.

Usage:
    python -m benchmarks.mmlu.run_mmlu \
        --base-model meta-llama/Llama-3.1-8B-Instruct \
        --adapter-dir results/llama-3.1-8b_dpo \
        --output-dir results/llama-3.1-8b_mmlu
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from meta_swag.adapters.state import restore_adapter_state, load_manifest, AdapterStateManifest
from meta_swag.posterior.base import AggregatedAdapterResult
from meta_swag.posterior.predictive import PosteriorPredictive
from meta_swag.evaluation.mmlu import (
    all_subjects,
    evaluate_mmlu_subject,
    evaluate_mmlu_subject_bma,
    load_mmlu_subject,
    SUBJECT_GROUPS,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MMLU OOD evaluation for Meta-SWAG")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter-dir", required=True, help="Directory with summary.csv and checkpoint data")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--subjects", nargs="+", default=None, help="Specific subjects (default: all 57)")
    p.add_argument("--num-few-shot", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--posterior-samples", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-bf16", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = Path(args.adapter_dir).resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = args.use_bf16 and torch.cuda.is_available()

    print(f"=== MMLU Evaluation ===")
    print(f"  Base model: {args.base_model}")
    print(f"  Adapter dir: {adapter_dir}")
    print(f"  Device: {device}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16 if use_bf16 else None,
    ).eval().to(device)

    subjects = args.subjects or all_subjects()
    print(f"  Subjects: {len(subjects)}")

    print("\nEvaluating base model (no adapter)...")
    base_results = []
    for subject in subjects:
        few_shot, test_data = load_mmlu_subject(subject, num_few_shot=args.num_few_shot)
        result = evaluate_mmlu_subject(
            base_model, tokenizer, subject, few_shot, test_data, device, args.batch_size,
        )
        base_results.append({
            "scheme": "base",
            "subject": result.subject,
            "subject_group": result.subject_group,
            "accuracy": result.accuracy,
            "n_questions": result.n_questions,
            "correct": result.correct,
        })
        print(f"  {subject}: {result.accuracy:.3f} ({result.correct}/{result.n_questions})")

    all_results = list(base_results)

    pd.DataFrame(all_results).to_csv(output_dir / "mmlu_results.csv", index=False)

    summary_rows = []
    results_df = pd.DataFrame(all_results)
    for scheme in results_df["scheme"].unique():
        scheme_df = results_df[results_df["scheme"] == scheme]
        total_correct = scheme_df["correct"].sum()
        total_questions = scheme_df["n_questions"].sum()
        overall_acc = total_correct / max(total_questions, 1)

        row = {
            "scheme": scheme,
            "overall_accuracy": overall_acc,
            "total_correct": int(total_correct),
            "total_questions": int(total_questions),
        }
        for group in SUBJECT_GROUPS:
            group_df = scheme_df[scheme_df["subject_group"] == group]
            if not group_df.empty:
                row[f"{group}_accuracy"] = group_df["correct"].sum() / max(group_df["n_questions"].sum(), 1)
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(output_dir / "mmlu_summary.csv", index=False)

    print(f"\n=== Results saved to {output_dir} ===")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
