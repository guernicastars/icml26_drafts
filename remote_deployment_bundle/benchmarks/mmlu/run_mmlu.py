"""Evaluate MMLU on aligned adapters produced by alignment or axbench benchmarks.

For each seed directory under --adapter-dir, iterates over every Meta-SWAG scheme
saved as ``seed_{seed}/adapters/{scheme}/mean_vector.npy`` and evaluates the
restored LoRA-wrapped policy model on MMLU. A "base" (no adapter) row is also
recorded for reference. Results sharded by --subject-group enable 4-GPU
parallelism (STEM / Humanities / Social Sciences / Other).

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m benchmarks.mmlu.run_mmlu \\
        --base-model meta-llama/Llama-3.1-8B-Instruct \\
        --adapter-dir results/alignment/llama-3.1-8b_dpo \\
        --output-dir results/mmlu/llama-3.1-8b \\
        --subject-group STEM
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from meta_swag.adapters.state import restore_adapter_state, load_manifest
from meta_swag.evaluation.mmlu import (
    all_subjects,
    evaluate_mmlu_subject,
    load_mmlu_subject,
    SUBJECT_GROUPS,
)
from meta_swag.utils import parse_dtype, supports_bf16


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MMLU OOD evaluation for Meta-SWAG")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter-dir", required=True,
                   help="Training output dir containing seed_*/adapters/{scheme}/ and seed_*/checkpoints/manifest.json")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--subjects", nargs="+", default=None,
                   help="Explicit subject list (overrides --subject-group).")
    p.add_argument("--subject-group", choices=list(SUBJECT_GROUPS.keys()) + ["all"], default="all",
                   help="Restrict evaluation to one MMLU group for GPU sharding.")
    p.add_argument("--num-few-shot", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", default="auto",
                   help="auto|fp16|bf16|fp32. auto picks bf16 on Ampere+, fp16 on Volta (V100).")
    p.add_argument("--schemes", nargs="+", default=None,
                   help="Restrict to a subset of schemes (default: every subdir of adapters/).")
    p.add_argument("--include-base", action="store_true", default=True,
                   help="Also evaluate base model with no adapter.")
    p.add_argument("--no-include-base", dest="include_base", action="store_false")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-targets", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"])
    return p.parse_args()


def select_subjects(args) -> list[str]:
    if args.subjects:
        return list(args.subjects)
    if args.subject_group == "all":
        return all_subjects()
    return sorted(SUBJECT_GROUPS[args.subject_group])


def discover_seed_dirs(adapter_dir: Path) -> list[Path]:
    return sorted(p for p in adapter_dir.glob("seed_*") if p.is_dir())


def discover_schemes(seed_dir: Path, requested: list[str] | None) -> list[str]:
    adapters_dir = seed_dir / "adapters"
    if not adapters_dir.is_dir():
        return []
    found = sorted(p.name for p in adapters_dir.iterdir()
                   if p.is_dir() and (p / "mean_vector.npy").exists())
    if requested:
        return [s for s in requested if s in found]
    return found


def build_lora_policy(base_model, args):
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=list(args.lora_targets),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(base_model, lora_config)


def evaluate_subjects(model, tokenizer, subjects, device, batch_size, num_few_shot, scheme_label, seed):
    rows = []
    for subject in subjects:
        few_shot, test_data = load_mmlu_subject(subject, num_few_shot=num_few_shot)
        result = evaluate_mmlu_subject(
            model, tokenizer, subject, few_shot, test_data, device, batch_size,
        )
        rows.append({
            "seed": seed,
            "scheme": scheme_label,
            "subject": result.subject,
            "subject_group": result.subject_group,
            "accuracy": result.accuracy,
            "n_questions": result.n_questions,
            "correct": result.correct,
        })
        print(f"    {subject}: {result.accuracy:.3f} ({result.correct}/{result.n_questions})")
    return rows


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = Path(args.adapter_dir).resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = parse_dtype(args.dtype, device)

    subjects = select_subjects(args)
    print(f"=== MMLU Evaluation ===")
    print(f"  Base model: {args.base_model}")
    print(f"  Adapter dir: {adapter_dir}")
    print(f"  Device: {device}, dtype: {dtype}, bf16 native: {supports_bf16(device)}")
    print(f"  Group: {args.subject_group} ({len(subjects)} subjects)")

    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype,
    ).eval().to(device)

    all_rows: list[dict] = []

    if args.include_base:
        print("\nEvaluating base model (no adapter)...")
        all_rows.extend(evaluate_subjects(
            base_model, tokenizer, subjects, device,
            args.batch_size, args.num_few_shot,
            scheme_label="base", seed=-1,
        ))

    seed_dirs = discover_seed_dirs(adapter_dir)
    if not seed_dirs:
        print(f"WARN: no seed_* directories under {adapter_dir}; only base results recorded.")
    else:
        policy = build_lora_policy(base_model, args).to(device).eval()

        for seed_dir in seed_dirs:
            seed = int(seed_dir.name.split("_", 1)[1])
            manifest_path = seed_dir / "checkpoints" / "manifest.json"
            if not manifest_path.exists():
                print(f"SKIP {seed_dir.name}: missing {manifest_path}")
                continue
            manifest = load_manifest(manifest_path)

            schemes = discover_schemes(seed_dir, args.schemes)
            if not schemes:
                print(f"SKIP {seed_dir.name}: no scheme adapters found")
                continue

            for scheme in schemes:
                mean_vec_path = seed_dir / "adapters" / scheme / "mean_vector.npy"
                print(f"\nSeed {seed} | scheme {scheme}")
                mean_vector = np.load(mean_vec_path)
                restore_adapter_state(policy, mean_vector, manifest)
                all_rows.extend(evaluate_subjects(
                    policy, tokenizer, subjects, device,
                    args.batch_size, args.num_few_shot,
                    scheme_label=scheme, seed=seed,
                ))

    group_tag = args.subject_group.lower().replace(" ", "_")
    results_csv = output_dir / f"mmlu_results_{group_tag}.csv"
    pd.DataFrame(all_rows).to_csv(results_csv, index=False)

    summary_rows = []
    results_df = pd.DataFrame(all_rows)
    for (seed, scheme), scheme_df in results_df.groupby(["seed", "scheme"]):
        total_correct = scheme_df["correct"].sum()
        total_questions = scheme_df["n_questions"].sum()
        row = {
            "seed": int(seed),
            "scheme": scheme,
            "overall_accuracy": total_correct / max(total_questions, 1),
            "total_correct": int(total_correct),
            "total_questions": int(total_questions),
        }
        for group in SUBJECT_GROUPS:
            group_df = scheme_df[scheme_df["subject_group"] == group]
            if not group_df.empty:
                row[f"{group}_accuracy"] = group_df["correct"].sum() / max(group_df["n_questions"].sum(), 1)
        summary_rows.append(row)

    summary_csv = output_dir / f"mmlu_summary_{group_tag}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    print(f"\n=== Results saved to {output_dir} ===")
    if summary_rows:
        print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
