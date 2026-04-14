"""Run a single Meta-SWAG AxBench experiment for one (model, layer, kind) configuration.

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_experiment.py \
        --model-name google/gemma-2-2b-it \
        --layer 20 \
        --model-kind lora \
        --output-dir results/gemma-2-2b-it_L20_lora

All Meta-SWAG weighting schemes (MAP, uniform, softmax, ESS, threshold) are
compared for each concept.  Results are written as CSVs to --output-dir.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch

BENCHMARK_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BENCHMARK_ROOT))

from meta_swag.adapters.state import restore_adapter_state, save_manifest, build_manifest
from meta_swag.posterior.predictive import PosteriorPredictive
from meta_swag.posterior.meta_swag import aggregate_adapter_checkpoints

try:
    from meta_swag.axbench_meta_swag import (
        FinalMethodResult,
        aggregate_checkpoint_records,
        attach_validation_metrics,
        choose_factor_from_factor_sweep,
        harmonic_mean,
        split_validation_test,
        train_lora_with_retention,
        train_preference_lora_with_retention,
    )
except ImportError:
    old_meta_swag = PROJECT_ROOT / "axbench_benchmark"
    sys.path.insert(0, str(old_meta_swag))
    from meta_swag.axbench_meta_swag import (
        FinalMethodResult,
        aggregate_checkpoint_records,
        attach_validation_metrics,
        choose_factor_from_factor_sweep,
        harmonic_mean,
        split_validation_test,
        train_lora_with_retention,
        train_preference_lora_with_retention,
    )

from axbench_runtime import describe_external_repo, import_axbench


DEFAULT_FACTORS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0]
DEFAULT_METHODS = [
    "map", "last_iterate", "uniform", "swa", "ema",
    "softmax", "ess", "threshold", "laplace",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single Meta-SWAG AxBench experiment.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-kind", choices=["lora", "preference_lora"], default="lora")
    p.add_argument("--model-name", default="google/gemma-2-2b-it")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--component", default="res")
    p.add_argument("--data-dir", default=None,
                   help="Data directory. Defaults to data/<model_tag>_L<layer>/")
    p.add_argument("--max-concepts", type=int, default=30)
    p.add_argument("--concept-ids", nargs="+", type=int, default=None)
    p.add_argument("--seed-count", type=int, default=3)
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--n-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--low-rank-dimension", type=int, default=4)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-components", nargs="+", default=["q_proj"])
    p.add_argument("--lora-layers", nargs="+", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--eval-output-length", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--steering-factors", nargs="+", type=float, default=DEFAULT_FACTORS)
    p.add_argument("--keep-last", type=int, default=20)
    p.add_argument("--tail-fraction", type=float, default=0.4)
    p.add_argument("--threshold-quantile", type=float, default=0.75)
    p.add_argument("--validation-ratio", type=float, default=0.5)
    p.add_argument("--max-validation-examples", type=int, default=32)
    p.add_argument("--max-test-examples", type=int, default=32)
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    p.add_argument("--posterior-samples", type=int, default=16)
    p.add_argument("--deterministic-mean", action="store_true",
                   help="Deploy the weighted mean only (old behavior), skip sampling")
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--mock-judge", action="store_true", default=True)
    p.add_argument("--real-judge", dest="mock_judge", action="store_false")
    p.add_argument("--use-bf16", action="store_true", default=True)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--gemma", type=float, default=0.0)
    p.add_argument("--simpo-scaler", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--reference-free", action="store_true")
    p.add_argument("--loss-type", default="dpo")
    p.add_argument("--preference-pairs", nargs="+", default=["orig_add"])
    p.add_argument("--steering-prompt-type", default="prepend")
    p.add_argument("--substraction-type", default="null_it_out")
    return p.parse_args()


def resolve_data_dir(args: argparse.Namespace) -> Path:
    if args.data_dir:
        return Path(args.data_dir)
    model_tag = args.model_name.split("/")[-1]
    return BENCHMARK_ROOT / "data" / f"{model_tag}_L{args.layer}"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_metadata(path: Path) -> list[dict[str, Any]]:
    metadata = []
    with path.open() as f:
        for line in f:
            metadata.append(json.loads(line))
    return metadata


def load_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise ValueError(f"Unsupported format: {path}")


def select_concept_ids(train_df: pd.DataFrame, args: argparse.Namespace) -> list[int]:
    if args.concept_ids:
        return [int(c) for c in args.concept_ids]
    concept_ids = sorted(int(v) for v in train_df["concept_id"].unique() if int(v) >= 0)
    if args.max_concepts is not None:
        concept_ids = concept_ids[:args.max_concepts]
    return concept_ids


def build_model_params(axbench, args: argparse.Namespace):
    params = axbench.ModelParams()
    params.batch_size = args.batch_size
    params.n_epochs = args.n_epochs
    params.lr = args.lr
    params.dropout = args.dropout
    params.low_rank_dimension = args.low_rank_dimension
    params.gradient_accumulation_steps = args.gradient_accumulation_steps
    params.lora_layers = args.lora_layers
    params.lora_components = args.lora_components
    params.lora_alpha = args.lora_alpha
    params.weight_decay = args.weight_decay
    params.topk = args.topk
    params.loss_type = args.loss_type
    params.beta = args.beta
    params.gemma = args.gemma
    params.reference_free = args.reference_free
    params.label_smoothing = args.label_smoothing
    params.steering_factors = list(args.steering_factors)
    params.simpo_scaler = args.simpo_scaler
    params.preference_pairs = list(args.preference_pairs)
    params.steering_prompt_type = args.steering_prompt_type
    params.substraction_type = args.substraction_type
    params.intervention_positions = "all"
    return params


def build_training_dataframe(axbench_train_module, train_df, negative_df,
                             metadata_entry, concept, tokenizer, args, model_params):
    prepared = train_df.copy()
    is_chat_model = args.model_name in axbench_train_module.CHAT_MODELS
    return axbench_train_module.prepare_df(
        prepared, negative_df, concept, metadata_entry, tokenizer,
        binarize=model_params.binarize_dataset,
        train_on_negative=model_params.train_on_negative,
        is_chat_model=is_chat_model,
        output_length=args.eval_output_length,
        model_name=args.model_name,
        max_num_of_examples=None,
        use_dpo_loss=args.model_kind == "preference_lora",
        steering_prompt_type=model_params.steering_prompt_type,
        keep_orig_axbench_format=True,
    )


def build_fallback_steering_df(concept_df, metadata_entry, factors, max_examples):
    base = concept_df[concept_df.get("category", pd.Series(["positive"] * len(concept_df))) == "positive"].copy()
    if base.empty:
        base = concept_df.copy()
    base = base.head(max_examples).copy()
    if "input" not in base.columns:
        raise KeyError("Need input column for steering data.")
    base["original_prompt"] = base["input"]
    base["input_concept"] = metadata_entry["concept"]
    base["dataset_name"] = "TrainFallback"
    base["input_id"] = np.arange(len(base))
    rows = []
    for factor in factors:
        cur = base[["concept_id", "input", "original_prompt", "input_concept",
                     "dataset_name", "input_id"]].copy()
        cur["factor"] = factor
        rows.append(cur)
    return pd.concat(rows, ignore_index=True)


def load_concept_steering_df(concept_df, metadata_entry, concept_id, data_dir, args):
    steer_path = data_dir / "steering_data.parquet"
    if steer_path.exists():
        full = pd.read_parquet(steer_path)
        filtered = full[full["concept_id"] == concept_id].copy()
        if not filtered.empty:
            return filtered
    return build_fallback_steering_df(
        concept_df, metadata_entry, list(args.steering_factors),
        max(args.max_validation_examples, args.max_test_examples),
    )


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def evaluate_mock_factor_sweep(results_df, model_name):
    scored = results_df.copy()

    def concept_score(row):
        concept = str(row.get("input_concept", "")).lower()
        gen = str(row.get(f"{model_name}_steered_generation", "")).lower()
        if concept in gen:
            return 2.0
        tokens = [t for t in _tokenize_words(concept) if len(t) > 2]
        if any(t in gen for t in tokens):
            return 1.0
        return 0.0

    def instruction_score(row):
        prompt_tokens = {t for t in _tokenize_words(str(row.get("original_prompt", ""))) if len(t) > 3}
        gen_tokens = set(_tokenize_words(str(row.get(f"{model_name}_steered_generation", ""))))
        overlap = len(prompt_tokens & gen_tokens)
        if overlap >= 3:
            return 2.0
        if overlap >= 1 or gen_tokens:
            return 1.0
        return 0.0

    def fluency_score(row):
        gen = str(row.get(f"{model_name}_steered_generation", ""))
        if len(gen.strip()) < 4:
            return 0.0
        ascii_ratio = sum(c.isascii() and (c.isalnum() or c.isspace()) for c in gen) / max(len(gen), 1)
        if ascii_ratio > 0.85 and len(gen.split()) >= 4:
            return 2.0
        return 1.0

    scored[f"{model_name}_concept"] = scored.apply(concept_score, axis=1)
    scored[f"{model_name}_instruction"] = scored.apply(instruction_score, axis=1)
    scored[f"{model_name}_fluency"] = scored.apply(fluency_score, axis=1)

    factor_rows = []
    for factor, group in scored.groupby("factor"):
        cr = float(group[f"{model_name}_concept"].mean())
        ir = float(group[f"{model_name}_instruction"].mean())
        fl = float(group[f"{model_name}_fluency"].mean())
        composite = harmonic_mean([cr, ir, fl])
        ppl_col = f"{model_name}_perplexity"
        ppl = float(group[ppl_col].mean()) if ppl_col in group.columns else np.nan
        factor_rows.append({
            "factor": float(factor), "composite": composite,
            "concept_relevance": cr, "instruction_relevance": ir,
            "fluency": fl, "perplexity": ppl,
        })
    return sorted(factor_rows, key=lambda r: r["factor"])


def compute_perplexities(base_model, tokenizer, sequences, device):
    if not sequences:
        return []
    encoded = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    with torch.no_grad():
        outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :].contiguous()
    targets = input_ids[:, 1:].contiguous()
    mask = attention_mask[:, 1:].contiguous()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    token_losses = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1)).view(targets.size(0), -1)
    seq_losses = (token_losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return torch.exp(seq_losses).detach().cpu().tolist()


def generate_unsteered_outputs(base_model, tokenizer, eval_df, batch_size, output_length,
                               temperature, device):
    tokenizer.padding_side = "left"
    all_gens, all_ppls = [], []
    for start in range(0, len(eval_df), batch_size):
        batch = eval_df.iloc[start:start + batch_size]
        inputs = tokenizer(batch["input"].tolist(), return_tensors="pt",
                          padding=True, truncation=True).to(device)
        with torch.no_grad():
            generations = base_model.generate(
                **inputs, max_new_tokens=output_length,
                do_sample=True, temperature=temperature,
            )
        input_lengths = [len(row) for row in inputs.input_ids]
        decoded = [
            tokenizer.decode(seq[il:], skip_special_tokens=True)
            for seq, il in zip(generations, input_lengths)
        ]
        all_gens.extend(decoded)
        full = [tokenizer.decode(seq, skip_special_tokens=True) for seq in generations]
        all_ppls.extend(compute_perplexities(base_model, tokenizer, full, device))
    return {"steered_generation": all_gens, "perplexity": all_ppls}


def evaluate_factor_sweep(model, eval_df, model_name, axbench, args, concept_id,
                          lm_model=None):
    working = eval_df.copy()
    results = model.predict_steer(
        working, concept_id=concept_id,
        batch_size=args.eval_batch_size,
        eval_output_length=args.eval_output_length,
        temperature=args.temperature,
        prefix_length=1, positions="all",
        use_synergy=False, disable_neuronpedia_max_act=True,
        intervene_on_prompt=True, return_vector=False,
    )
    for key, vals in results.items():
        working[f"{model_name}_{key}"] = vals
    factor_rows = evaluate_mock_factor_sweep(working, model_name)
    return factor_rows, working


def restore_record(model, record, manifest):
    restore_adapter_state(model.ax_model, record.adapter_vector, manifest)


def restore_aggregated(model, agg, manifest):
    restore_adapter_state(model.ax_model, agg.mean_vector, manifest)


def average_factor_sweep_over_posterior(
    model, eval_df, scheme, agg, manifest,
    axbench, args, concept_id, num_samples, rng_seed,
):
    """Sample S vectors from the posterior, evaluate factor sweep on each,
    and average composite metrics per-factor. S=1 for point estimates or
    --deterministic-mean."""
    if args.deterministic_mean or scheme.lower() in ("map", "last_iterate"):
        restore_adapter_state(model.ax_model, agg.mean_vector, manifest)
        rows, working = evaluate_factor_sweep(
            model, eval_df, model_name=scheme,
            axbench=axbench, args=args, concept_id=concept_id,
        )
        return rows, working, 1

    predictive = PosteriorPredictive(agg, manifest, num_samples=num_samples, seed=rng_seed)
    accumulated: dict[float, dict[str, float]] = {}
    last_working = None
    for _sample_idx, vector in predictive.deploy_iter(model.ax_model):
        rows, working = evaluate_factor_sweep(
            model, eval_df, model_name=scheme,
            axbench=axbench, args=args, concept_id=concept_id,
        )
        last_working = working
        for row in rows:
            factor = float(row["factor"])
            bucket = accumulated.setdefault(factor, {
                "factor": factor, "composite": 0.0,
                "concept_relevance": 0.0, "instruction_relevance": 0.0,
                "fluency": 0.0, "perplexity": 0.0,
                "_perplexity_count": 0,
            })
            bucket["composite"] += float(row["composite"])
            bucket["concept_relevance"] += float(row["concept_relevance"])
            bucket["instruction_relevance"] += float(row["instruction_relevance"])
            bucket["fluency"] += float(row["fluency"])
            ppl = row.get("perplexity", float("nan"))
            if ppl == ppl:
                bucket["perplexity"] += float(ppl)
                bucket["_perplexity_count"] += 1

    n = predictive.effective_num_samples
    averaged = []
    for factor in sorted(accumulated):
        bucket = accumulated[factor]
        ppl_count = bucket.pop("_perplexity_count")
        ppl = bucket["perplexity"] / ppl_count if ppl_count > 0 else float("nan")
        averaged.append({
            "factor": bucket["factor"],
            "composite": bucket["composite"] / n,
            "concept_relevance": bucket["concept_relevance"] / n,
            "instruction_relevance": bucket["instruction_relevance"] / n,
            "fluency": bucket["fluency"] / n,
            "perplexity": ppl,
        })
    return averaged, last_working, n


def summarize_method(scheme, agg, val_rows, test_rows, unsteered_composite):
    sel_factor, val_comp = choose_factor_from_factor_sweep(val_rows)
    test_row = next(r for r in test_rows if float(r["factor"]) == float(sel_factor))
    diag = {
        "retained_count": float(agg.retained_count),
        "ess": float(agg.effective_sample_size),
        "max_normalized_weight": float(agg.max_normalized_weight),
        "posterior_trace": float(agg.posterior_trace),
        "top_eigenvalue_ratio": float(agg.top_eigenvalue_ratio),
        "score_variance": float(agg.score_variance),
    }
    for i, v in enumerate(agg.top_eigenvalues, 1):
        diag[f"top_eigenvalue_{i}"] = float(v)

    return FinalMethodResult(
        scheme=scheme,
        selected_factor=float(sel_factor),
        validation_composite=float(val_comp),
        test_composite=float(test_row["composite"]),
        concept_relevance=float(test_row["concept_relevance"]),
        instruction_relevance=float(test_row["instruction_relevance"]),
        fluency=float(test_row["fluency"]),
        perplexity=None if pd.isna(test_row["perplexity"]) else float(test_row["perplexity"]),
        delta_over_unsteered=float(test_row["composite"] - unsteered_composite),
        diagnostics=diag,
    )


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = resolve_data_dir(args)

    print(f"=== Meta-SWAG Experiment ===")
    print(f"  Model: {args.model_name}")
    print(f"  Layer: {args.layer}")
    print(f"  Kind:  {args.model_kind}")
    print(f"  Data:  {data_dir}")
    print(f"  Out:   {output_dir}")
    print()

    set_global_seed(args.base_seed)
    axbench = import_axbench()
    import axbench.scripts.train as axbench_train_module  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer

    train_path = data_dir / "train_data.parquet"
    metadata_path = data_dir / "metadata.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    metadata = load_metadata(metadata_path)
    metadata_by_cid = {int(e["concept_id"]): e for e in metadata}
    train_df = load_dataframe(train_path)
    concept_ids = [c for c in select_concept_ids(train_df, args) if c in metadata_by_cid]
    print(f"  Concepts: {len(concept_ids)}")

    write_json(output_dir / "config.json", {
        "model_name": args.model_name,
        "layer": args.layer,
        "model_kind": args.model_kind,
        "max_concepts": args.max_concepts,
        "seed_count": args.seed_count,
        "methods": args.methods,
        "keep_last": args.keep_last,
        "tail_fraction": args.tail_fraction,
        "n_epochs": args.n_epochs,
        "lr": args.lr,
        "low_rank_dimension": args.low_rank_dimension,
    })

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=512)
    tokenizer.padding_side = "right"
    orig_vocab = len(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = args.use_bf16 and torch.cuda.is_available()
    print(f"  Device: {device}, bf16: {use_bf16}")

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if use_bf16 else None,
    ).eval().to(device)
    if len(tokenizer) != orig_vocab:
        base_model.resize_token_embeddings(len(tokenizer))

    model_params = build_model_params(axbench, args)
    negative_df = (train_df[train_df["category"] == "negative"].copy()
                   if "category" in train_df.columns
                   else train_df.iloc[0:0].copy())

    checkpoint_rows: list[dict] = []
    factor_sweep_rows: list[dict] = []
    final_summary_rows: list[dict] = []

    for seed_idx in range(args.seed_count):
        seed = args.base_seed + seed_idx
        set_global_seed(seed)
        print(f"\n--- Seed {seed} ({seed_idx+1}/{args.seed_count}) ---")

        for ci, concept_id in enumerate(concept_ids):
            me = metadata_by_cid[concept_id]
            concept_name = me["concept"]
            print(f"  [{ci+1}/{len(concept_ids)}] concept {concept_id}: {concept_name[:50]}")

            concept_train = train_df[train_df["concept_id"] == concept_id].copy()
            steering_df = load_concept_steering_df(concept_train, me, concept_id, data_dir, args)
            val_df, test_df = split_validation_test(steering_df, args.validation_ratio)
            val_df = val_df.head(args.max_validation_examples * max(1, len(args.steering_factors))).copy()
            test_df = test_df.head(args.max_test_examples * max(1, len(args.steering_factors))).copy()

            try:
                prepared = build_training_dataframe(
                    axbench_train_module, concept_train, negative_df,
                    me, concept_name, tokenizer, args, model_params,
                )
            except Exception as e:
                print(f"    SKIP: data prep failed: {e}")
                continue

            model_class = axbench.PreferenceLoRA if args.model_kind == "preference_lora" else axbench.LoRA
            model = model_class(
                base_model, tokenizer, layer=args.layer,
                training_args=model_params, lm_model_name=args.model_name,
                device=device, seed=seed,
            )

            try:
                model.make_model(
                    mode="train", embed_dim=base_model.config.hidden_size,
                    low_rank_dimension=args.low_rank_dimension,
                    concept_id=concept_id,
                    dtype=torch.bfloat16 if use_bf16 else None,
                    intervention_type="addition",
                    metadata_path=str(metadata_path),
                    dump_dir=str(output_dir),
                    model_params=model_params,
                    dropout=args.dropout,
                    intervention_positions_dropout=0.0,
                    preference_pairs=args.preference_pairs,
                )
            except Exception as e:
                print(f"    SKIP: model creation failed: {e}")
                continue

            trainer_kwargs = {
                "prefix_length": 1, "positions": "all", "exclude_bos": True,
                "metadata_path": str(metadata_path),
                "use_dpo_loss": args.model_kind == "preference_lora",
                "logging_metadata": {"concept_id": concept_id, "model_name": args.model_kind, "layer": args.layer},
                "negative_only": False,
                "preference_pairs": args.preference_pairs,
                "steering_prompt_type": args.steering_prompt_type,
                "substraction_type": args.substraction_type,
            }

            try:
                if args.model_kind == "preference_lora":
                    retained, manifest = train_preference_lora_with_retention(
                        model, prepared,
                        keep_last=args.keep_last, tail_fraction=args.tail_fraction,
                        checkpoint_id_prefix=f"s{seed}_c{concept_id}",
                        **trainer_kwargs,
                    )
                else:
                    retained, manifest = train_lora_with_retention(
                        model, prepared,
                        keep_last=args.keep_last, tail_fraction=args.tail_fraction,
                        checkpoint_id_prefix=f"s{seed}_c{concept_id}",
                        **trainer_kwargs,
                    )
            except Exception as e:
                print(f"    SKIP: training failed: {e}")
                continue

            if not retained:
                print(f"    SKIP: no checkpoints retained")
                continue

            for rec in retained:
                restore_record(model, rec, manifest)
                try:
                    vrows, _ = evaluate_factor_sweep(
                        model, val_df, model_name="checkpoint",
                        axbench=axbench, args=args, concept_id=concept_id,
                    )
                    attach_validation_metrics(rec, vrows)
                except Exception as e:
                    print(f"    WARN: checkpoint eval failed: {e}")
                    continue

                checkpoint_rows.append({
                    "seed": seed, "concept_id": concept_id,
                    "concept": concept_name, **rec.metadata(),
                })

            unsteered_eval = test_df.copy()
            unsteered_eval["factor"] = 0.0
            unsteered_out = generate_unsteered_outputs(
                base_model, tokenizer, unsteered_eval,
                args.eval_batch_size, args.eval_output_length, args.temperature, device,
            )
            for k, v in unsteered_out.items():
                unsteered_eval[f"unsteered_{k}"] = v
            unsteered_test_rows = evaluate_mock_factor_sweep(unsteered_eval, "unsteered")
            _, unsteered_composite = choose_factor_from_factor_sweep(unsteered_test_rows)

            valid_retained = [r for r in retained if r.weighting_metric is not None]
            if not valid_retained:
                print(f"    SKIP: no valid checkpoints after evaluation")
                continue

            seed_dir = output_dir / f"seed_{seed}" / f"concept_{concept_id}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            from meta_swag.adapters.state import save_manifest
            save_manifest(manifest, seed_dir / "manifest.json")
            c_vectors = np.stack([r.adapter_vector for r in retained], axis=0)
            np.savez_compressed(seed_dir / "retained_checkpoints.npz", vectors=c_vectors)
            pd.DataFrame([r.metadata() for r in retained]).to_csv(seed_dir / "checkpoint_metadata.csv", index=False)
            print(f"  Saved {len(retained)} checkpoint vectors -> {seed_dir}")

            for scheme in args.methods:
                try:
                    if scheme == "laplace":
                        best_idx = int(np.argmax([r.train_loss for r in valid_retained]))
                        restore_adapter_state(model.ax_model, valid_retained[best_idx].adapter_vector, manifest)
                        
                        from torch.utils.data import DataLoader
                        from meta_swag.training.dpo_trainer import DPODataset
                        from meta_swag.training.preference import get_batch_logps

                        def ax_loss_fn(mdl, batch):
                            if args.model_kind == "preference_lora":
                                c_ids = batch["chosen_input_ids"].to(device)
                                c_mask = batch["chosen_attention_mask"].to(device)
                                c_labels = batch["chosen_labels"].to(device)
                                outs = mdl(input_ids=c_ids, attention_mask=c_mask)
                                logps = get_batch_logps(outs.logits, c_labels)
                                return -logps.mean()
                            else:
                                ids = batch["input_ids"].to(device)
                                mask = batch["attention_mask"].to(device)
                                labels = batch["labels"].to(device)
                                outs = mdl(input_ids=ids, attention_mask=mask)
                                loss_f = torch.nn.CrossEntropyLoss()
                                return loss_f(outs.logits.view(-1, outs.logits.size(-1)), labels.view(-1))

                        # Use a subset of training data for Fisher
                        fisher_dataset = prepared if isinstance(prepared, torch.utils.data.Dataset) else model.get_dataset(prepared)
                        fisher_loader = DataLoader(fisher_dataset, batch_size=args.batch_size, shuffle=True)
                        
                        from meta_swag.posterior.laplace import compute_diagonal_fisher, tune_prior_precision, laplace_posterior
                        fisher = compute_diagonal_fisher(model.ax_model, ax_loss_fn, manifest, num_batches=10, dataloader=fisher_loader)
                        prior_p = tune_prior_precision(model.ax_model, manifest, fisher, loss_fn=ax_loss_fn, val_dataloader=fisher_loader, num_val_batches=5)
                        agg = laplace_posterior(model.ax_model, manifest, fisher, prior_p)
                    else:
                        agg = aggregate_checkpoint_records(
                            valid_retained, scheme=scheme, beta=1.0,
                            threshold_quantile=args.threshold_quantile,
                            low_rank_rank=min(args.keep_last, 20),
                        )

                    adapter_out = seed_dir / "adapters" / scheme
                    adapter_out.mkdir(parents=True, exist_ok=True)
                    restore_adapter_state(model.ax_model, agg.mean_vector, manifest)
                    try:
                        if hasattr(model.ax_model, "save_pretrained"):
                            model.ax_model.save_pretrained(str(adapter_out))
                        tokenizer.save_pretrained(str(adapter_out))
                    except Exception as e:
                        print(f"    WARN: could not save PEFT adapter for {scheme}: {e}")
                        
                    np.save(adapter_out / "mean_vector.npy", agg.mean_vector)
                    if hasattr(agg, "diagonal_variance") and agg.diagonal_variance is not None:
                        np.save(adapter_out / "diagonal_variance.npy", agg.diagonal_variance)

                    vrows, _, s_used = average_factor_sweep_over_posterior(
                        model, val_df, scheme, agg, manifest,
                        axbench=axbench, args=args, concept_id=concept_id,
                        num_samples=args.posterior_samples,
                        rng_seed=seed * 1_000_003 + concept_id,
                    )
                    trows, _, _ = average_factor_sweep_over_posterior(
                        model, test_df, scheme, agg, manifest,
                        axbench=axbench, args=args, concept_id=concept_id,
                        num_samples=args.posterior_samples,
                        rng_seed=seed * 1_000_003 + concept_id + 7,
                    )

                    for r in vrows:
                        factor_sweep_rows.append({
                            "seed": seed, "concept_id": concept_id,
                            "concept": concept_name, "partition": "validation",
                            "scheme": scheme, **r,
                        })
                    for r in trows:
                        factor_sweep_rows.append({
                            "seed": seed, "concept_id": concept_id,
                            "concept": concept_name, "partition": "test",
                            "scheme": scheme, **r,
                        })

                    summary = summarize_method(scheme, agg, vrows, trows, unsteered_composite)
                    final_summary_rows.append({
                        "seed": seed, "concept_id": concept_id,
                        "concept": concept_name, "scheme": scheme,
                        "selected_factor": summary.selected_factor,
                        "validation_composite": summary.validation_composite,
                        "test_composite": summary.test_composite,
                        "concept_relevance": summary.concept_relevance,
                        "instruction_relevance": summary.instruction_relevance,
                        "fluency": summary.fluency,
                        "perplexity": summary.perplexity,
                        "delta_over_unsteered": summary.delta_over_unsteered,
                        **summary.diagnostics,
                    })
                except Exception as e:
                    print(f"    WARN: {scheme} failed: {e}")
                    continue

            # free GPU memory for next concept
            del model
            torch.cuda.empty_cache()

    # save results
    pd.DataFrame(checkpoint_rows).to_csv(output_dir / "checkpoint_metrics.csv", index=False)
    pd.DataFrame(factor_sweep_rows).to_csv(output_dir / "factor_sweeps.csv", index=False)
    summary_df = pd.DataFrame(final_summary_rows)
    summary_df.to_csv(output_dir / "final_summary.csv", index=False)

    if not summary_df.empty:
        grouped = (
            summary_df.groupby("scheme", as_index=False)[
                ["test_composite", "delta_over_unsteered", "instruction_relevance", "fluency",
                 "ess", "max_normalized_weight", "posterior_trace", "top_eigenvalue_ratio"]
            ].agg(["mean", "std"])
        )
        grouped.columns = ["_".join(c).strip("_") for c in grouped.columns]
        grouped.to_csv(output_dir / "summary_by_scheme.csv", index=False)
        print(f"\n=== Results saved to {output_dir} ===")
        print(grouped.to_string(index=False))
    else:
        print("\n=== No results produced ===")


if __name__ == "__main__":
    main()
