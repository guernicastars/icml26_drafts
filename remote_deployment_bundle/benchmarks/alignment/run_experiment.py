"""Run a single DPO alignment experiment with Meta-SWAG posterior evaluation.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m benchmarks.alignment.run_experiment \
        --base-model meta-llama/Llama-3.1-8B-Instruct \
        --output-dir results/llama-3.1-8b_dpo
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from meta_swag.adapters.state import build_manifest, restore_adapter_state
from meta_swag.posterior.base import AggregatedAdapterResult
from meta_swag.posterior.laplace import compute_diagonal_fisher, laplace_posterior, tune_prior_precision
from meta_swag.posterior.meta_swag import aggregate_adapter_checkpoints
from meta_swag.posterior.predictive import PosteriorPredictive
from meta_swag.training.checkpoint import RetainedCheckpoint
from meta_swag.training.dpo_trainer import DPODataset, train_dpo_with_retention
from meta_swag.utils import parse_dtype, supports_bf16


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPO alignment + Meta-SWAG benchmark")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--dataset", default="HuggingFaceH4/ultrafeedback_binarized")
    p.add_argument("--train-split", default="train_prefs")
    p.add_argument("--test-split", default="test_prefs")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-targets", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"])
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--n-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--warmup-steps", type=int, default=150)
    p.add_argument("--keep-last", type=int, default=50)
    p.add_argument("--tail-fraction", type=float, default=0.5)
    p.add_argument("--loss-type", default="dpo")
    p.add_argument("--schemes", nargs="+",
                   default=["map", "last_iterate", "swa", "ema", "softmax", "ess", "threshold", "laplace"])
    p.add_argument("--posterior-samples", type=int, default=16)
    p.add_argument("--gold-rm", default="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")
    p.add_argument("--proxy-rm", default="internlm/internlm2-1_8b-reward")
    p.add_argument("--best-of-n", nargs="+", type=int, default=[1, 4, 16])
    p.add_argument("--num-eval-prompts", type=int, default=1000)
    p.add_argument("--eval-max-new-tokens", type=int, default=256)
    p.add_argument("--eval-temperature", type=float, default=1.0)
    p.add_argument("--seed-count", type=int, default=3)
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--dtype", default="auto",
                   help="auto|fp16|bf16|fp32. auto picks bf16 on Ampere+ and fp16 on Volta (V100).")
    p.add_argument("--reward-int8", action="store_true", default=True,
                   help="Load gold/proxy reward models in int8 (halves RM VRAM, required on 32GB cards).")
    p.add_argument("--no-reward-int8", dest="reward_int8", action="store_false")
    p.add_argument("--laplace-fisher-batches", type=int, default=100)
    p.add_argument("--max-train-samples", type=int, default=None)
    return p.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_ultrafeedback(dataset_name: str, split: str, max_samples: int | None = None) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=split)
    items = []
    for row in ds:
        chosen = row["chosen"]
        rejected = row["rejected"]
        if isinstance(chosen, list):
            prompt = chosen[0]["content"] if chosen[0]["role"] == "user" else ""
            chosen_text = chosen[-1]["content"] if len(chosen) > 1 else ""
            rejected_text = rejected[-1]["content"] if len(rejected) > 1 else ""
        else:
            prompt = row.get("prompt", "")
            chosen_text = chosen
            rejected_text = rejected
        items.append({"prompt": prompt, "chosen": chosen_text, "rejected": rejected_text})
        if max_samples and len(items) >= max_samples:
            break
    return items


def setup_lora_model(base_model, args):
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


def score_with_reward_model(rm_model, rm_tokenizer, prompts, responses, device, batch_size=8):
    scores = []
    rm_model.eval()
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_responses = responses[i:i + batch_size]
        texts = [p + r for p, r in zip(batch_prompts, batch_responses)]
        encoded = rm_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = rm_model(**encoded)
            if hasattr(outputs, "logits"):
                batch_scores = outputs.logits.squeeze(-1).float().cpu().tolist()
            else:
                batch_scores = outputs[0].squeeze(-1).float().cpu().tolist()
        if isinstance(batch_scores, float):
            batch_scores = [batch_scores]
        scores.extend(batch_scores)
    return scores


def best_of_n_eval(
    model, tokenizer, prompts, gold_rm, gold_tokenizer, proxy_rm, proxy_tokenizer,
    n_values, device, max_new_tokens, temperature, posterior_pred,
):
    results = []
    tokenizer.padding_side = "left"

    for n in n_values:
        all_gold_scores = []
        all_proxy_scores = []

        for prompt_idx, prompt in enumerate(prompts):
            encoded = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
            candidates = []
            for sample_idx in range(n):
                if n == 1:
                    for _sidx, _vec in posterior_pred.deploy_iter(model):
                        with torch.no_grad():
                            gen = model.generate(
                                **encoded, max_new_tokens=max_new_tokens,
                                do_sample=temperature > 0, temperature=max(temperature, 1e-6),
                            )
                        gen_text = tokenizer.decode(gen[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)
                        candidates.append(gen_text)
                        break
                else:
                    with torch.no_grad():
                        gen = model.generate(
                            **encoded, max_new_tokens=max_new_tokens,
                            do_sample=True, temperature=temperature,
                        )
                    gen_text = tokenizer.decode(gen[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)
                    candidates.append(gen_text)

            proxy_scores = score_with_reward_model(
                proxy_rm, proxy_tokenizer, [prompt] * len(candidates), candidates, device,
            )
            best_idx = int(np.argmax(proxy_scores))
            best_response = candidates[best_idx]

            gold_scores = score_with_reward_model(
                gold_rm, gold_tokenizer, [prompt], [best_response], device,
            )
            all_gold_scores.append(gold_scores[0])
            all_proxy_scores.append(proxy_scores[best_idx])

        results.append({
            "n": n,
            "gold_reward_mean": float(np.mean(all_gold_scores)),
            "gold_reward_std": float(np.std(all_gold_scores)),
            "proxy_reward_mean": float(np.mean(all_proxy_scores)),
            "proxy_reward_std": float(np.std(all_proxy_scores)),
            "gap": float(np.mean(all_proxy_scores) - np.mean(all_gold_scores)),
        })

    return results


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = parse_dtype(args.dtype, device)

    print(f"=== Alignment Experiment ===")
    print(f"  Model: {args.base_model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Device: {device}, dtype: {dtype}, bf16 native: {supports_bf16(device)}")
    print(f"  Output: {output_dir}")

    Path(output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, model_max_length=args.max_length)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype,
    ).to(device)

    rm_load_kwargs: dict = {"torch_dtype": dtype}
    if args.reward_int8 and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        rm_load_kwargs = {
            "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
            "device_map": {"": device.index if device.index is not None else 0},
        }

    print(f"Loading reward models (int8={args.reward_int8})...")
    gold_tokenizer = AutoTokenizer.from_pretrained(args.gold_rm)
    gold_rm = AutoModelForSequenceClassification.from_pretrained(
        args.gold_rm, **rm_load_kwargs,
    ).eval()
    if "device_map" not in rm_load_kwargs:
        gold_rm = gold_rm.to(device)

    proxy_tokenizer = AutoTokenizer.from_pretrained(args.proxy_rm)
    proxy_rm = AutoModelForSequenceClassification.from_pretrained(
        args.proxy_rm, **rm_load_kwargs,
    ).eval()
    if "device_map" not in rm_load_kwargs:
        proxy_rm = proxy_rm.to(device)

    print("Loading dataset...")
    train_data = load_ultrafeedback(args.dataset, args.train_split, args.max_train_samples)
    test_data = load_ultrafeedback(args.dataset, args.test_split, args.num_eval_prompts)
    eval_prompts = [d["prompt"] for d in test_data[:args.num_eval_prompts]]
    print(f"  Train: {len(train_data)}, Eval prompts: {len(eval_prompts)}")

    all_bon_rows = []
    all_summary_rows = []

    for seed_idx in range(args.seed_count):
        seed = args.base_seed + seed_idx
        set_global_seed(seed)
        print(f"\n--- Seed {seed} ({seed_idx + 1}/{args.seed_count}) ---")

        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        adapters_dir = seed_dir / "adapters"
        adapters_dir.mkdir(exist_ok=True)
        checkpoints_dir = seed_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)
        training_dir = seed_dir / "training"
        training_dir.mkdir(exist_ok=True)

        model = setup_lora_model(base_model, args)
        model.to(device)

        train_dataset = DPODataset(train_data, tokenizer, max_length=args.max_length)

        print("Training DPO with checkpoint retention (ref via disable_adapter)...")
        retained, manifest = train_dpo_with_retention(
            model=model,
            ref_model=None,
            train_dataset=train_dataset,
            device=device,
            lr=args.lr,
            beta=args.beta,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            keep_last=args.keep_last,
            tail_fraction=args.tail_fraction,
            loss_type=args.loss_type,
            checkpoint_id_prefix=f"s{seed}",
            save_dir=str(training_dir),
        )
        print(f"  Retained {len(retained)} checkpoints")

        from meta_swag.adapters.state import save_manifest
        save_manifest(manifest, checkpoints_dir / "manifest.json")

        vectors = np.stack([r.adapter_vector for r in retained], axis=0)
        meta_rows = [r.metadata() for r in retained]
        np.savez_compressed(
            checkpoints_dir / "retained_checkpoints.npz",
            vectors=vectors,
        )
        pd.DataFrame(meta_rows).to_csv(checkpoints_dir / "checkpoint_metadata.csv", index=False)
        print(f"  Saved {len(retained)} checkpoint vectors -> {checkpoints_dir}")

        torch.cuda.empty_cache()

        scores = np.array([r.train_loss for r in retained], dtype=np.float32)
        scores = -scores
        checkpoints = vectors

        for scheme in args.schemes:
            print(f"  Evaluating scheme: {scheme}")

            if scheme == "laplace":
                val_dataset = DPODataset(train_data[-500:], tokenizer, max_length=args.max_length)
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)

                best_idx = int(np.argmax(scores))
                restore_adapter_state(model, retained[best_idx].adapter_vector, manifest)

                def dpo_loss_fn(mdl, batch):
                    chosen_ids = batch["chosen_input_ids"].to(device)
                    chosen_mask = batch["chosen_attention_mask"].to(device)
                    chosen_labels = batch["chosen_labels"].to(device)
                    out = mdl(input_ids=chosen_ids, attention_mask=chosen_mask)
                    from meta_swag.training.preference import get_batch_logps
                    logps = get_batch_logps(out.logits, chosen_labels)
                    return -logps.mean()

                fisher = compute_diagonal_fisher(
                    model, dpo_loss_fn, manifest,
                    num_batches=args.laplace_fisher_batches,
                    dataloader=val_loader,
                )
                prior_precision = tune_prior_precision(
                    model, manifest, fisher,
                    loss_fn=dpo_loss_fn,
                    val_dataloader=val_loader,
                )
                agg = laplace_posterior(model, manifest, fisher, prior_precision)
            else:
                target_ess = max(8, int(math.ceil(len(retained) / 2)))
                agg = aggregate_adapter_checkpoints(
                    checkpoints=checkpoints,
                    scores=scores,
                    scheme=scheme,
                    beta=1.0,
                    target_ess=target_ess,
                )

            restore_adapter_state(model, agg.mean_vector, manifest)

            adapter_out = adapters_dir / scheme
            adapter_out.mkdir(exist_ok=True)
            try:
                model.save_pretrained(str(adapter_out))
                tokenizer.save_pretrained(str(adapter_out))
            except Exception as e:
                print(f"    WARN: could not save PEFT adapter for {scheme}: {e}")
            np.save(adapter_out / "mean_vector.npy", agg.mean_vector)
            np.save(adapter_out / "diagonal_variance.npy", agg.diagonal_variance)

            posterior_pred = PosteriorPredictive(
                result=agg,
                manifest=manifest,
                num_samples=args.posterior_samples,
                seed=seed,
            )

            bon_results = best_of_n_eval(
                model=model, tokenizer=tokenizer, prompts=eval_prompts[:args.num_eval_prompts],
                gold_rm=gold_rm, gold_tokenizer=gold_tokenizer,
                proxy_rm=proxy_rm, proxy_tokenizer=proxy_tokenizer,
                n_values=args.best_of_n, device=device,
                max_new_tokens=args.eval_max_new_tokens,
                temperature=args.eval_temperature,
                posterior_pred=posterior_pred,
            )

            for row in bon_results:
                all_bon_rows.append({"seed": seed, "scheme": scheme, **row})

            summary_row = {
                "seed": seed, "scheme": scheme,
                "ess": float(agg.effective_sample_size),
                "max_weight": float(agg.max_normalized_weight),
                "posterior_trace": float(agg.posterior_trace),
                "top_eigenvalue_ratio": float(agg.top_eigenvalue_ratio),
                "score_variance": float(agg.score_variance),
                "gold_reward_n1": bon_results[0]["gold_reward_mean"],
                "gold_reward_n256": bon_results[-1]["gold_reward_mean"] if len(bon_results) > 1 else float("nan"),
                "overopt_gap": bon_results[-1]["gap"] if len(bon_results) > 1 else float("nan"),
            }
            all_summary_rows.append(summary_row)

        pd.DataFrame(all_bon_rows).to_csv(seed_dir / "best_of_n.csv", index=False)
        pd.DataFrame(all_summary_rows).to_csv(seed_dir / "summary.csv", index=False)
        print(f"  Seed {seed} results -> {seed_dir}")

        del model
        torch.cuda.empty_cache()

    pd.DataFrame(all_bon_rows).to_csv(output_dir / "best_of_n.csv", index=False)
    pd.DataFrame(all_summary_rows).to_csv(output_dir / "summary.csv", index=False)
    print(f"\n=== Results saved to {output_dir} ===")


if __name__ == "__main__":
    main()
