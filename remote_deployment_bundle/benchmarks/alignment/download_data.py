"""Download UltraFeedback dataset and reward models for alignment experiments."""
from __future__ import annotations

import argparse
from pathlib import Path


def download_dataset(cache_dir: str | None = None) -> None:
    from datasets import load_dataset
    print("Downloading HuggingFaceH4/ultrafeedback_binarized...")
    load_dataset("HuggingFaceH4/ultrafeedback_binarized", cache_dir=cache_dir)
    print("Done.")


def download_reward_models(cache_dir: str | None = None) -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    models = [
        "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
        "internlm/internlm2-1_8b-reward",
    ]
    for model_name in models:
        print(f"Downloading {model_name}...")
        AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
        print(f"  Done: {model_name}")


def download_base_models(cache_dir: str | None = None) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    models = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "google/gemma-2-9b-it",
    ]
    for model_name in models:
        print(f"Downloading {model_name}...")
        AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        print(f"  Done: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--skip-base-models", action="store_true")
    parser.add_argument("--skip-reward-models", action="store_true")
    parser.add_argument("--skip-dataset", action="store_true")
    args = parser.parse_args()

    if not args.skip_dataset:
        download_dataset(args.cache_dir)
    if not args.skip_reward_models:
        download_reward_models(args.cache_dir)
    if not args.skip_base_models:
        download_base_models(args.cache_dir)


if __name__ == "__main__":
    main()
