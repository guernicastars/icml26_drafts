"""Download AxBench Concept500 data from HuggingFace and prepare per-config data dirs.

Creates:
  data/<model_tag>_L<layer>/
    train_data.parquet    - training examples (concept_id, input, output, category)
    dpo_train_data.parquet - DPO-format training examples (for preference_lora)
    metadata.jsonl        - concept metadata (concept_id, concept, ...)
    steering_data.parquet - steering evaluation data
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

BENCHMARK_ROOT = Path(__file__).resolve().parent
DATA_ROOT = BENCHMARK_ROOT / "data"

CONFIGS = [
    {"model_tag": "gemma-2-2b-it", "layer": 10},
    {"model_tag": "gemma-2-2b-it", "layer": 20},
    {"model_tag": "gemma-2-9b-it", "layer": 20},
    {"model_tag": "gemma-2-9b-it", "layer": 31},
    {"model_tag": "Llama-3.1-8B-Instruct", "layer": 10},
    {"model_tag": "Llama-3.1-8B-Instruct", "layer": 20},
]


def config_dir(model_tag: str, layer: int) -> Path:
    return DATA_ROOT / f"{model_tag}_L{layer}"


def download_concept500():
    """Download the full Concept500 dataset from HuggingFace."""
    print("Loading pyvene/axbench-concept500 from HuggingFace...")
    ds = load_dataset("pyvene/axbench-concept500", trust_remote_code=True)
    return ds


def download_concept_descriptions():
    """Download SAE concept description files needed for metadata."""
    desc_dir = DATA_ROOT / "concept_descriptions"
    desc_dir.mkdir(parents=True, exist_ok=True)

    urls = {
        "gemma-2-2b_10": "https://neuronpedia-exports.s3.amazonaws.com/explanations-only/gemma-2-2b_10-gemmascope-res-16k.json",
        "gemma-2-2b_20": "https://neuronpedia-exports.s3.amazonaws.com/explanations-only/gemma-2-2b_20-gemmascope-res-16k.json",
        "gemma-2-9b-it_20": "https://neuronpedia-exports.s3.amazonaws.com/explanations-only/gemma-2-9b-it_20-gemmascope-res-131k.json",
        "gemma-2-9b-it_31": "https://neuronpedia-exports.s3.amazonaws.com/explanations-only/gemma-2-9b-it_31-gemmascope-res-131k.json",
    }

    import urllib.request
    for name, url in urls.items():
        target = desc_dir / f"{name}.json"
        if target.exists():
            print(f"  Already have {name}.json")
            continue
        print(f"  Downloading {name}.json...")
        try:
            urllib.request.urlretrieve(url, str(target))
        except Exception as e:
            print(f"  WARN: Failed to download {name}: {e}")

    return desc_dir


def build_metadata_from_descriptions(desc_path: Path) -> list[dict]:
    """Parse neuronpedia SAE feature descriptions into metadata entries."""
    if not desc_path.exists():
        return []
    with open(desc_path) as f:
        raw = json.load(f)

    metadata = []
    for entry in raw:
        concept_id = entry.get("index", entry.get("feature_index", -1))
        description = entry.get("description", entry.get("explanation", ""))
        if concept_id < 0 or not description:
            continue
        metadata.append({
            "concept_id": int(concept_id),
            "concept": description.strip(),
            "source": str(desc_path.name),
        })
    return metadata


def build_metadata_from_dataframe(df: pd.DataFrame) -> list[dict]:
    """Build metadata from training dataframe concept columns."""
    metadata = []
    if "concept_id" not in df.columns:
        return metadata

    concept_col = "output_concept" if "output_concept" in df.columns else "concept"
    if concept_col not in df.columns:
        concept_col = None

    for cid in sorted(df["concept_id"].unique()):
        cid = int(cid)
        if cid < 0:
            continue
        concept_name = f"concept_{cid}"
        if concept_col:
            names = df[df["concept_id"] == cid][concept_col].dropna().unique()
            names = [n for n in names if n and str(n).strip() not in ("", "EEEEE")]
            if names:
                concept_name = str(names[0]).strip()
        metadata.append({
            "concept_id": cid,
            "concept": concept_name,
        })
    return metadata


def prepare_train_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize training dataframe to expected schema."""
    result = df.copy()

    if "output_concept" in result.columns and "category" not in result.columns:
        result["category"] = result["output_concept"].apply(
            lambda x: "negative" if str(x).strip() == "EEEEE" else "positive"
        )

    required = ["concept_id", "input"]
    for col in required:
        if col not in result.columns:
            raise KeyError(f"Training data missing required column: {col}")

    if "output" not in result.columns and "completion" in result.columns:
        result["output"] = result["completion"]

    return result


def prepare_dpo_data(df: pd.DataFrame) -> pd.DataFrame:
    """Build DPO-style preference pairs from training data."""
    pos = df[df["category"] == "positive"].copy()
    neg = df[df["category"] == "negative"].copy()

    if pos.empty or neg.empty:
        return pd.DataFrame()

    pairs = []
    for cid in pos["concept_id"].unique():
        cid_pos = pos[pos["concept_id"] == cid]
        cid_neg = neg[neg["concept_id"] == cid]
        if cid_pos.empty or cid_neg.empty:
            continue
        n = min(len(cid_pos), len(cid_neg))
        for i in range(n):
            p = cid_pos.iloc[i]
            n_row = cid_neg.iloc[i % len(cid_neg)]
            pairs.append({
                "concept_id": int(cid),
                "input": p["input"],
                "chosen": p.get("output", ""),
                "rejected": n_row.get("output", ""),
                "category": "positive",
            })
    return pd.DataFrame(pairs)


def prepare_steering_data(df: pd.DataFrame, factors: list[float]) -> pd.DataFrame:
    """Build steering evaluation dataframe from positive examples."""
    pos = df[df.get("category", pd.Series(["positive"] * len(df))) == "positive"].copy()
    if pos.empty:
        pos = df.copy()

    unique_concepts = pos["concept_id"].unique()
    rows = []
    for cid in unique_concepts:
        concept_rows = pos[pos["concept_id"] == cid].head(10)
        for idx, (_, row) in enumerate(concept_rows.iterrows()):
            for factor in factors:
                rows.append({
                    "concept_id": int(cid),
                    "input": row["input"],
                    "original_prompt": row["input"],
                    "input_concept": row.get("output_concept", row.get("concept", f"concept_{cid}")),
                    "dataset_name": "Concept500",
                    "input_id": idx,
                    "factor": factor,
                })
    return pd.DataFrame(rows)


def map_config_to_desc_key(model_tag: str, layer: int) -> str:
    """Map a benchmark config to a concept description file key."""
    if "gemma-2-2b" in model_tag:
        return f"gemma-2-2b_{layer}"
    if "gemma-2-9b" in model_tag:
        return f"gemma-2-9b-it_{layer}"
    if "Llama" in model_tag or "llama" in model_tag:
        return f"gemma-2-2b_{min(layer, 20)}"
    return f"gemma-2-2b_20"


def main():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    factors = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0]

    print("=== Downloading AxBench data ===\n")

    desc_dir = download_concept_descriptions()

    print("\nLoading Concept500 dataset from HuggingFace...")
    try:
        ds = download_concept500()
        splits = list(ds.keys())
        print(f"  Available splits: {splits}")

        if "train" in ds:
            full_df = ds["train"].to_pandas()
        elif splits:
            full_df = ds[splits[0]].to_pandas()
        else:
            raise ValueError("No splits found in dataset")

        print(f"  Loaded {len(full_df)} rows, columns: {list(full_df.columns)}")
    except Exception as e:
        print(f"  WARN: Could not load from HF datasets API: {e}")
        print("  Attempting direct parquet download...")
        full_df = None

    for cfg in CONFIGS:
        model_tag = cfg["model_tag"]
        layer = cfg["layer"]
        out_dir = config_dir(model_tag, layer)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Preparing {model_tag} L{layer} ---")

        desc_key = map_config_to_desc_key(model_tag, layer)
        desc_path = desc_dir / f"{desc_key}.json"

        if full_df is not None and "concept_id" in full_df.columns:
            df = full_df.copy()
        else:
            print(f"  No training data available, creating minimal placeholder")
            df = pd.DataFrame(columns=["concept_id", "input", "output", "category"])

        if not df.empty:
            train_df = prepare_train_data(df)

            concept_ids = sorted(
                int(c) for c in train_df["concept_id"].unique() if int(c) >= 0
            )
            print(f"  {len(concept_ids)} concepts, {len(train_df)} training rows")

            train_df.to_parquet(out_dir / "train_data.parquet", index=False)

            dpo_df = prepare_dpo_data(train_df)
            if not dpo_df.empty:
                dpo_df.to_parquet(out_dir / "dpo_train_data.parquet", index=False)
                print(f"  {len(dpo_df)} DPO pairs")

            steer_df = prepare_steering_data(train_df, factors)
            if not steer_df.empty:
                steer_df.to_parquet(out_dir / "steering_data.parquet", index=False)
                print(f"  {len(steer_df)} steering eval rows")

            if desc_path.exists():
                metadata = build_metadata_from_descriptions(desc_path)
                meta_cids = {m["concept_id"] for m in metadata}
                data_cids = set(concept_ids)
                missing = data_cids - meta_cids
                if missing:
                    extra = build_metadata_from_dataframe(
                        train_df[train_df["concept_id"].isin(missing)]
                    )
                    metadata.extend(extra)
            else:
                metadata = build_metadata_from_dataframe(train_df)

            with open(out_dir / "metadata.jsonl", "w") as f:
                for entry in sorted(metadata, key=lambda x: x["concept_id"]):
                    f.write(json.dumps(entry) + "\n")
            print(f"  {len(metadata)} metadata entries")
        else:
            print("  WARN: Empty dataset, check HF download")

    print("\n=== Data preparation complete ===")
    print(f"Data stored in: {DATA_ROOT}")


if __name__ == "__main__":
    main()
