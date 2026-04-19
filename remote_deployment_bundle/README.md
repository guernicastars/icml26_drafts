# Meta-SWAG Experiments Pipeline

This directory contains standalone scripts and code to deploy the Meta-SWAG benchmark experiments on a remote multi-GPU server. It is tuned for a **4× V100-SXM2-32GB** node (Volta, sm_70, 128 GB aggregate VRAM, no native bfloat16, no FlashAttention-2) and also runs unchanged on Ampere/Hopper (A100/H100). It runs the full method suite — point-estimates (MAP, last-iterate, SWA, EMA) and Bayesian approximations (Meta-SWAG softmax/ESS/threshold, diagonal Laplace-LoRA) — for an apples-to-apples comparison.

## Hardware notes (Volta / V100)

- Auto dtype (`--dtype auto`, default everywhere) selects **fp16** on sm_70 and **bf16** on sm_80+. Hard-coded bf16 was removed from training, MMLU, axbench, and reward-model wrappers because Volta emulates bf16 in fp32 — slow and NaN-prone on long DPO trajectories.
- Reward models load in **int8 via bitsandbytes** (`--reward-int8`, default on). On V100-32GB this is required; on A100-80GB it is optional and can be disabled with `--no-reward-int8`.
- The DPO trainer uses **PEFT `disable_adapter()`** as its reference path rather than loading a second copy of the base model — saves ~16 GB during training.
- Peak alignment footprint per GPU: ~26 GB (policy 16 GB + activations + optimizer state + int8 RMs ~5 GB). Fits 32 GB V100.

## Pipeline Execution

Run the scripts numbered 00 → 05 **sequentially**. Each `0x_run_*.sh` launcher parallelises across all 4 GPUs under the hood; per-job stdout is redirected to `logs/` and a summary prints to the terminal.

1. **`./00_setup.sh`** — creates `.venv`, installs `requirements.txt`, clones/installs Stanford AxBench, then probes every GPU (prints compute capability, VRAM, bf16 support) and warns if Volta cards are present.

2. **`./01_prestage.sh`** — downloads datasets and backbone checkpoints serially to prevent race conditions when parallel workers later call the HuggingFace cache.

3. **`./02_run_alignment.sh`** — DPO training + Meta-SWAG aggregation + best-of-n reward-overoptimisation curves.
   - Jobs = `{Llama-3.1-8B-Instruct, Gemma-2-9B-it} × {seed 42, 43, 44}` = **6 jobs in 2 waves of 4 GPUs** (external seed sharding; the Python script runs one seed per process).
   - Env knobs: `NUM_GPUS`, `SEEDS`, `DTYPE`, `REWARD_INT8`, `N_EPOCHS`, `POSTERIOR_SAMPLES`, `KEEP_LAST`, `MAX_TRAIN_SAMPLES`, `EXTRA_ARGS`.

4. **`./03_run_mmlu.sh`** — MMLU on the *aligned* adapters. For each seed under `results/alignment/<model>/seed_*/adapters/<scheme>/mean_vector.npy` the script restores the LoRA state via the saved `manifest.json` and evaluates every scheme (not just the base model). Also records a `base` row for reference.
   - Jobs = `{Llama-3.1-8B, Gemma-2-9B} × {STEM, Humanities, Social Sciences, Other}` = **8 jobs in 2 waves of 4 GPUs** (subject-group sharding). Per-group CSVs are merged into a single `mmlu_summary.csv` per model at the end.
   - Env knobs: `NUM_GPUS`, `BATCH_SIZE`, `DTYPE`, `EXTRA_ARGS`.

5. **`./04_run_axbench.sh`** — sweeps steering interventions across LLM layers using Meta-SWAG methods on AxBench concept data. Already 4-GPU parallel.

6. **`./05_collect_results.sh`** — synthesises raw metric tables into a unified summary dashboard.

## Outputs and Logging

- All raw execution logs live in **`logs/`** (one log per job).
- Final artefacts persist inside **`results/`**:
  - **`results/alignment/<model>_dpo/seed_<s>/`** — retained checkpoint vectors, `manifest.json`, per-scheme adapter weights + `mean_vector.npy`, `best_of_n.csv`, `summary.csv`, `training/loss_curve.{csv,png}`.
  - **`results/mmlu/<model>/`** — `mmlu_results.csv` (per-subject), `mmlu_summary.csv` (per-scheme overall + per-group accuracies). Feeds Fig 3 bottom-row ("BMA schemes stay within 0.5pp of MAP").
  - **`results/axbench/`** — per-seed / per-concept adapter state and scheme performance reports.
  - **`results/summary.txt`** — unified dashboard from `05_collect_results.sh`.

When the run is complete, zip up `results/` and transfer it back to your local machine for analysis and paper generation.
