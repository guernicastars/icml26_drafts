# Meta-SWAG AxBench Benchmark

Self-contained benchmark for Meta-SWAG posterior approximation over LoRA and
PreferenceLoRA adapters on the AxBench Concept500 steering task.

## Layout

```
axbench_benchmark/
  meta_swag/           # Meta-SWAG library (posterior + checkpoint retention)
  requirements.txt     # pinned dependencies
  setup.sh             # venv + pip + clone axbench + data + model download
  download_data.py     # pulls pyvene/axbench-concept500 + neuronpedia SAE descs
  run_experiment.py    # single (model, layer, kind) run -> CSVs
  run_all.sh           # 12-experiment grid scheduled over 4 GPUs in 3 waves
  collect_results.py   # aggregate final_summary.csv -> paper tables
  data/                # created by download_data.py
  external/axbench/    # cloned by setup.sh
  results/             # populated by run_all.sh
  logs/                # per-experiment stdout/stderr
  aggregates/          # populated by collect_results.py
```

## Procedure on the 4xA100 box

```bash
cd axbench_benchmark
bash setup.sh                    # one-time: venv, pip, data, weights
bash run_all.sh                  # ~8h wall for default config, mock judge
python collect_results.py        # aggregates -> aggregates/
```

## Experiment grid

Twelve runs, matching HyperSteer (arXiv:2506.03292) model/layer choices:

| model                         | layers        | kinds                |
|-------------------------------|---------------|----------------------|
| google/gemma-2-2b-it          | 10, 20        | lora, preference_lora|
| google/gemma-2-9b-it          | 20, 31        | lora, preference_lora|
| meta-llama/Llama-3.1-8B-Inst. | 10, 20        | lora, preference_lora|

Scheduling: 4 GPUs, 1 run per GPU (no tensor parallelism). Waves:

- Wave 1: all four Gemma-2B runs in parallel.
- Wave 2: Gemma-9B L20 (x2) + Llama-8B L10 (x2).
- Wave 3: Gemma-9B L31 (x2) + Llama-8B L20 (x2).

## Key env overrides for run_all.sh

```
NUM_GPUS=4            # 1 run per GPU
MAX_CONCEPTS=30       # concepts per experiment
SEED_COUNT=3
KEEP_LAST=20          # checkpoint retention window
N_EPOCHS=1
REAL_JUDGE=0          # 1 to call OpenAI judge (needs OPENAI_API_KEY)
EXTRA_ARGS=""         # passthrough to run_experiment.py
```

## Output per run

`results/<model_tag>_L<layer>_<kind>/`:

- `config.json`               captured CLI config
- `checkpoint_metrics.csv`    per-checkpoint validation metrics
- `factor_sweeps.csv`         per-factor composite for val and test
- `final_summary.csv`         selected factor + test composite per scheme
- `summary_by_scheme.csv`     mean/std across concepts per scheme

Meta-SWAG weighting schemes compared per concept:
`map`, `uniform`, `softmax`, `ess`, `threshold`.
