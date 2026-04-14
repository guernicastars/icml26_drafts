# Meta-SWAG Experiments Pipeline

This directory contains standalone scripts and code to deploy the Meta-SWAG benchmark experiments on a remote multi-GPU server (e.g., a 4xA100 node). It is configured to run the full suite of methods—including point-estimates (MAP, EMA, SWA) and Bayesian approximations (Meta-SWAG variants, Laplace-LoRA)—for a complete "apples-to-apples" comparison.

## Pipeline Execution

You should run the scripts numbered 00 through 05 **sequentially**. They have been written to automatically parallelize execution under-the-hood to fully utilize all 4 GPUs when possible, while tailing logs to your terminal so you can monitor progress.

Run the pipeline sequentially using these commands:

1. **`./00_setup.sh`**
   - Sets up the Python virtual environment (`.venv`).
   - Installs all local dependencies from `requirements.txt`.
   - Clones and installs the Stanford NLP AxBench repository.

2. **`./01_prestage.sh`**
   - Downloads datasets and foundational backbone models onto the disk natively. This prevents race conditions and redundant network calls from parallel workers.

3. **`./02_run_alignment.sh`**
   - Dispatches Meta-SWAG and DPO evaluations across all GPUs.
   - Evaluates combinations, sweeps configurations and performs best-of-n checks.

4. **`./03_run_mmlu.sh`**
   - Runs out-of-distribution capabilities preservation metrics via MMLU on the aligned models.

5. **`./04_run_axbench.sh`**
   - Sweeps steering interventions across LLM layers using meta-swag methods and concept data.

6. **`./05_collect_results.sh`**
   - Synthesizes raw metric tables into a unified high-level summary.

## Outputs and Logging

To avoid state clobbering or race conditions from parallel pipelines, results and logging are segmented statically.

- All raw execution logs are saved cleanly to the **`logs/`** directory. During benchmark execution, standard out will be hidden, but the `0x` launcher scripts will actively output a `tail -f logs/*.log` pipeline sequentially.
- Once completed, all outputs will persist inside the **`results/`** directory, cleanly partitioned into:
  - **`results/alignment/`**: Final checkpoint vectors, `manifest.json`, best-of-n generations (`best_of_n.csv`), and LoRA adapter weights for each DPO sweep.
  - **`results/mmlu/`**: MMLU scores.
  - **`results/axbench/`**: Adapter state data and performance reports separated into subdirectories per seed and concept ID. Adapter weights are explicitly dumped inside their specific scheme directories without conflicts.
  - **`results/summary.txt`**: Generates a unified dashboard view with metrics gathered from `05_collect_results.sh`.

When run is complete, zip up the `results/` folder and transfer it back to your local machine for analysis and paper generation.
