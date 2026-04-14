#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results}"
MMLU_RESULTS="${MMLU_RESULTS:-$PROJECT_ROOT/results/mmlu}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/mmlu}"
POSTERIOR_SAMPLES="${POSTERIOR_SAMPLES:-16}"
mkdir -p "$MMLU_RESULTS" "$LOG_DIR"

MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct|llama-3.1-8b"
    "google/gemma-2-9b-it|gemma-2-9b"
)

for spec in "${MODELS[@]}"; do
    IFS='|' read -r model_name model_tag <<< "$spec"

    adapter_dir="$RESULTS_DIR/alignment/${model_tag}_dpo"
    out_dir="$MMLU_RESULTS/${model_tag}"
    log_file="$LOG_DIR/${model_tag}_mmlu.log"

    if [ ! -d "$adapter_dir" ]; then
        echo "SKIP $model_tag: no adapter dir at $adapter_dir"
        continue
    fi

    echo "MMLU eval: $model_tag -> $log_file"
    mkdir -p "$out_dir"

    python -m benchmarks.mmlu.run_mmlu \
        --base-model "$model_name" \
        --adapter-dir "$adapter_dir" \
        --output-dir "$out_dir" \
        --posterior-samples "$POSTERIOR_SAMPLES" \
        > "$log_file" 2>&1

    if [ $? -eq 0 ]; then
        echo "[OK]   $model_tag"
    else
        echo "[FAIL] $model_tag (see $log_file)"
    fi
done

echo ""
echo "MMLU evaluation complete. Results: $MMLU_RESULTS"
