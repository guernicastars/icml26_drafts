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
export TRANSFORMERS_VERBOSITY=error

RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results/alignment}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/alignment}"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

NUM_GPUS="${NUM_GPUS:-4}"
SEED_COUNT="${SEED_COUNT:-3}"
POSTERIOR_SAMPLES="${POSTERIOR_SAMPLES:-16}"
KEEP_LAST="${KEEP_LAST:-50}"
N_EPOCHS="${N_EPOCHS:-3}"
NUM_EVAL_PROMPTS="${NUM_EVAL_PROMPTS:-1000}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

WAVE_1=(
    "meta-llama/Llama-3.1-8B-Instruct|llama-3.1-8b"
    "google/gemma-2-9b-it|gemma-2-9b"
)

launch_experiment() {
    local gpu_id="$1"
    local spec="$2"

    IFS='|' read -r model_name model_tag <<< "$spec"
    local exp_name="${model_tag}_dpo"
    local out_dir="$RESULTS_DIR/$exp_name"
    local log_file="$LOG_DIR/${exp_name}.log"

    mkdir -p "$out_dir"

    echo "[GPU $gpu_id] START $exp_name -> $log_file"

    local train_limit_arg=""
    if [ -n "$MAX_TRAIN_SAMPLES" ]; then
        train_limit_arg="--max-train-samples $MAX_TRAIN_SAMPLES"
    fi

    CUDA_VISIBLE_DEVICES="$gpu_id" \
        python -m benchmarks.alignment.run_experiment \
            --base-model "$model_name" \
            --output-dir "$out_dir" \
            --seed-count "$SEED_COUNT" \
            --posterior-samples "$POSTERIOR_SAMPLES" \
            --keep-last "$KEEP_LAST" \
            --n-epochs "$N_EPOCHS" \
            --num-eval-prompts "$NUM_EVAL_PROMPTS" \
            $train_limit_arg \
            $EXTRA_ARGS \
            > "$log_file" 2>&1 &

    echo "$!"
}

run_wave() {
    local wave_name="$1"
    shift
    local -a specs=("$@")

    echo ""
    echo "========================================"
    echo "  $wave_name  ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "  ${#specs[@]} experiments across $NUM_GPUS GPUs"
    echo "========================================"

    local -a pids=()
    local -a names=()

    for i in "${!specs[@]}"; do
        local gpu_id=$(( i % NUM_GPUS ))
        local spec="${specs[$i]}"
        IFS='|' read -r _ model_tag <<< "$spec"
        local exp_name="${model_tag}_dpo"

        local pid
        pid=$(launch_experiment "$gpu_id" "$spec")
        pids+=("$pid")
        names+=("$exp_name")
    done

    local failed=0
    for i in "${!pids[@]}"; do
        local pid="${pids[$i]}"
        local name="${names[$i]}"
        if wait "$pid"; then
            echo "[OK]   $name"
        else
            echo "[FAIL] $name (see $LOG_DIR/${name}.log)"
            failed=$((failed + 1))
        fi
    done

    echo "  $wave_name complete ($(date '+%Y-%m-%d %H:%M:%S')), $failed failure(s)"
    return $failed
}

TOTAL_FAIL=0
run_wave "DPO Alignment (Llama-8B + Gemma-9B)" "${WAVE_1[@]}" || TOTAL_FAIL=$((TOTAL_FAIL + $?))

echo ""
echo "========================================"
echo "  All experiments complete. Failures: $TOTAL_FAIL"
echo "  Results: $RESULTS_DIR"
echo "========================================"

exit $TOTAL_FAIL
