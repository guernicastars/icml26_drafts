#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error

RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

NUM_GPUS="${NUM_GPUS:-4}"
SEED_COUNT="${SEED_COUNT:-3}"
MAX_CONCEPTS="${MAX_CONCEPTS:-30}"
KEEP_LAST="${KEEP_LAST:-20}"
N_EPOCHS="${N_EPOCHS:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ "${REAL_JUDGE:-0}" = "1" ]; then
    JUDGE_FLAG="--real-judge"
else
    JUDGE_FLAG="--mock-judge"
fi

# Experiment grid: (model_name, layer, kind, model_tag)
# Wave 1: Gemma-2-2B (fastest, fits any A100)
WAVE_1=(
    "google/gemma-2-2b-it|10|lora|gemma-2-2b-it"
    "google/gemma-2-2b-it|10|preference_lora|gemma-2-2b-it"
    "google/gemma-2-2b-it|20|lora|gemma-2-2b-it"
    "google/gemma-2-2b-it|20|preference_lora|gemma-2-2b-it"
)

# Wave 2: Gemma-2-9B L20 + Llama-3.1-8B L10
WAVE_2=(
    "google/gemma-2-9b-it|20|lora|gemma-2-9b-it"
    "google/gemma-2-9b-it|20|preference_lora|gemma-2-9b-it"
    "meta-llama/Llama-3.1-8B-Instruct|10|lora|Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct|10|preference_lora|Llama-3.1-8B-Instruct"
)

# Wave 3: Gemma-2-9B L31 + Llama-3.1-8B L20
WAVE_3=(
    "google/gemma-2-9b-it|31|lora|gemma-2-9b-it"
    "google/gemma-2-9b-it|31|preference_lora|gemma-2-9b-it"
    "meta-llama/Llama-3.1-8B-Instruct|20|lora|Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct|20|preference_lora|Llama-3.1-8B-Instruct"
)

launch_experiment() {
    local gpu_id="$1"
    local spec="$2"

    IFS='|' read -r model_name layer kind model_tag <<< "$spec"
    local exp_name="${model_tag}_L${layer}_${kind}"
    local out_dir="$RESULTS_DIR/$exp_name"
    local log_file="$LOG_DIR/${exp_name}.log"

    mkdir -p "$out_dir"

    echo "[GPU $gpu_id] START $exp_name -> $log_file"

    CUDA_VISIBLE_DEVICES="$gpu_id" \
        python run_experiment.py \
            --model-name "$model_name" \
            --layer "$layer" \
            --model-kind "$kind" \
            --output-dir "$out_dir" \
            --max-concepts "$MAX_CONCEPTS" \
            --seed-count "$SEED_COUNT" \
            --keep-last "$KEEP_LAST" \
            --n-epochs "$N_EPOCHS" \
            $JUDGE_FLAG \
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
        IFS='|' read -r _ layer kind model_tag <<< "$spec"
        local exp_name="${model_tag}_L${layer}_${kind}"

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

run_wave "Wave 1 (Gemma-2-2B)"      "${WAVE_1[@]}" || TOTAL_FAIL=$((TOTAL_FAIL + $?))
run_wave "Wave 2 (Gemma-9B L20 + Llama-8B L10)" "${WAVE_2[@]}" || TOTAL_FAIL=$((TOTAL_FAIL + $?))
run_wave "Wave 3 (Gemma-9B L31 + Llama-8B L20)" "${WAVE_3[@]}" || TOTAL_FAIL=$((TOTAL_FAIL + $?))

echo ""
echo "========================================"
echo "  All waves complete. Failures: $TOTAL_FAIL"
echo "  Aggregate results: python collect_results.py --results-dir $RESULTS_DIR"
echo "========================================"

exit $TOTAL_FAIL
