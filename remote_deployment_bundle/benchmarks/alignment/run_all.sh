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
SEEDS="${SEEDS:-42 43 44}"
POSTERIOR_SAMPLES="${POSTERIOR_SAMPLES:-16}"
KEEP_LAST="${KEEP_LAST:-50}"
N_EPOCHS="${N_EPOCHS:-3}"
NUM_EVAL_PROMPTS="${NUM_EVAL_PROMPTS:-50}"
BON_N="${BON_N:-1 4 16 64}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
DTYPE="${DTYPE:-auto}"
REWARD_INT8="${REWARD_INT8:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Gemma excluded for workshop scope; add back with BON_N="1 4 16 64" Gemma=1 if time permits
MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct|llama-3.1-8b"
)

JOBS=()
for spec in "${MODELS[@]}"; do
    for seed in $SEEDS; do
        JOBS+=("${spec}|${seed}")
    done
done

launch_job() {
    local gpu_id="$1"
    local job="$2"

    IFS='|' read -r model_name model_tag seed <<< "$job"
    local exp_name="${model_tag}_dpo"
    local out_dir="$RESULTS_DIR/$exp_name"
    local log_file="$LOG_DIR/${exp_name}_seed${seed}.log"

    mkdir -p "$out_dir"

    echo "[GPU $gpu_id] START $exp_name seed=$seed -> $log_file"

    local train_limit_arg=""
    if [ -n "$MAX_TRAIN_SAMPLES" ]; then
        train_limit_arg="--max-train-samples $MAX_TRAIN_SAMPLES"
    fi

    local rm_arg="--reward-int8"
    if [ "$REWARD_INT8" != "1" ]; then
        rm_arg="--no-reward-int8"
    fi

    CUDA_VISIBLE_DEVICES="$gpu_id" \
        python -m benchmarks.alignment.run_experiment \
            --base-model "$model_name" \
            --output-dir "$out_dir" \
            --seed-count 1 \
            --base-seed "$seed" \
            --posterior-samples "$POSTERIOR_SAMPLES" \
            --keep-last "$KEEP_LAST" \
            --n-epochs "$N_EPOCHS" \
            --num-eval-prompts "$NUM_EVAL_PROMPTS" \
            --best-of-n $BON_N \
            --dtype "$DTYPE" \
            $rm_arg \
            $train_limit_arg \
            $EXTRA_ARGS \
            > "$log_file" 2>&1 &

    echo "$!"
}

# Partition JOBS into waves of NUM_GPUS parallel runs.
TOTAL=${#JOBS[@]}
echo "========================================"
echo "  DPO Alignment: $TOTAL jobs across $NUM_GPUS GPUs"
echo "  Models: ${MODELS[*]}"
echo "  Seeds : $SEEDS"
echo "========================================"

TOTAL_FAIL=0
wave_idx=0
for ((start=0; start<TOTAL; start+=NUM_GPUS)); do
    wave_idx=$((wave_idx + 1))
    end=$((start + NUM_GPUS))
    if [ $end -gt $TOTAL ]; then end=$TOTAL; fi

    pids=()
    names=()
    echo ""
    echo "----- wave $wave_idx ($(date '+%H:%M:%S')): jobs [$start..$((end-1))] -----"
    for ((i=start; i<end; i++)); do
        gpu_id=$(( (i - start) % NUM_GPUS ))
        job="${JOBS[$i]}"
        IFS='|' read -r _ model_tag seed <<< "$job"
        name="${model_tag}_seed${seed}"
        pid=$(launch_job "$gpu_id" "$job")
        pids+=("$pid")
        names+=("$name")
    done

    for idx in "${!pids[@]}"; do
        if wait "${pids[$idx]}"; then
            echo "[OK]   ${names[$idx]}"
        else
            echo "[FAIL] ${names[$idx]} (see $LOG_DIR/${names[$idx]}.log)"
            TOTAL_FAIL=$((TOTAL_FAIL + 1))
        fi
    done
done

echo ""
echo "========================================"
echo "  All jobs complete. Failures: $TOTAL_FAIL"
echo "  Results: $RESULTS_DIR"
echo "========================================"

exit $TOTAL_FAIL
