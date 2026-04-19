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

RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results}"
MMLU_RESULTS="${MMLU_RESULTS:-$PROJECT_ROOT/results/mmlu}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/mmlu}"
NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DTYPE="${DTYPE:-auto}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
mkdir -p "$MMLU_RESULTS" "$LOG_DIR"

MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct|llama-3.1-8b"
    "google/gemma-2-9b-it|gemma-2-9b"
)

GROUPS=("STEM" "Humanities" "Social Sciences" "Other")

JOBS=()
for spec in "${MODELS[@]}"; do
    IFS='|' read -r model_name model_tag <<< "$spec"
    adapter_dir="$RESULTS_DIR/alignment/${model_tag}_dpo"
    if [ ! -d "$adapter_dir" ]; then
        echo "SKIP $model_tag: no adapter dir at $adapter_dir"
        continue
    fi
    for group in "${GROUPS[@]}"; do
        JOBS+=("${model_name}|${model_tag}|${group}")
    done
done

TOTAL=${#JOBS[@]}
if [ "$TOTAL" -eq 0 ]; then
    echo "No MMLU jobs (all adapter dirs missing)."; exit 1
fi

echo "========================================"
echo "  MMLU: $TOTAL jobs across $NUM_GPUS GPUs"
echo "========================================"

launch_job() {
    local gpu_id="$1"
    local job="$2"

    IFS='|' read -r model_name model_tag group <<< "$job"
    local adapter_dir="$RESULTS_DIR/alignment/${model_tag}_dpo"
    local group_tag=$(echo "$group" | tr '[:upper:] ' '[:lower:]_')
    local out_dir="$MMLU_RESULTS/${model_tag}"
    local log_file="$LOG_DIR/${model_tag}_${group_tag}.log"
    mkdir -p "$out_dir"

    echo "[GPU $gpu_id] START $model_tag/$group -> $log_file"

    local include_base_arg="--include-base"
    if [ "$group_tag" != "stem" ]; then
        include_base_arg="--no-include-base"
    fi

    CUDA_VISIBLE_DEVICES="$gpu_id" \
        python -m benchmarks.mmlu.run_mmlu \
            --base-model "$model_name" \
            --adapter-dir "$adapter_dir" \
            --output-dir "$out_dir" \
            --subject-group "$group" \
            --batch-size "$BATCH_SIZE" \
            --dtype "$DTYPE" \
            $include_base_arg \
            $EXTRA_ARGS \
            > "$log_file" 2>&1 &

    echo "$!"
}

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
        IFS='|' read -r _ model_tag group <<< "$job"
        group_tag=$(echo "$group" | tr '[:upper:] ' '[:lower:]_')
        name="${model_tag}_${group_tag}"
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
python - <<'PY' "$MMLU_RESULTS"
import sys, glob, os
import pandas as pd
root = sys.argv[1]
for model_dir in sorted(glob.glob(os.path.join(root, "*"))):
    if not os.path.isdir(model_dir):
        continue
    frames = []
    for csv in sorted(glob.glob(os.path.join(model_dir, "mmlu_results_*.csv"))):
        frames.append(pd.read_csv(csv))
    if not frames:
        continue
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["seed", "scheme", "subject"], keep="last")
    merged.to_csv(os.path.join(model_dir, "mmlu_results.csv"), index=False)

    rows = []
    for (seed, scheme), df in merged.groupby(["seed", "scheme"]):
        row = {
            "seed": int(seed), "scheme": scheme,
            "overall_accuracy": df["correct"].sum() / max(df["n_questions"].sum(), 1),
            "total_correct": int(df["correct"].sum()),
            "total_questions": int(df["n_questions"].sum()),
        }
        for group, gdf in df.groupby("subject_group"):
            row[f"{group}_accuracy"] = gdf["correct"].sum() / max(gdf["n_questions"].sum(), 1)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(model_dir, "mmlu_summary.csv"), index=False)
    print(f"[merge] {os.path.basename(model_dir)}: {len(merged)} rows -> mmlu_summary.csv")
PY
echo "  Failures: $TOTAL_FAIL"
echo "  Results:  $MMLU_RESULTS"
echo "========================================"

exit $TOTAL_FAIL
