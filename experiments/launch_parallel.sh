#!/usr/bin/env bash
# Launch experiments in parallel across multiple GPUs.
#
# Usage:
#   bash experiments/launch_parallel.sh ipd 10 4    # IPD, 10 seeds, 4 GPUs
#   bash experiments/launch_parallel.sh rps 30 4    # RPS, 30 seeds, 4 GPUs
#   bash experiments/launch_parallel.sh ipd 30 4 "ew_lola_pg,meta_mapg"  # specific methods
#
# Results are written as individual JSON files per (method, seed).
# After completion, run:
#   python -m experiments.aggregate_results --env ipd --seeds 30

set -euo pipefail

ENV_NAME="${1:?Usage: $0 <env> <n_seeds> <n_gpus> [methods]}"
N_SEEDS="${2:-10}"
N_GPUS="${3:-4}"
METHODS="${4:-all}"

if [ "$METHODS" = "all" ]; then
    METHOD_LIST="reinforce meta_pg lola_dice meta_mapg ew_pg lola_pg ew_lola_pg"
else
    METHOD_LIST=$(echo "$METHODS" | tr ',' ' ')
fi

VENV_PYTHON=".venv/bin/python"
LOG_DIR="experiments/logs"
mkdir -p "$LOG_DIR" experiments/results

echo "=== Parallel Launch ==="
echo "  Env: $ENV_NAME"
echo "  Seeds: 0..$((N_SEEDS - 1))"
echo "  GPUs: $N_GPUS"
echo "  Methods: $METHOD_LIST"
echo ""

RUNNING=0
TOTAL=0

for method in $METHOD_LIST; do
    for seed in $(seq 0 $((N_SEEDS - 1))); do
        GPU_ID=$((TOTAL % N_GPUS))
        LOG_FILE="$LOG_DIR/${ENV_NAME}_${method}_seed${seed}.log"

        echo "  Starting ${method} seed=${seed} on GPU ${GPU_ID}"
        $VENV_PYTHON -m experiments.run_experiment \
            --env "$ENV_NAME" \
            --single-method "$method" \
            --single-seed "$seed" \
            --gpu "$GPU_ID" \
            > "$LOG_FILE" 2>&1 &

        TOTAL=$((TOTAL + 1))
        RUNNING=$((RUNNING + 1))

        if [ "$RUNNING" -ge "$N_GPUS" ]; then
            wait -n 2>/dev/null || wait
            RUNNING=$((RUNNING - 1))
        fi
    done
done

echo ""
echo "Waiting for remaining jobs..."
wait
echo "All $TOTAL jobs complete."
echo ""
echo "Aggregate with:"
echo "  $VENV_PYTHON -m experiments.aggregate_results --env $ENV_NAME --seeds $N_SEEDS"
