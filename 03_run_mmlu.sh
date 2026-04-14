#!/usr/bin/env bash
set -euo pipefail

echo "======================================"
echo " 03: Run MMLU Evaluation              "
echo "======================================"

source .venv/bin/activate

export RESULTS_DIR="${PWD}/results"
export MMLU_RESULTS="${PWD}/results/mmlu"
export LOG_DIR="${PWD}/logs/mmlu"

mkdir -p "$MMLU_RESULTS" "$LOG_DIR"
echo "Logs will be written to $LOG_DIR"
echo "Results will be saved in $MMLU_RESULTS"

touch "$LOG_DIR"/tail_combined.log
tail -f "$LOG_DIR"/*.log > "$LOG_DIR"/tail_combined.log 2>/dev/null &
TAIL_PID=$!

bash benchmarks/mmlu/run_all.sh

kill $TAIL_PID 2>/dev/null || true

echo "======================================"
echo " MMLU Complete. Proceed to 04.      "
echo "======================================"
