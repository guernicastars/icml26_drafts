#!/usr/bin/env bash
set -euo pipefail

echo "======================================"
echo " 04: Run AxBench Steering Benchmark   "
echo "======================================"

source .venv/bin/activate

export RESULTS_DIR="${PWD}/results/axbench"
export LOG_DIR="${PWD}/logs/axbench"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
echo "Logs will be written to $LOG_DIR"
echo "Results will be saved in $RESULTS_DIR"

touch "$LOG_DIR"/tail_combined.log
tail -f "$LOG_DIR"/*.log > "$LOG_DIR"/tail_combined.log 2>/dev/null &
TAIL_PID=$!

bash benchmarks/axbench/run_all.sh

kill $TAIL_PID 2>/dev/null || true

echo "======================================"
echo " AxBench Complete. Proceed to 05.   "
echo "======================================"
