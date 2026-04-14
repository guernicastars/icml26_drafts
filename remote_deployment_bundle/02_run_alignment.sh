#!/usr/bin/env bash
set -euo pipefail

echo "======================================"
echo " 02: Run DPO Alignment                "
echo "======================================"

source .venv/bin/activate

export RESULTS_DIR="${PWD}/results/alignment"
export LOG_DIR="${PWD}/logs/alignment"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
echo "Logs will be written to $LOG_DIR"
echo "Results will be saved in $RESULTS_DIR"

# Tail logs in the background so the user can see progress on the terminal
touch "$LOG_DIR"/tail_combined.log
tail -f "$LOG_DIR"/*.log > "$LOG_DIR"/tail_combined.log 2>/dev/null &
TAIL_PID=$!

bash benchmarks/alignment/run_all.sh

kill $TAIL_PID 2>/dev/null || true

echo "======================================"
echo " Alignment Complete. Proceed to 03. "
echo "======================================"
