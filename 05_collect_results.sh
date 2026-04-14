#!/usr/bin/env bash
set -euo pipefail

echo "======================================"
echo " 05: Collect and summarize results    "
echo "======================================"

source .venv/bin/activate

RESULTS_BASE="${PWD}/results"
echo "Summary of all outputs in $RESULTS_BASE/summary.txt"

{
    echo "================================="
    echo " ALIGNMENT RESULTS SUMMARY       "
    echo "================================="
    if [ -d "$RESULTS_BASE/alignment" ]; then
        if [ -f benchmarks/alignment/collect_results.py ]; then
            python benchmarks/alignment/collect_results.py --results-dir "$RESULTS_BASE/alignment"
        else
            echo "collect_results.py missing for alignment."
        fi
    else
        echo "No alignment results found."
    fi
    echo ""

    echo "================================="
    echo " MMLU RESULTS SUMMARY            "
    echo "================================="
    if [ -d "$RESULTS_BASE/mmlu" ]; then
        if [ -f benchmarks/mmlu/collect_results.py ]; then
            python benchmarks/mmlu/collect_results.py --results-dir "$RESULTS_BASE/mmlu"
        else
            echo "collect_results.py missing for mmlu."
        fi
    else
        echo "No mmlu results found."
    fi
    echo ""

    echo "================================="
    echo " AXBENCH RESULTS SUMMARY         "
    echo "================================="
    if [ -d "$RESULTS_BASE/axbench" ]; then
        if [ -f benchmarks/axbench/collect_results.py ]; then
            python benchmarks/axbench/collect_results.py --results-dir "$RESULTS_BASE/axbench"
        else
            echo "collect_results.py missing for axbench."
        fi
    else
        echo "No axbench results found."
    fi
} > "$RESULTS_BASE/summary.txt"

cat "$RESULTS_BASE/summary.txt"

echo "======================================"
echo " All tasks completed successfully.    "
echo " Detailed outputs are in results/     "
echo " Logs are in logs/                    "
echo "======================================"
