#!/bin/bash
set -euo pipefail

echo "[IPD] Installing dependencies..."
pip install numpy torch

echo "[IPD] Running Matrix Games Sweep..."
python run_matrix_ipd.py --out_dir results

echo "[IPD] Done. Checking convergence:"
grep "True" results/ipd_grid_results.csv | wc -l
