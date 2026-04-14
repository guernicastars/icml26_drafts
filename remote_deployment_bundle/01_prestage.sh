#!/usr/bin/env bash
set -euo pipefail

echo "======================================"
echo " 01: Prestage Data and Models         "
echo "======================================"

source .venv/bin/activate

bash scripts/prestage_data.sh
bash scripts/prestage_models.sh

echo "Downloading AxBench data..."
cd benchmarks/axbench
python download_data.py
cd ../..

echo "======================================"
echo " Prestaging Complete. Proceed to 02. "
echo "======================================"
