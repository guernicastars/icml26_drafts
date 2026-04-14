#!/usr/bin/env bash
set -euo pipefail

echo "======================================"
echo " 00: Project Setup for 4xA100 Server  "
echo "======================================"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment .venv..."
    python3 -m venv .venv
fi
source .venv/bin/activate

echo "Installing pip and requirements..."
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

echo "Setup axbench..."
mkdir -p benchmarks/axbench/external
if [ ! -d "benchmarks/axbench/external/axbench" ]; then
    git clone --depth 1 https://github.com/FlexCode29/axbench.git benchmarks/axbench/external/axbench
fi
echo "Installing axbench in editable mode..."
pip install -e benchmarks/axbench/external/axbench

echo "Checking system state..."
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'Count:', torch.cuda.device_count())"

echo "======================================"
echo " Setup Complete. Proceed to 01. "
echo "======================================"
