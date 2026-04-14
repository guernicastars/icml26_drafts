#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Meta-SWAG AxBench Benchmark Setup ==="

# 1. Create and activate venv
if [ ! -d ".venv" ]; then
    echo "[1/5] Creating Python virtual environment..."
    python3 -m venv .venv
else
    echo "[1/5] Virtual environment already exists."
fi
source .venv/bin/activate

# 2. Install Python dependencies
echo "[2/5] Installing Python dependencies..."
pip install --upgrade pip wheel setuptools > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1

# 3. Clone axbench into external/
mkdir -p external
if [ ! -d "external/axbench" ]; then
    echo "[3/5] Cloning axbench..."
    git clone --depth 1 https://github.com/FlexCode29/axbench.git external/axbench
else
    echo "[3/5] axbench already cloned."
fi

echo "[3/5] AxBench cloned. (No pip install needed; benchmark adds repo to sys.path.)"

# 4. Download AxBench data from HuggingFace
echo "[4/5] Downloading AxBench concept data..."
python download_data.py

# 5. Pre-download models (optional but avoids race conditions during parallel runs)
echo "[5/5] Pre-downloading model weights (this may take a while)..."
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

models = [
    'google/gemma-2-2b-it',
    'google/gemma-2-9b-it',
    'meta-llama/Llama-3.1-8B-Instruct',
]
for name in models:
    print(f'  Downloading {name}...')
    try:
        AutoTokenizer.from_pretrained(name)
        AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16)
        print(f'  OK: {name}')
    except Exception as e:
        print(f'  WARN: {name} failed: {e}')
        print(f'  (You may need to accept the license at huggingface.co/{name})')
"

echo ""
echo "=== Setup complete ==="
echo "Run experiments with: bash run_all.sh"
