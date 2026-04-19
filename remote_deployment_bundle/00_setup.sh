#!/usr/bin/env bash
set -euo pipefail

echo "========================================"
echo " 00: Project Setup (4x V100-SXM2 / A100) "
echo "========================================"

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

echo "Probing GPUs..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader
else
    echo "nvidia-smi not found"
fi

python3 - <<'PY'
import torch
print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()} | devices: {torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit(0)
volta = []
for i in range(torch.cuda.device_count()):
    major, minor = torch.cuda.get_device_capability(i)
    name = torch.cuda.get_device_name(i)
    free, total = torch.cuda.mem_get_info(i)
    bf16 = "bf16-native" if major >= 8 else "bf16-EMULATED (Volta)"
    print(f"  [{i}] {name} sm_{major}{minor} | {total/1e9:.1f} GB | {bf16}")
    if major < 8:
        volta.append(i)
if volta:
    print("NOTE: Volta GPUs detected (sm_70). Auto-dtype will pick fp16.")
    print("      Reward-model int8 (bitsandbytes) is required to fit on 32 GB cards.")
PY

echo "========================================"
echo " Setup Complete. Proceed to 01.         "
echo "========================================"
