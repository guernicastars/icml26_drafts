#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export HF_HOME="${HF_HOME:-$SCRIPT_DIR/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$SCRIPT_DIR/.hf_cache/datasets}"

_fail() {
    echo ""
    echo "ERROR: step failed at line $1 in 00_setup.sh"
    echo "       Fix the issue above and re-run ./00_setup.sh"
    exit 1
}
trap '_fail $LINENO' ERR

echo "========================================"
echo " 00: Project Setup (4x V100-SXM2 / A100)"
echo "========================================"

echo "[1/5] Creating virtual environment..."
# Validate existing venv: python3 inside it must exist and be ours.
VENV_OK=0
if [ -d ".venv" ] && [ -x ".venv/bin/python3" ]; then
    if ".venv/bin/python3" -c "import sys; sys.exit(0 if '.venv' in sys.prefix else 1)" 2>/dev/null; then
        VENV_OK=1
    fi
fi
if [ "$VENV_OK" -eq 0 ]; then
    if [ -d ".venv" ]; then
        echo "  Existing .venv is broken or stale — recreating..."
        rm -rf .venv
    fi
    python3 -m venv .venv
fi
source .venv/bin/activate
# Sanity-check: our python3 must be inside the venv.
ACTIVE_PY="$(which python3)"
if [[ "$ACTIVE_PY" != "$SCRIPT_DIR/.venv/"* ]]; then
    echo "ERROR: venv activation failed — 'python3' is $ACTIVE_PY, expected inside $SCRIPT_DIR/.venv/"
    echo "       Try: rm -rf .venv && ./00_setup.sh"
    exit 1
fi
echo "  python3: $ACTIVE_PY"

echo "[2/5] Installing pip and requirements..."
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

echo "[3/5] Cloning axbench (sys.path import — no package install, Python-version safe)..."
mkdir -p benchmarks/axbench/external
if [ ! -d "benchmarks/axbench/external/axbench" ]; then
    git clone --depth 1 https://github.com/FlexCode29/axbench.git \
        benchmarks/axbench/external/axbench
fi
# Install axbench's own dependencies only; skip the package itself (requires Python >=3.12).
AXBENCH_REQ="benchmarks/axbench/external/axbench/requirements.txt"
if [ -f "$AXBENCH_REQ" ]; then
    pip install -r "$AXBENCH_REQ" || echo "  WARN: some axbench deps failed — axbench benchmark may be affected"
fi

echo "[4/5] Probing GPUs..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader
else
    echo "  nvidia-smi not found — skipping hardware probe"
fi

echo "[5/5] Checking PyTorch + CUDA..."
python3 - <<'PY'
import sys, torch

print(f"  torch {torch.__version__} | CUDA: {torch.cuda.is_available()} | devices: {torch.cuda.device_count()}")
if not torch.cuda.is_available():
    print("  WARNING: no CUDA — jobs will run on CPU")
    sys.exit(0)

for i in range(torch.cuda.device_count()):
    major, minor = torch.cuda.get_device_capability(i)
    name = torch.cuda.get_device_name(i)
    free, total = torch.cuda.mem_get_info(i)
    bf16_label = "bf16-native" if major >= 8 else "bf16-EMULATED (use fp16)"
    print(f"  [{i}] {name} sm_{major}{minor} | {total/1e9:.1f} GB | {bf16_label}")

try:
    import bitsandbytes
    print(f"  bitsandbytes {bitsandbytes.__version__} OK")
except ImportError:
    print("  ERROR: bitsandbytes not installed — int8 reward models will fail")
    sys.exit(1)

try:
    import peft
    print(f"  peft {peft.__version__} OK")
except ImportError:
    print("  ERROR: peft not installed — LoRA training will fail")
    sys.exit(1)
PY

echo ""
echo "========================================"
echo " Setup complete. Proceed to 01.        "
echo "========================================"
