#!/usr/bin/env bash
# Setup script for remote A100 server.
# Run this ONCE after cloning the repo.
#
# Prerequisites (already installed on the server):
#   - NVIDIA drivers + CUDA toolkit
#   - Python 3.11+
#
# Usage:
#   bash experiments/setup_remote.sh

set -euo pipefail

echo "=== Remote A100 Setup ==="

# 1. Detect Python
PYTHON=""
for candidate in python3.11 python3.12 python3; do
    if command -v "$candidate" &>/dev/null; then
        VERSION=$("$candidate" --version 2>&1 | grep -oP '\d+\.\d+')
        MAJOR=$(echo "$VERSION" | cut -d. -f1)
        MINOR=$(echo "$VERSION" | cut -d. -f2)
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 11 ] && [ "$MINOR" -le 12 ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.11 or 3.12 required. Found:"
    python3 --version 2>&1 || echo "  no python3"
    exit 1
fi
echo "Using: $PYTHON ($($PYTHON --version))"

# 2. Create venv
if [ ! -d ".venv" ]; then
    echo "Creating venv..."
    $PYTHON -m venv .venv
else
    echo "Venv exists, reusing."
fi

source .venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Detect CUDA version for PyTorch
CUDA_VERSION=""
if command -v nvidia-smi &>/dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[\d.]+' | head -1)
    echo "CUDA driver version: $CUDA_VERSION"
fi

# Select PyTorch index based on CUDA version
TORCH_INDEX=""
if [ -n "$CUDA_VERSION" ]; then
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
    if [ "$CUDA_MAJOR" -ge 13 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 8 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    elif [ "$CUDA_MAJOR" -eq 12 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu126"
    elif [ "$CUDA_MAJOR" -eq 11 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    fi
fi

# 5. Install PyTorch
echo "Installing PyTorch..."
if [ -n "$TORCH_INDEX" ]; then
    pip install torch --index-url "$TORCH_INDEX"
else
    echo "WARNING: Could not detect CUDA. Installing CPU-only torch."
    pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

# 6. Install other deps
echo "Installing remaining dependencies..."
pip install numpy matplotlib pyyaml scipy

# 7. Verify installation
echo ""
echo "=== Verification ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem / 1e9
        print(f'  GPU {i}: {name} ({mem:.1f} GB)')
import numpy; print(f'NumPy: {numpy.__version__}')
import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')
import yaml; print(f'PyYAML: {yaml.__version__}')
"

# 8. Run smoke test
echo ""
echo "=== Smoke Test ==="
python -m experiments.smoke_test

echo ""
echo "=== Setup Complete ==="
echo "To run experiments:"
echo "  bash experiments/launch_parallel.sh ipd 10 4"
echo "  bash experiments/launch_parallel.sh rps 10 4"
