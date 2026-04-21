#!/bin/bash
set -euo pipefail

echo "[Setup] Installing TRL, PEFT, VLLM, Datasets..."
pip install -r requirements.txt

echo "[Setup] Checking LLM capability..."
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

echo "[Setup] Ready for LLM Debate."
