#!/bin/bash
set -e

echo "[Setup] Installing MuJoCo and RL deps..."
pip install -r requirements.txt

echo "[Setup] Checking GPU count..."
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

echo "[Setup] Done."
