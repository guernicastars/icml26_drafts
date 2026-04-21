#!/bin/bash
set -euo pipefail

OUT_DIR="results/debate_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUT_DIR

echo "[Run] Launching LLM Debate on 4 GPUs..."
# Llama-3-8B requires HF token. We assume it's in env or we use open alternative like Qwen.
# We will use Qwen2.5-7B or Llama-3-8B-Instruct. We specify standard HF path.

CUDA_VISIBLE_DEVICES=0,1,2,3 python debate/run_debate.py \
    --model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
    --out_dir $OUT_DIR \
    > $OUT_DIR/debate_run.log 2>&1 &

PID=$!
echo "[Run] Dispatch PID: $PID. Logs in $OUT_DIR/debate_run.log"
wait $PID
echo "[Run] Done."
