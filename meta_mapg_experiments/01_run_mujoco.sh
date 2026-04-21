#!/bin/bash
set -e

OUT_DIR="results/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUT_DIR

echo "[Run] Launching parallel 16 seeds across 4 GPUs to saturate V100s..."

for seed in {0..15}; do
    # Round-robin assign 4 processes per GPU
    GPU_ID=$((seed % 4))
    echo "Starting Seed $seed on GPU $GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID python mujoco/run_mujoco.py \
        --out_dir $OUT_DIR \
        --seed $seed \
        --device "cuda" \
        > $OUT_DIR/seed_${seed}.log 2>&1 &
done

echo "[Run] All 16 jobs dispatched. Waiting for completion..."
wait

echo "[Run] Done. Results and checkpoint traces in $OUT_DIR"
