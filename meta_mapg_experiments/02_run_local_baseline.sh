#!/bin/bash
# Local 3060 Run - Vanilla Meta-MAPG Baseline (Restarts Disabled)
OUT_DIR="results/baseline_local_3060_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUT_DIR

echo "Starting Vanilla Baseline (No Restarts) sweep on RTX 3060..."

for s in {0..3}; do
    # Setting threshold to -9999.0 ensures the global restart mechanism never triggers.
    # This mathematically isolates and degrades the algorithm to "Vanilla Meta-MAPG".
    python mujoco/run_mujoco.py --out_dir $OUT_DIR --seed $s --device "cuda" --episodes 1000 --threshold -9999.0 > $OUT_DIR/seed_$s.log 2>&1 &
done

wait
echo "Baseline Sweep Complete."
