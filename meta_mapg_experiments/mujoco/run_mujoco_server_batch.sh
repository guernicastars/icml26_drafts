#!/bin/bash
# runs all 30 MuJoCo seeds across 4 GPUs
# Usage: ./run_mujoco_server_batch.sh

mkdir -p results_server

# Activate virtual environment linked in this directory
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Warning: .venv/bin/activate not found! Running with system python."
fi

# Define configs
# baseline: tau=-9999
# high tau: tau=1500
# giannou: tau=-280
CONFIGS=("-9999 baseline" "1500 high_tau" "-280 giannou")
SEEDS=(0 1 2 3 4 5 6 7 8 9)

# A simple way to run jobs evenly across 4 GPUs
gpu_id=0
max_jobs=12 # 3 jobs per GPU simultaneously fits easily in V100 32GB

for seed in "${SEEDS[@]}"; do
    for config_str in "${CONFIGS[@]}"; do
        set -- $config_str
        tau=$1
        name=$2
        
        out_dir="results_server/${name}/seed_${seed}"
        
        CUDA_VISIBLE_DEVICES=$gpu_id python run_mujoco.py \
            --out_dir "results_server/${name}" \
            --episodes 1000 \
            --seed "$seed" \
            --threshold "$tau" \
            > "${name}_seed${seed}.log" 2>&1 &
        
        # cycle gpu
        gpu_id=$(( (gpu_id + 1) % 4 ))
        
        # check number of bg jobs
        while [ $(jobs -r | wc -l) -ge $max_jobs ]; do
            sleep 10
        done
    done
done

echo "All jobs dispatched. Waiting for completion..."
wait
echo "All done!"
