#!/bin/bash
# Self-contained: creates venv, installs deps, runs all MuJoCo experiments.
# All paths are relative to this script's directory — no /tmp or ~ used.
set -e

# Move to the directory containing this script
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"

echo "=== Working in: $SCRIPT_DIR ==="

# ── 1. Create venv inside the project directory ─────────────────────────────
VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "=== Creating venv at $VENV_DIR ==="
    python3 -m venv "$VENV_DIR"
else
    echo "=== Venv already exists at $VENV_DIR ==="
fi

source "$VENV_DIR/bin/activate"
echo "=== Python: $(which python) ==="

# ── 2. Install dependencies ──────────────────────────────────────────────────
echo "=== Installing deps (this may take a few minutes first time) ==="

# Torch with CUDA 12 — matches V100 driver on this server
pip install --quiet torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu121

# gymnasium[mujoco] pulls in mujoco + the physics bindings together
pip install --quiet \
    "gymnasium[mujoco]" \
    "numpy<2.0.0" \
    pandas \
    matplotlib \
    pettingzoo

echo "=== Verifying installs ==="
python -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.device_count()} GPUs')"
python -c "import mujoco; print(f'mujoco {mujoco.__version__}')"

# Attempt to run without forcing EGL hooks (since we don't call render())
export MUJOCO_GL=""
export PYOPENGL_PLATFORM=""

# Smoke test: make sure HalfCheetah-v4 actually loads
echo "=== Smoke test: HalfCheetah-v4 ==="
python - <<'SMOKETEST'
import sys, gymnasium as gym
try:
    env = gym.make("HalfCheetah-v4")
    obs, _ = env.reset(seed=0)
    env.close()
    print(f"HalfCheetah-v4 OK — obs shape: {obs.shape}")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)
SMOKETEST

# ── 3. Run MuJoCo experiments ─────────────────────────────────────────────────
echo "=== Launching MuJoCo experiments ==="
mkdir -p results_server logs_server

CONFIGS=("-9999 baseline" "1500 high_tau" "-280 giannou")
SEEDS=(0 1 2 3 4 5 6 7 8 9)

gpu_id=0
max_jobs=8   # conservative: 2 per GPU

for seed in "${SEEDS[@]}"; do
    for config_str in "${CONFIGS[@]}"; do
        set -- $config_str
        tau=$1
        name=$2

        mkdir -p "results_server/$name"

        CUDA_VISIBLE_DEVICES=$gpu_id \
        python run_mujoco.py \
            --out_dir "results_server/$name" \
            --episodes 1000 \
            --seed "$seed" \
            --threshold "$tau" \
            > "logs_server/${name}_seed${seed}.log" 2>&1 &

        echo "  Dispatched: $name seed=$seed GPU=$gpu_id (PID $!)"
        gpu_id=$(( (gpu_id + 1) % 4 ))

        # Throttle: wait if too many jobs running
        while [ "$(jobs -r | wc -l)" -ge "$max_jobs" ]; do
            sleep 15
        done
    done
done

echo "=== All jobs dispatched. Waiting for completion... ==="
wait
echo "=== All done! Results in: $SCRIPT_DIR/results_server/ ==="

# ── 4. Quick summary ─────────────────────────────────────────────────────────
echo ""
echo "=== Per-condition summary (last-50-ep mean reward) ==="
python - <<'EOF'
import os, glob, numpy as np

results_dir = "results_server"
for cond in sorted(os.listdir(results_dir)):
    cond_dir = os.path.join(results_dir, cond)
    if not os.path.isdir(cond_dir):
        continue
    rewards, restarts = [], []
    # run_mujoco.py writes: ep, ep_rew, restart_count, elapsed_time
    for f in sorted(glob.glob(f"{cond_dir}/seed_*/logs/metrics.csv")):
        try:
            data = np.loadtxt(f, delimiter=",")
            if data.ndim == 1:
                data = data[None]
            rewards.append(float(np.mean(data[-50:, 1])))
            restarts.append(float(data[-1, 2]))
        except Exception as e:
            pass
    if rewards:
        print(f"  {cond:12s}  reward={np.mean(rewards):+7.1f} ± {np.std(rewards):.1f}  restarts={np.mean(restarts):.1f}  (n={len(rewards)})")
    else:
        print(f"  {cond:12s}  no results found — check logs_server/")
EOF
