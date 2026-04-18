#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON_BIN:-/home/vlad/development/ICML26/.venv/bin/python}"
LOG="$SCRIPT_DIR/master_v3.log"

run_step() {
    local name="$1"; shift
    echo "[START] $name" | tee -a "$LOG"
    if "$PYTHON" "$@" 2>&1 | tee -a "$LOG"; then
        echo "[OK] $name" | tee -a "$LOG"
    else
        echo "[FAIL] $name" | tee -a "$LOG"
    fi
}

run_step "L1: predictive scaling (UET vs Kaplan vs Chinchilla)" \
    "$SCRIPT_DIR/scripts/run_predictive_scaling.py"

run_step "L2: double descent + deff" \
    "$SCRIPT_DIR/scripts/run_double_descent.py"

run_step "L3: noise dimension test" \
    "$SCRIPT_DIR/scripts/run_noise_dim_test.py"

run_step "L4: architecture diversity (MLP+CNN)" \
    "$SCRIPT_DIR/scripts/run_arch_diversity.py"

run_step "L5: discovery vs direct low-d" \
    "$SCRIPT_DIR/scripts/run_discovery_vs_direct.py"

echo "=== v3 complete ===" | tee -a "$LOG"
