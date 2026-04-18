#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

script_dir="$(cd "$(dirname "$0")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
python_bin="${PYTHON_BIN:-$project_root/.venv/bin/python}"
if [ ! -x "$python_bin" ]; then
    python_bin="$(which python3)"
fi

run_step() {
    local name="$1"; shift
    echo "=== $name ==="
    "$python_bin" "$@" && echo "  $name: ok" || echo "  $name: FAILED (continuing)"
}

# Real-data experiments -- sequential GPU runs
run_step "double_descent_v2"      scripts/run_double_descent_v2.py
run_step "arch_diversity_v2"      scripts/run_arch_diversity_v2.py
run_step "grokking_lambda_sweep"  scripts/run_grokking_lambda_sweep.py
run_step "deff_intervention"      scripts/run_deff_intervention.py
run_step "neural_collapse"        scripts/run_neural_collapse.py
run_step "fm_deff_probe"          scripts/run_fm_deff_probe.py

echo "=== all v4 experiments complete ==="
