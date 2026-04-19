#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

project_root="$(cd .. && pwd)"
if [ -x "$project_root/.venv/bin/python" ]; then
    PY="$project_root/.venv/bin/python"
else
    PY="${PYTHON_BIN:-python3}"
fi

LOG="run_v5_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

run_step() {
    local name="$1"; shift
    echo ""
    echo "=== [$name] starting at $(date) ==="
    if "$PY" "$@"; then
        echo "=== [$name] done ==="
    else
        echo "=== [$name] FAILED (continuing) ==="
    fi
}

"$PY" -m pip install ruptures --quiet 2>/dev/null || true

run_step "CP-detect"   scripts/run_changepoint.py
run_step "RMT-bulk"    scripts/run_rmt_bulk.py
run_step "NR-posthoc"  scripts/run_nr_posthoc.py
run_step "DIST-rank"   scripts/run_distill_rank.py

echo ""
echo "=== v5 complete at $(date) ==="
echo "    results: $(pwd)/results/"
