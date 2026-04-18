#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f "../uet-validation/.env" ]]; then set -a; source "../uet-validation/.env"; set +a; fi

VENV_PY="${VENV_PY:-/home/vlad/development/ICML26/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$OUTPUT_DIR/master_olmo_${RUN_STAMP}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER_LOG"; }
run_step() { local name="$1"; shift; log "=== START $name ==="; if "$@" 2>&1 | tee -a "$MASTER_LOG"; then log "=== OK $name ==="; else log "=== FAIL $name (continuing) ==="; fi; }

log "Starting OLMo-1B curriculum  RUN_STAMP=$RUN_STAMP"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null | tee -a "$MASTER_LOG" || true

OLMO_CURRICULUM="$OUTPUT_DIR/curriculum/${RUN_STAMP}_OLMo-1B"

run_step "olmo_curriculum" "$VENV_PY" scripts/run_curriculum_olmo.py \
    --n-checkpoints 14 \
    --device cuda \
    --max-tokens 150000 \
    --batch-size 2 \
    --seq-len 512 \
    --output-dir "$OUTPUT_DIR" \
    --run-name "${RUN_STAMP}_OLMo-1B"

# Cross-family UET fit: OLMo + Pythia 160M + 410M
V1_DIR="$SCRIPT_DIR/../uet-validation"
PYTHIA_160M="$V1_DIR/results/curriculum/20260418_020501_pythia-160m/curriculum.csv"
PYTHIA_410M="$V1_DIR/results/curriculum/20260418_020501_pythia-410m/curriculum.csv"
PYTHIA_70M="$OUTPUT_DIR/curriculum/20260418_053410_pythia-70m-deduped/curriculum.csv"
OLMO_CSV="$OLMO_CURRICULUM/curriculum.csv"

if [[ -f "$OLMO_CSV" ]]; then
    run_step "cross_family_fit" "$VENV_PY" scripts/run_uet_fit_cross_family.py \
        --curriculum-csvs "$OLMO_CSV" "$PYTHIA_70M" "$PYTHIA_160M" "$PYTHIA_410M" \
        --labels "OLMo-1B" "Pythia-70M" "Pythia-160M" "Pythia-410M" \
        --min-tokens 20000000000 \
        --output-dir "$OUTPUT_DIR" \
        --run-name "$RUN_STAMP"
else
    log "SKIP cross_family_fit: OLMo curriculum CSV missing"
fi

log "=== DONE ==="
