#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

VENV_PY="${VENV_PY:-/home/vlad/development/ICML26/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR"
MASTER_LOG="$OUTPUT_DIR/master_${RUN_STAMP}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER_LOG"; }
run_step() {
    local name="$1"; shift
    log "=== START $name ==="
    if "$@" 2>&1 | tee -a "$MASTER_LOG"; then
        log "=== OK $name ==="
    else
        log "=== FAIL $name (continuing) ==="
    fi
}

log "ROOT=$ROOT OUTPUT_DIR=$OUTPUT_DIR RUN_STAMP=$RUN_STAMP"
log "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv | tee -a "$MASTER_LOG" || true

run_step "failure_sweep" "$VENV_PY" scripts/run_failure_sweep.py \
    --d-values 16 32 64 128 256 512 \
    --k-values 2 4 8 16 32 64 128 \
    --gap-values 0.5 1.0 2.0 5.0 10.0 \
    --n-seeds 10 \
    --output-dir "$OUTPUT_DIR" \
    --run-name "$RUN_STAMP"

run_step "polymarket_embedding" "$VENV_PY" scripts/run_polymarket_embedding.py \
    --latent-dims 4 8 16 32 64 128 \
    --min-volume 5000 \
    --n-epochs 300 \
    --batch-size 256 \
    --device cuda \
    --output-dir "$OUTPUT_DIR" \
    --run-name "$RUN_STAMP"

run_step "art_embedding" "$VENV_PY" scripts/run_art_embedding.py \
    --sources christies sothebys \
    --latent-dims 4 8 16 32 64 128 \
    --n-epochs 200 \
    --batch-size 512 \
    --device cuda \
    --output-dir "$OUTPUT_DIR" \
    --run-name "$RUN_STAMP"

run_step "scaling_exponents" "$VENV_PY" scripts/run_scaling_exponents.py \
    --profile rtx3060 \
    --device cuda \
    --max-tokens 500000 \
    --batch-size 2 \
    --seq-len 512 \
    --output-dir "$OUTPUT_DIR" \
    --run-name "$RUN_STAMP"

run_step "curriculum_160m" "$VENV_PY" scripts/run_curriculum.py \
    --model EleutherAI/pythia-160m-deduped \
    --device cuda \
    --max-tokens 200000 \
    --batch-size 4 \
    --seq-len 512 \
    --output-dir "$OUTPUT_DIR" \
    --run-name "${RUN_STAMP}_pythia-160m"

run_step "curriculum_410m" "$VENV_PY" scripts/run_curriculum.py \
    --model EleutherAI/pythia-410m-deduped \
    --device cuda \
    --max-tokens 200000 \
    --batch-size 2 \
    --seq-len 512 \
    --output-dir "$OUTPUT_DIR" \
    --run-name "${RUN_STAMP}_pythia-410m"

POLY_Z="$OUTPUT_DIR/polymarket/$RUN_STAMP/models/latent64/Z.npy"
ART_Z="$OUTPUT_DIR/art/$RUN_STAMP/models/latent64/Z.npy"

if [[ -f "$POLY_Z" && -f "$ART_Z" ]]; then
    run_step "cross_domain" "$VENV_PY" scripts/run_cross_domain.py \
        --polymarket-z "$POLY_Z" \
        --art-z "$ART_Z" \
        --output-dir "$OUTPUT_DIR" \
        --run-name "$RUN_STAMP"
else
    log "SKIP cross_domain: missing embeddings (POLY_Z=$POLY_Z, ART_Z=$ART_Z)"
fi

log "=== ALL DONE. Master log: $MASTER_LOG ==="
log "Results tree:"
find "$OUTPUT_DIR" -maxdepth 3 -type d | tee -a "$MASTER_LOG"
