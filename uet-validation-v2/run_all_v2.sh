#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V1_DIR="$(dirname "$SCRIPT_DIR")/uet-validation"
cd "$SCRIPT_DIR"

if [[ -f "$V1_DIR/.env" ]]; then
    set -a
    source "$V1_DIR/.env"
    set +a
fi

VENV_PY="${VENV_PY:-/home/vlad/development/ICML26/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

# v1 curriculum dirs used for ablation (L2) and 3-model joint fit (L1 step 2)
V1_CURRICULUM_160M="$V1_DIR/results/curriculum/20260418_020501_pythia-160m"
V1_CURRICULUM_410M="$V1_DIR/results/curriculum/20260418_020501_pythia-410m"

mkdir -p "$OUTPUT_DIR"
MASTER_LOG="$OUTPUT_DIR/master_v2_${RUN_STAMP}.log"

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

log "ROOT=$SCRIPT_DIR  V1=$V1_DIR  RUN_STAMP=$RUN_STAMP"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null | tee -a "$MASTER_LOG" || true

# ── L1: Pythia-70M curriculum ──────────────────────────────────────────────
CURRICULUM_70M="$OUTPUT_DIR/curriculum/${RUN_STAMP}_pythia-70m-deduped"

run_step "L1_curriculum_70m" "$VENV_PY" "$V1_DIR/scripts/run_curriculum.py" \
    --model EleutherAI/pythia-70m-deduped \
    --device cuda \
    --max-tokens 200000 \
    --batch-size 4 \
    --seq-len 512 \
    --output-dir "$OUTPUT_DIR" \
    --run-name "${RUN_STAMP}_pythia-70m-deduped"

# ── L1 step 2: UET fit on 70M + v1 160M + v1 410M ─────────────────────────
if [[ -f "$CURRICULUM_70M/curriculum.csv" && \
      -f "$V1_CURRICULUM_160M/curriculum.csv" && \
      -f "$V1_CURRICULUM_410M/curriculum.csv" ]]; then
    run_step "L1_uet_fit_3models" "$VENV_PY" "$V1_DIR/scripts/run_uet_fit.py" \
        --curriculum-dirs \
            "$CURRICULUM_70M" \
            "$V1_CURRICULUM_160M" \
            "$V1_CURRICULUM_410M" \
        --min-step 1000 \
        --output-dir "$OUTPUT_DIR" \
        --run-name "${RUN_STAMP}_3models"
else
    log "SKIP L1_uet_fit_3models: missing curriculum CSVs"
fi

# ── L2: Functional-form ablation ──────────────────────────────────────────
run_step "L2_form_ablation" "$VENV_PY" scripts/run_form_ablation.py \
    --curriculum-dirs \
        "$V1_CURRICULUM_160M" \
        "$V1_CURRICULUM_410M" \
    --output-dir "$OUTPUT_DIR" \
    --run-name "$RUN_STAMP"

# ── L3: PCA-causality alignment sweep (CPU) ───────────────────────────────
run_step "L3_pca_alignment" "$VENV_PY" scripts/run_pca_causal_alignment.py \
    --d-values 64 128 256 \
    --k 8 \
    --gap-multipliers 0.1 0.2 0.5 1.0 2.0 5.0 10.0 20.0 \
    --n-values 200 500 2000 5000 20000 \
    --n-seeds 30 \
    --output-dir "$OUTPUT_DIR" \
    --run-name "$RUN_STAMP"

# ── L4: Sample complexity sweep (CPU) ─────────────────────────────────────
run_step "L4_sample_complexity" "$VENV_PY" scripts/run_sample_complexity.py \
    --d 256 \
    --k-values 4 8 16 32 \
    --n-seeds 50 \
    --n-grid 12 \
    --output-dir "$OUTPUT_DIR" \
    --run-name "$RUN_STAMP"

# ── L5: Per-layer d_eff (GPU) ──────────────────────────────────────────────
run_step "L5_layer_deff" "$VENV_PY" scripts/run_layer_deff.py \
    --models \
        EleutherAI/pythia-70m-deduped \
        EleutherAI/pythia-160m-deduped \
    --checkpoints 64 1000 143000 \
    --device cuda \
    --max-tokens 100000 \
    --batch-size 4 \
    --seq-len 512 \
    --output-dir "$OUTPUT_DIR" \
    --run-name "$RUN_STAMP"

# ── L6: Eval-set robustness (GPU) ─────────────────────────────────────────
run_step "L6_eval_robustness" "$VENV_PY" scripts/run_eval_robustness.py \
    --model EleutherAI/pythia-160m-deduped \
    --checkpoints 1000 2000 4000 8000 16000 32000 64000 100000 120000 143000 \
    --device cuda \
    --max-tokens 150000 \
    --batch-size 4 \
    --seq-len 512 \
    --output-dir "$OUTPUT_DIR" \
    --run-name "$RUN_STAMP"

# ── L7: Synthetic domain with known rank (GPU) ────────────────────────────
run_step "L7_synthetic_domain" "$VENV_PY" scripts/run_synthetic_domain.py \
    --d 100 \
    --n-samples 50000 \
    --k-true-values 3 5 10 20 50 \
    --latent-dims 2 4 8 16 32 64 100 \
    --n-seeds 2 \
    --snr 3.0 \
    --n-epochs 200 \
    --batch-size 512 \
    --device cuda \
    --output-dir "$OUTPUT_DIR" \
    --run-name "$RUN_STAMP"

log "=== ALL DONE. Master log: $MASTER_LOG ==="
log "Results tree:"
find "$OUTPUT_DIR" -maxdepth 3 -type d | sort | tee -a "$MASTER_LOG"
