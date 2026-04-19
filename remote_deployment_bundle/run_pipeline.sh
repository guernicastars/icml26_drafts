#!/usr/bin/env bash
# Master pipeline runner: executes stages 00-05 in sequence.
#
# Usage:
#   ./run_pipeline.sh                  # run all stages
#   ./run_pipeline.sh --from 02        # skip setup/prestage, start at alignment
#   ./run_pipeline.sh --only 02 03     # run specific stages only
#   ./run_pipeline.sh --dry-run        # print what would run, exit
#
# Environment knobs passed through to sub-scripts:
#   NUM_GPUS=4  SEEDS="42 43 44"  DTYPE=auto  REWARD_INT8=1
#   N_EPOCHS=3  POSTERIOR_SAMPLES=16  KEEP_LAST=50  EXTRA_ARGS=""
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PIPELINE_LOG="${SCRIPT_DIR}/logs/pipeline_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
exec > >(tee -a "$PIPELINE_LOG") 2>&1

declare -A STAGE_SCRIPTS=(
    [00]="./00_setup.sh"
    [01]="./01_prestage.sh"
    [02]="./02_run_alignment.sh"
    [03]="./03_run_mmlu.sh"
    [04]="./04_run_axbench.sh"
    [05]="./05_collect_results.sh"
)
STAGE_ORDER=(00 01 02 03 04 05)

STAGES_TO_RUN=("${STAGE_ORDER[@]}")
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)
            start="$2"; shift 2
            STAGES_TO_RUN=()
            found=0
            for s in "${STAGE_ORDER[@]}"; do
                [[ "$s" == "$start" ]] && found=1
                [[ $found -eq 1 ]] && STAGES_TO_RUN+=("$s")
            done
            if [[ ${#STAGES_TO_RUN[@]} -eq 0 ]]; then
                echo "ERROR: unknown stage '$start'. Valid: ${STAGE_ORDER[*]}"; exit 1
            fi
            ;;
        --only)
            shift
            STAGES_TO_RUN=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                STAGES_TO_RUN+=("$1"); shift
            done
            ;;
        --dry-run)
            DRY_RUN=1; shift
            ;;
        *)
            echo "Unknown argument: $1"; exit 1
            ;;
    esac
done

declare -A STAGE_STATUS
for s in "${STAGE_ORDER[@]}"; do STAGE_STATUS[$s]="skip"; done

_elapsed() {
    local secs=$(( $(date +%s) - $1 ))
    printf "%dm%02ds" $(( secs/60 )) $(( secs%60 ))
}

_print_summary() {
    echo ""
    echo "========================================"
    echo "  Pipeline Summary"
    echo "========================================"
    local any_fail=0
    for s in "${STAGE_ORDER[@]}"; do
        local st="${STAGE_STATUS[$s]}"
        local label
        case "$st" in
            ok)   label="[OK]  " ;;
            fail) label="[FAIL]"; any_fail=1 ;;
            skip) label="[SKIP]" ;;
        esac
        echo "  $label  $s  ${STAGE_SCRIPTS[$s]}"
    done
    echo "  Log: $PIPELINE_LOG"
    echo "========================================"
    return $any_fail
}

if [[ $DRY_RUN -eq 1 ]]; then
    echo "Dry run — stages that would execute:"
    for s in "${STAGES_TO_RUN[@]}"; do
        echo "  $s  ${STAGE_SCRIPTS[$s]}"
    done
    exit 0
fi

PIPELINE_START=$(date +%s)
echo "========================================"
echo "  Meta-SWAG Pipeline"
echo "  Stages: ${STAGES_TO_RUN[*]}"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Log: $PIPELINE_LOG"
echo "========================================"

for stage in "${STAGES_TO_RUN[@]}"; do
    script="${STAGE_SCRIPTS[$stage]}"
    echo ""
    echo "-------- Stage $stage: $script  ($(date '+%H:%M:%S')) --------"

    if [[ ! -f "$script" ]]; then
        echo "ERROR: $script not found — aborting"
        STAGE_STATUS[$stage]="fail"
        _print_summary || true
        exit 1
    fi

    stage_start=$(date +%s)
    set +e
    bash "$script"
    rc=$?
    set -e

    elapsed=$(_elapsed $stage_start)

    if [[ $rc -ne 0 ]]; then
        echo ""
        echo "FAILED: stage $stage exited with code $rc  (elapsed: $elapsed)"
        echo "        Check logs above or $PIPELINE_LOG for details."
        STAGE_STATUS[$stage]="fail"
        _print_summary || true
        exit $rc
    fi

    STAGE_STATUS[$stage]="ok"
    echo "OK: stage $stage done in $elapsed"
done

echo ""
echo "Total elapsed: $(_elapsed $PIPELINE_START)"
_print_summary
