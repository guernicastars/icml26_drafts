#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

_fail() {
    echo ""
    echo "ERROR: step failed at line $1 in 01_prestage.sh"
    echo "       Fix the issue above and re-run ./01_prestage.sh"
    exit 1
}
trap '_fail $LINENO' ERR

echo "========================================"
echo " 01: Prestage Data and Models          "
echo "========================================"

source .venv/bin/activate

if [ -f scripts/prestage_data.sh ]; then
    echo "[1/3] Prestaging datasets..."
    bash scripts/prestage_data.sh
else
    echo "[1/3] scripts/prestage_data.sh not found — skipping dataset prestage"
fi

if [ -f scripts/prestage_models.sh ]; then
    echo "[2/3] Prestaging model weights..."
    bash scripts/prestage_models.sh
else
    echo "[2/3] scripts/prestage_models.sh not found — skipping model prestage"
fi

if [ -f benchmarks/axbench/download_data.py ]; then
    echo "[3/3] Downloading AxBench concept data..."
    python benchmarks/axbench/download_data.py
else
    echo "[3/3] benchmarks/axbench/download_data.py not found — skipping axbench data"
fi

echo ""
echo "========================================"
echo " Prestaging complete. Proceed to 02.   "
echo "========================================"
