#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python_bin="${PYTHON_BIN:-python3}"
if [ -x "../../.venv/bin/python" ]; then
    python_bin="../../.venv/bin/python"
fi

echo "[1/4] generating .dat files"
cd .. && "$python_bin" generate_figures_v5.py && cd hild

echo "[2/4] pdflatex pass 1"
pdflatex -interaction=nonstopmode -halt-on-error hild.tex > build.log || { tail -40 build.log; exit 1; }

echo "[3/4] bibtex"
bibtex hild >> build.log 2>&1 || true

echo "[4/4] pdflatex passes 2+3"
pdflatex -interaction=nonstopmode -halt-on-error hild.tex >> build.log || { tail -40 build.log; exit 1; }
pdflatex -interaction=nonstopmode -halt-on-error hild.tex >> build.log || { tail -40 build.log; exit 1; }

echo "built: $(pwd)/hild.pdf"
