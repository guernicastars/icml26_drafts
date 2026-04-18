#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

../. 2>/dev/null || true
python_bin="${PYTHON_BIN:-python3}"
if [ -x "../.venv/bin/python" ]; then
    python_bin="../.venv/bin/python"
fi

echo "[1/5] regenerating data files"
"$python_bin" generate_figures.py
"$python_bin" generate_figures_v2.py
"$python_bin" generate_figures_v3.py

echo "[2/5] pdflatex pass 1"
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build.log || { tail -60 build.log; exit 1; }

echo "[3/5] bibtex"
bibtex main >> build.log || { tail -60 build.log; exit 1; }

echo "[4/5] pdflatex pass 2"
pdflatex -interaction=nonstopmode -halt-on-error main.tex >> build.log || { tail -60 build.log; exit 1; }

echo "[5/5] pdflatex pass 3"
pdflatex -interaction=nonstopmode -halt-on-error main.tex >> build.log || { tail -60 build.log; exit 1; }

echo "built: $(pwd)/main.pdf"
