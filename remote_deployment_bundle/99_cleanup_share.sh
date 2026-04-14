#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "======================================"
echo " 99: Cleanup for Sharing Bundle       "
echo "======================================"
echo "Root: $ROOT"
echo

rm_if_exists() {
  local path="$1"
  if [ -e "$path" ]; then
    echo "Removing: $path"
    rm -rf -- "$path"
  fi
}

echo "Removing virtual environments..."
rm_if_exists ".venv"
rm_if_exists ".venv311"

echo
echo "Removing local external repos (will be re-cloned by setup scripts)..."
rm_if_exists "external/axbench"
rm_if_exists "benchmarks/axbench/external/axbench"
rm_if_exists "axbench_benchmark/external/axbench"

echo
echo "Removing common run artifacts (if present)..."
rm_if_exists "results"
rm_if_exists "logs"

echo
echo "Removing python bytecode caches..."
python3 - <<'PY'
from __future__ import annotations

import os
import shutil
from pathlib import Path

root = Path(".").resolve()

def rm_tree(p: Path) -> None:
    shutil.rmtree(p, ignore_errors=True)

removed_dirs = 0
removed_files = 0

for dirpath, dirnames, filenames in os.walk(root):
    dp = Path(dirpath)

    # prune venvs/results/logs just in case they exist
    for prune in {".venv", ".venv311", "results", "logs"}:
        if prune in dirnames:
            dirnames.remove(prune)

    if dp.name == "__pycache__":
        rm_tree(dp)
        removed_dirs += 1
        dirnames[:] = []
        continue

    for fn in filenames:
        if fn.endswith(".pyc"):
            try:
                (dp / fn).unlink()
                removed_files += 1
            except FileNotFoundError:
                pass

print(f"Removed __pycache__ dirs: {removed_dirs}")
print(f"Removed .pyc files: {removed_files}")
PY

echo
echo "Cleanup complete."
