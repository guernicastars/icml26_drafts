#!/bin/bash
set -euo pipefail

echo "[Eval] Analyzing Debate JSONL logs for Truthfulness collapses..."
python -c "
import json, glob
logs = glob.glob('results/*/metrics.jsonl')
for l in logs:
    with open(l) as f:
        data = [json.loads(line) for line in f]
    print(f'Log: {l} | Rounds: {len(data)}')
    if len(data) > 0:
        print(f'Final Truthfulness: {data[-1].get(\"truth_score\", 0)}')
"
echo "[Eval] Done."
