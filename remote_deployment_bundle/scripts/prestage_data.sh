#!/usr/bin/env bash
set -euo pipefail

echo "=== Pre-downloading datasets for Meta-SWAG benchmarks ==="

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

python -c "
from datasets import load_dataset

datasets_to_download = [
    ('HuggingFaceH4/ultrafeedback_binarized', None),
    ('cais/mmlu', 'abstract_algebra'),
    ('cais/mmlu', 'machine_learning'),
]

for name, config in datasets_to_download:
    label = f'{name}/{config}' if config else name
    print(f'Downloading {label}...')
    load_dataset(name, config)
    print(f'  Done: {label}')

print('All datasets downloaded.')
"
