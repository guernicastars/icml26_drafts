#!/usr/bin/env bash
set -euo pipefail

echo "=== Pre-downloading models for Meta-SWAG benchmarks ==="

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

models_causal = [
    'meta-llama/Llama-3.1-8B-Instruct',
    'google/gemma-2-9b-it',
    'google/gemma-2-2b-it',
]

models_reward = [
    'Skywork/Skywork-Reward-Llama-3.1-8B-v0.2',
    'internlm/internlm2-1_8b-reward',
]

for name in models_causal:
    print(f'Downloading {name}...')
    AutoTokenizer.from_pretrained(name)
    AutoModelForCausalLM.from_pretrained(name)
    print(f'  Done: {name}')

for name in models_reward:
    print(f'Downloading {name}...')
    AutoTokenizer.from_pretrained(name)
    AutoModelForSequenceClassification.from_pretrained(name)
    print(f'  Done: {name}')

print('All models downloaded.')
"
