#!/usr/bin/env bash
set -euo pipefail

echo "=== Local smoke test (RTX 3060 6GB / CPU) ==="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo ""
echo "1/3: Unit tests for meta_swag library..."
python -c "
import numpy as np
from meta_swag.posterior.base import AggregatedAdapterResult
from meta_swag.posterior.meta_swag import aggregate_adapter_checkpoints
from meta_swag.training.retention import build_retention_schedule

# Retention schedule
schedule = build_retention_schedule(100, 10, 0.4)
assert len(schedule) > 0, 'Empty retention schedule'
assert all(s >= 60 for s in schedule), 'Schedule starts before tail'
print(f'  Retention schedule: {schedule}')

# Aggregation with MAP (argmax)
rng = np.random.default_rng(42)
checkpoints = rng.normal(size=(20, 100)).astype(np.float32)
scores = rng.uniform(size=20).astype(np.float32)

for scheme in ['map', 'last_iterate', 'uniform', 'swa', 'ema', 'softmax', 'ess', 'threshold']:
    agg = aggregate_adapter_checkpoints(checkpoints, scores, scheme=scheme)
    assert agg.mean_vector.shape == (100,)
    assert agg.effective_sample_size > 0
    print(f'  {scheme}: ESS={agg.effective_sample_size:.2f}, trace={agg.posterior_trace:.4f}')

# MAP = argmax
agg_map = aggregate_adapter_checkpoints(checkpoints, scores, scheme='map')
assert np.argmax(agg_map.weights) == np.argmax(scores), 'MAP should select argmax(scores)'

# Posterior sampling
agg = aggregate_adapter_checkpoints(checkpoints, scores, scheme='softmax')
samples = agg.sample(32, rng)
assert samples.shape == (32, 100)
sample_mean = samples.mean(axis=0)
assert np.allclose(sample_mean, agg.mean_vector, atol=0.5), 'Sample mean should be near posterior mean'
print(f'  Sampling: 32 samples, mean deviation = {np.abs(sample_mean - agg.mean_vector).mean():.4f}')

print('  All unit tests passed.')
"

echo ""
echo "2/3: Laplace posterior test..."
python -c "
import numpy as np
from meta_swag.posterior.laplace import laplace_posterior
from meta_swag.adapters.state import AdapterStateManifest, AdapterParameterSpec

# Synthetic test
fisher = np.array([1.0, 4.0, 0.5, 10.0], dtype=np.float32)
manifest = AdapterStateManifest(parameters=(
    AdapterParameterSpec(name='weight', shape=(1, 4), dtype='float32', numel=4),
))

import torch
model = torch.nn.Linear(4, 1, bias=False)
model.weight.data = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
model.weight.requires_grad_(True)

result = laplace_posterior(model, manifest, fisher, prior_precision=1.0)
expected_var = 1.0 / (fisher + 1.0)
assert np.allclose(result.diagonal_variance, expected_var, atol=1e-5), \
    f'Laplace variance mismatch: {result.diagonal_variance} vs {expected_var}'
print(f'  Laplace posterior: trace={result.posterior_trace:.4f}')
print(f'  Variance: {result.diagonal_variance}')
print('  Laplace test passed.')
"

echo ""
echo "3/3: PosteriorPredictive test..."
python -c "
import numpy as np
from meta_swag.posterior.base import AggregatedAdapterResult
from meta_swag.posterior.predictive import PosteriorPredictive
from meta_swag.adapters.state import AdapterStateManifest, AdapterParameterSpec

manifest = AdapterStateManifest(parameters=(
    AdapterParameterSpec(name='weight', shape=(2, 3), dtype='float32', numel=6),
))
result = AggregatedAdapterResult(
    scheme='softmax', mean_vector=np.zeros(6, dtype=np.float32),
    weights=np.array([0.5, 0.5], dtype=np.float32),
    retained_count=2, effective_sample_size=2.0,
    beta=1.0, threshold=None, posterior_trace=1.0,
    top_eigenvalues=(0.5, 0.3), top_eigenvalue_ratio=0.5,
    max_normalized_weight=0.5, score_variance=0.1,
    diagonal_variance=np.ones(6, dtype=np.float32) * 0.1,
    deviations=np.zeros((0, 6), dtype=np.float32),
)

pred = PosteriorPredictive(result, manifest, num_samples=8, seed=42)
assert pred.effective_num_samples == 8
vectors = pred.sample_vectors()
assert vectors.shape == (8, 6)
print(f'  PosteriorPredictive: {vectors.shape[0]} samples of dim {vectors.shape[1]}')

# Point estimate
result_map = AggregatedAdapterResult(
    scheme='map', mean_vector=np.ones(6, dtype=np.float32),
    weights=np.array([1.0], dtype=np.float32),
    retained_count=1, effective_sample_size=1.0,
    beta=None, threshold=None, posterior_trace=0.0,
    top_eigenvalues=(), top_eigenvalue_ratio=0.0,
    max_normalized_weight=1.0, score_variance=0.0,
    diagonal_variance=np.zeros(6, dtype=np.float32),
    deviations=np.zeros((0, 6), dtype=np.float32),
)
pred_map = PosteriorPredictive(result_map, manifest, num_samples=16, seed=0)
assert pred_map.effective_num_samples == 1
print('  Point estimate: S=1 as expected.')
print('  PosteriorPredictive test passed.')
"

echo ""
echo "=== All smoke tests passed ==="
