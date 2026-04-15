# UET Validation — Empirical Tests for the Universal Embedding Theorem

Standalone experiment suite validating quantitative predictions of the Universal
Embedding Theorem (Shcherbinin 2026) on pretrained language models and synthetic
data.

## Experiments

| # | Name | What it tests | Compute |
|---|------|---------------|---------|
| 1 | Scaling exponents | Predict Kaplan-style loss scaling from d_eff alone | RTX 3060 (up to 1B) / A100 (full sweep) |
| 3 | Failure prediction | PCA alignment degrades as theorem conditions are violated | CPU only |
| 7 | Curriculum tracking | d_eff(t) over Pythia training checkpoints | RTX 3060 / A100 |

## Layout

```
uet/                    # library (import, don't modify)
  eigendecomp.py        # covariance, d_eff, spectral gap, PCA alignment
  pretrained.py         # HF model loading, activation harvesting, val loss
  scaling.py            # scaling law fitting and prediction
  failure.py            # synthetic failure mode generators
  plotting.py           # paper-quality figures
scripts/                # CLI runners (parse args, call library, save results)
  run_scaling_exponents.py
  run_failure_sweep.py
  run_curriculum.py
tests/
results/                # output CSVs and PNGs (gitignored)
```

## Hardware profiles

**RTX 3060 Mobile 6GB** — Pythia up to 1B in fp16, synthetic experiments on CPU.

**4x A100 40GB** — Full Pythia sweep (70M to 12B), all experiments.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Exp 3: synthetic failure sweep (CPU, ~2 min)
python scripts/run_failure_sweep.py --output-dir results

# Exp 1: scaling exponents on RTX 3060 (small models)
python scripts/run_scaling_exponents.py --profile rtx3060 --output-dir results

# Exp 1: full sweep on A100
python scripts/run_scaling_exponents.py --profile a100 --output-dir results

# Exp 7: curriculum tracking
python scripts/run_curriculum.py --model EleutherAI/pythia-160m-deduped --output-dir results
```

## Outputs

Each script writes to `results/`:
- `scaling_exponents.csv` + `scaling_exponents.png`
- `failure_sweep.csv` + `failure_sweep.png`
- `curriculum_{model}.csv` + `curriculum_{model}.png`
