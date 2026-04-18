# UET Validation — Empirical Tests for the Universal Embedding Theorem

Standalone experiment suite for the ICML 2026 Structured FM Workshop submission.
Tests quantitative predictions of the Universal Embedding Theorem (Shcherbinin 2026)
on pretrained language models, synthetic data, and domain embeddings.

## Target hardware

Designed for **4×A100 40GB** server. RTX 3060 6GB works for the smaller
Pythia models and CPU-only experiments.

## Experiments

| # | Name | Script | Data source | Compute |
|---|------|--------|-------------|---------|
| 1 | Scaling exponents | `run_scaling_exponents.py` | Pythia + WikiText | GPU |
| 2 | Polymarket embedding | `run_polymarket_embedding.py` | ClickHouse `polymarket.markets` | GPU |
| 3 | Failure prediction | `run_failure_sweep.py` | Synthetic | CPU |
| 4a | Art embedding | `run_art_embedding.py` | ClickHouse `christies/sothebys.gold_features` | GPU |
| 4b | Cross-domain spectra | `run_cross_domain.py` | Outputs of 2 + 4a | CPU |
| 7 | Curriculum d_eff(t) | `run_curriculum.py` | Pythia training checkpoints | GPU |

## Layout

```
uet/                       # library (import only, do not modify)
  eigendecomp.py           # covariance, d_eff, spectral gap, PCA alignment
  pretrained.py            # HF model loading, activation harvest, val loss
  scaling.py               # Chinchilla + UET scaling law fitting
  failure.py               # synthetic failure mode generators
  polymarket_data.py       # ClickHouse fetcher + feature builder
  art_data.py              # ClickHouse fetcher for art gold_features
  embedding_train.py       # small AE trainer (GPU)
  clickhouse.py            # config + client factory (reads env vars)
  plotting.py              # paper-quality figures
scripts/                   # CLI runners
  run_scaling_exponents.py
  run_failure_sweep.py
  run_curriculum.py
  run_polymarket_embedding.py
  run_art_embedding.py
  run_cross_domain.py
tests/                     # pytest suite
results/                   # output (gitignored)
requirements.txt
.env.example               # copy to .env, fill in credentials
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env with ClickHouse credentials
set -a; source .env; set +a
```

## One-shot runner (RTX 3060 6GB)

```bash
bash scripts/run_all_rtx3060.sh
```

Runs all experiments sequentially, logs to `results/master_<stamp>.log`, each
experiment writes to `results/<experiment>/<stamp>/` with:
- `config.json` - argparse snapshot
- `metadata.json` - summary metrics
- `logs/run.log` - full per-experiment log
- `*.csv` - metrics tables
- `*.png` - figures
- `models/<variant>/autoencoder.pt` - trained weights
- `models/<variant>/Z.npy` - encoded embeddings
- `models/<variant>/train_loss.npy`, `val_loss.npy` - loss curves
- `models/<variant>/eigenvalues.npy`, `spectrum.json` - spectral stats

Expected wall clock on RTX 3060 + i9-11900H: ~4-6 hours total
(dominated by Pythia scaling + curriculum runs).

Continue even if a step fails - each experiment is independent.

## Run order (4xA100)

```bash
# 0. Smoke test — verify everything loads
pytest tests/ -v
python scripts/run_failure_sweep.py --help
python scripts/run_scaling_exponents.py --help

# 1. Synthetic (CPU, fast) — Exp 3
python scripts/run_failure_sweep.py \
    --d-values 16 32 64 128 256 512 \
    --k-values 2 4 8 16 32 64 128 \
    --gap-values 0.5 1.0 2.0 5.0 10.0 \
    --n-seeds 10 \
    --output-dir results/failure

# 2. Pythia scaling sweep — Exp 1 (A100)
python scripts/run_scaling_exponents.py \
    --profile a100 \
    --device cuda \
    --max-tokens 2000000 \
    --batch-size 16 \
    --output-dir results/scaling

# 3. Pythia curriculum — Exp 7 (A100). Loop over model sizes if desired
for sz in pythia-70m-deduped pythia-160m-deduped pythia-410m-deduped pythia-1b-deduped; do
    python scripts/run_curriculum.py \
        --model "EleutherAI/$sz" \
        --device cuda \
        --max-tokens 500000 \
        --output-dir "results/curriculum"
done

# 4. Polymarket embedding — Exp 2 (A100)
python scripts/run_polymarket_embedding.py \
    --latent-dims 8 16 32 64 128 256 \
    --min-volume 5000 \
    --n-epochs 300 \
    --device cuda \
    --output-dir results/polymarket

# 5. Art embedding — Exp 4a (A100)
python scripts/run_art_embedding.py \
    --sources christies sothebys \
    --latent-dims 8 16 32 64 128 256 \
    --n-epochs 200 \
    --device cuda \
    --output-dir results/art

# 6. Cross-domain summary — Exp 4b
python scripts/run_cross_domain.py \
    --polymarket-z results/polymarket/Z_latent64.npy \
    --art-z results/art/Z_latent64.npy \
    --output-dir results/cross_domain
```

## Environment variables

ClickHouse credentials are read from env vars (see `.env.example`). The
`uet.clickhouse.ClickHouseConfig.from_env(database=...)` factory:

- `CH_HOST`, `CH_PORT`, `CH_USER`, `CH_PASSWORD`, `CH_SECURE`
- database defaults to the one passed to `from_env(database=...)`.

## Multi-GPU note

For Pythia-6.9B / 12B on a single A100 40GB, use fp16 (default). For
model-parallel across 4×A100, set `--device cuda` and replace
`device_map=device` in `uet/pretrained.py` line 59 with `device_map="auto"`
if needed; `accelerate` package handles sharding.

## Outputs

Each script writes to `results/<experiment>/`:
- `*.csv` — main metrics table
- `*.png` — paper figures
- `*.npy` — raw embeddings/eigenvalues for downstream use

## Smoke test checklist

```bash
pytest tests/ -v                                # all unit tests
python -c "from uet import eigendecomp, scaling, failure"
python -c "from uet import polymarket_data, art_data, embedding_train, clickhouse"
for s in scripts/run_*.py; do python "$s" --help > /dev/null || echo "FAIL: $s"; done
```
