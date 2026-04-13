# A100 Runbook: EW-LOLA-PG Experiments

Target paper: "The Omega-Gradient: Evidence-Weighted Multi-Agent Policy Optimisation with Opponent Shaping"
Venue: NExT-Game @ ICML 2026, deadline April 24 AoE

## Quick start (5 commands)

```bash
git clone <this-repo> && cd ICML26
bash experiments/setup_remote.sh        # creates .venv, installs torch, runs smoke test
bash experiments/launch_parallel.sh ipd 10 4    # IPD, 10 seeds, 4 GPUs
bash experiments/launch_parallel.sh rps 10 4    # RPS, 10 seeds, 4 GPUs
.venv/bin/python -m experiments.plot_results --env all
```

## What the server MUST NOT have overwritten

The setup script does NOT touch:
- System Python
- NVIDIA drivers
- CUDA toolkit
- Any system packages
- Anything outside this repo's directory

It only creates a `.venv/` directory and installs Python packages into it.

---

## Server requirements

- Python 3.11 or 3.12 (NOT 3.14 -- PyTorch has no wheels for it)
- NVIDIA drivers installed
- CUDA toolkit (11.8, 12.x, or 13.x -- script auto-detects)
- 4x A100 80GB (or any multi-GPU NVIDIA setup)
- ~50GB disk for venv + results + checkpoints

## Step 1: Setup (once, ~10 min)

```bash
bash experiments/setup_remote.sh
```

This will:
1. Find Python 3.11/3.12
2. Create `.venv/`
3. Detect CUDA version and install matching PyTorch
4. Install numpy, matplotlib, pyyaml, scipy
5. Verify GPU access
6. Run `experiments/smoke_test.py` to validate the full pipeline

IF the smoke test fails, check the error. Common issues:
- `ModuleNotFoundError: No module named 'torch'` -- pip install failed, check CUDA version
- GPU not visible -- check `nvidia-smi`, ensure CUDA toolkit matches driver
- Python version -- PyTorch needs 3.11 or 3.12

## Step 2: Run IPD experiments (~1-2 hours with 4 GPUs)

```bash
bash experiments/launch_parallel.sh ipd 10 4
```

This runs all 7 methods x 10 seeds = 70 jobs, parallelized across 4 GPUs.
Each job writes `experiments/results/ipd_<method>_seed<N>.json`.

After completion, aggregate:
```bash
.venv/bin/python -m experiments.aggregate_results --env ipd --seeds 10
```

This creates `experiments/results/ipd_results.json` (the combined file).

## Step 3: Run RPS experiments (~2-3 hours with 4 GPUs)

```bash
bash experiments/launch_parallel.sh rps 10 4
.venv/bin/python -m experiments.aggregate_results --env rps --seeds 10
```

## Step 4: Generate figures

```bash
.venv/bin/python -m experiments.plot_results --env all
```

Produces:
- `experiments/figures/ipd_adaptation.pdf` -- adaptation curves, all 7 methods
- `experiments/figures/ipd_ablation.pdf` -- ablation (Meta-MAPG vs EW-PG vs LOLA-PG vs EW-LOLA-PG)
- `experiments/figures/rps_adaptation.pdf`
- `experiments/figures/rps_ablation.pdf`
- AUC table printed to stdout

## Step 5: Copy results back

```bash
scp -r user@server:ICML26/experiments/results/ experiments/results/
scp -r user@server:ICML26/experiments/figures/ experiments/figures/
```

---

## Extended experiments (if time permits)

### More seeds (30 instead of 10) -- ~4 hours

```bash
bash experiments/launch_parallel.sh ipd 30 4
bash experiments/launch_parallel.sh rps 30 4
.venv/bin/python -m experiments.aggregate_results --env ipd --seeds 30
.venv/bin/python -m experiments.aggregate_results --env rps --seeds 30
```

### Longer training (2000 meta-steps) -- ~6 hours

Edit the config files before running:
```bash
sed -i 's/n_meta_steps: 500/n_meta_steps: 2000/' experiments/configs/ipd.yaml
sed -i 's/n_meta_steps: 500/n_meta_steps: 2000/' experiments/configs/rps.yaml
sed -i 's/eval_interval: 50/eval_interval: 100/' experiments/configs/ipd.yaml
sed -i 's/eval_interval: 50/eval_interval: 100/' experiments/configs/rps.yaml
```

Then rerun.

### Sample complexity sweep (K = 4, 8, 16, 32, 64)

Run manually for each batch size:
```bash
for K in 4 8 16 32 64; do
    sed -i "s/batch_size: .*/batch_size: $K/" experiments/configs/ipd.yaml
    mkdir -p experiments/results/sample_complexity_K${K}
    bash experiments/launch_parallel.sh ipd 10 4
    .venv/bin/python -m experiments.aggregate_results --env ipd --seeds 10
    cp experiments/results/ipd_results.json experiments/results/sample_complexity_K${K}/
done
# Restore default
sed -i 's/batch_size: .*/batch_size: 64/' experiments/configs/ipd.yaml
```

---

## What experiments produce for the paper

### Minimum viable (Steps 1-5 above)

Produces Figures 1-4 and Table 1 for the paper:
- Fig 1: IPD adaptation curves (cooperating + defecting peers mixed)
- Fig 2: RPS adaptation curves
- Fig 3: Ablation (EW-PG vs LOLA-PG vs EW-LOLA-PG vs Meta-MAPG)
- Table 1: AUC comparison across all methods

### What to check in the results

1. REINFORCE should be worst (no meta-learning)
2. Meta-PG and LOLA-DiCE should be intermediate
3. Meta-MAPG should match Kim et al. (this validates our implementation)
4. EW-LOLA-PG should match or beat Meta-MAPG (our main claim)
5. Ablation: EW-PG alone < EW-LOLA-PG, LOLA-PG alone < EW-LOLA-PG

### If EW-LOLA-PG underperforms Meta-MAPG

Don't panic. Check:
1. Are the CIs overlapping? If yes, "comparable with convergence guarantees" is still a valid paper
2. Try tuning `lola_lambda_init` and `evidence_alpha` (our params are Kim's defaults, not tuned for EW-LOLA-PG)
3. Check if EW-PG alone beats REINFORCE (validates the variance reduction theory independently)

---

## Architecture reference

```
experiments/
    envs/
        iterated_matrix.py     # IPD and RPS environments, persona generation
    agents/
        policy.py              # LSTMPolicy: FC(obs,64)->LSTM(64)->FC(64,n_act)->softmax
        value.py               # LSTMValue: FC(obs,64)->LSTM(64)->FC(64,1)
    training/
        meta_learner.py        # Core: meta_train(), meta_test(), collect_rollout()
        dice.py                # DiCE operator for differentiable PG
        gae.py                 # Generalized Advantage Estimation
        evidence.py            # EvidenceTracker: w_i = V_min/V_i
        pcgrad.py              # Projecting Conflicting Gradients
    configs/
        ipd.yaml               # IPD hyperparameters (from Kim et al. appendix)
        rps.yaml               # RPS hyperparameters
    run_experiment.py          # CLI: --env, --methods, --seeds, --gpu
    aggregate_results.py       # Combine per-seed JSONs into one file
    plot_results.py            # Publication figures
    smoke_test.py              # Validates all components
    launch_parallel.sh         # Parallel launcher across GPUs
    setup_remote.sh            # One-time setup
```

### Methods implemented

| Method | Terms | Evidence | LOLA | Reference |
|--------|-------|----------|------|-----------|
| reinforce | 1 | no | no | Williams 1992 |
| meta_pg | 1+2 | no | no | Al-Shedivat 2018 |
| lola_dice | 1+3 | no | fixed | Foerster 2018 |
| meta_mapg | 1+2+3 | no | fixed | Kim 2021 |
| ew_pg | 1 | yes | no | Dissertation (ablation) |
| lola_pg | 1+3 | no | annealed | Dissertation (ablation) |
| ew_lola_pg | 1+3 | yes | annealed | Dissertation (main) |

Term 1: current policy gradient
Term 2: own learning gradient (through the chain)
Term 3: peer learning gradient (through DiCE)

### Hyperparameters (from Kim et al. Appendix E-G)

| Param | IPD | RPS |
|-------|-----|-----|
| K (batch) | 64 | 64 |
| H (horizon) | 150 | 150 |
| L (chain) | 7 | 7 |
| gamma | 0.96 | 0.90 |
| lr_inner | 0.1 | 0.01 |
| lr_outer_actor | 1e-4 | 1e-5 |
| lr_outer_critic | 1.5e-4 | 1.5e-5 |
| GAE lambda | 0.95 | 0.95 |
| n_meta_steps | 500 | 500 |
| Personas | 480 (400/40/40) | 720 (600/60/60) |

EW-LOLA-PG additional params:
- evidence_alpha: 0.99
- lola_lambda_init: 0.1
- lola_anneal_rate: 0.5
- evidence_w_min: 0.01

---

## Comparison with dissertation experiments

The dissertation (Exp 9-12) uses:
- Tabular policies (numpy, direct gradient)
- 16 named personas (TFT, Always-C, Always-D, etc.)
- 300 episodes per persona, 5 runs
- 5 methods: Standard PG, EW-PG, LOLA, Coop, Omega-PG

The ICML experiments use:
- LSTM policies (PyTorch, meta-learning with DiCE)
- 480/720 random personas (cooperating/defecting population)
- Meta-training with K=64 trajectories, H=150, L=7 chain, 500 outer steps
- 7 methods: REINFORCE, Meta-PG, LOLA-DiCE, Meta-MAPG, EW-PG, LOLA-PG, EW-LOLA-PG

Key differences:
1. LSTM vs tabular: LSTM matches Kim et al. and is more representative of real MARL
2. Meta-learning vs direct PG: meta-learning optimizes the INITIAL policy across personas
3. More methods: we include Kim's baselines for direct comparison
4. Random personas vs named strategies: population sampling matches Kim's protocol

The two experiment sets are complementary:
- Dissertation shows Omega-PG works on tabular games with known strategies
- ICML shows EW-LOLA-PG works in the LSTM meta-learning setting from Kim et al.
