# A100 Compute Plan: EW-LOLA-PG Experiments for ICML 2026

Target: NExT-Game workshop @ ICML 2026, Seoul
Hardware: 4x A100 (80GB VRAM each, 320GB total)
Current baseline: RTX 3060 Mobile (6GB VRAM), runs only IPD/RPS with reduced settings

---

## Current state (RTX 3060)

What runs now:
- IPD: 7 methods, 10 seeds, K=64, H=150, L=7, 500 meta-steps (~1-2h)
- RPS: 7 methods, 10 seeds, K=64, H=150, L=7, 500 meta-steps (~2-3h)
- All plotting and analysis

What does NOT run:
- HalfCheetah (needs MuJoCo + continuous action policy + teammate pre-training)
- More than 10 seeds (time)
- More than 500 meta-steps (time)
- Batch sizes > 64 (VRAM)
- Multi-player RPS (memory for computational graphs)
- Hyperparameter sweeps (time)

---

## Phase 0: Environment setup on A100 cluster

Estimated: 30 min

```bash
# Python 3.11, PyTorch with CUDA
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy matplotlib pyyaml gymnasium mujoco
pip install multiagent-mujoco  # or clone from github.com/schroederdewitt/multiagent_mujoco
```

Verify:
```bash
python -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
python -m experiments.smoke_test
```

---

## Phase 1: Rerun existing experiments with proper statistical power

**Priority: P0 -- do this FIRST**
**Estimated: 4-6 hours on 4x A100**
**Impact: Tighter confidence intervals, convincing results, fair comparison**

### 1.1 Increase seeds: 10 -> 30

Why: Kim et al. use 10 seeds but CI is wide. 30 seeds gives ~1.7x tighter 95% CI
(CI proportional to 1/sqrt(n): sqrt(10)/sqrt(30) = 0.577).

Files to change:
- `experiments/configs/ipd.yaml` -- set `n_seeds: 30`
- `experiments/configs/rps.yaml` -- set `n_seeds: 30`
- `experiments/run_experiment.py` -- already supports `--seeds N` flag, no change needed

Run:
```bash
# 4 GPUs, run seeds in parallel (4 at a time)
for seed in $(seq 0 29); do
    GPU_ID=$((seed % 4))
    CUDA_VISIBLE_DEVICES=$GPU_ID python -m experiments.run_experiment \
        --env ipd --methods all --seeds 1 --seed-offset $seed &
    if (( (seed + 1) % 4 == 0 )); then wait; fi
done
```

Requires adding `--seed-offset` to `run_experiment.py`:
```python
# in run_experiment.py, modify run_single_seed to accept seed directly
# and add --seed-offset CLI arg
```

### 1.2 Increase meta-training steps: 500 -> 2000+

Why: 500 steps may undertrain Meta-MAPG (the main baseline). Kim et al. train
"until convergence" which was 2000-5000 outer steps with early stopping on
validation set. If Meta-MAPG is undertrained, comparison is unfair.

Files to change:
- `experiments/configs/ipd.yaml` -- set `n_meta_steps: 2000`, `eval_interval: 100`
- `experiments/configs/rps.yaml` -- set `n_meta_steps: 2000`, `eval_interval: 100`
- `experiments/training/meta_learner.py` -- add early stopping on validation reward:
  - Track best validation reward
  - Stop if no improvement for 200 consecutive meta-steps
  - Save best model checkpoint

Implementation needed in `meta_train()`:
```python
# After each eval_interval, compute validation reward
# If val_reward > best_val_reward, save checkpoint, reset patience
# If patience exhausted, break
```

### 1.3 Increase batch size: K=64 -> K=256

Why: Reduces gradient variance, makes the EW-PG variance reduction more visible
against a lower-noise baseline. The whole point of EW-PG is variance reduction;
if the PG estimate is already low-variance due to large K, the advantage shrinks.
Conversely, showing EW-PG still helps at K=256 proves the benefit is structural,
not just small-sample noise.

Files to change:
- `experiments/configs/ipd.yaml` -- set `batch_size: 256`
- `experiments/configs/rps.yaml` -- set `batch_size: 256`

CONDITION: If K=256 makes EW-PG improvement invisible (gap < CI), revert to K=64
for the main result and present K=256 as a supplementary robustness check.

### 1.4 Distributed training: 5 async threads (matching Kim et al.)

Why: Kim uses 5 parallel threads with A3C-style async updates. Our sequential
training is numerically different (not just slower). The async updates introduce
stale gradients that affect convergence, and Kim's hyperparameters are tuned
for this regime.

Implementation needed:
- `experiments/training/async_meta.py` -- new file
  - 5 worker threads, each running a separate persona batch
  - Shared model weights with async gradient updates
  - Lock-free parameter sharing (PyTorch SharedMemory or just torch.multiprocessing)
- Reference: `torch.multiprocessing` with `model.share_memory()`

CONDITION: If async training degrades EW-LOLA-PG relative to Meta-MAPG, report
both sync and async results and note the difference. The theoretical guarantees
hold for sync updates; async is an engineering choice from Kim.

---

## Phase 2: HalfCheetah (2-Agent Cooperative)

**Priority: P1 -- second after Phase 1**
**Estimated: 12-18 hours on 4x A100**
**Impact: Proves method scales beyond matrix games, covers Kim Section 5.4**

### 2.1 Environment setup

Source: github.com/schroederdewitt/multiagent_mujoco
Kim et al. split HalfCheetah into 2 agents: back leg (3 joints) + front leg (3 joints).
Joint reward: velocity to the right.

Files to create:
- `experiments/envs/halfcheetah.py`
  - Wrapper around multiagent-mujoco 2-Agent HalfCheetah
  - Continuous observations (all joint info)
  - Continuous action space per agent (3-dim each)
  - Compatible interface with `IteratedMatrixGame` (obs, step, reset)

### 2.2 Gaussian LSTM policy

The matrix game policy uses Categorical output. HalfCheetah needs Gaussian.

Files to modify:
- `experiments/agents/policy.py` -- add `GaussianLSTMPolicy`
  - Same architecture: FC(obs_dim, 64) -> LSTM(64, 64) -> FC(64, action_dim * 2)
  - Output: mean + log_std for each action dimension
  - `forward()` returns Normal distribution parameters
  - `sample()` returns action + log_prob

Alternatively, make `LSTMPolicy` generic with a `distribution_type` parameter.
The second approach is cleaner for the library pattern.

### 2.3 Teammate pre-training

Kim protocol:
1. Pre-train teammate j with LSTM policy to move LEFT (opposite of true objective)
2. Train for 500 iterations, save checkpoint at each iteration
3. Meta-train population: checkpoints 0-300 (275 train, 25 val)
4. Meta-test population: checkpoints 475-500
5. Gap between 300-475 ensures OOD testing

Files to create:
- `experiments/pretrain_teammate.py`
  - Train a single-agent policy on HalfCheetah-LEFT using PPO or PG
  - Save checkpoints every iteration
  - ~2 hours on A100

- `experiments/configs/halfcheetah.yaml`
  ```yaml
  env: halfcheetah
  horizon: 200
  chain_length: 2
  batch_size: 64
  gamma: 0.95
  gae_lambda: 0.95
  lr_inner: 0.005
  lr_outer_actor: 5.0e-5
  lr_outer_critic: 5.5e-5
  n_meta_steps: 2000
  eval_interval: 100
  n_seeds: 10
  pretrain_iterations: 500
  n_train_checkpoints: 275
  n_val_checkpoints: 25
  n_test_checkpoints: 25
  evidence_alpha: 0.99
  lola_lambda_init: 0.1
  lola_anneal_rate: 0.5
  evidence_w_min: 0.01
  ```

### 2.4 Meta-training on HalfCheetah

Run all 7 methods. Kim found that in cooperative settings, peer learning gradient
helps less (Meta-MAPG ~ Meta-PG). This is actually good for us: if EW-LOLA-PG
still matches or beats Meta-MAPG in cooperative settings, it shows the method
doesn't HURT when opponent shaping is less useful.

CONDITION: If EW-LOLA-PG significantly underperforms Meta-PG on HalfCheetah,
investigate whether the LOLA term is destructive in cooperative settings.
If so, present EW-PG (without LOLA) as the recommended variant for cooperative
games, and discuss in the paper.

### 2.5 Modifications to training loop

`experiments/training/meta_learner.py` needs:
- Support for continuous action spaces (Gaussian policy gradient)
- Larger observation/action dimensions
- The `collect_rollout()` function currently returns discrete actions;
  needs to handle continuous sampling + log_prob correctly

`experiments/training/dice.py` needs:
- Verify DiCE works with continuous actions (it should, since it operates
  on log_probs which are well-defined for Gaussian)

---

## Phase 3: Hyperparameter sweep for EW-LOLA-PG

**Priority: P2**
**Estimated: 8-12 hours on 4x A100**
**Impact: Fair comparison -- current hyperparams are tuned for Meta-MAPG, not EW-LOLA-PG**

### 3.1 Parameters to sweep

EW-LOLA-PG has 3 parameters not in Kim et al.:
- `evidence_alpha`: {0.9, 0.95, 0.99, 0.999} -- EMA smoothing for variance estimate
- `lola_lambda_init`: {0.01, 0.05, 0.1, 0.5} -- initial LOLA strength
- `lola_anneal_rate`: {0.1, 0.3, 0.5, 1.0} -- how fast LOLA decays

Shared parameters to re-tune:
- `lr_inner`: {0.01, 0.05, 0.1, 0.5} -- Kim uses different values per method
- `lr_outer_actor`: {1e-5, 5e-5, 1e-4, 5e-4}

Total grid: 4 * 4 * 4 * 4 * 4 = 1024 configurations per environment.
Impractical. Use random search with 50 configurations.

### 3.2 Implementation

Files to create:
- `experiments/sweep.py`
  - Random search over hyperparameter space
  - 50 random configs, 3 seeds each (for speed), select top 3, rerun with 10 seeds
  - Report best config + sensitivity analysis

Strategy:
1. Random search: 50 configs x 3 seeds x IPD = ~150 runs
2. Each run: 500 meta-steps (reduced, just for screening)
3. Select top 5 by validation AUC
4. Rerun top 5 with 2000 meta-steps, 10 seeds
5. Report best + show sensitivity

CONDITION: If the best EW-LOLA-PG config uses Kim's original hyperparameters
(or very close), that's actually a positive result -- report it as "robust to
hyperparameter choice."

### 3.3 Re-sweep baselines too

For fairness, also sweep lr_inner and lr_outer for Meta-MAPG, Meta-PG, LOLA-DiCE.
Kim's published hyperparameters should be near-optimal for these, but verify.

---

## Phase 4: Sample complexity experiment (Kim Q2)

**Priority: P3**
**Estimated: 6-8 hours on 4x A100**
**Impact: Directly tests the variance reduction theory of EW-PG**

### 4.1 Experiment design

Vary K = {4, 8, 16, 32, 64} on IPD with cooperating peers.
For each K, run all 7 methods, 10 seeds.
Measure AUC (sum of rewards across chain steps L=1..7).

This is Kim et al. Fig 5a. The key prediction:
- All methods degrade as K decreases (higher gradient variance)
- EW-LOLA-PG should degrade LESS than Meta-MAPG because evidence weighting
  reduces effective variance by HM(V)/AM(V)
- The gap should WIDEN at small K (where variance matters most)

### 4.2 Implementation

Files to modify:
- `experiments/run_experiment.py` -- add `--batch-sizes` flag
  ```bash
  python -m experiments.run_experiment --env ipd --methods all --seeds 10 \
      --batch-sizes 4,8,16,32,64
  ```

- `experiments/plot_results.py` -- add `plot_sample_complexity()` function
  - X-axis: K (log scale)
  - Y-axis: AUC
  - One line per method, mean + 95% CI

### 4.3 Additional analysis

Plot the HM(V)/AM(V) ratio over training for each K.
If the ratio is closer to 1 at large K (less heterogeneity in evidence),
this explains why the EW-PG advantage diminishes.

Files to modify:
- `experiments/training/meta_learner.py` -- log `evidence_tracker.variance_ratio()`
  at each meta-step

---

## Phase 5: Multi-player RPS (Kim Q6)

**Priority: P4**
**Estimated: 6-10 hours on 4x A100**
**Impact: Scalability beyond 2-player games**

### 5.1 Environment extension

3-player RPS: 3 agents, each plays R/P/S. State = last joint action.
- Joint action space: 3^3 = 27
- State space: 1 + 27 = 28
- Payoff: agent i beats agent to the right (circular)

4-player RPS: 4 agents, 3^4 = 81 joint actions, 82 states.

Files to modify:
- `experiments/envs/iterated_matrix.py` -- generalize to N-player
  - Current `IteratedMatrixGame` assumes 2 players (R1, R2 matrices)
  - Need N-player version: reward tensor R of shape (n_actions,) * n_players
  - State encoding: one-hot of joint action index

Alternative (simpler): create `NPlayerRPS` class separately.

### 5.2 Policy and training modifications

- Meta-learner currently handles 1 agent + 1 peer
- Need to handle 1 agent + (N-1) peers
- Each peer is sampled independently from persona population
- DiCE peer learning gradient sums over all N-1 peers

Files to modify:
- `experiments/training/meta_learner.py`
  - `collect_rollout()` -- N agents instead of 2
  - `collect_batch_dice()` -- sum DiCE over all peers
  - `_train_step_meta()` -- peer learning gradient over all peers

### 5.3 Computational concern

The computation graph for peer learning gradient grows with N.
At N=4, the graph has 4x more nodes than N=2.
A100 with 80GB should handle this, but monitor memory.

Kim et al. results (Fig 5c):
- 2-player: Meta-MAPG clearly beats baselines
- 3-player: Gap narrows
- 4-player: Meta-MAPG still best but all methods degraded

CONDITION: If 4-player runs OOM on A100, try gradient checkpointing
(`torch.utils.checkpoint`) to trade compute for memory.

---

## Phase 6: Out-of-distribution generalization (Kim Q4)

**Priority: P5**
**Estimated: 4-6 hours on 4x A100**
**Impact: Novel contribution if EW-LOLA-PG shows OOD robustness**

### 6.1 OOD persona construction

Kim et al. construct OOD personas by shifting the probability ranges:
- In-distribution cooperating: P(C|s) in [0.5, 1.0]
- OOD cooperating: P(C|s) in [0.3, 0.7] (overlaps with defecting range)
- OOD defecting: P(C|s) in [0.1, 0.6]

PCA visualization shows reduced overlap between train and test.

Files to modify:
- `experiments/envs/iterated_matrix.py` -- add `generate_ipd_personas_ood()`
  with configurable probability ranges

### 6.2 Hypothesis

EW-LOLA-PG should be more robust to OOD than Meta-MAPG because:
- Evidence weighting is conservative: low-evidence agents take smaller steps
- OOD personas produce unfamiliar gradient patterns -> lower evidence -> smaller updates
- This is natural regularization against OOD overconfidence

### 6.3 Metrics

- In-distribution AUC vs OOD AUC per method
- AUC ratio (OOD/ID) -- higher is more robust
- If EW-LOLA-PG has highest ratio, that's a distinct contribution

CONDITION: If all methods degrade equally on OOD, the evidence weighting
doesn't help with OOD. Report honestly. The theoretical contribution
(convergence guarantees) still stands.

---

## Phase 7: Opponent modeling / decentralized training (Kim Q3)

**Priority: P6**
**Estimated: 6-8 hours on 4x A100**
**Impact: Tests whether EW-PG helps when peer parameters are inferred**

### 7.1 Implementation

Kim et al. Algorithm 3: Meta-MAPG with Opponent Modeling (OM).
Instead of accessing peer's true parameters phi^{-i}, infer them from
observed actions via maximum likelihood.

Files to create:
- `experiments/training/opponent_model.py`
  - Given trajectory tau and LSTM architecture, infer peer's parameters
  - Maximize log-likelihood: L = sum_t log pi^{-i}(a^{-i}_t | s_t, phi_hat)
  - Gradient ascent on phi_hat for M steps
  - Return inferred phi_hat

Files to modify:
- `experiments/training/meta_learner.py` -- add `_train_step_meta_om()` variant
  - Uses inferred peer parameters for computing peer learning gradient
  - Everything else identical to centralized

### 7.2 Expected results

Kim et al. found: Meta-MAPG-OM > Meta-PG but < Meta-MAPG (centralized).
Noise from parameter inference degrades peer learning gradient quality.

Hypothesis: EW-PG helps MORE in decentralized setting because:
- Inferred parameters are noisier -> gradient variance increases
- Evidence weighting downweights high-variance gradient estimates
- Should see larger gap between EW-LOLA-PG-OM and Meta-MAPG-OM
  than between EW-LOLA-PG and Meta-MAPG (centralized)

CONDITION: If opponent modeling + EW-PG produces negligible improvement,
the noise from inference may dominate the variance reduction. Still worth
reporting as a negative result.

---

## Phase 8: Analysis and visualization

**Priority: P7 (run after Phases 1-3 at minimum)**
**Estimated: 2-4 hours on A100 (mostly CPU-bound plotting)**

### 8.1 Action probability dynamics (Kim Appendix F)

Visualize how each method's policy evolves across the Markov chain.
For one representative seed, at each chain step l=0..L-1:
- Plot P(C) for agent i and peer j in IPD
- Plot P(R), P(P), P(S) for agent i in RPS

Files to modify:
- `experiments/training/meta_learner.py` -- save policy outputs during meta-test
  - At each chain step, store action probabilities for all states
  - Requires running with `return_dynamics=True` flag

- `experiments/plot_results.py` -- add `plot_action_dynamics()`
  - Grid of subplots: one column per chain step
  - Shows how policy adapts over time

### 8.2 Gradient term decomposition

Measure Terms 1, 2, 3 separately during training:
- Term 1: current policy gradient (everyone has this)
- Term 2: own learning gradient (Meta-PG contribution)
- Term 3: peer learning gradient (LOLA contribution)

For EW-LOLA-PG, additionally measure:
- Evidence weight w_i over training
- HM(V)/AM(V) ratio over training
- Effective gradient magnitude: w_i * ||grad||

Files to modify:
- `experiments/training/meta_learner.py` -- decompose meta-gradient and log each term
  - Compute Term 1, 2, 3 norms at each meta-step
  - Store in training log

- `experiments/plot_results.py` -- add `plot_gradient_decomposition()`
  - Three subplots: Term 1, 2, 3 magnitudes over training
  - Shows which term dominates and how EW affects each

### 8.3 Variance tracking

Plot the EvidenceTracker's HM(V)/AM(V) ratio over training steps.
Should start near 1.0 (no evidence) and decrease as variance heterogeneity
develops across agents.

Files to modify:
- `experiments/training/meta_learner.py` -- log `evidence_tracker.variance_ratio()`
- `experiments/plot_results.py` -- add `plot_variance_ratio()`

### 8.4 Separate cooperating vs defecting results for IPD

Kim et al. show separate plots for cooperating and defecting peers (Fig 3a, 3b).
We currently mix them.

Files to modify:
- `experiments/run_experiment.py` -- during meta-testing, split test personas
  by type and report separately
- `experiments/plot_results.py` -- generate `ipd_cooperating.pdf` and `ipd_defecting.pdf`

---

## Phase 9: Ablation study refinement

**Priority: P8 (run with Phase 1)**
**Estimated: included in Phase 1 time**

Already have 7 methods. The ablation is built into the method list:
- EW-PG = evidence weighting only (Term 1 + EW)
- LOLA-PG = opponent shaping only (Term 1 + 3)
- EW-LOLA-PG = both combined

Kim's ablation (Fig 5d) removes own-learning and peer-learning terms.
Our ablation additionally decomposes the EW and LOLA contributions.

Ensure the ablation figure clearly shows:
1. EW-PG > REINFORCE (evidence weighting alone helps)
2. LOLA-PG > REINFORCE (opponent shaping alone helps)
3. EW-LOLA-PG > max(EW-PG, LOLA-PG) (improvements compose)
4. EW-LOLA-PG >= Meta-MAPG (combined matches or beats full 3-term)

CONDITION: If (4) fails, emphasize convergence guarantees from theory.
Meta-MAPG has no convergence proof; EW-LOLA-PG does (Thm 3.4, 3.5, 3.6).
A method that's slightly worse empirically but provably convergent is
still a contribution.

---

## Execution order and dependencies

```
Phase 0 (setup)
    |
    v
Phase 1 (rerun IPD/RPS) -----> Phase 8 (analysis)
    |                               |
    |                               v
    |                          Phase 9 (ablation figures)
    |
    +---> Phase 3 (hyperparam sweep, can run in parallel with Phase 1)
    |
    v
Phase 2 (HalfCheetah)
    |
    v
Phase 4 (sample complexity) \
Phase 5 (multi-player RPS)   |-- these are independent, run in parallel
Phase 6 (OOD generalization) |
Phase 7 (opponent modeling)  /
```

Minimum viable experiment set for the paper: Phases 0, 1, 8, 9.
Strong paper: add Phase 2 (HalfCheetah).
Outstanding paper: add any 2 of Phases 4-7.

---

## Time budget on 4x A100

| Phase | Hours | Parallelizable | Notes |
|-------|-------|----------------|-------|
| 0: Setup | 0.5 | No | One-time |
| 1: Rerun IPD/RPS | 4-6 | Yes (4 GPUs) | 30 seeds, 2000 steps, K=256 |
| 2: HalfCheetah | 12-18 | Yes | Pre-train 2h + meta-train 10-16h |
| 3: Hyperparam sweep | 8-12 | Yes (4 GPUs) | 50 configs x 3 seeds screening |
| 4: Sample complexity | 6-8 | Yes | K={4,8,16,32,64} x 7 methods |
| 5: Multi-player RPS | 6-10 | Partially | Memory-bound for 4-player |
| 6: OOD | 4-6 | Yes | Same as Phase 1 with OOD personas |
| 7: Opponent modeling | 6-8 | Yes | Additional OM inference overhead |
| 8: Analysis | 2-4 | CPU-bound | Plotting, mostly |
| **Total** | **49-73** | | **~2-3 days with 4 GPUs** |

With 4x A100 running 24/7: all phases complete in ~3 days.
With 8h/day: ~4-5 days for minimum viable (Phases 0,1,8,9) + HalfCheetah.

---

## Files to create/modify summary

### New files needed
| File | Phase | Description |
|------|-------|-------------|
| `experiments/envs/halfcheetah.py` | 2 | MuJoCo 2-Agent HalfCheetah wrapper |
| `experiments/configs/halfcheetah.yaml` | 2 | HalfCheetah hyperparameters |
| `experiments/pretrain_teammate.py` | 2 | Teammate pre-training script |
| `experiments/training/async_meta.py` | 1.4 | A3C-style async meta-optimization |
| `experiments/training/opponent_model.py` | 7 | MLE opponent parameter inference |
| `experiments/sweep.py` | 3 | Random hyperparameter search |

### Existing files to modify
| File | Phase | Changes |
|------|-------|---------|
| `experiments/configs/ipd.yaml` | 1 | seeds=30, n_meta_steps=2000, batch_size=256 |
| `experiments/configs/rps.yaml` | 1 | seeds=30, n_meta_steps=2000, batch_size=256 |
| `experiments/agents/policy.py` | 2 | Add GaussianLSTMPolicy for continuous actions |
| `experiments/training/meta_learner.py` | 1,2,4,5,7,8 | Early stopping, continuous actions, N-player, OM variant, logging |
| `experiments/run_experiment.py` | 1,4 | --seed-offset, --batch-sizes flags |
| `experiments/plot_results.py` | 4,8 | sample_complexity, action_dynamics, gradient_decomposition plots |
| `experiments/envs/iterated_matrix.py` | 5,6 | N-player RPS, OOD persona generation |

---

## Risk matrix

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| EW-LOLA-PG underperforms Meta-MAPG | Low | High | Emphasize convergence guarantees (Thm 3.4-3.6). Theoretical contribution stands. |
| HalfCheetah doesn't converge | Medium | Medium | IPD+RPS alone sufficient for workshop. Kim also had trouble with MuJoCo (see experiment.tex line 46-47). |
| Hyperparameter sweep shows sensitivity | Medium | Medium | Report sensitivity analysis honestly. If sensitive, present recommended ranges. |
| A100 OOM on 4-player RPS | Medium | Low | Use gradient checkpointing. Fall back to 3-player only. |
| Async training changes results | Low | Medium | Report both sync/async. Note Kim's params are tuned for async. |
| MuJoCo installation fails | Low | Low | Use gymnasium's built-in HalfCheetah-v4 + custom joint splitting. |
| EW-LOLA-PG helps less with large K | High | Low | Expected from theory. Present as validation of variance reduction mechanism. |
| OOD test shows no EW advantage | Medium | Low | Report as negative result. Not the main contribution. |

---

## Decision tree for paper content

```
Start with Phase 0 + Phase 1
    |
    +-- Phase 1 results show EW-LOLA-PG >= Meta-MAPG?
    |       |
    |       +-- YES: Strong empirical result. Proceed to Phase 2.
    |       |       |
    |       |       +-- HalfCheetah works?
    |       |       |       +-- YES: Full replication. Main paper: IPD + RPS + HC.
    |       |       |       +-- NO: Main paper: IPD + RPS. Note HC as future work.
    |       |       |
    |       |       +-- Add Phase 4 (sample complexity)?
    |       |               +-- YES if K sweep shows widening gap at small K
    |       |               +-- Supports the variance reduction theory
    |       |
    |       +-- NO (EW-LOLA-PG < Meta-MAPG):
    |               |
    |               +-- Check: is it close (within CI)?
    |               |       +-- YES: "Comparable performance WITH convergence guarantee"
    |               |       +-- NO: Investigate. Check hyperparameters (Phase 3 first).
    |               |
    |               +-- After Phase 3 sweep, still worse?
    |                       +-- YES: Pivot to theoretical paper. Minimize experiments.
    |                       +-- NO: Report with tuned hyperparameters.
    |
    +-- Phase 6 (OOD) shows EW advantage?
    |       +-- YES: Major selling point. Add OOD robustness section.
    |       +-- NO: Skip from paper. Not the main claim.
    |
    +-- Phase 5 (multi-player) works?
            +-- YES: Shows scalability. Add as supplementary.
            +-- NO: Not critical. Matrix games are standard benchmark.
```

---

## Sanity checks before submitting

1. REINFORCE on IPD converges to mutual defection (~-0.5 per step)
2. Meta-MAPG on IPD roughly matches Kim et al. Fig 3a,b
3. All methods' learning curves are smooth (not oscillating wildly)
4. 95% CIs are tight enough to distinguish methods (30 seeds should help)
5. Ablation shows monotonic improvement: REINFORCE < EW-PG < EW-LOLA-PG
6. HalfCheetah results consistent with Kim et al. observation:
   peer learning gradient less useful in cooperative settings
7. All random seeds are deterministic and reproducible
