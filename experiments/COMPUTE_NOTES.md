# Compute Notes: Current vs 4xA100

## Current Setup (RTX 3060 Mobile, 6GB VRAM, 64GB RAM)

### What we can run now
- **IPD experiments**: All 7 methods, 10 seeds each. K=64 trajectories, H=150 horizon, L=7 chain.
  LSTM policy is tiny (~50K params). Fits easily in 6GB. Training ~1-2 hours total.
- **RPS experiments**: Same as IPD but with 3 actions and 10 states instead of 5.
  Slightly heavier but still fits. Training ~2-3 hours total.
- **Plotting and analysis**: All figures, AUC tables, confidence intervals.

### What is limited
- **Batch size**: K=64 is the Kim et al. default but we could benefit from K=128 or K=256
  for tighter gradient estimates. VRAM limits this.
- **Seeds**: 10 seeds is the Kim et al. standard. 30+ seeds would give tighter CIs.
- **Meta-training steps**: 500 is practical. Kim et al. trained until convergence
  (~2000-5000 outer steps) with early stopping on validation set.
- **HalfCheetah**: 2-Agent HalfCheetah needs MuJoCo + continuous action spaces.
  Gaussian policy LSTM is heavier. Teammate pre-training for 500 iterations.
  Feasible on CPU but slow (days). Not practical for April 24 deadline on this hardware.
- **Parallel threads**: Kim uses 5 async threads for meta-optimization. We run
  sequentially on single GPU — slower but numerically equivalent.

## With 4xA100 (80GB VRAM each, 320GB total)

### Immediately recalculate
1. **More seeds**: Run 30 seeds instead of 10. Tighter 95% CIs, more convincing plots.
2. **Larger batches**: K=256 trajectories per inner step. Reduces gradient variance,
   may show even clearer EW-PG advantage (the variance reduction becomes more visible
   with less noise from sampling).
3. **Longer training**: 2000+ meta-steps with proper early stopping on validation set.
   Current 500 steps may undertrain Meta-MAPG baseline, making comparison unfair.
4. **Full hyperparameter sweep**: Grid search over lr_inner, lr_outer, lola_lambda_init,
   evidence_alpha across all methods. Currently using Kim's published hyperparameters
   which are tuned for Meta-MAPG, not for EW-LOLA-PG.

### New experiments enabled
5. **HalfCheetah (2-Agent cooperative)**:
   - Pre-train teammate for 500 iterations (GPU-accelerated, ~2 hours on A100)
   - Train all 7 methods with K=64, H=200, L=2
   - 10 seeds, full evaluation
   - This is the continuous control experiment that shows the method scales beyond matrix games
6. **3-player and 4-player RPS** (Kim et al. Q6):
   - State space grows exponentially: 3^3 = 27 joint actions for 3-player
   - Tests scalability of peer learning gradient computation
   - A100 needed for the larger computational graphs
7. **Out-of-distribution generalization** (Kim et al. Q4):
   - Train on in-distribution personas, test on OOD
   - Compare EW-LOLA-PG vs Meta-MAPG on OOD robustness
   - Hypothesis: evidence weighting provides natural OOD robustness
     (low-evidence agents are conservative, avoiding overconfident OOD behavior)
8. **Sample complexity sweep** (Kim et al. Q2):
   - Vary K = {4, 8, 16, 32, 64} and measure AUC
   - Tests how the HM/AM variance reduction translates to sample efficiency
9. **Opponent modeling (decentralized)** (Kim et al. Appendix C):
   - Meta-MAPG with opponent modeling vs centralized training
   - Tests whether EW-PG helps when peer parameters must be inferred
10. **PettingZoo environments**:
    - Test on more diverse environments beyond matrix games
    - Waterworld, simple_spread, etc.

### Distributed training improvements
11. **Async meta-optimization**: 5 parallel threads (matching Kim et al.) across GPUs.
    Each thread runs a separate persona batch, computes meta-gradients, and async-updates
    shared model weights (A3C-style). 4x speedup on 4 GPUs.
12. **Data parallelism across seeds**: Run 4 seeds simultaneously, one per GPU.
    10 seeds completes in time of 3 sequential runs.
13. **Mixed precision**: FP16 for forward pass, FP32 for gradient accumulation.
    2x memory efficiency, enabling K=512 batch sizes.

### Paper improvements with more compute
14. **Action probability dynamics plots** (Kim et al. Appendix F):
    Visualize how each method's policy evolves across the Markov chain.
    Requires storing full policy outputs at every chain step, memory-intensive.
15. **Gradient term decomposition**: Measure the magnitude of Terms 1, 2, 3
    separately to show exactly where the improvement comes from.
16. **Variance tracking over training**: Plot HM(V)/AM(V) ratio over training
    to show how evidence heterogeneity evolves and how EW-PG adapts.

## Priority order (if compute arrives before deadline)

| Priority | Experiment | Time on 4xA100 | Impact |
|----------|-----------|-----------------|--------|
| P0 | Rerun IPD/RPS with 30 seeds, 2000 steps | ~4 hours | Tighter CIs, convincing |
| P1 | HalfCheetah all methods | ~12 hours | Shows continuous control |
| P2 | Hyperparameter sweep for EW-LOLA-PG | ~8 hours | Fair comparison |
| P3 | Sample complexity K={4,8,16,32,64} | ~6 hours | Tests variance theory |
| P4 | 3-player and 4-player RPS | ~6 hours | Tests scalability |
| P5 | OOD generalization | ~4 hours | Novel contribution |
| P6 | Gradient decomposition analysis | ~2 hours | Theoretical insight |
