---
name: ICML26 EW-LOLA-PG Paper Status
description: NExT-Game workshop paper on precision-weighted + opponent-shaping PG. Deadline April 24 AoE. Based on Eugene's dissertation.
type: project
---

Paper: "Tighter Convergence Constants for Multi-Agent PG via Precision Weighting and Opponent-Shaping Basin Enlargement"
Target: NExT-Game workshop @ ICML 2026, deadline 2026-04-24 AoE

**Why:** P1 priority. Theoretical anchor combining Giannou et al. framework with two orthogonal improvements from Eugene's LSE dissertation. Workshop speakers include Tardos, Sandholm, Ratliff, Niao He.

**Current state (2026-04-14):**
- Draft paper exists at `dissertation/papers/ew-lola-pg/src/main.tex` with full theory (Theorems 1-3), proofs, related work. Experiments section references existing figures.
- Dissertation simulation code at `dissertation/dissertation/simulations/` has: evidence_weighted_pg.py, lola_basin.py, full_experiments.py, games.py
- Meta-SWAG experiments at `dissertation/ICML Sprint/meta-swag/` — separate paper, different method
- PW weight normalization discrepancy: paper uses sum-to-N normalization, code uses V_min/V_est (slower drift)
- Previous session narrowed experiments from 5 games to 3: Matching Pennies, RPS, Prisoner's Dilemma
- Plan had 15 tasks, 5 completed, wanted to build a `pwlola` library + run on LLMs (Qwen-2.5-1.5B + LoRA via GRPO)
- LLM angle: advantage alignment approximation to LOLA (O(1) memory), maps to GRPO pipeline
- Workshop angle: "algorithmic monoculture" — LLMs locking onto pure strategies, LOLA's basin enlargement promotes diversity

**How to apply:** 10 days to deadline. Focus on strengthening the existing draft, not rewriting experiments from scratch. The tabular matrix-game experiments already validate the theory. The LLM fine-tuning angle is ambitious and optional.
