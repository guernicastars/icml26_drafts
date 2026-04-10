Overview
This is a sprint plan, not a recruitment letter. The objective is to submit seven papers to ICML 2026
workshops in Seoul (July 10–11) between April 24 and May 20, combining theoretical results from
Eugene's LSE dissertation with applied empirical work leveraging Bloomsbury Technology's
proprietary datasets and methodological expertise. Most of the underlying research is already done
— the work is reframing, writing, and experiment re-runs, which is what makes a seven-paper sprint
tractable with BT's grunt-work capacity.
Strategic thesis
This sprint rests on two distinct edges that compound when run in parallel:
The theoretical edge. Eugene's dissertation The Ω-Gradient: Evidence-Weighted Multi-Agent Policy
Optimisation in Stochastic Games introduces four orthogonal modifications to policy gradient
methods that provably improve convergence within the Giannou et al. (2022) framework —
evidence weighting, LOLA-based opponent shaping, cooperative communication under bounded
self-knowledge, and sparse-policy regularisation. All are proved and validated on matrix games.
The empirical edge. Bloomsbury Technology operates across four domains where most academic
groups have no data access: the art market (primary and secondary transactions, gallery networks,
provenance chains), Polymarket (the complete historical record of a real prediction market with
resolved ground-truth outcomes), financial markets, and energy markets. BT's methodological stack
— causal intelligence (Bayesian networks, DAGs, SCMs, counterfactual reasoning), custom
embeddings, graph neural networks, and classical ML with rigorous validation — is purpose-built for
these domains.
Why this matters for the sprint. Most ICML workshop submissions run on public benchmarks.
Proprietary datasets with ground-truth labels are a serious differentiator, especially in workshops on
forecasting, hypothesis testing, and causal/cultural AI. The empirical papers are in many cases easier
to produce than the theoretical ones because the research has already been done at BT — the
bottleneck is writeup, not method development.
Sprint objectives
• Submit seven papers to ICML 2026 workshops between April 24 and May 20.
• Three priority-1 papers lock first (NExT-Game, Hypothesis Testing, Forecasting) — these are
the highest-value-per-hour placements and drive sprint success.
• Zero technical overlap across submissions — each paper owns a distinct theorem, dataset,
or empirical question.
• Shared infrastructure: one monorepo, one BibTeX, one experiment-tracking setup across all
seven papers.
• Every coauthor has an attributable contribution — section, proof, experiment, or framing.
No vanity authorship.

# THE TASK

Ω — The Ω-Gradient: Evidence-Weighted Multi-Agent Policy Optimisation with Opponent
Shaping
Priority: P1 — theoretical anchor. Target workshop is a near-perfect fit (speaker list includes Tardos,
Sandholm, Ratliff, Niao He). Tight deadline but the research is complete.
Venue: NExT-Game: New Frontiers in Game-Theoretic Learning @ ICML 2026
Deadline: April 24, 2026 (AoE) — first binding constraint
Source: Dissertation Ch. 7 (primary), Ch. 14 (experiments).
Draft abstract. Policy gradient methods converge to Nash equilibria in general stochastic games at
O(1/√n) rates (Giannou et al., 2022), but the high variance of REINFORCE-type estimators and
narrow basins of attraction around equilibria limit practical performance. We introduce two
orthogonal modifications to the multi-agent policy gradient that provably improve convergence
within the Giannou et al. framework. An evidence-weighted variant scales gradient updates by a
Keynesian evidence weight proportional to effective sample size, reducing effective variance by a
factor HM(V)/AM(V) ≤ 1 via an AM-HM inequality argument. A LOLA-style opponent-shaping term
(Foerster et al., 2018) enlarges the basin of attraction around second-order stationary policies under
a spectral reinforcement condition we identify on the best-response Jacobian. Both modifications
compose cleanly: we prove convergence of the combined EW-LOLA-PG with strictly improved rates
over the baseline, and validate on a suite of matrix games where the combined method achieves
measurably faster convergence and substantial basin enlargement.
Contribution slots.
• Lead author / framing — Eugene.
• Proof polishing (EW-PG convergence, spectral reinforcement condition) — theory-heavy
role.
• Experimental re-runs of matrix-game benchmarks.
• Related work against Giannou, LOLA, Meta-PG, Mertikopoulos line.
Skills sought: MARL, stochastic games, optimisation theory; familiarity with Giannou et al. (2022) or
the LOLA line is a strong signal.
Estimated effort: ~60–80 hours. Highest theoretical density of any paper in the slate.

Need to replicate \subsection{IPD Environment: Mixed Incentive}
\subsection{RPS Environment: Competitive} and \subsection{2-Agent HalfCheetah Environment: Cooperative} (5.2, 5.3, 5.4) from arXiv-2011.00382v5/experiment.tex
with Eugene's dissertation model dissertation/dissertation/technical_report.tex
