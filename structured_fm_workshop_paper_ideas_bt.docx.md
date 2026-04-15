**Structured FM Workshop Paper Ideas**

Bloomsbury Technology • ICML 2026 Structured Foundation Models Workshop • April 2026

| Purpose. This memo turns the workshop opportunity into a shortlist of plausible BT papers built around the art market, energy trading, and Polymarket / prediction market datasets. The workshop is a strong fit for papers framed as structured-data foundation models, especially cross-dataset transfer, multimodality, real-world evaluation, and benchmark design. |
| :---- |

**1\. Workshop fit**

**What matters for fit:** the workshop is centered on structured-data FMs rather than graph FMs in the abstract. The strongest BT submissions should therefore be framed around tabular / time-series predictive tasks, with graph structure treated as an input modality, schema, or feature source rather than the entire story.

**•** The three BT datasets share an unusually useful common shape: they are market-like, temporal, heterogeneous, and outcome-linked.

**•** That creates a credible transfer-learning story across domains that look different on the surface but share participants, assets, prices, events, and evolving state.

**•** A narrow art-only or graph-only paper would underuse the portfolio; a cross-domain structured-model paper uses the full moat.

**2\. Ranked paper concepts**

| Rank | Paper concept | Why it is strong | Risk |
| ----- | :---- | :---- | ----- |
| 1 | Cross-domain transfer in structured market models | Best balance of novelty, workshop fit, and use of all three datasets. Clean central claim: pretraining on heterogeneous market data transfers across domains. | Medium |
| 2 | Multimodal structured FM for markets | Best match to the workshop’s multimodal emphasis. Combines time series with text such as auction catalogues, market questions, and event descriptions. | Medium-high |
| 3 | MarketFM-Bench benchmark paper | Safest option. Strong even if the modelling story is still early, because the unified benchmark and leakage-aware evaluation are themselves valuable. | Low |
| 4 | Scaling laws for structured market FMs | Elegant if executed cleanly: test whether model size, data volume, or domain diversity matters most. | High |
| 5 | Structured FMs versus LLM agents on market prediction | Interesting and timely, but only worth doing with disciplined latency / cost / numerical-reliability evaluation. | High |

**A. Cross-domain transfer in structured market models**

*Working title: Cross-Domain Transfer in Structured Foundation Models for Real-World Markets*

**•** Core thesis: a pretrained model on heterogeneous structured market data transfers across art auctions, prediction markets, and energy markets better than domain-specific pretraining alone, especially in few-shot settings.

**•** Representation: each domain is expressed as time series plus metadata, with graph structure entering through typed entity features, topology summaries, and event relations.

**•** Minimal task set: art hammer-price residual prediction; Polymarket resolution or calibration forecasting; energy spike or load forecasting.

**•** Key experiment: pretrain on two domains and adapt to the third. This gives a clear held-out transfer test and a strong workshop narrative.

**B. Multimodal structured FM**

*Working title: Multimodal Foundation Models for Structured Markets*

**•** Core thesis: adding text to structured pretraining improves generalisation under sparse labels and schema shift.

**•** Art can contribute catalogue descriptions, provenance text, exhibition notes, and artist metadata.

**•** Polymarket can contribute market questions, resolution criteria, and linked event text.

**•** Energy may contribute weather reports, plant metadata, system notices, or regulatory text if available.

**C. Benchmark paper**

*Working title: MarketFM-Bench: A Cross-Domain Benchmark for Real-World Structured Market Data*

**•** Core thesis: release a unified evaluation protocol across the three proprietary domains with leakage-aware splits and standard downstream tasks.

**•** The strongest twist is leave-one-domain-out evaluation: train a structured FM on two domains and score adaptation on the third.

**•** This is the best fallback if the full modelling story is not ready by the workshop deadline.

**3\. Recommended lead submission**

**Best single choice:** the cross-domain transfer paper.

**Why this one:** it is the strongest synthesis of the workshop brief and BT’s real asset base. It uses all three datasets, tells a clean story in four pages, and still produces a publishable result if transfer is only partial rather than universal.

**Suggested minimal experiment package**

| Component | What to include |
| :---- | :---- |
| Model | One structured foundation-model backbone with a simple, defensible architecture. |
| Domains | Art market, Polymarket, and energy trading. |
| Tasks | One forecasting or classification task per domain. |
| Ablations | Single-domain, two-domain, and all-domain pretraining; with vs without text / metadata. |
| Baselines | From-scratch training plus one strong domain baseline for each task. |
| Main figure | Transfer matrix showing where cross-domain pretraining helps, where it fails, and by how much. |

**4\. What to avoid**

**•** A pure graph-FM framing with no structured prediction story. That is a weaker fit to this workshop.

**•** A paper that is mostly art economics or Polymarket interpretation with only a thin FM wrapper.

**•** An overly ambitious architecture paper that spreads across too many claims for a 4-page format.

**5\. Practical next move**

**Decision rule:** if the team can support one focused empirical submission, pursue the cross-domain transfer paper. If the modelling work is too early, convert it into the benchmark paper and preserve the broader transfer agenda for a later venue.

Prepared from the workshop call and BT internal scoping notes for coauthor discussion.