from __future__ import annotations

import numpy as np
from scipy import stats


def paired_wilcoxon(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str = "two-sided",
) -> dict[str, float]:
    x, y = np.asarray(x), np.asarray(y)
    diff = x - y
    nonzero = diff[diff != 0]
    if len(nonzero) < 10:
        return {"statistic": float("nan"), "p_value": float("nan"), "n_nonzero": len(nonzero)}
    result = stats.wilcoxon(nonzero, alternative=alternative)
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_nonzero": len(nonzero),
    }


def cluster_bootstrap_ci(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    statistic_fn=np.mean,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)
    bootstrap_stats = np.empty(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        sampled_values = np.concatenate([values[cluster_ids == c] for c in sampled_clusters])
        bootstrap_stats[b] = statistic_fn(sampled_values)

    lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))
    point = float(statistic_fn(values))

    return {
        "point_estimate": point,
        "ci_lower": lower,
        "ci_upper": upper,
        "alpha": alpha,
        "n_bootstrap": n_bootstrap,
        "n_clusters": n_clusters,
    }
