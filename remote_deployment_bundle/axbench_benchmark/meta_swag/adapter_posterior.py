from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AggregatedAdapterResult:
    scheme: str
    mean_vector: np.ndarray
    weights: np.ndarray
    retained_count: int
    effective_sample_size: float
    beta: float | None
    threshold: float | None
    posterior_trace: float
    top_eigenvalues: tuple[float, ...]
    top_eigenvalue_ratio: float
    max_normalized_weight: float
    score_variance: float
    diagonal_variance: np.ndarray
    deviations: np.ndarray

    def sample(self, num_samples: int, rng: np.random.Generator) -> np.ndarray:
        if self.mean_vector.size == 0:
            return np.zeros((num_samples, 0), dtype=np.float32)

        diag_noise = rng.normal(size=(num_samples, self.mean_vector.size)).astype(np.float32)
        diag_term = np.sqrt(0.5 * self.diagonal_variance)[None, :] * diag_noise

        if self.deviations.size == 0:
            low_rank_term = 0.0
        else:
            scale = np.sqrt(0.5 / max(self.deviations.shape[0] - 1, 1))
            coeffs = rng.normal(size=(num_samples, self.deviations.shape[0])).astype(np.float32)
            low_rank_term = scale * coeffs @ self.deviations

        return self.mean_vector[None, :] + diag_term + low_rank_term


def effective_sample_size(weights: np.ndarray) -> float:
    numerator = float(weights.sum() ** 2)
    denominator = float(np.square(weights).sum())
    return numerator / denominator if denominator > 0 else 0.0


def softmax_weights(scores: np.ndarray, beta: float) -> np.ndarray:
    centered = scores - float(np.max(scores))
    logits = beta * centered
    weights = np.exp(logits)
    return weights / weights.sum()


def threshold_weights(scores: np.ndarray, quantile: float) -> tuple[np.ndarray, float]:
    threshold = float(np.quantile(scores, quantile))
    mask = scores >= threshold
    if not np.any(mask):
        mask[np.argmax(scores)] = True
    weights = mask.astype(np.float32)
    weights /= weights.sum()
    return weights, threshold


def find_beta_for_target_ess(
    scores: np.ndarray,
    target_ess: float,
    beta_min: float = 0.0,
    beta_max: float = 100.0,
    steps: int = 60,
) -> float:
    target = float(np.clip(target_ess, 1.0, len(scores)))
    lo, hi = beta_min, beta_max
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        current = effective_sample_size(softmax_weights(scores, mid))
        if current < target:
            hi = mid
        else:
            lo = mid
    return lo


def build_retention_schedule(total_steps: int, keep_last: int, tail_fraction: float) -> list[int]:
    if total_steps <= 0:
        return []
    if keep_last <= 0:
        return [total_steps]
    tail_fraction = float(np.clip(tail_fraction, 0.0, 1.0))
    start_step = max(1, int(np.floor(total_steps * (1.0 - tail_fraction))))
    candidate_steps = np.linspace(start_step, total_steps, num=min(keep_last, total_steps - start_step + 1))
    return sorted({int(round(step)) for step in candidate_steps})


def _resolve_weights(
    scores: np.ndarray,
    scheme: str,
    beta: float,
    target_ess: float,
    threshold_quantile: float,
) -> tuple[np.ndarray, float | None, float | None]:
    normalized_scheme = scheme.lower()
    if normalized_scheme == "map":
        weights = np.zeros(len(scores), dtype=np.float32)
        weights[-1] = 1.0
        return weights, None, None
    if normalized_scheme == "uniform":
        weights = np.ones(len(scores), dtype=np.float32)
        weights /= weights.sum()
        return weights, None, None
    if normalized_scheme == "softmax":
        return softmax_weights(scores, beta).astype(np.float32), beta, None
    if normalized_scheme == "ess":
        resolved_beta = find_beta_for_target_ess(scores, target_ess)
        return softmax_weights(scores, resolved_beta).astype(np.float32), resolved_beta, None
    if normalized_scheme == "threshold":
        weights, threshold = threshold_weights(scores, threshold_quantile)
        return weights.astype(np.float32), None, threshold
    raise ValueError(f"Unknown aggregation scheme: {scheme}")


def aggregate_adapter_checkpoints(
    checkpoints: np.ndarray,
    scores: np.ndarray,
    scheme: str,
    beta: float = 1.0,
    target_ess: float | None = None,
    threshold_quantile: float = 0.75,
    low_rank_rank: int | None = None,
    num_score_samples: int = 0,
    score_fn=None,
    rng: np.random.Generator | None = None,
) -> AggregatedAdapterResult:
    if checkpoints.ndim != 2:
        raise ValueError("checkpoints must be a 2D array of shape (num_checkpoints, adapter_dim).")
    if checkpoints.shape[0] == 0:
        raise ValueError("At least one checkpoint is required.")
    if len(scores) != checkpoints.shape[0]:
        raise ValueError("scores must have the same length as checkpoints.")

    target = float(target_ess if target_ess is not None else max(8, int(np.ceil(checkpoints.shape[0] / 2))))
    weights, resolved_beta, threshold = _resolve_weights(
        np.asarray(scores, dtype=np.float32),
        scheme=scheme,
        beta=beta,
        target_ess=target,
        threshold_quantile=threshold_quantile,
    )
    mean_vector = np.sum(checkpoints * weights[:, None], axis=0).astype(np.float32, copy=False)
    centered = checkpoints - mean_vector
    diagonal_variance = np.sum(weights[:, None] * centered * centered, axis=0).astype(np.float32, copy=False)

    retain = checkpoints.shape[0] if low_rank_rank is None else min(low_rank_rank, checkpoints.shape[0])
    centered_tail = centered[-retain:]
    weighted_tail = centered_tail * np.sqrt(weights[-retain:])[:, None]

    if weighted_tail.size == 0:
        eigenvalues = np.zeros(0, dtype=np.float32)
    else:
        gram = weighted_tail @ weighted_tail.T / max(retain - 1, 1)
        eigenvalues = np.linalg.eigvalsh(gram).astype(np.float32)
        eigenvalues = np.sort(np.clip(eigenvalues, a_min=0.0, a_max=None))[::-1]

    low_rank_trace = float(eigenvalues.sum())
    posterior_trace = float(0.5 * diagonal_variance.sum() + 0.5 * low_rank_trace)
    top_eigenvalues = tuple(float(value) for value in eigenvalues[:5])
    top_eigenvalue_ratio = float(top_eigenvalues[0] / posterior_trace) if top_eigenvalues and posterior_trace > 0 else 0.0

    score_variance = 0.0
    if score_fn is not None and num_score_samples > 0:
        resolved_rng = rng or np.random.default_rng(0)
        probe = AggregatedAdapterResult(
            scheme=scheme,
            mean_vector=mean_vector,
            weights=weights,
            retained_count=int(checkpoints.shape[0]),
            effective_sample_size=effective_sample_size(weights),
            beta=resolved_beta,
            threshold=threshold,
            posterior_trace=posterior_trace,
            top_eigenvalues=top_eigenvalues,
            top_eigenvalue_ratio=top_eigenvalue_ratio,
            max_normalized_weight=float(weights.max()),
            score_variance=0.0,
            diagonal_variance=diagonal_variance,
            deviations=weighted_tail.astype(np.float32, copy=False),
        )
        sampled_scores = np.asarray(score_fn(probe.sample(num_score_samples, resolved_rng)), dtype=np.float32)
        if sampled_scores.size > 0:
            score_variance = float(np.var(sampled_scores))

    return AggregatedAdapterResult(
        scheme=scheme,
        mean_vector=mean_vector,
        weights=weights,
        retained_count=int(checkpoints.shape[0]),
        effective_sample_size=effective_sample_size(weights),
        beta=resolved_beta,
        threshold=threshold,
        posterior_trace=posterior_trace,
        top_eigenvalues=top_eigenvalues,
        top_eigenvalue_ratio=top_eigenvalue_ratio,
        max_normalized_weight=float(weights.max()),
        score_variance=score_variance,
        diagonal_variance=diagonal_variance,
        deviations=weighted_tail.astype(np.float32, copy=False),
    )
