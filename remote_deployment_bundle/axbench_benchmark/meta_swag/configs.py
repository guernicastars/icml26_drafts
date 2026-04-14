from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    steps: int
    burn_in: int
    lr: float
    beta: float
    rank: int
    seeds: int
    posterior_samples: int


REPORT_EXPERIMENT_1_VARIANCES: tuple[tuple[float, float], ...] = (
    (1.0, 1.0),
    (5.0, 1.0),
    (20.0, 1.0),
)


DEFAULT_CONFIG = ExperimentConfig(
    steps=400,
    burn_in=200,
    lr=0.22,
    beta=3.0,
    rank=20,
    seeds=12,
    posterior_samples=48,
)
