from __future__ import annotations

from typing import Callable

import numpy as np

from .posterior.base import AggregatedAdapterResult
from .posterior.meta_swag import aggregate_adapter_checkpoints

SchemeFactory = Callable[..., AggregatedAdapterResult]

_REGISTRY: dict[str, SchemeFactory] = {}

BUILTIN_SCHEMES = (
    "map", "last_iterate", "uniform", "swa", "ema",
    "softmax", "ess", "threshold",
)

REQUIRES_SPECIAL_CONSTRUCTION = ("laplace",)


def register_scheme(name: str, factory: SchemeFactory) -> None:
    _REGISTRY[name.lower()] = factory


def get_scheme(name: str) -> SchemeFactory:
    key = name.lower()
    if key in _REGISTRY:
        return _REGISTRY[key]
    if key in BUILTIN_SCHEMES:
        return _builtin_factory(key)
    raise KeyError(f"Unknown scheme: {name!r}. Registered: {list_schemes()}")


def list_schemes() -> list[str]:
    return sorted(set(BUILTIN_SCHEMES) | set(_REGISTRY.keys()))


def _builtin_factory(scheme: str) -> SchemeFactory:
    def factory(
        checkpoints: np.ndarray,
        scores: np.ndarray,
        beta: float = 1.0,
        target_ess: float | None = None,
        threshold_quantile: float = 0.75,
        low_rank_rank: int | None = None,
    ) -> AggregatedAdapterResult:
        return aggregate_adapter_checkpoints(
            checkpoints=checkpoints,
            scores=scores,
            scheme=scheme,
            beta=beta,
            target_ess=target_ess,
            threshold_quantile=threshold_quantile,
            low_rank_rank=low_rank_rank,
        )
    return factory
