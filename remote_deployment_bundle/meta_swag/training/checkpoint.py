from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RetainedCheckpoint:
    checkpoint_id: str
    step: int
    epoch: int
    train_loss: float
    adapter_vector: np.ndarray
    adapter_dimension: int
    selected_factor: float | None = None
    weighting_metric: float | None = None
    validation_factor_sweep: list[dict[str, float]] = field(default_factory=list)

    def metadata(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "step": self.step,
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "selected_factor": self.selected_factor,
            "weighting_metric": self.weighting_metric,
            "adapter_dimension": self.adapter_dimension,
            "validation_factor_sweep": self.validation_factor_sweep,
        }
