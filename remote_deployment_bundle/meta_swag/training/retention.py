from __future__ import annotations

import numpy as np


def build_retention_schedule(total_steps: int, keep_last: int, tail_fraction: float) -> list[int]:
    if total_steps <= 0:
        return []
    if keep_last <= 0:
        return [total_steps]
    tail_fraction = float(np.clip(tail_fraction, 0.0, 1.0))
    start_step = max(1, int(np.floor(total_steps * (1.0 - tail_fraction))))
    candidate_steps = np.linspace(start_step, total_steps, num=min(keep_last, total_steps - start_step + 1))
    return sorted({int(round(step)) for step in candidate_steps})
