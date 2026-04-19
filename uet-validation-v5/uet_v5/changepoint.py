"""Discovery→formalisation changepoint detection.

Strategy: the discovery phase ends and formalisation begins at the PEAK
of d_eff(t).  Peak detection is more robust than PELT for these
monotone-with-bump trajectories:

  Pythia curriculum: d_eff rises from ~10 early to peak ~130 at step ~8k,
  then compresses to stable ~50-100 by step 143k.

  Grokking: d_eff is already high at first measurement, then drops sharply
  at/before grokking step; peak is near the first logged step.

Reports:
  tau_step       — step of peak d_eff
  d_eff_peak     — max d_eff (discovery height)
  d_eff_final    — last d_eff (formalisation endpoint)
  drop_fraction  — (peak - final) / peak
  has_interior_peak — True if peak is not at the series boundary
"""
from __future__ import annotations

import numpy as np


def detect(steps: np.ndarray, deff: np.ndarray) -> dict:
    if len(steps) == 0:
        return {}

    idx_peak = int(np.argmax(deff))
    peak_val = float(deff[idx_peak])
    final_val = float(deff[-1])
    drop = (peak_val - final_val) / max(peak_val, 1e-9)

    # interior = peak is not at the last point
    has_interior = idx_peak < len(steps) - 1

    return {
        "tau_step": int(steps[idx_peak]),
        "d_eff_peak": round(peak_val, 3),
        "d_eff_final": round(final_val, 3),
        "drop_fraction": round(float(drop), 4),
        "has_interior_peak": bool(has_interior),
    }
