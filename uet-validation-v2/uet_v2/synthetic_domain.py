from __future__ import annotations

import numpy as np


def generate_structured_data(
    n: int,
    d: int,
    k_true: int,
    snr: float = 3.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data with planted causal subspace.

    Returns (X, U) where:
      X = U @ Z + noise,  shape (n, d)
      U = orthonormal causal directions, shape (d, k_true)

    snr scales signal eigenvalues vs noise eigenvalues.
    With snr=3, spectral gap between signal and noise = 3.
    """
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((d, k_true)))
    Z = rng.standard_normal((k_true, n)) * snr
    noise = rng.standard_normal((d, n))
    X = (U @ Z + noise).T
    return X, U
