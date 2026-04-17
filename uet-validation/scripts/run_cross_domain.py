from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uet.eigendecomp import (
    covariance,
    effective_dimension,
    eigenspectrum,
    stable_rank,
)

logger = logging.getLogger(__name__)


def spectrum_summary(evals: np.ndarray) -> dict:
    total = evals.sum()
    cumulative = np.cumsum(evals) / max(total, 1e-12)
    return {
        "n_eigs": len(evals),
        "lambda_max": float(evals[0]) if len(evals) else 0.0,
        "d_eff": effective_dimension(evals),
        "stable_rank": stable_rank(evals),
        "variance_top1": float(cumulative[0]) if len(cumulative) else 0.0,
        "variance_top5": float(cumulative[min(4, len(cumulative) - 1)]) if len(cumulative) else 0.0,
        "variance_top10": float(cumulative[min(9, len(cumulative) - 1)]) if len(cumulative) else 0.0,
    }


def load_domain_embedding(path: Path, domain: str) -> tuple[np.ndarray, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing embedding file: {path}")
    Z = np.load(path)
    evals = eigenspectrum(covariance(Z))
    summary = spectrum_summary(evals)
    summary["domain"] = domain
    summary["path"] = str(path)
    return Z, summary


def main():
    parser = argparse.ArgumentParser(description="Exp 4: Cross-domain Rosetta Stone spectra")
    parser.add_argument("--polymarket-z", type=Path, required=True, help="Path to Z_latent*.npy from polymarket")
    parser.add_argument("--art-z", type=Path, required=True)
    parser.add_argument("--language-z", type=Path, default=None, help="Optional language-model embedding")
    parser.add_argument("--output-dir", type=Path, default=Path("results/cross_domain"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []

    _, poly_summary = load_domain_embedding(args.polymarket_z, "polymarket")
    summaries.append(poly_summary)
    logger.info("polymarket: d_eff=%.2f / %d", poly_summary["d_eff"], poly_summary["n_eigs"])

    _, art_summary = load_domain_embedding(args.art_z, "art")
    summaries.append(art_summary)
    logger.info("art: d_eff=%.2f / %d", art_summary["d_eff"], art_summary["n_eigs"])

    if args.language_z is not None:
        _, lang_summary = load_domain_embedding(args.language_z, "language")
        summaries.append(lang_summary)
        logger.info("language: d_eff=%.2f / %d", lang_summary["d_eff"], lang_summary["n_eigs"])

    df = pd.DataFrame(summaries)
    df.to_csv(args.output_dir / "cross_domain_spectra.csv", index=False)
    logger.info("Saved %s", args.output_dir / "cross_domain_spectra.csv")


if __name__ == "__main__":
    main()
