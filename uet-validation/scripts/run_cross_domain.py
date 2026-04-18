from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uet.eigendecomp import covariance, effective_dimension, eigenspectrum, stable_rank
from uet.run_utils import dump_config, dump_metadata, setup_logging, setup_run_dir

logger = logging.getLogger(__name__)


def spectrum_summary(evals: np.ndarray) -> dict:
    total = evals.sum()
    cumulative = np.cumsum(evals) / max(total, 1e-12)
    return {
        "n_eigs": int(len(evals)),
        "lambda_max": float(evals[0]) if len(evals) else 0.0,
        "d_eff": effective_dimension(evals),
        "stable_rank": stable_rank(evals),
        "variance_top1": float(cumulative[0]) if len(cumulative) else 0.0,
        "variance_top5": float(cumulative[min(4, len(cumulative) - 1)]) if len(cumulative) else 0.0,
        "variance_top10": float(cumulative[min(9, len(cumulative) - 1)]) if len(cumulative) else 0.0,
    }


def load_domain_embedding(path: Path, domain: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    Z = np.load(path)
    evals = eigenspectrum(covariance(Z))
    summary = spectrum_summary(evals)
    summary["domain"] = domain
    summary["path"] = str(path)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Exp 4: Cross-domain Rosetta Stone spectra")
    parser.add_argument("--polymarket-z", type=Path, required=True)
    parser.add_argument("--art-z", type=Path, required=True)
    parser.add_argument("--language-z", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    run_dir = setup_run_dir(args.output_dir, experiment="cross_domain", run_name=args.run_name)
    setup_logging(run_dir)
    dump_config(run_dir, args)

    summaries = [
        load_domain_embedding(args.polymarket_z, "polymarket"),
        load_domain_embedding(args.art_z, "art"),
    ]
    if args.language_z is not None:
        summaries.append(load_domain_embedding(args.language_z, "language"))

    for s in summaries:
        logger.info("%s: d_eff=%.2f sr=%.2f n=%d", s["domain"], s["d_eff"], s["stable_rank"], s["n_eigs"])

    df = pd.DataFrame(summaries)
    df.to_csv(run_dir / "cross_domain_spectra.csv", index=False)

    dump_metadata(run_dir, {"n_domains": len(summaries), "summaries": summaries})
    logger.info("Done. Run dir: %s", run_dir)


if __name__ == "__main__":
    main()
