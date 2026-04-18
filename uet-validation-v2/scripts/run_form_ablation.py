from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "uet-validation"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uet.run_utils import dump_config, dump_metadata, setup_logging, setup_run_dir
from uet.scaling_fit import fit_uet_curriculum, pythia_step_to_tokens, uet_predict
from uet_v2.fitting_models import (
    ModelFit, compute_aic_bic, fit_free_uet, fit_kaplan,
)

logger = logging.getLogger(__name__)
MIN_STEP = 1000


def load_curriculum(curriculum_dir: Path) -> pd.DataFrame:
    csv = curriculum_dir / "curriculum.csv"
    if not csv.exists():
        raise FileNotFoundError(csv)
    df = pd.read_csv(csv)
    df["model"] = curriculum_dir.name
    df["n_tokens"] = pythia_step_to_tokens(df["step"].values)
    return df[df["step"] >= MIN_STEP].reset_index(drop=True)


def uet_to_model_fit(df: pd.DataFrame) -> ModelFit:
    fit = fit_uet_curriculum(
        d_eff=df["d_eff"].values,
        d=df["hidden_dim"].values,
        n=df["n_tokens"].values,
        L=df["val_loss"].values,
    )
    aic, bic = compute_aic_bic(fit.residuals, n_params=2)
    return ModelFit(
        name="uet", n_params=2, r_squared=fit.r_squared, rmse=fit.rmse,
        aic=aic, bic=bic,
        params={"c": fit.c, "L_inf": fit.L_inf},
        predicted=fit.predicted, residuals=fit.residuals,
    )


def ablate_one(df: pd.DataFrame, label: str) -> list[dict]:
    rows = []
    base = {"model": label, "n_points": len(df)}

    for fit_fn, name, kwargs in [
        (lambda: uet_to_model_fit(df), "uet", {}),
        (lambda: fit_kaplan(df["n_tokens"].values, df["val_loss"].values), "kaplan", {}),
        (
            lambda: fit_free_uet(
                df["d_eff"].values, df["hidden_dim"].values,
                df["n_tokens"].values, df["val_loss"].values,
            ),
            "free_uet", {},
        ),
    ]:
        try:
            fit = fit_fn()
        except Exception as e:
            logger.warning("%s %s fit failed: %s", label, name, e)
            continue
        rows.append({
            **base,
            "fit": fit.name,
            "n_params": fit.n_params,
            "r_squared": fit.r_squared,
            "rmse": fit.rmse,
            "aic": fit.aic,
            "bic": fit.bic,
            **{f"param_{k}": v for k, v in fit.params.items()},
        })
        logger.info(
            "%s %-10s  R2=%.4f  RMSE=%.4f  AIC=%.2f  BIC=%.2f  n_params=%d",
            label, fit.name, fit.r_squared, fit.rmse, fit.aic, fit.bic, fit.n_params,
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="L2: Functional-form ablation (UET vs Kaplan vs Free)")
    parser.add_argument("--curriculum-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    run_dir = setup_run_dir(args.output_dir, "form_ablation", args.run_name)
    setup_logging(run_dir)
    dump_config(run_dir, args)

    all_rows = []
    for d in args.curriculum_dirs:
        try:
            df = load_curriculum(d)
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", d, e)
            continue
        label = d.name
        logger.info("Fitting %s (%d points)", label, len(df))
        all_rows.extend(ablate_one(df, label))

    # Joint cross-model ablation — the critical test of UET vs Kaplan.
    # Kaplan has no d_eff term, so it cannot predict across model sizes.
    # Joint UET should win here because d_eff encodes model-specific capacity.
    all_dfs = []
    for d in args.curriculum_dirs:
        try:
            all_dfs.append(load_curriculum(d))
        except FileNotFoundError:
            pass
    if len(all_dfs) >= 2:
        df_joint = pd.concat(all_dfs, ignore_index=True)
        logger.info("Joint fit across %d models (%d points total)", len(all_dfs), len(df_joint))
        all_rows.extend(ablate_one(df_joint, "JOINT"))

    results = pd.DataFrame(all_rows)
    results.to_csv(run_dir / "ablation.csv", index=False)
    logger.info("Saved ablation.csv (%d rows)", len(results))

    summary = (
        results.groupby(["fit", "n_params"])
        .agg(r2_mean=("r_squared", "mean"), aic_mean=("aic", "mean"), bic_mean=("bic", "mean"))
        .reset_index()
    )
    logger.info("\n%s", summary.to_string(index=False))

    dump_metadata(run_dir, {
        "n_models": len(args.curriculum_dirs),
        "n_rows": int(len(results)),
        "fits": results[["model", "fit", "r_squared", "rmse", "aic", "bic"]].to_dict("records"),
    })


if __name__ == "__main__":
    main()
