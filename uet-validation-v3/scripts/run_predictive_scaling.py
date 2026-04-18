"""Predictive scaling test: fit on first 60% points, evaluate on last 40%.

Compares UET vs Kaplan vs Chinchilla in true hold-out RMSE.
A scaling law must PREDICT, not merely fit.
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

V3 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V3))
sys.path.insert(0, str(V3.parent / "uet-validation"))
sys.path.insert(0, str(V3.parent / "uet-validation-v2"))

from uet.run_utils import setup_run_dir, setup_logging, dump_metadata
from uet.scaling_fit import fit_uet_curriculum, uet_predict, PYTHIA_TOKENS_PER_STEP
from uet_v2.fitting_models import fit_kaplan
from uet_v3.chinchilla import fit_chinchilla, chinchilla_predict

logger = logging.getLogger(__name__)


MODEL_CURRICULUMS = {
    "pythia-70m": {
        "path": V3.parent / "uet-validation-v2/results/curriculum/20260418_053410_pythia-70m-deduped/curriculum.csv",
        "n_params": 70_426_624,
        "token_col": None,
    },
    "pythia-160m": {
        "path": V3.parent / "uet-validation/results/curriculum/20260418_020501_pythia-160m/curriculum.csv",
        "n_params": 162_322_944,
        "token_col": None,
    },
    "pythia-410m": {
        "path": V3.parent / "uet-validation/results/curriculum/20260418_020501_pythia-410m/curriculum.csv",
        "n_params": 405_334_016,
        "token_col": None,
    },
    "olmo-1b": {
        "path": V3.parent / "uet-validation-v2/results/curriculum/20260418_140612_OLMo-1B/curriculum.csv",
        "n_params": 1_176_832_000,
        "token_col": "n_tokens",
    },
}

TRAIN_FRAC = 0.6
MIN_N_FOR_FIT = 4


def load_curriculum(cfg: dict, min_tokens: float = 2e9) -> pd.DataFrame:
    df = pd.read_csv(cfg["path"])
    if cfg["token_col"] is None:
        df["n_tokens"] = df["step"].astype(float) * PYTHIA_TOKENS_PER_STEP
    df = df[df["n_tokens"] >= min_tokens].copy()
    df = df.sort_values("n_tokens").reset_index(drop=True)
    return df


def hold_out_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_train = max(MIN_N_FOR_FIT, int(np.floor(n * TRAIN_FRAC)))
    return df.iloc[:n_train].copy(), df.iloc[n_train:].copy()


def rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    r = pred - actual
    return float(np.sqrt(np.mean(r ** 2)))


def evaluate_uet(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    d_eff_tr = train_df["d_eff"].to_numpy()
    d_tr = train_df["hidden_dim"].to_numpy()
    n_tr = train_df["n_tokens"].to_numpy()
    L_tr = train_df["val_loss"].to_numpy()

    fit = fit_uet_curriculum(d_eff_tr, d_tr, n_tr, L_tr)

    d_eff_te = test_df["d_eff"].to_numpy()
    d_te = test_df["hidden_dim"].to_numpy()
    n_te = test_df["n_tokens"].to_numpy()
    L_te = test_df["val_loss"].to_numpy()
    pred_te = uet_predict(fit.c, fit.L_inf, d_eff_te, d_te, n_te)
    return {
        "model": "UET", "n_params_fit": 2,
        "r2_train": fit.r_squared, "rmse_train": fit.rmse,
        "rmse_test": rmse(pred_te, L_te),
        "params": {"c": fit.c, "L_inf": fit.L_inf},
        "pred_test": pred_te.tolist(), "actual_test": L_te.tolist(),
    }


def evaluate_kaplan(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    fit = fit_kaplan(train_df["n_tokens"].to_numpy(), train_df["val_loss"].to_numpy())
    n_te = test_df["n_tokens"].to_numpy()
    L_te = test_df["val_loss"].to_numpy()
    A, alpha, L_inf = fit.params["A"], fit.params["alpha"], fit.params["L_inf"]
    pred_te = A * np.maximum(n_te, 1.0) ** (-alpha) + L_inf
    return {
        "model": "Kaplan", "n_params_fit": 3,
        "r2_train": fit.r_squared, "rmse_train": fit.rmse,
        "rmse_test": rmse(pred_te, L_te),
        "params": fit.params,
        "pred_test": pred_te.tolist(), "actual_test": L_te.tolist(),
    }


def evaluate_chinchilla_single_model(train_df: pd.DataFrame, test_df: pd.DataFrame, n_params: int) -> dict:
    """Chinchilla needs (N, D). Single-model curriculum: N is constant, so B/D^beta+E dominates."""
    N_tr = np.full(len(train_df), n_params, dtype=float)
    D_tr = train_df["n_tokens"].to_numpy()
    L_tr = train_df["val_loss"].to_numpy()
    try:
        fit = fit_chinchilla(N_tr, D_tr, L_tr)
    except RuntimeError as e:
        logger.warning("chinchilla fit failed: %s", e)
        return {
            "model": "Chinchilla", "n_params_fit": 5,
            "r2_train": float("nan"), "rmse_train": float("nan"),
            "rmse_test": float("nan"),
            "params": {}, "pred_test": [], "actual_test": [],
        }
    N_te = np.full(len(test_df), n_params, dtype=float)
    D_te = test_df["n_tokens"].to_numpy()
    L_te = test_df["val_loss"].to_numpy()
    pred_te = chinchilla_predict(fit.A, fit.B, fit.alpha, fit.beta, fit.E, N_te, D_te)
    return {
        "model": "Chinchilla", "n_params_fit": 5,
        "r2_train": fit.r_squared, "rmse_train": fit.rmse,
        "rmse_test": rmse(pred_te, L_te),
        "params": {
            "A": fit.A, "B": fit.B, "alpha": fit.alpha,
            "beta": fit.beta, "E": fit.E,
        },
        "pred_test": pred_te.tolist(), "actual_test": L_te.tolist(),
    }


def evaluate_chinchilla_joint(all_train: pd.DataFrame, all_test: pd.DataFrame) -> dict:
    """Joint fit across all models using N and D."""
    N_tr = all_train["n_params"].to_numpy()
    D_tr = all_train["n_tokens"].to_numpy()
    L_tr = all_train["val_loss"].to_numpy()
    fit = fit_chinchilla(N_tr, D_tr, L_tr)
    N_te = all_test["n_params"].to_numpy()
    D_te = all_test["n_tokens"].to_numpy()
    L_te = all_test["val_loss"].to_numpy()
    pred_te = chinchilla_predict(fit.A, fit.B, fit.alpha, fit.beta, fit.E, N_te, D_te)
    return {
        "model": "Chinchilla-joint", "n_params_fit": 5,
        "r2_train": fit.r_squared, "rmse_train": fit.rmse,
        "rmse_test": rmse(pred_te, L_te),
        "params": {
            "A": fit.A, "B": fit.B, "alpha": fit.alpha,
            "beta": fit.beta, "E": fit.E,
        },
    }


def main() -> None:
    run_dir = setup_run_dir(V3 / "results", "predictive_scaling")
    setup_logging(run_dir)
    logger.info("predictive scaling test (hold-out last 40%%)")

    per_model: list[dict] = []
    joint_train_rows = []
    joint_test_rows = []

    for label, cfg in MODEL_CURRICULUMS.items():
        df = load_curriculum(cfg)
        if len(df) < MIN_N_FOR_FIT + 2:
            logger.warning("%s: only %d points, skipping", label, len(df))
            continue
        train_df, test_df = hold_out_split(df)
        logger.info("%s: %d train + %d test", label, len(train_df), len(test_df))

        uet_res = evaluate_uet(train_df, test_df)
        kap_res = evaluate_kaplan(train_df, test_df)
        chin_res = evaluate_chinchilla_single_model(train_df, test_df, cfg["n_params"])

        for r in (uet_res, kap_res, chin_res):
            r["curriculum"] = label
            r["n_train"] = len(train_df)
            r["n_test"] = len(test_df)
            per_model.append(r)

        for _, row in train_df.iterrows():
            joint_train_rows.append({
                "curriculum": label, "n_params": cfg["n_params"],
                "n_tokens": row["n_tokens"], "val_loss": row["val_loss"],
                "d_eff": row["d_eff"], "hidden_dim": row["hidden_dim"],
            })
        for _, row in test_df.iterrows():
            joint_test_rows.append({
                "curriculum": label, "n_params": cfg["n_params"],
                "n_tokens": row["n_tokens"], "val_loss": row["val_loss"],
                "d_eff": row["d_eff"], "hidden_dim": row["hidden_dim"],
            })

    joint_train = pd.DataFrame(joint_train_rows)
    joint_test = pd.DataFrame(joint_test_rows)

    joint_uet_fit = fit_uet_curriculum(
        joint_train["d_eff"].to_numpy(),
        joint_train["hidden_dim"].to_numpy(),
        joint_train["n_tokens"].to_numpy(),
        joint_train["val_loss"].to_numpy(),
    )
    joint_uet_pred = uet_predict(
        joint_uet_fit.c, joint_uet_fit.L_inf,
        joint_test["d_eff"].to_numpy(),
        joint_test["hidden_dim"].to_numpy(),
        joint_test["n_tokens"].to_numpy(),
    )
    joint_uet_test_rmse = rmse(joint_uet_pred, joint_test["val_loss"].to_numpy())
    logger.info("joint UET: train R²=%.3f, test RMSE=%.4f",
                joint_uet_fit.r_squared, joint_uet_test_rmse)

    joint_chin = evaluate_chinchilla_joint(joint_train, joint_test)
    logger.info("joint Chinchilla: train R²=%.3f, test RMSE=%.4f",
                joint_chin["r2_train"], joint_chin["rmse_test"])

    summary_path = run_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["curriculum", "model", "n_params_fit", "n_train", "n_test",
                    "r2_train", "rmse_train", "rmse_test"])
        for r in per_model:
            w.writerow([r["curriculum"], r["model"], r["n_params_fit"],
                        r["n_train"], r["n_test"],
                        f"{r['r2_train']:.4f}", f"{r['rmse_train']:.4f}",
                        f"{r['rmse_test']:.4f}"])
        w.writerow(["joint", "UET", 2, len(joint_train), len(joint_test),
                    f"{joint_uet_fit.r_squared:.4f}", f"{joint_uet_fit.rmse:.4f}",
                    f"{joint_uet_test_rmse:.4f}"])
        w.writerow(["joint", "Chinchilla-joint", 5, len(joint_train), len(joint_test),
                    f"{joint_chin['r2_train']:.4f}", f"{joint_chin['rmse_train']:.4f}",
                    f"{joint_chin['rmse_test']:.4f}"])
    logger.info("wrote %s", summary_path)

    with (run_dir / "per_point_predictions.json").open("w") as f:
        json.dump(per_model, f, indent=2)

    dump_metadata(run_dir, {
        "train_frac": TRAIN_FRAC,
        "models": list(MODEL_CURRICULUMS.keys()),
        "joint_uet": {"c": joint_uet_fit.c, "L_inf": joint_uet_fit.L_inf},
        "joint_chinchilla": joint_chin["params"],
    })

    print(f"\n=== Predictive Scaling: hold-out RMSE (lower = better) ===\n")
    print(pd.read_csv(summary_path).to_string(index=False))


if __name__ == "__main__":
    main()
