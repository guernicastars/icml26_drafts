from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from uet.clickhouse import ClickHouseConfig, get_client

logger = logging.getLogger(__name__)


NUMERIC_FEATURES = [
    "volume_24h",
    "volume_total",
    "volume_1wk",
    "volume_1mo",
    "liquidity",
    "competitive_score",
    "one_day_price_change",
    "one_week_price_change",
    "neg_risk",
]

DERIVED_FEATURES = [
    "duration_days",
    "n_outcomes",
    "max_outcome_price",
    "min_outcome_price",
    "price_spread",
    "price_entropy",
    "has_winning_outcome",
    "log_volume_total",
    "log_liquidity",
    "log_volume_24h",
]


@dataclass(frozen=True)
class PolymarketFeatures:
    condition_ids: np.ndarray
    feature_names: list[str]
    X: np.ndarray
    categories: np.ndarray
    winning_outcomes: np.ndarray


def fetch_resolved_markets(
    config: ClickHouseConfig | None = None,
    min_volume: float = 1000.0,
    limit: int | None = None,
) -> pd.DataFrame:
    config = config or ClickHouseConfig.from_env(database="polymarket")
    client = get_client(config)

    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
    SELECT
        condition_id,
        category,
        tags,
        outcomes,
        outcome_prices,
        neg_risk,
        volume_24h,
        volume_total,
        volume_1wk,
        volume_1mo,
        liquidity,
        competitive_score,
        one_day_price_change,
        one_week_price_change,
        winning_outcome,
        toUnixTimestamp(start_date) AS start_ts,
        toUnixTimestamp(end_date) AS end_ts,
        resolved,
        closed
    FROM polymarket.markets
    WHERE (resolved = 1 OR closed = 1)
      AND volume_total >= {min_volume}
      AND end_date > '2020-01-01'
    ORDER BY end_date DESC
    {limit_clause}
    """
    logger.info("Fetching resolved markets from ClickHouse")
    rows = client.query_df(query)
    logger.info("Fetched %d markets", len(rows))
    return rows


def _price_entropy(prices: list[float]) -> float:
    arr = np.asarray(prices, dtype=float)
    arr = arr[arr > 0]
    if arr.size == 0:
        return 0.0
    p = arr / arr.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


def build_features(df: pd.DataFrame) -> PolymarketFeatures:
    rows = []
    for _, r in df.iterrows():
        prices = list(r["outcome_prices"]) if r["outcome_prices"] is not None else []
        outcomes = list(r["outcomes"]) if r["outcomes"] is not None else []
        prices_arr = np.asarray(prices, dtype=float) if prices else np.array([0.0])

        duration = max((r["end_ts"] - r["start_ts"]) / 86400.0, 0.0)
        row = {
            "volume_24h": r["volume_24h"],
            "volume_total": r["volume_total"],
            "volume_1wk": r["volume_1wk"],
            "volume_1mo": r["volume_1mo"],
            "liquidity": r["liquidity"],
            "competitive_score": r["competitive_score"],
            "one_day_price_change": r["one_day_price_change"],
            "one_week_price_change": r["one_week_price_change"],
            "neg_risk": float(r["neg_risk"]),
            "duration_days": duration,
            "n_outcomes": float(len(outcomes)),
            "max_outcome_price": float(prices_arr.max()) if prices_arr.size else 0.0,
            "min_outcome_price": float(prices_arr.min()) if prices_arr.size else 0.0,
            "price_spread": float(prices_arr.max() - prices_arr.min()) if prices_arr.size else 0.0,
            "price_entropy": _price_entropy(prices),
            "has_winning_outcome": 1.0 if r["winning_outcome"] else 0.0,
            "log_volume_total": float(np.log1p(max(r["volume_total"], 0.0))),
            "log_liquidity": float(np.log1p(max(r["liquidity"], 0.0))),
            "log_volume_24h": float(np.log1p(max(r["volume_24h"], 0.0))),
        }
        rows.append(row)

    feat_df = pd.DataFrame(rows)
    feature_names = NUMERIC_FEATURES + DERIVED_FEATURES
    X = feat_df[feature_names].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return PolymarketFeatures(
        condition_ids=df["condition_id"].to_numpy(),
        feature_names=feature_names,
        X=X,
        categories=df["category"].to_numpy(),
        winning_outcomes=df["winning_outcome"].to_numpy(),
    )


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return (X - mu) / sigma, mu, sigma
