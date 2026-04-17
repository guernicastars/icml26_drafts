from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from uet.clickhouse import ClickHouseConfig, get_client

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArtFeatures:
    lot_ids: np.ndarray
    feature_names: list[str]
    X: np.ndarray
    sources: np.ndarray
    targets: np.ndarray


JOIN_QUERY = """
SELECT
    l.lot_uuid AS lot_uuid,
    l.estimate_low AS estimate_low,
    l.estimate_high AS estimate_high,
    l.lot_number AS lot_number,
    l.accepts_crypto AS accepts_crypto,
    s.num_bids AS num_bids,
    s.starting_bid AS starting_bid,
    s.hammer_price AS hammer_price,
    s.final_price AS final_price,
    s.reserve_met AS reserve_met,
    s.is_sold AS is_sold,
    toUnixTimestamp(s.closing_time) AS closing_ts,
    g.surface_area_cm2 AS surface_area_cm2,
    g.log_surface_area AS log_surface_area,
    g.is_rare_artist AS is_rare_artist,
    g.artist_id AS artist_id,
    g.vital_status AS vital_status,
    se.creator_birth_year AS creator_birth_year,
    se.creator_death_year AS creator_death_year,
    se.creator_nationality AS creator_nationality,
    se.medium AS medium,
    se.date_created AS date_created,
    se.provenance AS provenance,
    se.exhibitions AS exhibitions,
    se.literature AS literature,
    se.condition_summary AS condition_summary,
    se.signed_inscribed AS signed_inscribed,
    se.style_period AS style_period,
    se.origin AS origin,
    se.lot_category AS lot_category,
    v.artist_lot_count AS artist_lot_count
FROM {src}.lots l
LEFT JOIN {src}.sales s ON s.lot_uuid = l.lot_uuid
LEFT JOIN {src}.gold_features g ON g.lot_uuid = l.lot_uuid
LEFT JOIN {src}.silver_extractions se ON se.lot_uuid = l.lot_uuid
LEFT JOIN {src}.v_gold_with_artist v ON v.lot_uuid = l.lot_uuid
WHERE s.is_sold = 1
  AND s.hammer_price > 0
{limit_clause}
"""


def fetch_art_features(
    source: str,
    config: ClickHouseConfig | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    if source not in {"christies", "sothebys", "phillips"}:
        raise ValueError(f"Unknown source: {source}")

    config = config or ClickHouseConfig.from_env(database=source)
    client = get_client(config)

    query = JOIN_QUERY.format(
        src=source,
        limit_clause=f"LIMIT {limit}" if limit else "",
    )
    logger.info("Fetching %s lots via join", source)
    df = client.query_df(query)
    logger.info("Fetched %d sold lots from %s", len(df), source)
    df["_source"] = source
    return df


def fetch_all_sources(
    sources: list[str] | None = None,
    per_source_limit: int | None = None,
    config: ClickHouseConfig | None = None,
) -> pd.DataFrame:
    sources = sources or ["christies", "sothebys"]
    frames = []
    for src in sources:
        cfg = config or ClickHouseConfig.from_env(database=src)
        frames.append(fetch_art_features(src, config=cfg, limit=per_source_limit))
    return pd.concat(frames, ignore_index=True)


def _safe_float(x, default: float = 0.0) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _year_from_string(x) -> float:
    if not isinstance(x, str):
        return 0.0
    m = re.search(r"\b(1[5-9]\d{2}|20[0-2]\d)\b", x)
    return float(m.group(1)) if m else 0.0


def _parse_dimensions(dims: str | None) -> tuple[float, float]:
    if not isinstance(dims, str):
        return 0.0, 0.0
    nums = re.findall(r"(\d+(?:\.\d+)?)", dims)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    if len(nums) == 1:
        return float(nums[0]), 0.0
    return 0.0, 0.0


def _array_len(x) -> float:
    if isinstance(x, (list, tuple, np.ndarray)):
        return float(len(x))
    return 0.0


def _text_len(x) -> float:
    return float(len(x)) if isinstance(x, str) else 0.0


def _str_or_empty(x) -> str:
    if isinstance(x, str):
        return x
    return ""


def build_features(df: pd.DataFrame) -> ArtFeatures:
    rows = []
    for _, r in df.iterrows():
        est_low = _safe_float(r.get("estimate_low"))
        est_high = _safe_float(r.get("estimate_high"))
        hammer = _safe_float(r.get("hammer_price"))
        num_bids = _safe_float(r.get("num_bids"))
        starting = _safe_float(r.get("starting_bid"))
        surface = _safe_float(r.get("surface_area_cm2"))
        birth = _safe_float(r.get("creator_birth_year"))
        death = _safe_float(r.get("creator_death_year"))
        year_created = _year_from_string(r.get("date_created"))
        artist_lot_count = _safe_float(r.get("artist_lot_count"))

        career = max(death - birth, 0.0) if birth > 0 and death > 0 else 0.0
        age_at_creation = year_created - birth if birth > 0 and year_created > 0 else 0.0

        vital = _str_or_empty(r.get("vital_status"))
        medium = _str_or_empty(r.get("medium")).lower()
        origin = _str_or_empty(r.get("origin")).lower()
        nationality = _str_or_empty(r.get("creator_nationality")).lower()

        row = {
            "log_estimate_low": np.log1p(max(est_low, 0.0)),
            "log_estimate_high": np.log1p(max(est_high, 0.0)),
            "log_estimate_mid": np.log1p(max((est_low + est_high) / 2.0, 0.0)),
            "estimate_spread": np.log1p(max(est_high - est_low, 0.0)),
            "estimate_ratio": (est_high / est_low) if est_low > 0 else 1.0,
            "log_surface_area": _safe_float(r.get("log_surface_area")),
            "surface_area_cm2": surface,
            "num_bids": num_bids,
            "log_starting_bid": np.log1p(max(starting, 0.0)),
            "hammer_start_ratio": (hammer / starting) if starting > 0 else 1.0,
            "reserve_met": _safe_float(r.get("reserve_met")),
            "is_rare_artist": _safe_float(r.get("is_rare_artist")),
            "artist_lot_count": np.log1p(artist_lot_count),
            "is_living": 1.0 if vital == "alive" else 0.0,
            "is_deceased": 1.0 if vital == "dead" else 0.0,
            "creator_birth_year": birth / 2000.0 if birth > 0 else 0.0,
            "creator_career_length": career,
            "year_created": year_created / 2000.0 if year_created > 0 else 0.0,
            "age_at_creation": age_at_creation,
            "has_provenance": 1.0 if _array_len(r.get("provenance")) > 0 else 0.0,
            "provenance_count": _array_len(r.get("provenance")),
            "exhibitions_count": _array_len(r.get("exhibitions")),
            "literature_count": _array_len(r.get("literature")),
            "has_condition_report": 1.0 if _text_len(r.get("condition_summary")) > 0 else 0.0,
            "has_signed_inscribed": 1.0 if _text_len(r.get("signed_inscribed")) > 0 else 0.0,
            "has_style_period": 1.0 if _text_len(r.get("style_period")) > 0 else 0.0,
            "has_origin": 1.0 if _text_len(origin) > 0 else 0.0,
            "is_painting": 1.0 if ("oil" in medium or "canvas" in medium or "paint" in medium) else 0.0,
            "is_work_on_paper": 1.0 if ("paper" in medium or "drawing" in medium or "print" in medium) else 0.0,
            "is_sculpture": 1.0 if ("bronze" in medium or "sculpt" in medium or "marble" in medium) else 0.0,
            "is_photograph": 1.0 if ("photo" in medium) else 0.0,
            "is_european": 1.0 if any(k in nationality for k in ("french", "italian", "british", "german", "spanish")) else 0.0,
            "is_american": 1.0 if "american" in nationality else 0.0,
            "is_asian": 1.0 if any(k in nationality for k in ("chinese", "japanese", "korean", "asian")) else 0.0,
            "accepts_crypto": _safe_float(r.get("accepts_crypto")),
            "lot_number": _safe_float(r.get("lot_number")),
            "closing_ts": _safe_float(r.get("closing_ts")) / 1e9,
        }
        rows.append(row)

    feat_df = pd.DataFrame(rows)
    feature_names = list(feat_df.columns)
    X = feat_df.to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    targets = np.log1p(
        np.nan_to_num(df["hammer_price"].to_numpy(dtype=np.float64), nan=0.0)
    ).astype(np.float32)

    return ArtFeatures(
        lot_ids=df["lot_uuid"].to_numpy(),
        feature_names=feature_names,
        X=X,
        sources=df["_source"].to_numpy(),
        targets=targets,
    )


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return (X - mu) / sigma, mu, sigma
