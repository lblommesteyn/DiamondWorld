"""Data pipeline: load DiamondWorld parquets → enriched pitch rows."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import polars as pl

from diamondworldjax.paths import processed_root
from diamondworldjax.schema import DW_REMAP, PITCH_TYPE_IDX, PA_OUTCOME_IDX

_DATA_ROOT = processed_root()

# Seasons with full Statcast pitch tracking
TRACKING_SEASONS = list(range(2017, 2025))


def _rule_era(season: int) -> dict[str, int]:
    return {
        "shift_restricted": int(season >= 2023),
        "pitch_clock":      int(season >= 2023),
    }


def _encode_pitch_type(s: str | None) -> int:
    if s is None:
        return PITCH_TYPE_IDX["UNK"]
    return PITCH_TYPE_IDX.get(s[:2], PITCH_TYPE_IDX["UNK"])


def _encode_outcome(s: str | None) -> int:
    if s is None:
        return -1
    return PA_OUTCOME_IDX.get(s, -1)


def load_season(season: int, data_root: Path = _DATA_ROOT) -> pl.DataFrame:
    """Load one season's pitch parquet and apply schema enrichment."""
    path = data_root / f"pitches_{season}.parquet"
    df = pl.read_parquet(path)

    # Rename DiamondWorld columns to DWJAX schema
    rename_map = {k: v for k, v in DW_REMAP.items() if k in df.columns and k != v}
    if rename_map:
        df = df.rename(rename_map)

    # Rule-era indicators (derived from season)
    era = _rule_era(season)
    df = df.with_columns([
        pl.lit(era["shift_restricted"]).cast(pl.Int8).alias("shift_restricted"),
        pl.lit(era["pitch_clock"]).cast(pl.Int8).alias("pitch_clock"),
        pl.lit(season).cast(pl.Int16).alias("season"),
    ])

    # Encode pitch_type string → int
    if "pitch_type" in df.columns:
        df = df.with_columns([
            pl.col("pitch_type").map_elements(
                _encode_pitch_type, return_dtype=pl.Int8
            ).alias("pitch_type_idx")
        ])

    # Encode pa_outcome string → int
    if "pa_outcome" in df.columns:
        df = df.with_columns([
            pl.col("pa_outcome").map_elements(
                _encode_outcome, return_dtype=pl.Int8
            ).alias("pa_outcome_idx")
        ])

    # Pre-PA fatigue must be computed before filtering to terminal rows.
    if "pitch_count_game" in df.columns:
        df = df.with_columns(
            (pl.col("pitch_count_game").min().over(["game_pk", "at_bat_number", "pitcher_id"]) - 1)
            .clip(lower_bound=0).alias("pitch_count_before_pa")
        )

    # Derived fields
    if "inning" in df.columns:
        df = df.with_columns([
            ((pl.col("inning") - 1) / 8.0).cast(pl.Float32).alias("inning_norm"),
        ])

    if "half" in df.columns:
        df = df.with_columns([
            (pl.col("half") == "bot").cast(pl.Int8).alias("half_bin"),
        ])

    # Spray angle from hc_x / hc_y if available and not present
    if "hc_x" in df.columns and "hc_y" in df.columns and "spray_angle" not in df.columns:
        df = df.with_columns([
            (pl.arctan2(pl.col("hc_x") - 125.42, 198.27 - pl.col("hc_y"))
             * (180.0 / np.pi)).cast(pl.Float32).alias("spray_angle")
        ])

    # Encode hurdle booleans → int8 (0/1) so batching can use them as obs
    for col in ("swing", "contact", "foul"):
        if col in df.columns:
            df = df.with_columns([
                pl.col(col).cast(pl.Int8).alias(f"{col}_obs")
            ])

    return df


def load_seasons(seasons: list[int], data_root: Path = _DATA_ROOT) -> pl.DataFrame:
    return pl.concat([load_season(s, data_root) for s in seasons])
