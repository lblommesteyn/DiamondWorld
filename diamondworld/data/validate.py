from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl

from diamondworld.data.paths import processed_root
from diamondworld.data.schema import PITCH_COLUMNS, PITCH_SCHEMA


def null_rates(frame: pl.DataFrame) -> dict[str, float]:
    n_rows = frame.height
    if n_rows == 0:
        return {column: 1.0 for column in frame.columns}
    return {
        column: frame.select(pl.col(column).is_null().sum()).item() / n_rows
        for column in frame.columns
    }


def base_state_after_mismatches(frame: pl.DataFrame) -> int:
    checked = frame.sort(["game_pk", "at_bat_number", "pitch_number"]).with_columns(
        pl.col("base_state").shift(-1).over("game_pk").alias("_next_base_state"),
        pl.col("game_pk").shift(-1).alias("_next_game_pk"),
    )
    checked = checked.filter(pl.col("game_pk") == pl.col("_next_game_pk"))
    checked = checked.filter(pl.col("base_state_after").is_not_null())
    return checked.filter(pl.col("base_state_after") != pl.col("_next_base_state")).height


def validate_frame(frame: pl.DataFrame, *, season: int | None = None) -> dict[str, Any]:
    missing = [column for column in PITCH_COLUMNS if column not in frame.columns]
    extra = [column for column in frame.columns if column not in PITCH_COLUMNS]
    type_mismatches = {}
    for column, dtype in PITCH_SCHEMA.items():
        if column in frame.columns and frame.schema[column] != dtype:
            type_mismatches[column] = {"expected": str(dtype), "actual": str(frame.schema[column])}

    report = {
        "season": season,
        "row_count": frame.height,
        "game_count": frame.select(pl.col("game_pk").n_unique()).item() if "game_pk" in frame.columns else 0,
        "missing_columns": missing,
        "extra_columns": extra,
        "type_mismatches": type_mismatches,
        "null_rates": null_rates(frame),
        "base_state_after_next_pitch_mismatches": base_state_after_mismatches(frame)
        if all(c in frame.columns for c in ["game_pk", "at_bat_number", "pitch_number", "base_state", "base_state_after"])
        else None,
    }
    report["ok"] = (
        report["row_count"] > 0
        and not missing
        and not extra
        and not type_mismatches
        and report["base_state_after_next_pitch_mismatches"] == 0
    )
    return report


def validate_parquet(path: Path, *, season: int | None = None, report_path: Path | None = None) -> dict[str, Any]:
    frame = pl.read_parquet(path)
    report = validate_frame(frame, season=season)
    report_path = report_path or processed_root() / f"validation_{season or path.stem}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate one DiamondWorld season Parquet.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.path or processed_root() / f"pitches_{args.season}.parquet"
    report = validate_parquet(path, season=args.season)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
