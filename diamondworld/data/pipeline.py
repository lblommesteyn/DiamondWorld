from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import polars as pl
from tqdm import tqdm

from diamondworld.data.enrich import AtBatRunnerState, build_at_bat_runner_states
from diamondworld.data.mlb_api import fetch_live_feed, fetch_play_by_play, home_plate_umpire_id
from diamondworld.data.paths import ensure_data_dirs, processed_root
from diamondworld.data.schema import OUTCOME_BY_EVENT, PITCH_COLUMNS, PITCH_SCHEMA
from diamondworld.data.statcast import fetch_statcast_season
from diamondworld.data.validate import validate_parquet

SWING_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "foul_bunt",
    "hit_into_play",
    "hit_into_play_score",
    "hit_into_play_no_out",
    "missed_bunt",
}
CONTACT_DESCRIPTIONS = {
    "foul",
    "foul_tip",
    "foul_bunt",
    "hit_into_play",
    "hit_into_play_score",
    "hit_into_play_no_out",
}
FOUL_DESCRIPTIONS = {"foul", "foul_tip", "foul_bunt"}
IN_PLAY_DESCRIPTIONS = {"hit_into_play", "hit_into_play_score", "hit_into_play_no_out"}


def base_state(row: dict[str, Any]) -> int:
    mask = 0
    if row.get("on_1b") is not None:
        mask |= 1
    if row.get("on_2b") is not None:
        mask |= 2
    if row.get("on_3b") is not None:
        mask |= 4
    return mask


def pa_outcome(event: str | None) -> str | None:
    if event is None:
        return None
    return OUTCOME_BY_EVENT.get(event)


def compute_first_inning_velo(rows: list[dict[str, Any]]) -> dict[tuple[int, int], float]:
    speeds: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        speed = row.get("release_speed")
        if row.get("inning") == 1 and speed is not None:
            speeds[(int(row["game_pk"]), int(row["pitcher"]))].append(float(speed))
    return {key: sum(values) / len(values) for key, values in speeds.items() if values}


def tracking_era(season: int) -> int:
    return int(season >= 2020)


def pitcher_tto(pa_seen: int) -> int:
    return min(3, pa_seen // 9 + 1)


def build_pitch_rows(
    statcast_rows: list[dict[str, Any]],
    runner_states: dict[int, dict[int, AtBatRunnerState]],
    umpire_ids: dict[int, int | None],
    season: int,
) -> list[dict[str, Any]]:
    statcast_rows = sorted(
        statcast_rows,
        key=lambda r: (
            int(r["game_pk"]),
            int(r["at_bat_number"]),
            int(r["pitch_number"]),
        ),
    )
    first_inning_velo = compute_first_inning_velo(statcast_rows)
    pitch_count_game: dict[tuple[int, int], int] = defaultdict(int)
    pitch_count_inning: dict[tuple[int, int, int], int] = defaultdict(int)
    pitcher_pa_seen: dict[tuple[int, int], int] = defaultdict(int)
    inning_runs: dict[tuple[int, int, str], int] = defaultdict(int)
    reached_history: dict[tuple[int, str], list[bool]] = defaultdict(list)
    base_sources: dict[tuple[int, int, str], dict[int, str | None]] = defaultdict(
        lambda: {1: None, 2: None, 3: None}
    )
    output: list[dict[str, Any]] = []

    for raw in statcast_rows:
        game_pk = int(raw["game_pk"])
        pitcher_id = int(raw["pitcher"])
        inning = int(raw["inning"])
        half = str(raw["inning_topbot"]).lower()
        at_bat = int(raw["at_bat_number"])
        description = raw.get("description")
        event = raw.get("events")
        terminal = event is not None
        outcome = pa_outcome(event) if terminal else None

        pitch_count_game[(game_pk, pitcher_id)] += 1
        pitch_count_inning[(game_pk, pitcher_id, inning)] += 1

        score_diff = int((raw.get("bat_score") or 0) - (raw.get("fld_score") or 0))
        pre_bat_score = raw.get("bat_score") or 0
        post_bat_score = raw.get("post_bat_score")
        pitch_runs = max(0, int((post_bat_score if post_bat_score is not None else pre_bat_score) - pre_bat_score))

        runner_state = runner_states.get(game_pk, {}).get(at_bat, AtBatRunnerState())
        current_sources = base_sources[(game_pk, inning, half)]
        history = reached_history[(game_pk, half)]
        reached = outcome in {"BB", "HBP", "1B", "2B", "3B", "HR"}
        mean_velo = first_inning_velo.get((game_pk, pitcher_id))
        release_speed = raw.get("release_speed")

        row = {
            "game_pk": game_pk,
            "at_bat_number": at_bat,
            "pitch_number": int(raw["pitch_number"]),
            "pitcher_id": pitcher_id,
            "batter_id": int(raw["batter"]),
            "umpire_id": umpire_ids.get(game_pk),
            "inning": inning,
            "half": half,
            "balls": int(raw["balls"]),
            "strikes": int(raw["strikes"]),
            "outs": int(raw["outs_when_up"]),
            "base_state": base_state(raw),
            "score_diff": score_diff,
            "pitch_count_game": pitch_count_game[(game_pk, pitcher_id)],
            "pitch_count_inning": pitch_count_inning[(game_pk, pitcher_id, inning)],
            "tto": pitcher_tto(pitcher_pa_seen[(game_pk, pitcher_id)]),
            "prior_batter_reached": bool(history[-1]) if len(history) >= 1 else False,
            "prior_two_reached": bool(history[-1] and history[-2]) if len(history) >= 2 else False,
            "how_on_1b": current_sources.get(1),
            "how_on_2b": current_sources.get(2),
            "how_on_3b": current_sources.get(3),
            "runs_this_inning": inning_runs[(game_pk, inning, half)],
            "velo_delta": float(release_speed - mean_velo) if release_speed is not None and mean_velo else None,
            "pitch_type": raw.get("pitch_type"),
            "release_speed": release_speed,
            "pfx_x": raw.get("pfx_x"),
            "pfx_z": raw.get("pfx_z"),
            "plate_x": raw.get("plate_x"),
            "plate_z": raw.get("plate_z"),
            "swing": description in SWING_DESCRIPTIONS,
            "contact": description in CONTACT_DESCRIPTIONS,
            "foul": description in FOUL_DESCRIPTIONS,
            "in_play": description in IN_PLAY_DESCRIPTIONS,
            "launch_speed": raw.get("launch_speed"),
            "launch_angle": raw.get("launch_angle"),
            "pa_terminal": terminal,
            "pa_outcome": outcome,
            "base_state_after": runner_state.base_after,
            "runs_scored": pitch_runs,
            "park_id": raw.get("home_team"),
            "stand": raw.get("stand"),
            "p_throws": raw.get("p_throws"),
            "season": season,
            "tracking_era": tracking_era(season),
        }
        output.append(row)

        if terminal:
            inning_runs[(game_pk, inning, half)] += pitch_runs
            reached_history[(game_pk, half)].append(reached)
            pitcher_pa_seen[(game_pk, pitcher_id)] += 1
            base_sources[(game_pk, inning, half)] = runner_state.how_on.copy()

    for current, nxt in zip(output, output[1:]):
        if current["game_pk"] == nxt["game_pk"]:
            current["base_state_after"] = nxt["base_state"]

    return output


def cast_to_schema(frame: pl.DataFrame) -> pl.DataFrame:
    for column, dtype in PITCH_SCHEMA.items():
        if column not in frame.columns:
            frame = frame.with_columns(pl.lit(None).alias(column))
        frame = frame.with_columns(pl.col(column).cast(dtype, strict=False))
    return frame.select(PITCH_COLUMNS)


def build_season(
    season: int,
    *,
    output_path: Path | None = None,
    force_statcast: bool = False,
    force_mlb_api: bool = False,
    api_sleep_seconds: float = 0.0,
    validate: bool = True,
) -> Path:
    ensure_data_dirs()
    output_path = output_path or processed_root() / f"pitches_{season}.parquet"

    statcast = fetch_statcast_season(season, force=force_statcast)
    statcast = statcast.filter(pl.col("game_type") == "R") if "game_type" in statcast.columns else statcast
    statcast = statcast.sort(["game_pk", "at_bat_number", "pitch_number"])
    game_pks = statcast.select("game_pk").unique().to_series().to_list()

    runner_states: dict[int, dict[int, AtBatRunnerState]] = {}
    umpire_ids: dict[int, int | None] = {}
    for game_pk in tqdm(game_pks, desc=f"MLB playByPlay {season}"):
        payload = fetch_play_by_play(
            int(game_pk),
            force=force_mlb_api,
            sleep_seconds=api_sleep_seconds,
        )
        runner_states[int(game_pk)] = build_at_bat_runner_states(payload)
        live_feed = fetch_live_feed(
            int(game_pk),
            force=force_mlb_api,
            sleep_seconds=api_sleep_seconds,
        )
        umpire_ids[int(game_pk)] = home_plate_umpire_id(live_feed)

    rows = build_pitch_rows(statcast.to_dicts(), runner_states, umpire_ids, season)
    frame = cast_to_schema(pl.DataFrame(rows, infer_schema_length=None))
    frame.write_parquet(output_path)
    if validate:
        validate_parquet(output_path, season=season)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one DiamondWorld season Parquet.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force-statcast", action="store_true")
    parser.add_argument("--force-mlb-api", action="store_true")
    parser.add_argument("--api-sleep-seconds", type=float, default=0.0)
    parser.add_argument("--no-validate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = build_season(
        args.season,
        output_path=args.output,
        force_statcast=args.force_statcast,
        force_mlb_api=args.force_mlb_api,
        api_sleep_seconds=args.api_sleep_seconds,
        validate=not args.no_validate,
    )
    print(path)


if __name__ == "__main__":
    main()
