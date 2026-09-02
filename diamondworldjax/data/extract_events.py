"""Extract the transition-event labels Transformer C needs, from the raw feed.

WHY THIS EXISTS

model/transition.py declares heads for error, wild pitch, passed ball, balk, steals
and outs added, but the processed parquet has none of those columns, so every one
of those heads has been running with obs=None: sampling from its prior, learning
nothing, and contributing a plausible-looking number to the simulator. That is the
same silent-failure shape as the three defects the external review found, and it
means Transformer C cannot be trained until these labels exist.

They are all present in data/raw/mlb_api/feed_live, which is why the raw tier was
worth keeping. Each game's JSON carries allPlays[].playEvents[] with a details.
eventType per non-pitch event, plus allPlays[].runners[] with per-runner movement
and credit.

JOIN KEYS

Output is keyed (game_pk, at_bat_number, pitch_number) so it left-joins onto the
processed parquet. MLB's atBatIndex is 0-based and the parquet's at_bat_number is
1-based, which `--verify` checks against real data rather than assuming, because
an off-by-one here would attach every event to the wrong pitch and still produce a
full, plausible-looking table.

EVENT TIMING

A non-pitch event (a steal, a balk) sits BETWEEN pitches in playEvents. It is
attributed to the pitch that PRECEDES it within the plate appearance, so the label
answers "did this follow the pitch just thrown", which is the causal direction a
sequence model can use. An event before the first pitch of a PA attaches to pitch
number 0, meaning "before the PA started".
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl

FEED_DIR = "data/raw/mlb_api/feed_live"

# eventType values in playEvents that matter to the transition model.
PITCH_EVENTS = {
    "wild_pitch": "wild_pitch",
    "passed_ball": "passed_ball",
    "balk": "balk",
    "stolen_base_2b": "steal",
    "stolen_base_3b": "steal",
    "stolen_base_home": "steal",
    "caught_stealing_2b": "caught_stealing",
    "caught_stealing_3b": "caught_stealing",
    "caught_stealing_home": "caught_stealing",
    "pickoff_1b": "pickoff",
    "pickoff_2b": "pickoff",
    "pickoff_3b": "pickoff",
    "pickoff": "pickoff",
    "error": "error",
    "defensive_indiff": "defensive_indiff",
}

FLAGS = ("wild_pitch", "passed_ball", "balk", "steal", "caught_stealing",
         "pickoff", "error", "defensive_indiff")


def _game(path: str):
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return []
    gpk = d.get("gamePk")
    if gpk is None:
        return []
    rows = []
    for play in d.get("liveData", {}).get("plays", {}).get("allPlays", []):
        ab = play.get("atBatIndex")
        if ab is None:
            continue
        # Errors are recorded on the runners, not as a playEvent, so they are
        # collected separately and attributed to the last pitch of the play.
        play_error = any(
            (r.get("details", {}) or {}).get("event", "").lower().startswith("error")
            or (r.get("details", {}) or {}).get("isOut") is False
            and "error" in ((r.get("details", {}) or {}).get("event", "").lower())
            for r in play.get("runners", []))

        last_pitch = 0
        acc = defaultdict(int)
        per_pitch = {}
        for ev in play.get("playEvents", []):
            if ev.get("isPitch"):
                # Flush whatever accumulated since the previous pitch onto that
                # previous pitch, then start a fresh bucket.
                if acc:
                    per_pitch.setdefault(last_pitch, defaultdict(int))
                    for k, v in acc.items():
                        per_pitch[last_pitch][k] += v
                    acc = defaultdict(int)
                last_pitch = ev.get("pitchNumber", last_pitch + 1)
                per_pitch.setdefault(last_pitch, defaultdict(int))
            else:
                et = (ev.get("details", {}) or {}).get("eventType") or ev.get("type")
                key = PITCH_EVENTS.get(et)
                if key:
                    acc[key] += 1
        if acc:
            per_pitch.setdefault(last_pitch, defaultdict(int))
            for k, v in acc.items():
                per_pitch[last_pitch][k] += v
        if play_error and last_pitch:
            per_pitch.setdefault(last_pitch, defaultdict(int))
            per_pitch[last_pitch]["error"] += 1

        for pn, flags in per_pitch.items():
            if not flags:
                continue
            row = {"game_pk": int(gpk), "at_bat_number": int(ab) + 1,
                   "pitch_number": int(pn)}
            for f in FLAGS:
                row[f] = int(flags.get(f, 0) > 0)
            rows.append(row)
    return rows


def extract(paths, workers=8):
    out = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, rows in enumerate(ex.map(_game, paths, chunksize=32)):
            out.extend(rows)
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(paths)} games, {len(out):,} event rows",
                      flush=True)
    if not out:
        return pl.DataFrame()
    return pl.DataFrame(out)


def verify(df: pl.DataFrame, season: int = 2024) -> None:
    """Check the join actually lands, instead of trusting the index convention.

    A silent off-by-one in at_bat_number would still produce a full table that
    joins to real rows, so the check is on MATCH RATE against the processed
    parquet, and on whether the matched rate of each event is physically sane.
    """
    pq = pl.read_parquet(f"data/processed/pitches_{season}.parquet",
                         columns=["game_pk", "at_bat_number", "pitch_number"])
    sub = df.filter(pl.col("game_pk").is_in(pq["game_pk"].unique().to_list()))
    if sub.height == 0:
        print("VERIFY: no overlapping games, cannot check")
        return
    j = pq.join(sub, on=["game_pk", "at_bat_number", "pitch_number"], how="inner")
    print(f"VERIFY: {sub.height:,} event rows for {season}, "
          f"{j.height:,} joined onto real pitches "
          f"({100.0 * j.height / sub.height:.1f}% matched)")

    # Same check with the off-by-one, to prove the chosen convention is the
    # better one rather than merely adequate.
    alt = sub.with_columns((pl.col("at_bat_number") - 1).alias("at_bat_number"))
    ja = pq.join(alt, on=["game_pk", "at_bat_number", "pitch_number"], how="inner")
    print(f"VERIFY: with at_bat_number-1 the match rate would be "
          f"{100.0 * ja.height / max(alt.height,1):.1f}%")

    n_pitch = pq.height
    for f in FLAGS:
        if f in j.columns:
            print(f"  {f:18} {j[f].sum():7,}  "
                  f"{100.0 * j[f].sum() / n_pitch:6.3f}% of pitches")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="data/processed/events.parquet")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--verify-season", type=int, default=2024)
    ap.add_argument("--only-season", type=int, default=None,
                    help="restrict to games present in that season's parquet")
    args = ap.parse_args()

    paths = sorted(
        os.path.join(FEED_DIR, f) for f in os.listdir(FEED_DIR)
        if f.endswith(".json"))
    if args.only_season:
        keep = set(int(x) for x in pl.read_parquet(
            f"data/processed/pitches_{args.only_season}.parquet",
            columns=["game_pk"])["game_pk"].unique().to_list())
        paths = [p for p in paths
                 if os.path.basename(p)[:-5].isdigit()
                 and int(os.path.basename(p)[:-5]) in keep]
    if args.limit:
        paths = paths[:args.limit]
    print(f"{len(paths):,} feed_live games", flush=True)

    df = extract(paths, args.workers)
    print(f"{df.height:,} event rows", flush=True)
    if df.height:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        df.write_parquet(args.out)
        print(f"saved -> {args.out}")
        if args.verify:
            verify(df, args.verify_season)


if __name__ == "__main__":
    main()
