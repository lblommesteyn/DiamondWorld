"""Per-game environment: weather, wind, altitude, roof, and MLB's own park dimensions.

WHAT THIS IS FOR

Everything here acts on BATTED BALL FLIGHT, with one exception. That matters, because
the park-geometry experiment already showed what happens when you feed flight physics
to heads that do not model flight: A predicts which pitch is thrown and where it
crosses the plate, B predicts whether the batter offers at it, and neither cares about
the wind. Geometry came back a clean null for exactly that reason.

The exception is worth testing: air density affects pitch MOVEMENT. Thin air at
altitude, and hot air anywhere, both reduce break. That is the real reason breaking
balls misbehave at Coors, and it acts on A's stuff model rather than on the batted
ball. So temperature and elevation are the two columns here with a plausible path to
improving the current stack; the rest should wait for a batted-ball head.

WHY fieldInfo SUPERSEDES data/parks/geometry.csv FOR DISTANCES

MLB publishes each park's outfield distances in every game's own feed. That is
authoritative, and being per-game it tracks mid-season fence changes, which a static
scraped table cannot. The scraped table disagrees with it already (it has Chase Field
at 330 down the left line; MLB says 328). What fieldInfo does NOT carry is wall
heights, so the two are complementary: distances from here, heights from the CSV.

WHAT IS NOT AVAILABLE

Humidity and UV are not in the feed. They would need an external weather service joined
on latitude, longitude and first-pitch timestamp. Nothing here fabricates them.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import polars as pl

FEED_DIR = "data/raw/mlb_api/feed_live"

# Wind is recorded as "10 mph, Out To RF". The direction is resolved into two
# components rather than a category, so a model sees a continuous field instead of
# nine arbitrary one-hot slots:
#   wind_out   + blows from home toward centre field (carries a fly ball out)
#   wind_cross + blows from the left-field side toward the right-field side
_S = 0.7071
WIND_VECTORS = {
    "out to cf": (1.0, 0.0),
    "out to lf": (_S, -_S),
    "out to rf": (_S, _S),
    "in from cf": (-1.0, 0.0),
    "in from lf": (-_S, _S),
    "in from rf": (-_S, -_S),
    "l to r": (0.0, 1.0),
    "r to l": (0.0, -1.0),
    "left to right": (0.0, 1.0),
    "right to left": (0.0, -1.0),
    "calm": (0.0, 0.0),
    "none": (0.0, 0.0),
    "varies": (0.0, 0.0),
}

# A closed or fixed roof means the weather columns describe indoor air, so the model
# must be able to tell that apart from a calm outdoor day.
ROOF_CLOSED_HINTS = ("roof closed", "dome")


def _parse_wind(w: str):
    """'10 mph, Out To RF' -> (speed_mph, out_component, cross_component, known)."""
    if not w:
        return None, 0.0, 0.0, 0
    m = re.match(r"\s*(\d+)\s*mph", w.lower())
    speed = float(m.group(1)) if m else None
    direction = ""
    if "," in w:
        direction = w.split(",", 1)[1].strip().lower()
    vec = WIND_VECTORS.get(direction)
    if vec is None:
        return speed, 0.0, 0.0, 0
    return speed, vec[0], vec[1], 1


def _game(path: str):
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return None
    gd = d.get("gameData", {})
    gpk = d.get("gamePk") or gd.get("game", {}).get("pk")
    if gpk is None:
        return None

    wx = gd.get("weather", {}) or {}
    ven = gd.get("venue", {}) or {}
    loc = ven.get("location", {}) or {}
    fi = ven.get("fieldInfo", {}) or {}
    dt = gd.get("datetime", {}) or {}

    cond = (wx.get("condition") or "").strip()
    speed, w_out, w_cross, w_known = _parse_wind(wx.get("wind") or "")

    try:
        temp = float(wx.get("temp"))
    except (TypeError, ValueError):
        temp = None

    roof_type = (fi.get("roofType") or "").strip()
    closed = int(any(h in cond.lower() for h in ROOF_CLOSED_HINTS)
                 or roof_type.lower() == "dome")

    return {
        "game_pk": int(gpk),
        "temp_f": temp,
        "condition": cond,
        "wind_raw": (wx.get("wind") or "").strip(),
        "wind_mph": speed,
        "wind_out": w_out,
        "wind_cross": w_cross,
        "wind_known": w_known,
        "roof_closed": closed,
        "roof_type": roof_type,
        "turf_type": (fi.get("turfType") or "").strip(),
        "day_night": (dt.get("dayNight") or "").strip(),
        "elevation_ft": loc.get("elevation"),
        "azimuth_deg": loc.get("azimuthAngle"),
        "latitude": (loc.get("defaultCoordinates") or {}).get("latitude"),
        "longitude": (loc.get("defaultCoordinates") or {}).get("longitude"),
        # MLB's own park dimensions, authoritative and per game.
        "fi_left_line": fi.get("leftLine"),
        "fi_left_center": fi.get("leftCenter"),
        "fi_center": fi.get("center"),
        "fi_right_center": fi.get("rightCenter"),
        "fi_right_line": fi.get("rightLine"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="data/processed/game_context.parquet")
    ap.add_argument("--scan", action="store_true",
                    help="report the observed condition/wind vocabulary and exit")
    args = ap.parse_args()

    paths = sorted(os.path.join(FEED_DIR, f) for f in os.listdir(FEED_DIR)
                   if f.endswith(".json"))
    if args.limit:
        paths = paths[:args.limit]
    print(f"{len(paths):,} feed_live games", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(_game, paths, chunksize=32)):
            if r:
                rows.append(r)
            if (i + 1) % 4000 == 0:
                print(f"  {i+1}/{len(paths)}", flush=True)

    df = pl.DataFrame(rows)
    print(f"{df.height:,} games with context")

    if args.scan:
        for col in ("condition", "wind_raw", "roof_type", "turf_type", "day_night"):
            c = Counter(df[col].to_list())
            print(f"\n{col}: {len(c)} distinct")
            for k, v in c.most_common(14):
                print(f"  {v:7,}  {k!r}")
        # Anything the wind parser could not resolve is reported, not silently zeroed.
        unknown = df.filter((pl.col("wind_known") == 0) & (pl.col("wind_raw") != ""))
        print(f"\nunparsed wind strings: {unknown.height:,}")
        for k, v in Counter(unknown["wind_raw"].to_list()).most_common(10):
            print(f"  {v:7,}  {k!r}")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.write_parquet(args.out)
    print(f"saved -> {args.out}")
    print(f"  temp present:      {df['temp_f'].is_not_null().sum():,}")
    print(f"  wind resolved:     {int(df['wind_known'].sum()):,}")
    print(f"  elevation present: {df['elevation_ft'].is_not_null().sum():,}")
    print(f"  roof closed:       {int(df['roof_closed'].sum()):,}")


if __name__ == "__main__":
    main()
