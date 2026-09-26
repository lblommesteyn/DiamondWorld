"""Fetch minor-league season lines (AAA, AA) from the MLB stats API, for translated priors.

One row per player, season, level and team stint, with the counts needed to build per-PA rates
(hit, walk+HBP, strikeout, home run). Pitching lines are batters faced and the same outcomes
allowed. Written to data/cache/milb/milb_{group}.csv. Re-running skips seasons already fetched.

  python -m diamondworldjax.scripts.fetch_milb --seasons 2014-2025
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.request
from pathlib import Path

OUT = Path("data/cache/milb")
LEVELS = {11: "AAA", 12: "AA"}
URL = ("https://statsapi.mlb.com/api/v1/stats?stats=season&group={group}&sportId={sport}"
       "&season={season}&playerPool=ALL&limit={limit}&offset={offset}")
FIELDS = ["season", "level", "player_id", "name", "team", "age", "pa", "h", "bb", "hbp", "so", "hr"]


def fetch(group, sport, season, limit=1000):
    rows, offset = [], 0
    while True:
        for attempt in range(4):
            try:
                req = urllib.request.Request(URL.format(group=group, sport=sport, season=season,
                                                        limit=limit, offset=offset),
                                             headers={"User-Agent": "Mozilla/5.0"})
                d = json.load(urllib.request.urlopen(req, timeout=60))
                break
            except Exception as e:  # transient API errors: back off and retry
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
                print(f"  retry {group} {sport} {season} @{offset}: {e}", flush=True)
        block = d["stats"][0] if d.get("stats") else {"splits": []}
        splits = block.get("splits", [])
        for sp in splits:
            st = sp["stat"]
            pa = st.get("plateAppearances") if group == "hitting" else st.get("battersFaced")
            if not pa:
                continue
            rows.append(dict(season=season, level=LEVELS[sport], player_id=sp["player"]["id"],
                             name=sp["player"].get("fullName", ""), team=sp.get("team", {}).get("name", ""),
                             age=st.get("age", sp.get("player", {}).get("currentAge", "")), pa=pa,
                             h=st.get("hits", 0), bb=st.get("baseOnBalls", 0), hbp=st.get("hitByPitch", 0),
                             so=st.get("strikeOuts", 0), hr=st.get("homeRuns", 0)))
        total = block.get("totalSplits", 0)
        offset += limit
        if offset >= total or not splits:
            return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2014-2025")
    args = ap.parse_args()
    a, b = (int(x) for x in args.seasons.split("-"))
    OUT.mkdir(parents=True, exist_ok=True)
    for group in ("hitting", "pitching"):
        path = OUT / f"milb_{group}.csv"
        have = set()
        if path.exists():
            with open(path) as f:
                have = {(int(r["season"]), r["level"]) for r in csv.DictReader(f)}
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            for season in range(a, b + 1):
                for sport, level in LEVELS.items():
                    if (season, level) in have:
                        continue
                    rows = fetch(group, sport, season)
                    w.writerows(rows)
                    f.flush()
                    print(f"{group} {season} {level}: {len(rows)} lines", flush=True)


if __name__ == "__main__":
    main()
