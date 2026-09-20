"""Item 7 groundwork: fetch player birth dates so age curves are possible.

The project has no age information at all. Steamer's edge over Marcel is usually
attributed to age curves plus minor-league translation, and RESULTS.md names both
as the remaining 0.019-0.047 to Steamer, so the first thing needed is a birth date
per player.

Source: MLB StatsAPI /api/v1/people, batched. Writes data/cache/birthdates.json as
{mlbam_id: "YYYY-MM-DD"}. Cached; rerunning only fetches ids that are missing.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import polars as pl

from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.paths import processed_root

CACHE = Path("data/cache/birthdates.json")
BATCH = 100
URL = ("https://statsapi.mlb.com/api/v1/people?personIds={ids}"
       "&fields=people,id,birthDate,primaryPosition,abbreviation")


def fetch_batch(ids):
    url = URL.format(ids=",".join(str(i) for i in ids))
    raw = urllib.request.urlopen(url, timeout=60).read()
    people = json.loads(raw).get("people", [])
    out = {}
    for p in people:
        if "id" in p and p.get("birthDate"):
            out[str(p["id"])] = {
                "birthDate": p["birthDate"],
                "pos": (p.get("primaryPosition") or {}).get("abbreviation", ""),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2015-2025")
    ap.add_argument("--out", default=str(CACHE))
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.seasons.split("-"))
    seasons = list(range(lo, hi + 1))
    print("collecting batter and pitcher ids from " + str(seasons) + " ...", flush=True)
    ids = set()
    for y in seasons:
        try:
            d = load_seasons([y], data_root=processed_root())
        except Exception as exc:
            print("  skip " + str(y) + ": " + str(exc))
            continue
        for col in ("batter_id", "pitcher_id"):
            if col in d.columns:
                ids.update(int(v) for v in d[col].unique().to_list() if v is not None)
    ids = sorted(i for i in ids if i > 0)
    print("  " + str(len(ids)) + " distinct player ids", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    have = json.loads(out.read_text()) if out.exists() else {}
    todo = [i for i in ids if str(i) not in have]
    print("  " + str(len(have)) + " cached, " + str(len(todo)) + " to fetch", flush=True)

    for k in range(0, len(todo), BATCH):
        chunk = todo[k:k + BATCH]
        try:
            have.update(fetch_batch(chunk))
        except Exception as exc:
            print("  batch at " + str(k) + " failed: " + str(exc), flush=True)
            continue
        if (k // BATCH) % 5 == 0:
            out.write_text(json.dumps(have))
            print("  " + str(min(k + BATCH, len(todo))) + "/" + str(len(todo)), flush=True)

    out.write_text(json.dumps(have))
    print("wrote " + str(out) + " with " + str(len(have)) + " players")
    miss = [i for i in ids if str(i) not in have]
    print("missing birth date for " + str(len(miss)) + " ids")


if __name__ == "__main__":
    main()
