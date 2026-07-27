"""Extract 2025 consensus closing moneylines from the reactiv/delphi dataset (which
covers 2021-2025) into data/eval2/odds_2025.csv keyed by MLB game_pk, so the causal
validation can use 2025 as an out-of-sample season with a real sportsbook baseline."""
from __future__ import annotations

import json
import statistics as st
import urllib.request
from pathlib import Path

import polars as pl

SRC = "https://raw.githubusercontent.com/reactiv/delphi/HEAD/mlb/deprecated/artifacts/mlb_odds_dataset.json"


def consensus(ml, key):
    hs, as_ = [], []
    for b in ml:
        ln = b.get(key)
        if not ln:
            continue
        h, a = ln.get("homeOdds"), ln.get("awayOdds")
        if h is not None and a is not None and abs(h) >= 100 and abs(a) >= 100:
            hs.append(h); as_.append(a)
    if not hs:
        return None, None
    return st.median(hs), st.median(as_)


def main():
    p = Path("/tmp/mlbodds.json")
    if not p.exists():
        p.write_bytes(urllib.request.urlopen(SRC, timeout=120).read())
    odds = json.loads(p.read_text())
    sp = Path("/tmp/sched_2025.json")
    if not sp.exists():
        sp.write_bytes(urllib.request.urlopen(
            "https://statsapi.mlb.com/api/v1/schedule?sportId=1&season=2025&gameType=R", timeout=60).read())
    sched = json.loads(sp.read_text())
    xwalk = {}
    for day in sched["dates"]:
        for g in day["games"]:
            key = (g["officialDate"], g["teams"]["home"]["team"]["name"], g["teams"]["away"]["team"]["name"])
            xwalk.setdefault(key, []).append(g["gamePk"])
    rows = []
    for date, games in odds.items():
        if not date.startswith("2025"):
            continue
        for g in games:
            gv = g["gameView"]; ml = g.get("odds", {}).get("moneyline", [])
            if not ml:
                continue
            hc, ac = consensus(ml, "currentLine")
            if hc is None:
                continue
            key = (date, gv["homeTeam"]["fullName"], gv["awayTeam"]["fullName"])
            pks = xwalk.get(key)
            if not pks or len(pks) > 1:
                continue
            rows.append(dict(game_pk=int(pks[0]), ml_home=hc, ml_away=ac))
    df = pl.DataFrame(rows)
    Path("data/eval2").mkdir(parents=True, exist_ok=True)
    df.write_csv("data/eval2/odds_2025.csv")
    print(f"wrote {len(df)} games -> data/eval2/odds_2025.csv")


if __name__ == "__main__":
    main()
