"""Build a game_pk-keyed closing-odds CSV from the reactiv/delphi odds dump.

Joins the historical MLB moneyline odds (per-book opening + closing lines) to MLB
game_pk via the MLB Stats API schedule (matched on date + full team names), then
writes the schema backtest.py consumes: game_pk, ml_home, ml_away (closing
consensus), plus opening consensus for CLV. 2023-2024 regular season.

Inputs (fetch first):
  curl -sL "https://raw.githubusercontent.com/reactiv/delphi/HEAD/mlb/deprecated/artifacts/mlb_odds_dataset.json" -o /tmp/mlb_odds_dataset.json
  curl -sL "https://statsapi.mlb.com/api/v1/schedule?sportId=1&season=2023&gameType=R" -o /tmp/sched_2023.json
  curl -sL "https://statsapi.mlb.com/api/v1/schedule?sportId=1&season=2024&gameType=R" -o /tmp/sched_2024.json
Data source: reactiv/delphi (free, per-book opening+closing MLB moneyline, 2021-2025);
game_pk crosswalk from the public MLB Stats API. Free/open data only.
Output: data/eval2/odds_2023_2024.csv
"""
from __future__ import annotations
import json
import statistics as st
from pathlib import Path


def consensus(book_lines, which):
    """Median home/away american odds across books for openingLine/currentLine."""
    hs, as_ = [], []
    for b in book_lines:
        ln = b.get(which)
        if not ln:
            continue
        h, a = ln.get("homeOdds"), ln.get("awayOdds")
        # valid american moneyline odds are always |odds| >= 100; smaller values
        # are malformed scrapes (runline/spread leakage) that inflate payouts.
        if h is not None and a is not None and abs(h) >= 100 and abs(a) >= 100:
            hs.append(h); as_.append(a)
    if not hs:
        return None, None
    return st.median(hs), st.median(as_)


def totals_consensus(book_lines, which):
    """Median (total line, over odds, under odds) across books."""
    tl, oo, uo = [], [], []
    for b in book_lines:
        ln = b.get(which)
        if not ln:
            continue
        t, o, u = ln.get("total"), ln.get("overOdds"), ln.get("underOdds")
        if t is not None and o is not None and u is not None and abs(o) >= 100 and abs(u) >= 100:
            tl.append(t); oo.append(o); uo.append(u)
    if not tl:
        return None, None, None
    return st.median(tl), st.median(oo), st.median(uo)


def runline_consensus(book_lines, which):
    """Median (home spread, home odds, away odds). MLB runline is ~always +-1.5."""
    hsp, ho, ao = [], [], []
    for b in book_lines:
        ln = b.get(which)
        if not ln:
            continue
        s, h, a = ln.get("homeSpread"), ln.get("homeOdds"), ln.get("awayOdds")
        if s is not None and h is not None and a is not None and abs(h) >= 100 and abs(a) >= 100:
            hsp.append(s); ho.append(h); ao.append(a)
    if not hsp:
        return None, None, None
    return st.median(hsp), st.median(ho), st.median(ao)


def main():
    odds = json.load(open("/tmp/mlb_odds_dataset.json"))
    # crosswalk: (date, home_full, away_full) -> [game_pks]
    xwalk = {}
    for y in (2023, 2024):
        sched = json.load(open(f"/tmp/sched_{y}.json"))
        for day in sched["dates"]:
            for g in day["games"]:
                key = (g["officialDate"], g["teams"]["home"]["team"]["name"],
                       g["teams"]["away"]["team"]["name"])
                xwalk.setdefault(key, []).append(g["gamePk"])

    rows = []
    matched = dup = nomatch = 0
    for date, games in odds.items():
        if not (date.startswith("2023") or date.startswith("2024")):
            continue
        for g in games:
            gv = g["gameView"]
            if gv.get("gameType") not in (None, "REGULAR", "R", "Regular Season"):
                pass  # keep; gameType label varies
            o = g.get("odds", {})
            ml = o.get("moneyline", [])
            if not ml:
                continue
            hc, ac = consensus(ml, "currentLine")   # closing
            ho, ao = consensus(ml, "openingLine")    # opening (for CLV)
            if hc is None:
                continue
            tot = o.get("totals", [])
            tl, oo_, uo = totals_consensus(tot, "currentLine") if tot else (None, None, None)
            psl = o.get("pointspread", [])
            rsp, rho, rao = runline_consensus(psl, "currentLine") if psl else (None, None, None)
            key = (date, gv["homeTeam"]["fullName"], gv["awayTeam"]["fullName"])
            pks = xwalk.get(key)
            if not pks:
                nomatch += 1
                continue
            if len(pks) > 1:
                dup += 1
                continue  # doubleheader: ambiguous date+teams join, skip
            matched += 1
            rows.append((pks[0], hc, ac, ho, ao, tl, oo_, uo, rsp, rho, rao))

    out = Path("data/eval2/odds_2023_2024.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    def s(v): return "" if v is None else v
    with open(out, "w") as f:
        f.write("game_pk,ml_home,ml_away,ml_home_open,ml_away_open,total_line,over_odds,under_odds,"
                "rl_home_spread,rl_home_odds,rl_away_odds\n")
        for r in rows:
            pk, hc, ac, ho, ao, tl, oo_, uo, rsp, rho, rao = r
            f.write(f"{pk},{hc},{ac},{s(ho)},{s(ao)},{s(tl)},{s(oo_)},{s(uo)},{s(rsp)},{s(rho)},{s(rao)}\n")
    print(f"matched={matched}  skipped_doubleheader={dup}  no_pk_match={nomatch}")
    print(f"wrote {len(rows)} games -> {out}")


if __name__ == "__main__":
    main()
