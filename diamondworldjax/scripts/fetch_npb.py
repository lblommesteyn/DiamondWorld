"""Fetch NPB (Japan) individual season lines from the official English stats pages on npb.jp.

One request per team, season and group, one second apart. Pitching lines keep batters faced and the
outcomes allowed; batting lines keep PA and outcomes. Names are NPB's romanization ("Last, First");
matching to MLB ids happens in npb_translation.py. Written to data/cache/npb/npb_{group}.csv.

  python -m diamondworldjax.scripts.fetch_npb --seasons 2014-2025
"""
from __future__ import annotations

import argparse
import csv
import re
import time
import urllib.request
from pathlib import Path

OUT = Path("data/cache/npb")
TEAMS = ["b", "bs", "c", "d", "db", "e", "f", "g", "h", "l", "m", "s", "t"]
URL = "https://npb.jp/bis/eng/{season}/stats/id{g}1_{team}.html"
FIELDS = ["season", "team", "name", "pa", "h", "bb", "hbp", "so", "hr"]


def cells(row):
    return [re.sub(r"<[^>]+>", "", c).replace("&nbsp;", " ").strip()
            for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.S)]


def parse(html, group):
    out, header = [], None
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
        c = cells(row)
        if len(c) > 5 and c[1] in ("Pitcher", "Player"):
            header = c
            continue
        if not header or len(c) != len(header):
            continue
        rec = dict(zip(header, c))

        def n(k):
            try:
                return int(rec.get(k, "0") or 0)
            except ValueError:
                return 0
        if group == "p":
            out.append(dict(name=c[1], pa=n("BF"), h=n("H"), bb=n("BB"), hbp=n("HB"), so=n("SO"), hr=n("HR")))
        else:
            out.append(dict(name=c[1], pa=n("PA"), h=n("H"), bb=n("BB"), hbp=n("HP"), so=n("SO"), hr=n("HR")))
    return [r for r in out if r["pa"] > 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2014-2025")
    args = ap.parse_args()
    a, b = (int(x) for x in args.seasons.split("-"))
    OUT.mkdir(parents=True, exist_ok=True)
    for g, group in (("p", "pitching"), ("b", "batting")):
        path = OUT / f"npb_{group}.csv"
        have = set()
        if path.exists():
            with open(path) as f:
                have = {(int(r["season"]), r["team"]) for r in csv.DictReader(f)}
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            for season in range(a, b + 1):
                for team in TEAMS:
                    if (season, team) in have:
                        continue
                    try:
                        req = urllib.request.Request(URL.format(season=season, g=g, team=team),
                                                     headers={"User-Agent": "Mozilla/5.0"})
                        html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
                    except Exception as e:
                        print(f"  {group} {season} {team}: {e}", flush=True)
                        time.sleep(1)
                        continue
                    rows = parse(html, g)
                    for r in rows:
                        r.update(season=season, team=team)
                    w.writerows(rows)
                    f.flush()
                    print(f"{group} {season} {team}: {len(rows)}", flush=True)
                    time.sleep(1)


if __name__ == "__main__":
    main()
