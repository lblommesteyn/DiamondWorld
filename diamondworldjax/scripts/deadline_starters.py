"""Find 2024 starting pitchers who started for two different teams (mid-season acquisitions).

A game's home team pitches the top of each inning, so the starter of the top half belongs to the
home team and the starter of the bottom half to the away team. Team ids come from the MLB schedule
(team_rates_2024), since the processed parquet carries no team column.

  python -m diamondworldjax.scripts.deadline_starters
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import polars as pl

from diamondworldjax.scripts.simulator_benchmarks import team_rates_2024
from diamondworldjax.scripts.starter_value import starters_2024


def starts_by_team():
    """pitcher id -> list of (game_pk, team id, home?) for every 2024 start, in game order."""
    st = starters_2024()
    _, pkt = team_rates_2024()
    dates = game_dates()
    out = defaultdict(list)
    # game_pk is not chronological (postponed and makeup games keep their original ids), so
    # order by the scheduled date
    for pk in sorted(st, key=lambda p: (dates.get(p, "9999"), p)):
        if pk not in pkt or "top" not in st[pk] or "bot" not in st[pk]:
            continue
        home, away = pkt[pk]
        out[st[pk]["top"]].append((pk, home, True))
        out[st[pk]["bot"]].append((pk, away, False))
    return out


def game_dates():
    """game_pk -> ISO date from the cached 2024 schedule."""
    sched = json.loads(Path("data/cache/sched_2024.json").read_text())
    return {int(g["gamePk"]): d["date"] for d in sched["dates"] for g in d["games"]}


def team_names():
    sched = json.loads(Path("data/cache/sched_2024.json").read_text())
    return {g["teams"][s]["team"]["id"]: g["teams"][s]["team"]["name"]
            for d in sched["dates"] for g in d["games"] for s in ("home", "away")}


def main():
    names = {int(k): v for k, v in json.loads(
        Path("data/cache/projections/mlb_names.json").read_text()).items()}
    dates, tnames = game_dates(), team_names()
    for pid, starts in starts_by_team().items():
        teams = [t for _, t, _ in starts]
        if len(set(teams)) < 2:
            continue
        # the switch point is the first start for a team other than the first one
        first = teams[0]
        k = next(i for i, t in enumerate(teams) if t != first)
        after = [s for s in starts[k:]]
        new = after[0][1]
        n_new = sum(1 for _, t, _ in after if t == new)
        print(f"{names.get(pid, pid)!s:22s} {tnames.get(first, first)!s:22s} -> "
              f"{tnames.get(new, new)!s:22s} switch {dates.get(after[0][0], '?')}  "
              f"starts before {k:2d} after {n_new:2d}")


if __name__ == "__main__":
    main()
