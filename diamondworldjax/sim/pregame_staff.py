"""Leak-free staff selection for pre-game simulation.

`game_extract.extract_games` reads each team's pitching staff off the COMPLETED game,
in the order the relievers actually appeared. The fitted hook decides when the starter
is pulled, but who follows is taken from the finished game, and a manager's bullpen
choices are endogenous to how that game went. That is look-ahead, and it biases every
game-level result built on those arrays.

This module replaces the realized staff with one that could have been written down
before first pitch.

WHAT IS AND IS NOT LEAKAGE

The starter is public days in advance, so keeping the actual starter is legitimate and
is what a real pre-game forecast would do. What is not legitimate is knowing which
relievers appeared, in what order, in the game being predicted.

HOW THE POOL IS BUILT

Team labels are not in the processed schema, and deriving them by co-appearance
union-find collapses to a single component because mid-season trades link every club.
So team identity is sidestepped: for each game we take the pitchers who appeared
alongside THIS GAME'S STARTER in STRICTLY EARLIER games, ranked by how often. That is
knowable pre-game, tracks the roster through trades without needing team labels, and
degrades gracefully.

Ordering by prior co-appearance frequency is a real pre-game prediction of who is most
likely to be used, which is the honest stand-in for the realized order.

Games are ordered by game_pk, which increases with date within a season. That is the
only time key the processed data carries.

CORRECTION (Oct 2026): game_pk is NOT chronological. A postponed game keeps its original,
low game_pk and is played weeks later, so under game_pk order its relievers were folded
into the "prior" history of games actually played before it: a small look-ahead. Passing
`dates` (game_pk -> ISO date, from the schedule) orders by date and only lets a game see
games from strictly earlier DAYS. Off by default so earlier results reproduce.

BULLPEN FATIGUE (opt-in). Whether a reliever is available is public before first pitch:
it follows from who pitched, and how much, on the previous days. With `fatigue=True` a
reliever is moved to the back of the pool if, before this game's date, he
  (a) pitched on each of the two previous days, or
  (b) threw >= 30 pitches the previous day, or
  (c) threw >= 50 pitches over the previous three days.
These are the common public availability heuristics, not tuned. Requires `dates`.
"""
from __future__ import annotations

import numpy as np
import polars as pl

MAX_STAFF = 12


def build_prior_cooccurrence(pa: pl.DataFrame) -> dict:
    """Per (game, half): the starter and the relievers, in game_pk order.

    Returns {"order": [(game_pk, half_bin, starter_id, [reliever_id, ...]), ...]}.
    """
    pcol = "pitcher_id" if "pitcher_id" in pa.columns else "pitcher_idx"
    df = pa.select(["game_pk", "half_bin", pcol, "at_bat_number"]).sort(
        ["game_pk", "half_bin", "at_bat_number"]
    )
    order = []
    for key, g in df.group_by(["game_pk", "half_bin"], maintain_order=True):
        gp, hb = int(key[0]), int(key[1])
        seen, seq = set(), []
        for p in g[pcol].to_list():
            p = int(p)
            if p not in seen:
                seen.add(p)
                seq.append(p)
        if seq:
            order.append((gp, hb, seq[0], seq[1:]))
    order.sort(key=lambda t: (t[0], t[1]))
    return {"order": order}


FATIGUE_YESTERDAY_PITCHES = 30
FATIGUE_THREE_DAY_PITCHES = 50


def _pitches_by_game(pa: pl.DataFrame) -> dict:
    """(game_pk, pitcher_id) -> pitches thrown (max cumulative in-game count)."""
    pcol = "pitcher_id" if "pitcher_id" in pa.columns else "pitcher_idx"
    if "pitch_count_game" not in pa.columns:
        return {}
    g = pa.group_by(["game_pk", pcol]).agg(pl.col("pitch_count_game").max().alias("pc"))
    return {(int(a), int(b)): float(c or 0) for a, b, c in g.iter_rows()}


def _tired(log, day, ordinal):
    """Public availability heuristics from a reliever's log of (day ordinal, pitches)."""
    d1 = sum(p for d, p in log if d == ordinal - 1)
    d2 = any(d == ordinal - 2 for d, _ in log)
    three = sum(p for d, p in log if ordinal - 3 <= d <= ordinal - 1)
    return (d1 > 0 and d2) or d1 >= FATIGUE_YESTERDAY_PITCHES or three >= FATIGUE_THREE_DAY_PITCHES


def pregame_staffs(
    pa: pl.DataFrame, id_to_idx: dict, unknown_idx: int | None = None,
    dates: dict | None = None, fatigue: bool = False, stats: dict | None = None,
) -> dict:
    """(game_pk, half_bin) -> staff of player INDICES, leak-free.

    The staff is [starter] + relievers ranked by how often they have followed this
    starter in strictly earlier games. Counts are accumulated as we sweep forward, so
    a game's pool never sees its own outcome or any later game.

    dates: game_pk -> ISO date. When given, games are ordered by date and history is
    folded in only after each DAY, so same-day games (doubleheaders) never see each other.
    fatigue: needs dates; tired relievers (see module docstring) go to the back of the pool.
    stats: optional dict, filled with counts of how often fatigue changed a staff.
    """
    if fatigue and dates is None:
        raise ValueError("fatigue=True needs dates")
    if unknown_idx is None:
        unknown_idx = max(id_to_idx.values(), default=-1) + 1
    co = build_prior_cooccurrence(pa)
    # starter_id -> {reliever_id: prior appearances behind him}
    hist: dict[int, dict[int, int]] = {}
    # league-wide reliever usage so far, the fallback for an unseen starter
    league: dict[int, int] = {}
    pitches = _pitches_by_game(pa) if fatigue else {}
    usage: dict[int, list] = {}            # pitcher -> [(day ordinal, pitches)]

    order = co["order"]
    if dates is not None:
        from datetime import date as _date
        day = {gp: _date.fromisoformat(dates[gp]).toordinal() if gp in dates else 10**9
               for gp, _, _, _ in order}
        order = sorted(order, key=lambda t: (day[t[0]], t[0], t[1]))
    else:
        day = None

    def select(gp, hb, starter):
        prior = hist.get(starter, {})
        ranked = sorted(prior.items(), key=lambda kv: (-kv[1], kv[0]))
        pool = [p for p, _ in ranked]
        if len(pool) < MAX_STAFF - 1:
            # Early-season, a debut starter, or a thin history: top up from the
            # league's most-used relievers so far. Still strictly prior information.
            lg = sorted(league.items(), key=lambda kv: (-kv[1], kv[0]))
            for p, _ in lg:
                if p != starter and p not in pool:
                    pool.append(p)
                if len(pool) >= MAX_STAFF - 1:
                    break
        if fatigue:
            o = day[gp]
            fresh = [p for p in pool if not _tired(usage.get(p, ()), None, o)]
            tired = [p for p in pool if _tired(usage.get(p, ()), None, o)]
            if stats is not None:
                top = pool[:3]
                stats["halves"] = stats.get("halves", 0) + 1
                stats["top3_tired"] = stats.get("top3_tired", 0) + sum(p in tired for p in top)
                stats["changed"] = stats.get("changed", 0) + int(fresh + tired != pool)
            pool = fresh + tired
        staff_ids = [starter] + pool[: MAX_STAFF - 1]
        return [id_to_idx.get(int(p), unknown_idx) for p in staff_ids]

    def fold(gp, starter, relievers):
        d = hist.setdefault(starter, {})
        for r in relievers:
            d[r] = d.get(r, 0) + 1
            league[r] = league.get(r, 0) + 1
        if fatigue:
            for p in [starter] + list(relievers):
                usage.setdefault(p, []).append((day[gp], pitches.get((gp, p), 15.0)))

    out: dict[tuple[int, int], list[int]] = {}
    if day is None:
        for gp, hb, starter, relievers in order:
            # --- select using ONLY what has accumulated from earlier games ---
            out[(gp, hb)] = select(gp, hb, starter)
            # --- only now fold this game into the history ---
            fold(gp, starter, relievers)
        return out

    i = 0
    while i < len(order):
        j = i
        while j < len(order) and day[order[j][0]] == day[order[i][0]]:
            j += 1
        for gp, hb, starter, relievers in order[i:j]:      # select for the whole day first
            out[(gp, hb)] = select(gp, hb, starter)
        for gp, hb, starter, relievers in order[i:j]:      # then fold the day in
            fold(gp, starter, relievers)
        i = j
    return out


def overlap_with_realized(pa: pl.DataFrame) -> dict:
    """Diagnostic: how much of the realized staff the leak-free pool recovers.

    Reported so the cost of removing the leak is visible rather than assumed. A high
    overlap means the realized order carried little information the pool lacks; a low
    overlap means the previous game-level numbers were leaning on it.
    """
    co = build_prior_cooccurrence(pa)
    ident = {int(p): int(p) for _, _, s, rs in co["order"] for p in [s] + rs}
    staffs = pregame_staffs(pa, ident)

    jac, top1, n_used = [], [], []
    for gp, hb, starter, relievers in co["order"]:
        if not relievers:
            continue
        pred = staffs[(gp, hb)][1:]                 # predicted relievers, in order
        real = relievers
        k = len(real)
        inter = len(set(pred[:k]) & set(real))
        jac.append(inter / k)
        top1.append(1.0 if pred and pred[0] == real[0] else 0.0)
        n_used.append(k)
    return {
        "halves_with_relief": len(jac),
        "mean_recall_at_k": float(np.mean(jac)) if jac else float("nan"),
        "first_reliever_accuracy": float(np.mean(top1)) if top1 else float("nan"),
        "mean_relievers_used": float(np.mean(n_used)) if n_used else float("nan"),
    }
