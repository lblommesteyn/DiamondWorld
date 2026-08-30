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


def pregame_staffs(pa: pl.DataFrame, id_to_idx: dict) -> dict:
    """(game_pk, half_bin) -> staff of player INDICES, leak-free.

    The staff is [starter] + relievers ranked by how often they have followed this
    starter in strictly earlier games. Counts are accumulated as we sweep forward, so
    a game's pool never sees its own outcome or any later game.
    """
    co = build_prior_cooccurrence(pa)
    # starter_id -> {reliever_id: prior appearances behind him}
    hist: dict[int, dict[int, int]] = {}
    # league-wide reliever usage so far, the fallback for an unseen starter
    league: dict[int, int] = {}

    out: dict[tuple[int, int], list[int]] = {}
    for gp, hb, starter, relievers in co["order"]:
        # --- select using ONLY what has accumulated from earlier games ---
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
        staff_ids = [starter] + pool[: MAX_STAFF - 1]
        out[(gp, hb)] = [id_to_idx.get(int(p), 0) for p in staff_ids]

        # --- only now fold this game into the history ---
        d = hist.setdefault(starter, {})
        for r in relievers:
            d[r] = d.get(r, 0) + 1
            league[r] = league.get(r, 0) + 1
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
