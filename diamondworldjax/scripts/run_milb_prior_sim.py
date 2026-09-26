"""Season simulation where rookie hitters get a translated minor-league prior instead of the blank.

The locked model maps every player absent from its training table to one shared placeholder (a zero
embedding), so a rookie with a strong AAA season is simulated exactly like one with a weak one. This
appends a row for each such hitter who appears in a test-season lineup, filled from his prior-season
AAA/AA lines translated with MLE factors (milb_translation.py; fit on MLB seasons before the test
season) and shrunk toward the league with the model's per-stat constants. Column 4 carries the
effective minor-league PA, as the batting-PA column does for major leaguers. The new rows get a zero
skill vector (the prior mean); nothing else about the model changes, and nothing from the test season
is used except each player's batting side, which is known before the game.

Pitchers are not given priors: the locked model has no pitcher rate features.

  python -m diamondworldjax.scripts.run_milb_prior_sim --ckpt <ckpt> --season 2024 --r 2000 --tag X
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.paths import processed_root
from diamondworldjax.scripts.milb_translation import SHRINK, STATS, milb_rates, mlb_rates
from diamondworldjax.scripts.run_inseason_sim import install
from diamondworldjax.scripts.run_pregame_sim import real_runs
from diamondworldjax.scripts.scenario_sim import Sim

RATE = {"hit": "hit", "bb": "bb_rate", "k": "k", "hr": "hr_rate"}
COUNT = {"hit": "n_hit", "bb": "n_bb", "k": "n_k", "hr": "n_hr"}


def fit_factors(milb, mlb, T, min_milb=200, min_mlb=100):
    pairs = (milb.filter(pl.col("pa") >= min_milb).with_columns((pl.col("season") + 1).alias("next"))
             .join(mlb.filter(pl.col("pa") >= min_mlb), left_on=["player_id", "next"],
                   right_on=["player_id", "season"], suffix="_mlb")
             .filter(pl.col("next") < T))
    out = {}
    for lvl in ("AAA", "AA"):
        p = pairs.filter(pl.col("level") == lvl)
        out[lvl] = {s: float(p[COUNT[s]].sum() / max((p["pa_mlb"] * p[RATE[s]]).sum(), 1e-9))
                    for s in STATS}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--r", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    T = args.season

    s = Sim(ckpt=args.ckpt, hook_model=True)
    if T <= s.train_end:
        raise SystemExit(f"--season {T} is inside training (train_end={s.train_end})")
    base = s.ptab
    milb = milb_rates()
    mlb = mlb_rates(list(range(2015, T + 1)))
    factors = fit_factors(milb, mlb, T)
    prev = mlb.filter(pl.col("season") < T)
    league = {k: float(prev[COUNT[k]].sum() / prev["pa"].sum()) for k in STATS}

    te = load_seasons([T], data_root=processed_root()).filter(pl.col("pa_terminal"))
    known = base["id_to_idx"]
    hitters = (te.filter(~pl.col("batter_id").is_in(list(known)))
                 .group_by("batter_id").agg((pl.col("batter_hand") == "R").mean().alias("r")))
    mm = milb.filter(pl.col("season").is_in([T - 1, T - 2]))
    new_ids, rows, bat_hand = [], [], []
    stats = np.asarray(base["stats"])
    for r in hitters.iter_rows(named=True):
        lines = mm.filter(pl.col("player_id") == r["batter_id"])
        if len(lines) == 0:
            continue
        pa_eff, acc = 0.0, dict.fromkeys(STATS, 0.0)
        for ln in lines.iter_rows(named=True):
            w = 1.0 if ln["season"] == T - 1 else 0.5
            for k in STATS:
                acc[k] += w * ln["pa"] * ln[RATE[k]] * factors[ln["level"]][k]
            pa_eff += w * ln["pa"]
        row = np.zeros(stats.shape[1], dtype=stats.dtype)
        row[:4] = [(acc[k] + SHRINK[k] * league[k]) / (pa_eff + SHRINK[k]) for k in STATS]
        row[4] = pa_eff
        # contact-quality columns: league value, since there is no MLB batted-ball data
        seen = stats[:, 4] > 0
        row[5:7] = np.average(stats[seen, 5:7], axis=0, weights=stats[seen, 4])
        new_ids.append(int(r["batter_id"]))
        rows.append(row)
        bat_hand.append(1.0 if (r["r"] or 0) >= 0.5 else 0.0)

    n = len(new_ids)
    table = {k: np.asarray(base[k]) for k in ("stats", "league", "hand", "bat_hand", "pit_hand")}
    table["stats"] = np.concatenate([table["stats"], np.array(rows).reshape(n, -1)], 0)
    table["league"] = np.concatenate([table["league"], np.zeros(n, table["league"].dtype)])
    bh = np.array(bat_hand, dtype=np.float32)
    table["bat_hand"] = np.concatenate([table["bat_hand"], bh])
    table["pit_hand"] = np.concatenate([table["pit_hand"], np.full(n, 0.5, np.float32)])
    table["hand"] = np.concatenate([table["hand"], (bh >= 0.5).astype(table["hand"].dtype)])
    all_ids = np.concatenate([np.asarray(base["all_ids"], dtype=np.int64), np.array(new_ids, np.int64)])
    table.update(all_ids=all_ids, id_to_idx={int(p): i for i, p in enumerate(all_ids)},
                 unknown_index=len(all_ids))
    install(s, table, dict(s.params), n)
    print(f"factors {factors}; {n} rookie hitters given translated priors", flush=True)

    outcomes = real_runs(T)
    games = [g for g in s.real_games(T, limit=100000, pregame_staff=True)
             if g["park"] != 0 and int(g["game_pk"]) in outcomes]
    if args.limit:
        games = games[:args.limit]
    Hs, As = [], []
    for i in range(0, len(games), args.chunk):
        H, A = s.run(games[i:i + args.chunk], R=args.r, seed=0, skill_mode="mean", crn=False)
        Hs.append(H); As.append(A)
    sh, sa = np.concatenate(Hs, 0), np.concatenate(As, 0)
    pk = np.array([int(g["game_pk"]) for g in games])
    rh = np.array([outcomes[p][0] for p in pk], float)
    ra = np.array([outcomes[p][1] for p in pk], float)
    out = f"data/eval2/calib_{args.tag}_arrays.npz"
    np.savez(out, sim_home=sh, sim_away=sa, sim_total=sh + sa,
             real_home=rh, real_away=ra, real_total=rh + ra, game_pk=pk)
    print(f"saved -> {out}  ({len(pk)} games, sim mean total {(sh + sa).mean():.2f}, real {(rh + ra).mean():.2f})")


if __name__ == "__main__":
    main()
