"""Run the simulator pre-game (hook model = realistic, look-ahead-free bullpen) over
all 2024 games and save the full per-game replica distributions, so
simulator_benchmarks.py can score it against Log5, the market, and independent-Poisson.

This is the clean foundation the benchmarks want: the current calibrated model, the
fitted starter-pull hazard instead of the actual bullpen (no leakage), and enough
replicas (R=100) that per-game win probability is not dominated by sampling noise.

Saves data/eval2/calib_<tag>_arrays.npz with the same schema the benchmark reads.

  python -m diamondworldjax.scripts.run_pregame_sim --tag v15-pregame-hook --r 100
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.paths import processed_root
from diamondworldjax.scripts.scenario_sim import Sim


def real_runs(season=2024):
    te = (load_seasons([season], data_root=processed_root()).filter(pl.col("pa_terminal"))
          .group_by(["game_pk", "half_bin"]).agg(pl.col("runs_scored").sum().alias("r")))
    home = {int(r["game_pk"]): r["r"] for r in te.filter(pl.col("half_bin") == 1).iter_rows(named=True)}
    away = {int(r["game_pk"]): r["r"] for r in te.filter(pl.col("half_bin") == 0).iter_rows(named=True)}
    return {g: (home[g], away[g]) for g in home if g in away}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v15-pregame-hook")
    ap.add_argument("--r", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=250)
    args = ap.parse_args()

    s = Sim(hook_model=True)                       # pre-game-legit bullpen via the fitted hazard
    outcomes = real_runs(2024)
    games = [g for g in s.real_games(2024, limit=10000)
             if g["park"] != 0 and int(g["game_pk"]) in outcomes]
    if args.limit:
        games = games[:args.limit]
    pk = np.array([int(g["game_pk"]) for g in games])
    rh = np.array([outcomes[int(g["game_pk"])][0] for g in games], float)
    ra = np.array([outcomes[int(g["game_pk"])][1] for g in games], float)
    print(f"pre-game sim over {len(games)} games x R={args.r} (hook model, crn off)", flush=True)

    Hs, As = [], []
    for i in range(0, len(games), args.chunk):
        H, A = s.run(games[i:i + args.chunk], R=args.r, seed=0, skill_mode="mean", crn=False)
        Hs.append(H); As.append(A)
        print(f"  {min(i + args.chunk, len(games))}/{len(games)}", flush=True)
    sh = np.concatenate(Hs, 0); sa = np.concatenate(As, 0)

    out = f"data/eval2/calib_{args.tag}_arrays.npz"
    np.savez(out, sim_home=sh, sim_away=sa, sim_total=sh + sa,
             real_home=rh, real_away=ra, real_total=rh + ra, game_pk=pk)
    print(f"saved -> {out}  (sim mean total {(sh+sa).mean():.2f}, real {(rh+ra).mean():.2f})")


if __name__ == "__main__":
    main()
