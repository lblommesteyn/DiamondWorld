"""Reliever what-if for one game: how much does a bullpen plan change win probability?

Simulates the game twice from first pitch with common random numbers: once with the default
leak-free bullpen, once with a plan that brings named relievers in at named innings (see the
relief-plan hook in simulate_games). Replica r of both runs sees the same randomness, so the
difference isolates the plan. Reports win probability for the team whose bullpen changes,
the paired difference with a 95% CI over replicas, and who pitched when.

Pitchers are MLBAM ids; anyone absent from the model's table is refused rather than silently
mapped to the unknown placeholder.

  python -m diamondworldjax.scripts.reliever_scenario --ckpt <ckpt> --season 2024 \
      --game 745000 --side home --plan 8:621242,9:661403 --r 4000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from diamondworldjax.scripts.scenario_sim import Sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--game", type=int, required=True, help="game_pk")
    ap.add_argument("--side", choices=["home", "away"], required=True,
                    help="team whose bullpen the plan changes")
    ap.add_argument("--plan", required=True, help="inning:mlbam_id[,inning:mlbam_id...]")
    ap.add_argument("--r", type=int, default=4000)
    ap.add_argument("--fatigue", action="store_true", help="fatigue-aware default bullpen")
    args = ap.parse_args()

    s = Sim(ckpt=args.ckpt, hook_model=True)
    games = {int(g["game_pk"]): g for g in s.real_games(args.season, limit=100000, pregame_staff=True,
                                                         staff_by_date=True,
                                                         bullpen_fatigue=args.fatigue)}
    if args.game not in games:
        raise SystemExit(f"game {args.game} not in {args.season}")
    g = games[args.game]
    plan = []
    for item in args.plan.split(","):
        inn, pid = item.split(":")
        if int(pid) not in s.id2i:
            raise SystemExit(f"pitcher {pid} is not in the model's player table")
        plan.append((int(inn), s.id2i[int(pid)]))
    planned = dict(g, **{f"{args.side}_relief_plan": plan})
    H, A, raw = s.run([g, planned], R=args.r, seed=11, skill_mode="mean", crn=True, return_raw=True)
    own = (H > A) if args.side == "home" else (A > H)          # (2, R) wins for the side
    wp0, wp1 = own[0].mean(), own[1].mean()
    d = own[1].astype(float) - own[0].astype(float)
    se = d.std(ddof=1) / np.sqrt(len(d))
    names = {}
    nf = Path("data/cache/projections/mlb_names.json")
    if nf.exists():
        names = {int(k): v for k, v in json.loads(nf.read_text()).items()}
    i2id = {i: p for p, i in s.id2i.items()}
    print(f"game {args.game} ({args.season}), {args.side} bullpen, R={args.r}, common random numbers")
    print(f"  default bullpen  WP {wp0:.3f}")
    print(f"  plan             WP {wp1:.3f}   change {100 * (wp1 - wp0):+.1f} pts "
          f"(95% CI {100 * (d.mean() - 1.96 * se):+.1f} to {100 * (d.mean() + 1.96 * se):+.1f})")
    for inn, idx in plan:
        pid = i2id.get(idx, idx)
        print(f"  inning {inn}: {names.get(int(pid), pid)}")


if __name__ == "__main__":
    main()
