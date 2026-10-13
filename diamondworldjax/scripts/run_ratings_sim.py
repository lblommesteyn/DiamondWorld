"""Season simulation with hitters' rate inputs replaced by an external rating system.

Our player layer trails Steamer (0.648 vs 0.671 cross-player correlation), and a Steamer-built
starter+lineup index ties the simulator at predicting market moves. If the gap is the ratings,
feeding better hitter ratings into the same simulator should raise its market agreement. This
tests that without retraining: the model's batting-rate input columns 0..3 (hit, BB+HBP, K, HR
per PA) are overwritten, for every hitter the source covers, with
  marcel_cq  Marcel (5/4/3 weights over the three seasons before the test season, per-stat
             regression tuned on an earlier season) with hit and HR from contact quality;
  steamer    Steamer's preseason projection (2024 only).
Everything else (PA-count column, contact-quality columns, pitchers, skill vectors, staff) is
unchanged. Only information available before the season enters.

  python -m diamondworldjax.scripts.run_ratings_sim --ckpt <ckpt> --season 2024 --ratings marcel_cq \
      --tag X [--staff-by-date]
"""
from __future__ import annotations

import argparse

import numpy as np

from diamondworldjax.scripts.projection_levers import (
    _load, build_projection, contact_quality_table, counts, expected_counts, league_rates,
)
from diamondworldjax.scripts.run_inseason_sim import install
from diamondworldjax.scripts.run_pregame_sim import real_runs
from diamondworldjax.scripts.scenario_sim import Sim

COLS = {"hit": 0, "bb": 1, "k": 2, "hr": 3}
REG = {"k": 200, "bb": 400, "hit": 2200, "hr": 2200}   # tuned on 2023 in blend_projections


def marcel_cq(T):
    S = _load([T - 3, T - 2, T - 1])
    hist = [counts(S[y]) for y in (T - 1, T - 2, T - 3)]
    lg = league_rates(S[T - 1])
    marcel = build_projection(hist, lg, REG)
    tbl = contact_quality_table([S[y] for y in (T - 3, T - 2, T - 1)])
    xc = [expected_counts(S[y], tbl) for y in (T - 1, T - 2, T - 3)]
    cq = build_projection(xc, lg, REG)
    out = {s: dict(marcel[s]) for s in COLS}
    for s in ("hit", "hr"):                     # contact quality replaces luck-heavy hit/HR
        out[s].update(cq[s])
    return out


def steamer(T):
    if T != 2024:
        raise SystemExit("only 2024 Steamer projections are cached")
    from diamondworldjax.scripts.blend_projections import steamer_rates
    return steamer_rates("data/projections_2024.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--ratings", choices=["marcel_cq", "steamer"], required=True)
    ap.add_argument("--r", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--staff-by-date", action="store_true")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    T = args.season

    s = Sim(ckpt=args.ckpt, hook_model=True)
    if T <= s.train_end:
        raise SystemExit(f"--season {T} is inside training (train_end={s.train_end})")
    proj = marcel_cq(T) if args.ratings == "marcel_cq" else steamer(T)
    table = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in s.ptab.items()}
    stats = table["stats"]
    before = stats[:, :4].copy()
    covered = 0
    for pid, idx in table["id_to_idx"].items():
        if all(pid in proj[st] for st in COLS):
            for st, c in COLS.items():
                stats[idx, c] = proj[st][pid]
            covered += 1
    moved = np.abs(stats[:, :4] - before).max(axis=1) > 0
    print(f"{args.ratings}: replaced batting rates for {covered} players "
          f"(mean |change| hit {np.abs(stats[moved, 0] - before[moved, 0]).mean():.4f}, "
          f"K {np.abs(stats[moved, 2] - before[moved, 2]).mean():.4f})", flush=True)
    install(s, table, dict(s.params), 0)

    outcomes = real_runs(T)
    games = [g for g in s.real_games(T, limit=100000, pregame_staff=True,
                                     staff_by_date=args.staff_by_date)
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
    print(f"saved -> {out}  ({len(pk)} games, sim mean total {(sh + sa).mean():.2f}, "
          f"real {(rh + ra).mean():.2f})")


if __name__ == "__main__":
    main()
