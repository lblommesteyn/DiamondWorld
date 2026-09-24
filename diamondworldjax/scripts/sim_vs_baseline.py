"""Does the simulator beat a simple rate baseline at predicting the market's within-series moves?

The within-series validation shows the simulator's forecast changes agree with the market's. A
reviewer's first question: would a one-line starter-quality and lineup-quality index do as well?
This answers it on identical games.

Baseline: the two channels of whatif_channels.py (the starters' allowed-run index difference and
the lineups' rate-value difference, from the same training-season player table the simulator
reads), combined by a linear regression FIT TO THE MARKET. That is a supervised baseline that sees
the target, so it is cross-fitted: series are split in two, each half is predicted by weights fit
on the other half. The simulator is not fitted to the market at all.

Reported on identical game-deviations, paired cluster bootstrap by series:
  r(sim, market), r(baseline, market), and their difference;
  whether each adds to the other (partial correlation, both directions).

  python -m diamondworldjax.scripts.sim_vs_baseline --arrays data/eval2/calib_v22L_s42-pregame-leakfree-r500_arrays.npz
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.paths import processed_root
from diamondworldjax.scripts.seq_models import pitcher_rates
from diamondworldjax.scripts.simulator_benchmarks import american_implied, team_rates_2024
from diamondworldjax.scripts.train_pa import _build_park_index, _build_player_table, apply_park_idx
from diamondworldjax.scripts.whatif_channels import allowed_idx, woba_bat
from diamondworldjax.sim.game_extract import extract_games


def _partial(y, x, z):
    """Correlation of y with x after removing z from both."""
    def res(a):
        Z = np.column_stack([np.ones_like(z), z])
        return a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]
    return float(np.corrcoef(res(y), res(x))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", default="data/eval2/calib_v22L_s42-pregame-leakfree-r500_arrays.npz")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--reps", type=int, default=4000)
    args = ap.parse_args()
    tag = args.tag or Path(args.arrays).stem.removeprefix("calib_").removesuffix("_arrays")

    train = load_seasons(list(range(2015, 2024)), data_root=processed_root())
    ptab = _build_player_table(train, recency_halflife=2.0, contact_quality=True)
    park_map = _build_park_index(train)
    pit = pitcher_rates(train.filter(pl.col("pa_terminal")), ptab["id_to_idx"], len(ptab["hand"]))
    stats, unknown = ptab["stats"], ptab["unknown_index"]
    del train
    te = apply_park_idx(load_seasons([2024], data_root=processed_root()).filter(pl.col("pa_terminal")),
                        park_map)
    games = extract_games(te, ptab["id_to_idx"], park_map=park_map, unknown_idx=unknown)

    d = np.load(args.arrays)
    simwp = dict(zip(d["game_pk"].astype(int), (d["sim_home"] > d["sim_away"]).mean(1)))
    _, pkt = team_rates_2024()
    odds = pl.read_csv("data/eval2/odds_2023_2024.csv")
    om = {int(r["game_pk"]): (r["ml_home"], r["ml_away"]) for r in odds.iter_rows(named=True)
          if r["ml_home"] is not None and r["ml_away"] is not None}

    rows = []
    for g in games:
        pk = int(g["game_pk"])
        if pk not in om or pk not in pkt or pk not in simwp or g["park"] == 0:
            continue
        if (not g["home_staff"] or not g["away_staff"]
                or g["home_staff"][0] == unknown or g["away_staff"][0] == unknown):
            continue
        hl = [i for i in g["home_lineup"] if i != unknown]
        al = [i for i in g["away_lineup"] if i != unknown]
        if len(hl) < 8 or len(al) < 8:
            continue
        ih, ia = american_implied(om[pk][0]), american_implied(om[pk][1])
        rows.append((pkt[pk][0], pkt[pk][1], ih / (ih + ia), simwp[pk],
                     allowed_idx(pit[g["home_staff"][0]]) - allowed_idx(pit[g["away_staff"][0]]),
                     np.mean([woba_bat(stats[i]) for i in hl]) - np.mean([woba_bat(stats[i]) for i in al])))

    grp = defaultdict(list)
    for i, r in enumerate(rows):
        grp[(r[0], r[1])].append(i)
    series = []                                   # per series: (market dev, sim dev, X dev)
    for idx in grp.values():
        if len(idx) < 2:
            continue
        a = np.array([rows[i][2:] for i in idx], dtype=float)
        a = a - a.mean(0)
        series.append(a)
    ns = len(series)

    # cross-fitted baseline: weights fit on one half of series, applied to the other
    rng = np.random.default_rng(0)
    order = rng.permutation(ns)
    halves = [order[: ns // 2], order[ns // 2:]]
    base_pred = [None] * ns
    for k in (0, 1):
        fit = np.concatenate([series[s] for s in halves[1 - k]])
        X = fit[:, 2:4]
        w = np.linalg.lstsq(X, fit[:, 0], rcond=None)[0]
        for s in halves[k]:
            base_pred[s] = series[s][:, 2:4] @ w
    M = [s[:, 0] for s in series]
    S = [s[:, 1] for s in series]
    B = base_pred

    def stats_for(pick):
        m = np.concatenate([M[i] for i in pick]); s = np.concatenate([S[i] for i in pick])
        b = np.concatenate([B[i] for i in pick])
        rs, rb = np.corrcoef(s, m)[0, 1], np.corrcoef(b, m)[0, 1]
        both = np.column_stack([s, b])
        yhat = both @ np.linalg.lstsq(both, m, rcond=None)[0]
        return rs, rb, rs - rb, _partial(m, s, b), _partial(m, b, s), np.corrcoef(yhat, m)[0, 1]

    point = stats_for(np.arange(ns))
    boots = np.array([stats_for(rng.integers(0, ns, ns)) for _ in range(args.reps)])
    ci = np.percentile(boots, [2.5, 97.5], axis=0)
    names = ["r(simulator, market)", "r(baseline, market)  [cross-fit]", "simulator - baseline",
             "sim partial | baseline", "baseline partial | sim", "r(sim + baseline, market)"]
    n_dev = sum(len(x) for x in M)
    L = [f"SIMULATOR vs RATE BASELINE on market within-series moves ({tag})",
         f"  {ns} series, {n_dev} game-deviations (games with a known starter and >= 8 known batters)",
         "  baseline = starter allowed-run index diff + lineup rate-value diff, linear, fit to the",
         "  market on the other half of series; simulator is not fit to the market at all", ""]
    for nm, p, lo, hi in zip(names, point, ci[0], ci[1]):
        L.append(f"  {nm:34s} {p:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    rep = "\n".join(L)
    print(rep)
    Path(f"data/eval2/sim_vs_baseline_{tag}.txt").write_text(rep + "\n")


if __name__ == "__main__":
    main()
