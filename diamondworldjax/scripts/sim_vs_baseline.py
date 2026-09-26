"""Does the simulator beat simple rate baselines at predicting the market's within-series moves?

The within-series validation shows the simulator's forecast changes agree with the market's. Two
reviewer questions follow, and this answers both on identical games:

  (a) Would a one-line starter-quality and lineup-quality index do as well?
  (b) The market is partly built from public projections. Does the simulator merely agree with the
      market because it agrees with Steamer?

Baselines, each the same two channels (the starters' allowed-run index difference and the lineups'
rate-value difference, see whatif_channels.py), from different player ratings:
  own      the training-season player table the simulator itself reads;
  steamer  Steamer's 2024 preseason projections (data/projections_2024.csv; the pitcher file is
           dated Jan 2024, the hitter file mid-April 2024, i.e. about two weeks into the season,
           which if anything favours the baseline);
  both     all four channels together.
Each is combined by a linear regression FIT TO THE MARKET, cross-fitted: series are split in two
and each half is predicted by weights fit on the other half. The simulator is fit to nothing.

With --steamer, games are restricted to those where both starters and at least eight batters per
side have a Steamer projection, and every row of the report is on that same set.

  python -m diamondworldjax.scripts.sim_vs_baseline --arrays data/eval2/calib_X_arrays.npz [--steamer]
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
from diamondworldjax.scripts.simulator_benchmarks import ODDS, american_implied, team_rates_2024
from diamondworldjax.scripts.train_pa import _build_park_index, _build_player_table, apply_park_idx
from diamondworldjax.scripts.whatif_channels import allowed_idx, woba_bat
from diamondworldjax.sim.game_extract import extract_games


def _partial(y, x, z):
    """Correlation of y with x after linearly removing z (one or more columns) from both."""
    Z = np.column_stack([np.ones(len(y)), z])

    def res(a):
        return a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]
    return float(np.corrcoef(res(y), res(x))[0, 1])


def steamer_tables(path="data/projections_2024.csv"):
    """mlbam id -> [hit, bb, k, hr] per PA (hitters) / per batter faced (pitchers)."""
    df = pl.read_csv(path).filter(pl.col("system") == "steamer")
    out = {}
    for grp in ("bat", "pit"):
        sub = df.filter(pl.col("group") == grp)
        out[grp] = {int(r["mlbam_id"]): np.array([r["hit_rate"], r["bb_rate"], r["k_rate"], r["hr_rate"]])
                    for r in sub.iter_rows(named=True) if r["mlbam_id"] is not None}
    return out["bat"], out["pit"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", default="data/eval2/calib_v22L_s42-pregame-leakfree-r2000_arrays.npz")
    ap.add_argument("--steamer", action="store_true",
                    help="add Steamer-built baselines and restrict to games Steamer covers")
    ap.add_argument("--season", type=int, default=2024, choices=sorted(ODDS),
                    help="test season; the baseline's ratings use every season before it")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--reps", type=int, default=4000)
    args = ap.parse_args()
    if args.steamer and args.season != 2024:
        ap.error("only 2024 Steamer projections are cached")
    tag = args.tag or Path(args.arrays).stem.removeprefix("calib_").removesuffix("_arrays")
    if args.steamer:
        tag += "_steamer"

    train = load_seasons(list(range(2015, args.season)), data_root=processed_root())
    ptab = _build_player_table(train, recency_halflife=2.0, contact_quality=True)
    park_map = _build_park_index(train)
    pit = pitcher_rates(train.filter(pl.col("pa_terminal")), ptab["id_to_idx"], len(ptab["hand"]))
    stats, unknown = ptab["stats"], ptab["unknown_index"]
    idx2id = {i: pid for pid, i in ptab["id_to_idx"].items()}
    del train
    te = apply_park_idx(load_seasons([args.season], data_root=processed_root())
                        .filter(pl.col("pa_terminal")), park_map)
    games = extract_games(te, ptab["id_to_idx"], park_map=park_map, unknown_idx=unknown)
    st_bat, st_pit = steamer_tables() if args.steamer else ({}, {})

    d = np.load(args.arrays)
    simwp = dict(zip(d["game_pk"].astype(int), (d["sim_home"] > d["sim_away"]).mean(1)))
    _, pkt = team_rates_2024(args.season)
    odds = pl.read_csv(ODDS[args.season])
    om = {int(r["game_pk"]): (r["ml_home"], r["ml_away"]) for r in odds.iter_rows(named=True)
          if r["ml_home"] is not None and r["ml_away"] is not None}

    rows = []           # (home team, away team, market, sim, own_p, own_h[, st_p, st_h])
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
        hs, as_ = g["home_staff"][0], g["away_staff"][0]
        ih, ia = american_implied(om[pk][0]), american_implied(om[pk][1])
        row = [pkt[pk][0], pkt[pk][1], ih / (ih + ia), simwp[pk],
               allowed_idx(pit[hs]) - allowed_idx(pit[as_]),
               np.mean([woba_bat(stats[i]) for i in hl]) - np.mean([woba_bat(stats[i]) for i in al])]
        if args.steamer:
            sp = [st_pit.get(idx2id.get(int(x))) for x in (hs, as_)]
            sh = [st_bat[idx2id[i]] for i in hl if idx2id.get(i) in st_bat]
            sa = [st_bat[idx2id[i]] for i in al if idx2id.get(i) in st_bat]
            if sp[0] is None or sp[1] is None or len(sh) < 8 or len(sa) < 8:
                continue
            row += [allowed_idx(sp[0]) - allowed_idx(sp[1]),
                    np.mean([woba_bat(v) for v in sh]) - np.mean([woba_bat(v) for v in sa])]
        rows.append(row)

    grp = defaultdict(list)
    for i, r in enumerate(rows):
        grp[(r[0], r[1])].append(i)
    series = []                       # per series, demeaned: [market, sim, features...]
    for idx in grp.values():
        if len(idx) < 2:
            continue
        a = np.array([rows[i][2:] for i in idx], dtype=float)
        series.append(a - a.mean(0))
    ns = len(series)

    feats = {"own": [2, 3]}
    if args.steamer:
        feats.update({"steamer": [4, 5], "own+steamer": [2, 3, 4, 5]})

    # cross-fitted baselines: weights fit on one half of series, applied to the other
    rng = np.random.default_rng(0)
    order = rng.permutation(ns)
    halves = [order[: ns // 2], order[ns // 2:]]
    preds = {}
    for name, cols in feats.items():
        pr = [None] * ns
        for k in (0, 1):
            fit = np.concatenate([series[s] for s in halves[1 - k]])
            w = np.linalg.lstsq(fit[:, cols], fit[:, 0], rcond=None)[0]
            for s in halves[k]:
                pr[s] = series[s][:, cols] @ w
        preds[name] = pr
    M = [s[:, 0] for s in series]
    S = [s[:, 1] for s in series]

    def stats_for(pick):
        m = np.concatenate([M[i] for i in pick]); s = np.concatenate([S[i] for i in pick])
        out = [np.corrcoef(s, m)[0, 1]]
        for name in feats:
            b = np.concatenate([preds[name][i] for i in pick])
            rb = np.corrcoef(b, m)[0, 1]
            out += [rb, out[0] - rb, _partial(m, s, b), _partial(m, b, s)]
        return out

    point = stats_for(np.arange(ns))
    boots = np.array([stats_for(rng.integers(0, ns, ns)) for _ in range(args.reps)])
    ci = np.percentile(boots, [2.5, 97.5], axis=0)
    names = ["r(simulator, market)"]
    for name in feats:
        names += [f"r({name} baseline, market)", f"simulator - {name}",
                  f"sim partial | {name}", f"{name} partial | sim"]
    n_dev = sum(len(x) for x in M)
    L = [f"SIMULATOR vs RATE BASELINES on market within-series moves ({tag})",
         f"  {ns} series, {n_dev} game-deviations"
         + (" (games where Steamer covers both starters and >= 8 batters a side)" if args.steamer else
            " (games with a known starter and >= 8 known batters)"),
         "  baselines = starter allowed-run index diff + lineup rate-value diff, linear, fit to the",
         "  market on the other half of series (cross-fit); the simulator is fit to nothing", ""]
    for nm, p, lo, hi in zip(names, point, ci[0], ci[1]):
        L.append(f"  {nm:34s} {p:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    rep = "\n".join(L)
    print(rep)
    Path(f"data/eval2/sim_vs_baseline_{tag}.txt").write_text(rep + "\n")


if __name__ == "__main__":
    main()
