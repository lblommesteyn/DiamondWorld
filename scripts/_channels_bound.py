"""Two cheap bounds, before committing effort to items 7 and 8.

(1) MiLB translation upside. Translation can only help players whose MLB history
    is thin. The headline metric only scores batters with >= 150 PA in the test
    season, and most of those have thousands of prior MLB PAs, so the reachable
    population bounds the gain no matter how good the translation is.

(2) The missing win-probability channels. Log5 is built from season runs scored
    AND runs allowed, so it already contains team defence, bullpen quality and
    baserunning. The simulator contains none of them. If adding those two
    aggregates to the simulator's win probability reaches Log5, then the missing
    channels ARE those aggregates and item 8 is a well-defined modelling task. If
    it overshoots or undershoots badly, the diagnosis is wrong.
"""
import json
from pathlib import Path

import numpy as np
import polars as pl

from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.paths import processed_root
from diamondworldjax.scripts.projection_levers import MIN_PA, _load, counts
from diamondworldjax.scripts.simulator_benchmarks import (
    team_rates_2024, log5, logloss, auc, ece, american_implied)
from diamondworldjax.scripts.wp_blend import logit, sigmoid, fit_logistic, predict_logistic

print("=" * 70)
print("(1) MiLB translation upside bound")
print("=" * 70)
hist = {}
for y in range(2015, 2024):
    c = counts(_load([y])[y])
    for r in c.iter_rows(named=True):
        hist[int(r["batter_id"])] = hist.get(int(r["batter_id"]), 0) + int(r["pa"])
c24 = counts(_load([2024])[2024]).filter(pl.col("pa") >= MIN_PA)
cohort = [int(r["batter_id"]) for r in c24.iter_rows(named=True)]
prior = np.array([hist.get(b, 0) for b in cohort])
print("scored cohort: " + str(len(cohort)) + " batters with >= " + str(MIN_PA) + " PA in 2024")
for thr in (0, 150, 300, 500, 1000, 2000):
    k = int((prior <= thr).sum())
    print("  prior MLB PA <= " + str(thr).rjust(4) + ":  " + str(k).rjust(3)
          + " batters (" + format(100.0 * k / len(cohort), "4.1f") + "%)")
print()
thin = prior <= 500
bound = 0.10 * thin.mean()
print("Even a PERFECT projection for every batter with <= 500 prior MLB PA touches "
      + format(100.0 * thin.mean(), ".1f") + "% of the cohort.")
print("Correlation is computed over the whole cohort, so a generous +0.10 improvement")
print("confined to that slice bounds the pooled gain at roughly "
      + format(bound, ".4f") + ".")
print("That bound is ABOVE the ~0.021 pooled MDE and comparable to the 0.017-0.020")
print("remaining gap to Steamer, so MiLB translation is NOT ruled out on this metric.")
print("But treat the bound as loose and optimistic: it assumes a +0.10 lift on a")
print("slice where we currently have no minor-league data at all, and correlation")
print("does not decompose additively across sub-populations. Realistic gain is well")
print("below the bound. Verdict: plausible but unquantified, and it needs a MiLB data")
print("fetch before it can be measured, so it ranks behind the levers already measured.")
print()

print("=" * 70)
print("(2) Which channels does Log5 have that the simulator lacks?")
print("=" * 70)
d = np.load("data/eval2/calib_v16-pregame-leakfree_arrays.npz")
sh, sa = d["sim_home"], d["sim_away"]
rh, ra_ = d["real_home"], d["real_away"]
pk = d["game_pk"].astype(int)

# Season team run-scored / run-allowed rates, the information Pythagorean encodes.
sched = json.loads(Path("data/cache/sched_2024.json").read_text())
rs, ra, g = {}, {}, {}
pk_teams = {}
for day in sched.get("dates", []):
    for gm in day.get("games", []):
        h, a = gm["teams"]["home"], gm["teams"]["away"]
        if "score" not in h or "score" not in a:
            continue
        hid, aid = h["team"]["id"], a["team"]["id"]
        pk_teams[int(gm["gamePk"])] = (hid, aid)
        for t, sf, sa_ in ((hid, h["score"], a["score"]), (aid, a["score"], h["score"])):
            rs[t] = rs.get(t, 0) + sf
            ra[t] = ra.get(t, 0) + sa_
            g[t] = g.get(t, 0) + 1
rs_pg = {t: rs[t] / g[t] for t in g if g[t] >= 20}
ra_pg = {t: ra[t] / g[t] for t in g if g[t] >= 20}
pyth = {t: (rs[t] ** 1.83) / (rs[t] ** 1.83 + ra[t] ** 1.83) for t in rs_pg}

keep = np.array([p in pk_teams for p in pk])
sh, sa, rh, ra_, pk = (x[keep] for x in (sh, sa, rh, ra_, pk))
y = (rh > ra_).astype(float)
nz = rh != ra_
sim_wp = (sh > sa).mean(1)
HFA = np.log(0.521 / 0.479)

n = len(pk)
log5_wp = np.full(n, np.nan)
d_off = np.full(n, np.nan)   # home minus away runs scored per game
d_def = np.full(n, np.nan)   # away minus home runs allowed per game
for i, p in enumerate(pk):
    hid, aid = pk_teams[int(p)]
    if hid in pyth and aid in pyth:
        base = log5(pyth[hid], pyth[aid])
        log5_wp[i] = sigmoid(np.log(base / (1 - base + 1e-9) + 1e-12) + HFA)
        d_off[i] = rs_pg[hid] - rs_pg[aid]
        d_def[i] = ra_pg[aid] - ra_pg[hid]

m = nz & np.isfinite(log5_wp)
idx = np.flatnonzero(m)
rng = np.random.default_rng(0)
perm = rng.permutation(len(idx))
tr, te = idx[perm[: len(idx) // 2]], idx[perm[len(idx) // 2:]]
zs = logit(sim_wp)

specs = [
    ("sim alone", lambda i: zs[i, None]),
    ("Log5 alone", lambda i: logit(log5_wp)[i, None]),
    ("sim + offence diff", lambda i: np.column_stack([zs[i], d_off[i]])),
    ("sim + defence diff", lambda i: np.column_stack([zs[i], d_def[i]])),
    ("sim + offence + defence", lambda i: np.column_stack([zs[i], d_off[i], d_def[i]])),
    ("offence + defence only", lambda i: np.column_stack([d_off[i], d_def[i]])),
]
print("half-sample fit, scored on the held-out half (" + str(len(te)) + " games):")
print()
print("  " + "model".ljust(26) + " logloss     AUC")
for name, f in specs:
    w = fit_logistic(f(tr), y[tr])
    p = predict_logistic(w, f(te))
    print("  " + name.ljust(26) + format(logloss(p, y[te]), ".4f") + "  " + format(auc(p, y[te]), ".3f"))
print()
print("Reading it: if 'sim + offence + defence' reaches 'Log5 alone', the simulator's")
print("win-probability deficit IS the two season-level team aggregates it never sees,")
print("and giving the model team defence and bullpen quality is the right fix. If")
print("'offence + defence only' already matches Log5 while the sim adds nothing on")
print("top, then the simulator contributes no independent signal at all and the game")
print("axis is not worth further work.")
