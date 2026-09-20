"""Item 8: build the team-defence channel and measure its ceiling.

Why this is a different test from the runs-allowed one. Runs allowed conflates
pitching with fielding, and the simulator ALREADY has pitcher identity, so finding
that runs-allowed adds nothing on top of the simulator does not settle whether
FIELDING adds anything. Fielding is the channel the simulator genuinely lacks: the
rules engine converts a batted ball to an out with league-average empirical rates,
with no notion that some defences are better than others.

So this computes a real defensive-efficiency ratio per team and asks whether it buys
win probability the simulator does not already have.

The measurement is deliberately an ORACLE UPPER BOUND, not a usable feature. DER is
computed from the same season being predicted, so it is the best possible estimate
of each team's 2024 defence, better than anything a pre-game model could have. If
even the oracle adds nothing, a learned or prior-season channel certainly cannot.
That makes a null here decisive rather than suggestive.

DER = outs / (singles + doubles + triples + outs + errors) over balls in play,
excluding home runs, which are not fieldable. The fielding team is the home team in
the top half and the away team in the bottom half.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.paths import processed_root
from diamondworldjax.scripts.simulator_benchmarks import (
    team_rates_2024, log5, logloss, auc, ece)
from diamondworldjax.scripts.wp_blend import logit, sigmoid, fit_logistic, predict_logistic

BIP_OUT = ("out",)
BIP_HIT = ("1B", "2B", "3B")
BIP_ERR = ("E",)


def defensive_efficiency(season, pk_teams):
    """DER per team id, from balls in play. Home fields the top half."""
    d = load_seasons([season], data_root=processed_root()).filter(
        pl.col("pa_terminal") & pl.col("pa_outcome").is_not_null())
    keep = set(pk_teams)
    d = d.filter(pl.col("game_pk").is_in(list(keep)))

    home = {pk: t[0] for pk, t in pk_teams.items()}
    away = {pk: t[1] for pk, t in pk_teams.items()}
    d = d.with_columns([
        pl.col("game_pk").replace_strict(home, default=None).alias("_home"),
        pl.col("game_pk").replace_strict(away, default=None).alias("_away"),
    ])
    # half == "top" means the away team is batting, so the HOME team fields.
    d = d.with_columns(
        pl.when(pl.col("half") == "top").then(pl.col("_home"))
          .otherwise(pl.col("_away")).alias("fielding_team"))

    bip = d.filter(pl.col("pa_outcome").is_in(list(BIP_OUT + BIP_HIT + BIP_ERR)))
    g = bip.group_by("fielding_team").agg([
        pl.col("pa_outcome").is_in(list(BIP_OUT)).sum().alias("outs"),
        pl.len().alias("bip"),
    ])
    return {int(r["fielding_team"]): r["outs"] / r["bip"]
            for r in g.iter_rows(named=True) if r["bip"] >= 500}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", default="data/eval2/calib_v16-pregame-leakfree_arrays.npz")
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/eval2/defence_channel.txt")
    args = ap.parse_args()

    pyth, pk_teams = team_rates_2024()
    print("computing defensive efficiency ...", flush=True)
    der = defensive_efficiency(args.season, pk_teams)
    print("  " + str(len(der)) + " teams", flush=True)

    # Season runs scored / allowed per game, for the comparison rows.
    sched = json.loads(Path("data/cache/sched_2024.json").read_text())
    rs, ra, g = {}, {}, {}
    for day in sched.get("dates", []):
        for gm in day.get("games", []):
            h, a = gm["teams"]["home"], gm["teams"]["away"]
            if "score" not in h or "score" not in a:
                continue
            for t, sf, sa_ in ((h["team"]["id"], h["score"], a["score"]),
                               (a["team"]["id"], a["score"], h["score"])):
                rs[t] = rs.get(t, 0) + sf
                ra[t] = ra.get(t, 0) + sa_
                g[t] = g.get(t, 0) + 1
    rs_pg = {t: rs[t] / g[t] for t in g if g[t] >= 20}
    ra_pg = {t: ra[t] / g[t] for t in g if g[t] >= 20}

    d = np.load(args.arrays)
    sh, sa = d["sim_home"], d["sim_away"]
    rh, ra_r = d["real_home"], d["real_away"]
    pk = d["game_pk"].astype(int)
    keep = np.array([p in pk_teams for p in pk])
    sh, sa, rh, ra_r, pk = (x[keep] for x in (sh, sa, rh, ra_r, pk))

    n = len(pk)
    y = (rh > ra_r).astype(float)
    nz = rh != ra_r
    sim_wp = (sh > sa).mean(1)
    HFA = np.log(0.521 / 0.479)

    l5 = np.full(n, np.nan)
    d_der = np.full(n, np.nan)
    d_off = np.full(n, np.nan)
    d_ra = np.full(n, np.nan)
    for i, p in enumerate(pk):
        hid, aid = pk_teams[int(p)]
        if hid in pyth and aid in pyth and hid in der and aid in der:
            b = log5(pyth[hid], pyth[aid])
            l5[i] = sigmoid(np.log(b / (1 - b + 1e-9) + 1e-12) + HFA)
            d_der[i] = der[hid] - der[aid]
            d_off[i] = rs_pg[hid] - rs_pg[aid]
            d_ra[i] = ra_pg[aid] - ra_pg[hid]

    m = nz & np.isfinite(l5) & np.isfinite(d_der)
    idx = np.flatnonzero(m)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(idx))
    tr, te = idx[perm[: len(idx) // 2]], idx[perm[len(idx) // 2:]]
    zs, zl = logit(sim_wp), logit(l5)

    specs = [
        ("sim alone", lambda i: zs[i, None]),
        ("sim + DER (oracle)", lambda i: np.column_stack([zs[i], d_der[i]])),
        ("sim + runs allowed", lambda i: np.column_stack([zs[i], d_ra[i]])),
        ("sim + DER + runs allowed", lambda i: np.column_stack([zs[i], d_der[i], d_ra[i]])),
        ("sim + DER + off + RA", lambda i: np.column_stack([zs[i], d_der[i], d_off[i], d_ra[i]])),
        ("DER alone", lambda i: d_der[i, None]),
        ("Log5 alone", lambda i: zl[i, None]),
    ]

    L = []
    L.append("# Item 8: the team-defence channel, measured as an oracle upper bound")
    L.append("")
    L.append("DER = outs / balls in play (HR excluded), fielding team = home in the top half.")
    L.append("Computed on the SAME season being predicted, so it is a best-case estimate of")
    L.append("team defence and strictly better than any pre-game feature could be.")
    L.append("")
    L.append("DER spread across " + str(len(der)) + " teams: min "
             + format(min(der.values()), ".4f") + "  max " + format(max(der.values()), ".4f")
             + "  sd " + format(float(np.std(list(der.values()))), ".4f"))
    L.append("")
    L.append("half-sample fit, scored on the held-out " + str(len(te)) + " games:")
    L.append("")
    L.append("| " + "model".ljust(26) + " |  logloss |    AUC |")
    L.append("|" + "-" * 28 + "|" + "-" * 10 + "|" + "-" * 8 + "|")
    res = {}
    for name, f in specs:
        w = fit_logistic(f(tr), y[tr])
        p = predict_logistic(w, f(te))
        res[name] = (logloss(p, y[te]), auc(p, y[te]), w)
        L.append("| " + name.ljust(26) + " | " + format(res[name][0], "8.4f")
                 + " | " + format(res[name][1], "6.3f") + " |")
    L.append("")
    base = res["sim alone"][0]
    L.append("coefficient on DER in 'sim + DER': "
             + format(res["sim + DER (oracle)"][2][1], "+.4f"))
    L.append("")
    L.append("deltas against sim alone (negative = better):")
    for name in ("sim + DER (oracle)", "sim + runs allowed", "sim + DER + runs allowed",
                 "sim + DER + off + RA"):
        L.append("  " + name.ljust(26) + format(res[name][0] - base, "+.4f") + " nats")
    L.append("")
    L.append("gap from the best sim-based row to Log5 alone: "
             + format(min(res[k][0] for k in res if k.startswith("sim"))
                      - res["Log5 alone"][0], "+.4f") + " nats")
    L.append("")
    L.append("Reading it. DER is an ORACLE here. If it does not close the gap to Log5 when")
    L.append("handed the answer for the season being predicted, then building a learned or")
    L.append("prior-season defence channel into the model cannot close it either, and the")
    L.append("team-defence route is shut on evidence rather than on assumption.")

    rep = "\n".join(L)
    print(rep)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(rep + "\n")
    print("\nwrote " + args.out)


if __name__ == "__main__":
    main()
