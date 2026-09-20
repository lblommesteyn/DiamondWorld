"""Item 8 (first half): blend the simulator's win probability with Log5.

RESULTS.md measures the leak-free simulator at win-probability log-loss 0.6985
against a 0.6923 home-field base rate, 0.6709 for Log5 and 0.6707 for the market,
with AUC 0.543 against ~0.615. The documented mechanism is that a per-PA process
washes team-level strength out toward 0.5, while Log5 encodes it directly.

Two things follow, and only the second is actionable without new features:

  * Recalibration cannot help. The simulator is already well calibrated, and AUC
    is invariant to any monotone transform, so sharpening the probabilities
    cannot improve the ranking. This script prints the recalibrated row anyway,
    so that dead end is on the record rather than assumed.

  * The simulator is bottom-up and Log5 is top-down, so their errors should be
    largely decorrelated. A blend is then expected to beat Log5 alone even though
    the simulator loses to it outright.

Blend weights are fitted by logistic regression on half the games and every
reported number comes from the held-out half.

One caveat that must travel with any Log5 comparison: Log5 here uses SAME-SEASON
team Pythagorean rates, which is a mild in-sample peek, so a blend that includes
Log5 inherits it. The clean pre-game comparison is against the market, and that
row is reported too.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from diamondworldjax.scripts.simulator_benchmarks import (
    american_implied, team_rates_2024, log5, logloss, auc, ece,
)


def logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic(X, y, l2=1e-3, iters=100):
    """Newton-Raphson logistic regression with an intercept and light ridge."""
    A = np.column_stack([X, np.ones(len(y))])
    w = np.zeros(A.shape[1])
    for _ in range(iters):
        p = sigmoid(A @ w)
        g = A.T @ (p - y) + l2 * w
        W = p * (1 - p)
        H = A.T @ (A * W[:, None]) + l2 * np.eye(A.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w


def predict_logistic(w, X):
    return sigmoid(np.column_stack([X, np.ones(len(X))]) @ w)


def report(name, p, y):
    return (name, len(y), logloss(p, y), float(np.mean((p - y) ** 2)), auc(p, y), ece(p, y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", default="data/eval2/calib_v16-pregame-leakfree_arrays.npz")
    ap.add_argument("--tag", default="v16-leakfree")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--odds", default="data/eval2/odds_2023_2024.csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = np.load(args.arrays)
    sh, sa = d["sim_home"], d["sim_away"]
    rh, ra_ = d["real_home"], d["real_away"]
    pk = d["game_pk"].astype(int)

    pyth, pk_teams = team_rates_2024()
    keep = np.array([p in pk_teams for p in pk])
    sh, sa, rh, ra_, pk = (x[keep] for x in (sh, sa, rh, ra_, pk))
    n = len(pk)

    y = (rh > ra_).astype(float)
    nz = rh != ra_

    sim_wp = (sh > sa).mean(1)
    HFA = np.log(0.521 / 0.479)
    log5_wp = np.full(n, np.nan)
    for i, p in enumerate(pk):
        hid, aid = pk_teams[int(p)]
        if hid in pyth and aid in pyth:
            base = log5(pyth[hid], pyth[aid])
            log5_wp[i] = sigmoid(np.log(base / (1 - base + 1e-9) + 1e-12) + HFA)

    mkt_wp = np.full(n, np.nan)
    odds = pl.read_csv(args.odds)
    om = {int(r["game_pk"]): (r["ml_home"], r["ml_away"]) for r in odds.iter_rows(named=True)
          if r["ml_home"] is not None and r["ml_away"] is not None}
    for i, p in enumerate(pk):
        if int(p) in om:
            mh, ma = om[int(p)]
            ih, ia = american_implied(mh), american_implied(ma)
            mkt_wp[i] = ih / (ih + ia)

    # Work on decided games where sim, Log5 and the market all exist, so every
    # row in the table is scored on exactly the same games.
    m = nz & np.isfinite(log5_wp) & np.isfinite(mkt_wp)
    idx = np.flatnonzero(m)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(idx))
    tr, te = idx[perm[: len(idx) // 2]], idx[perm[len(idx) // 2:]]

    zs, zl = logit(sim_wp), logit(log5_wp)

    w_recal = fit_logistic(zs[tr, None], y[tr])
    w_blend = fit_logistic(np.column_stack([zs, zl])[tr], y[tr])
    w_l5cal = fit_logistic(zl[tr, None], y[tr])

    base_rate = float(y[tr].mean())
    rows = [
        report("base rate (train)", np.full(len(te), base_rate), y[te]),
        report("DiamondWorld sim", sim_wp[te], y[te]),
        report("sim, recalibrated", predict_logistic(w_recal, zs[te, None]), y[te]),
        report("Log5 (Pythagorean)", log5_wp[te], y[te]),
        report("Log5, recalibrated", predict_logistic(w_l5cal, zl[te, None]), y[te]),
        report("Market (devig close)", mkt_wp[te], y[te]),
        report("BLEND sim + Log5", predict_logistic(w_blend, np.column_stack([zs, zl])[te]), y[te]),
    ]

    L = []
    L.append("# Item 8: win-probability blend of the simulator with Log5")
    L.append("")
    L.append("arrays: " + Path(args.arrays).name)
    L.append("games with sim, Log5 and market all present, decided: " + str(len(idx)))
    L.append("weights fitted on " + str(len(tr)) + " games, scored on the held-out " + str(len(te)) + ".")
    L.append("")
    L.append("| " + "model".ljust(22) + " |     n |  logloss |   brier |    AUC |    ECE |")
    L.append("|" + "-" * 24 + "|" + "-" * 7 + "|" + "-" * 10 + "|" + ("-" * 9 + "|") + ("-" * 8 + "|") * 2)
    for name, nn, ll, br, au, ec in rows:
        au_s = format(au, "6.3f") if np.isfinite(au) else "     -"
        L.append("| " + name.ljust(22) + " | " + format(nn, "5d") + " | " + format(ll, "8.4f")
                 + " | " + format(br, "7.4f") + " | " + au_s + " | " + format(ec, "6.3f") + " |")
    L.append("")
    L.append("blend coefficients (logit scale): sim " + format(w_blend[0], "+.3f")
             + ", Log5 " + format(w_blend[1], "+.3f")
             + ", intercept " + format(w_blend[2], "+.3f"))
    L.append("")

    by = {r[0]: r for r in rows}
    sim_ll, l5_ll = by["DiamondWorld sim"][2], by["Log5 (Pythagorean)"][2]
    bl_ll, mk_ll = by["BLEND sim + Log5"][2], by["Market (devig close)"][2]
    rc_au, sm_au = by["sim, recalibrated"][4], by["DiamondWorld sim"][4]
    L.append("Reading it:")
    L.append("  recalibration moved AUC " + format(sm_au, ".3f") + " -> " + format(rc_au, ".3f")
             + ", i.e. not at all, exactly as the monotone-invariance argument predicts.")
    L.append("  blend vs Log5:   " + format(bl_ll - l5_ll, "+.4f") + " nats")
    L.append("  blend vs market: " + format(bl_ll - mk_ll, "+.4f") + " nats")
    L.append("  blend vs sim:    " + format(bl_ll - sim_ll, "+.4f") + " nats")
    if bl_ll < l5_ll:
        L.append("  The simulator carries win-probability information Log5 does not, even though")
        L.append("  it loses to Log5 outright. That is the decorrelated-errors prediction holding.")
    else:
        L.append("  The blend does not improve on Log5, so the simulator's win-probability signal")
        L.append("  is subsumed by season-level team strength. That closes the blend route.")
    L.append("")
    L.append("Caveat: Log5 uses same-season Pythagorean rates, a mild in-sample peek, so the")
    L.append("blend inherits it. The market row is the clean pre-game reference.")

    rep = "\n".join(L)
    print(rep)
    out = args.out or ("data/eval2/wp_blend_" + args.tag + ".txt")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(rep + "\n")
    print("\nwrote " + out)


if __name__ == "__main__":
    main()
