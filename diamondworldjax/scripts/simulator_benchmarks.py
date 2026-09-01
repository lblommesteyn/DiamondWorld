"""Benchmark the SIMULATOR's game-level outputs against real baselines.

The pitch is "a generative game model answers distributional questions projection
systems cannot." Until now that rested on the mechanism being obviously true. This
measures it, on two outputs, against baselines a reviewer would demand:

  1. WIN PROBABILITY. Is the simulator's pre-game P(home win) actually good, not
     just calibrated? Compare against Log5 (the canonical talent-only baseline, from
     each team's Pythagorean win rate + home field) and the devigged closing
     moneyline (the market, the gold standard). Metrics: log-loss, Brier, AUC.

  2. RUN-TOTAL DISTRIBUTION. The headline claim is correlated overdispersion a
     summed model misses. Compare the simulator's full predictive distribution of
     total runs against an INDEPENDENT two-Poisson model with the SAME per-game mean
     (so only the shape differs) and a league negative-binomial. Metrics: interval
     coverage (50/80/90%), randomized PIT calibration, tail probabilities vs
     empirical, and the log-score on the actual total.

Consumes a saved arrays file (sim_home/away/total replicas + real outcomes +
game_pk); no GPU. Team rates come from the MLB schedule; market from odds_2023_2024.

  python -m diamondworldjax.scripts.simulator_benchmarks --arrays data/eval2/calib_v13_nobp_arrays.npz --tag v13-pregame
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import polars as pl

CACHE = Path("data/cache")


def american_implied(ml):
    ml = np.asarray(ml, float)
    ml = np.where(np.abs(ml) < 100, np.sign(ml) * 100 + (ml == 0) * 100, ml)  # guard bad odds
    return np.where(ml > 0, 100.0 / (ml + 100.0), -ml / (-ml + 100.0))


def team_rates_2024():
    """Per-team runs scored / allowed / games from the MLB schedule, for Log5."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / "sched_2024.json"
    if not p.exists() or p.stat().st_size < 1000:
        url = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&season=2024&gameType=R"
        p.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    sched = json.loads(p.read_text())
    rs, ra, w, g = {}, {}, {}, {}
    pk_teams = {}
    for d in sched.get("dates", []):
        for gm in d.get("games", []):
            h = gm["teams"]["home"]; a = gm["teams"]["away"]
            if "score" not in h or "score" not in a:
                continue
            hid, aid = h["team"]["id"], a["team"]["id"]
            hs, as_ = h["score"], a["score"]
            pk_teams[int(gm["gamePk"])] = (hid, aid)
            for t, sf, sa, win in ((hid, hs, as_, hs > as_), (aid, as_, hs, as_ > hs)):
                rs[t] = rs.get(t, 0) + sf; ra[t] = ra.get(t, 0) + sa
                w[t] = w.get(t, 0) + int(win); g[t] = g.get(t, 0) + 1
    pyth = {t: (rs[t] ** 1.83) / (rs[t] ** 1.83 + ra[t] ** 1.83) for t in rs if g.get(t, 0) >= 20}
    return pyth, pk_teams


def log5(pa, pb):
    return (pa - pa * pb) / (pa + pb - 2 * pa * pb + 1e-12)


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(p, y):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(len(p))
    npos, nneg = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - npos * (npos - 1) / 2) / (npos * nneg + 1e-9))


def ece(p, y, nb=10):
    e = 0.0
    for a, b in zip(np.linspace(0, 1, nb + 1)[:-1], np.linspace(0, 1, nb + 1)[1:]):
        m = (p >= a) & (p < b) if b < 1 else (p >= a) & (p <= b)
        if m.sum():
            e += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return float(e)


def randomized_pit(samples, y, rng):
    """PIT of an integer observation under an empirical sample distribution, with
    the standard uniform randomization over the probability mass at y so a
    well-specified forecast yields Uniform(0,1)."""
    below = (samples < y[:, None]).mean(1)
    at = (samples == y[:, None]).mean(1)
    return below + rng.random(len(y)) * at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", default="data/eval2/calib_v13_nobp_arrays.npz")
    # No default tag. It used to default to "v13-pregame" independently of --arrays,
    # so running with only --arrays scored the new arrays and then wrote the report
    # over simulator_benchmarks_v13-pregame.txt: the baseline was destroyed, and the
    # surviving file claimed a model it did not contain. Deriving the tag from the
    # arrays filename keeps the output named after its own input.
    ap.add_argument("--tag", default=None)
    ap.add_argument("--season-2024-only", action="store_true", default=True,
                    help="Keep only 2024 game_pks (the test season).")
    args = ap.parse_args()
    if args.tag is None:
        stem = Path(args.arrays).name
        for pre, suf in (("calib_", ""), ("", "_arrays.npz")):
            stem = stem[len(pre):] if pre and stem.startswith(pre) else stem
            stem = stem[:-len(suf)] if suf and stem.endswith(suf) else stem
        args.tag = stem

    d = np.load(args.arrays)
    sh, sa = d["sim_home"], d["sim_away"]
    st = d.get("sim_total", sh + sa)
    rh, ra_, rt = d["real_home"], d["real_away"], d["real_total"]
    pk = d["game_pk"].astype(int)

    pyth, pk_teams = team_rates_2024()
    in2024 = np.array([p in pk_teams for p in pk])
    if args.season_2024_only:
        sh, sa, st, rh, ra_, rt, pk = (x[in2024] for x in (sh, sa, st, rh, ra_, rt, pk))
    n = len(pk)
    y = (rh > ra_).astype(float)                   # home win (drop ties below)
    nz = rh != ra_
    L = [f"SIMULATOR BENCHMARKS ({args.tag}) — {n} games, {int(nz.sum())} decided", ""]

    # ---------------- 1. WIN PROBABILITY ----------------
    sim_wp = (sh > sa).mean(1)
    HFA = np.log(0.521 / 0.479)                    # league home-field logit bump
    log5_wp = np.full(n, np.nan)
    for i, p in enumerate(pk):
        hid, aid = pk_teams[int(p)]
        if hid in pyth and aid in pyth:
            base = log5(pyth[hid], pyth[aid])
            log5_wp[i] = 1 / (1 + np.exp(-(np.log(base / (1 - base + 1e-9) + 1e-12) + HFA)))
    odds = pl.read_csv("data/eval2/odds_2023_2024.csv")
    om = {int(r["game_pk"]): (r["ml_home"], r["ml_away"]) for r in odds.iter_rows(named=True)
          if r["ml_home"] is not None and r["ml_away"] is not None}
    mkt_wp = np.full(n, np.nan)
    for i, p in enumerate(pk):
        if int(p) in om:
            mh, ma = om[int(p)]
            ih, ia = american_implied(mh), american_implied(ma)
            mkt_wp[i] = ih / (ih + ia)             # devig

    L.append("1. WIN PROBABILITY (home), lower log-loss / Brier better, higher AUC better:")
    L.append(f"   {'model':22s} {'n':>6s} {'logloss':>9s} {'brier':>8s} {'AUC':>7s} {'ECE':>7s}")
    base_home = y[nz].mean()
    L.append(f"   {'base rate (home '+f'{base_home:.3f}'+')':22s} {int(nz.sum()):6d} "
             f"{logloss(np.full(nz.sum(), base_home), y[nz]):9.4f} "
             f"{np.mean((base_home - y[nz])**2):8.4f} {'-':>7s} {'-':>7s}")
    for name, wp in (("DiamondWorld sim", sim_wp), ("Log5 (Pythagorean)", log5_wp), ("Market (devig close)", mkt_wp)):
        m = nz & np.isfinite(wp)
        L.append(f"   {name:22s} {int(m.sum()):6d} {logloss(wp[m], y[m]):9.4f} "
                 f"{np.mean((wp[m]-y[m])**2):8.4f} {auc(wp[m], y[m]):7.3f} {ece(wp[m], y[m]):7.3f}")
    # head-to-head on the common set (all three available)
    common = nz & np.isfinite(log5_wp) & np.isfinite(mkt_wp)
    L.append(f"   (common set n={int(common.sum())}: sim logloss {logloss(sim_wp[common],y[common]):.4f}, "
             f"Log5 {logloss(log5_wp[common],y[common]):.4f}, market {logloss(mkt_wp[common],y[common]):.4f})")
    L.append("")

    # ---------------- 2. RUN-TOTAL DISTRIBUTION ----------------
    rng = np.random.default_rng(0)
    R = st.shape[1]
    # independent two-Poisson with the SAME per-game means (only the shape differs)
    lam_h, lam_a = sh.mean(1), sa.mean(1)
    pois_tot = rng.poisson(lam_h[:, None], (n, R)) + rng.poisson(lam_a[:, None], (n, R))
    # league negative-binomial matched to the marginal mean+variance of real totals
    mu, var = rt.mean(), rt.var()
    rr = mu ** 2 / (var - mu) if var > mu else 1e6
    pp = rr / (rr + mu)
    nb_tot = rng.negative_binomial(rr, pp, (n, R))

    L.append("2. RUN-TOTAL DISTRIBUTION. mean/variance and how well each covers reality:")
    L.append(f"   real total: mean {rt.mean():.2f}  var {rt.var():.2f}")
    L.append(f"   {'model':26s} {'mean':>6s} {'var':>7s} {'P(>=10)':>8s} {'P(<=5)':>7s} {'logscore':>9s}")
    L.append(f"   {'real (empirical)':26s} {rt.mean():6.2f} {rt.var():7.2f} "
             f"{(rt>=10).mean():8.3f} {(rt<=5).mean():7.3f} {'-':>9s}")

    def logscore(samples):
        # mean over games of -log P(actual total), P from the empirical sample pmf (Laplace-smoothed)
        ls = 0.0
        for i in range(n):
            p = (samples[i] == rt[i]).mean()
            ls += -np.log((p * R + 1) / (R + 20))
        return ls / n

    def cover(samples, lo, hi):
        ql = np.quantile(samples, lo, axis=1); qh = np.quantile(samples, hi, axis=1)
        return float(((rt >= ql) & (rt <= qh)).mean())

    for name, S in (("DiamondWorld sim", st), ("independent 2-Poisson", pois_tot), ("league neg-binomial", nb_tot)):
        L.append(f"   {name:26s} {S.mean():6.2f} {S.var(1).mean():7.2f} "
                 f"{(S>=10).mean():8.3f} {(S<=5).mean():7.3f} {logscore(S):9.3f}")
    L.append("")
    L.append("   central-interval coverage (should equal the nominal level):")
    L.append(f"   {'model':26s} {'50%':>7s} {'80%':>7s} {'90%':>7s}")
    for name, S in (("DiamondWorld sim", st), ("independent 2-Poisson", pois_tot), ("league neg-binomial", nb_tot)):
        L.append(f"   {name:26s} {cover(S,.25,.75):7.3f} {cover(S,.10,.90):7.3f} {cover(S,.05,.95):7.3f}")
    L.append("")
    # randomized PIT uniformity (KS distance from Uniform(0,1))
    L.append("   randomized-PIT uniformity (KS distance from Uniform, lower = better calibrated):")
    for name, S in (("DiamondWorld sim", st), ("independent 2-Poisson", pois_tot), ("league neg-binomial", nb_tot)):
        u = np.sort(randomized_pit(S, rt, rng))
        ks = float(np.max(np.abs(u - (np.arange(1, n + 1) / n))))
        L.append(f"   {name:26s} KS {ks:.4f}")
    L.append("")
    ratio = st.var(1).mean() / (lam_h + lam_a).mean()
    L.append(f"   overdispersion: sim within-game total variance is {ratio:.2f}x the independent-Poisson")
    L.append(f"   value (var = mean); real is {rt.var()/rt.mean():.2f}x. The summed model cannot produce this.")

    rep = "\n".join(L)
    print(rep)
    Path("data/eval2").mkdir(parents=True, exist_ok=True)
    Path(f"data/eval2/simulator_benchmarks_{args.tag}.txt").write_text(rep + "\n")


if __name__ == "__main__":
    main()
