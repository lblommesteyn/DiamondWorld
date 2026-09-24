"""Airtight statistics for the within-series validation (addresses reviewer rigor points).

Games within a series and repeated team appearances are not independent, so we report
cluster-bootstrap confidence intervals (resampling whole series), on both the probability
and logit scales, with sign accuracy, R^2, MAE of forecast changes, Pearson and Spearman
correlations, and slope/intercept with intervals. We also separate calibration from
evaluation: the magnitude-calibration slope is fit on one random half of the series and
evaluated, frozen, on the held-out half.

  python -m diamondworldjax.scripts.validation_stats
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from diamondworldjax.scripts.simulator_benchmarks import ODDS, team_rates_2024, american_implied


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _series_pairs(arrays, odds_csv, season=2024):
    d = np.load(arrays)
    sh, sa, pk = d["sim_home"], d["sim_away"], d["game_pk"].astype(int)
    _, pkt = team_rates_2024(season)
    keep = np.array([p in pkt for p in pk]); sh, sa, pk = sh[keep], sa[keep], pk[keep]
    sim = (sh > sa).mean(1)
    od = pl.read_csv(odds_csv)
    om = {int(r["game_pk"]): (r["ml_home"], r["ml_away"]) for r in od.iter_rows(named=True)
          if r["ml_home"] is not None and r["ml_away"] is not None}
    mk = np.array([np.nan if int(p) not in om else american_implied(om[int(p)][0]) /
                   (american_implied(om[int(p)][0]) + american_implied(om[int(p)][1])) for p in pk])
    m = np.isfinite(mk) & np.isfinite(sim)
    sim, mk, pk = sim[m], mk[m], pk[m]
    H = np.array([pkt[int(p)][0] for p in pk]); A = np.array([pkt[int(p)][1] for p in pk])
    grp = defaultdict(list)
    for r in range(len(pk)):
        grp[(int(H[r]), int(A[r]))].append(r)
    series = []                                    # list of (sim_dev, mkt_dev, sim_dev_logit, mkt_dev_logit)
    n_games = 0; teams = set()
    for (h, a), idx in grp.items():
        if len(idx) < 2:
            continue
        idx = np.array(idx); n_games += len(idx); teams |= {h, a}
        sd = sim[idx] - sim[idx].mean(); md = mk[idx] - mk[idx].mean()
        sl = logit(sim[idx]); ml = logit(mk[idx])
        sld = sl - sl.mean(); mld = ml - ml.mean()
        series.append((sd, md, sld, mld))
    return series, n_games, len(teams)


def _flat(series, i, j):
    return np.concatenate([s[i] for s in series]), np.concatenate([s[j] for s in series])


def _stats(sx, my):
    r = float(np.corrcoef(sx, my)[0, 1])
    from scipy.stats import spearmanr
    rho = float(spearmanr(sx, my).statistic)
    b1, b0 = np.polyfit(sx, my, 1)
    yhat = b0 + b1 * sx
    ss_res = float(((my - yhat) ** 2).sum()); ss_tot = float(((my - my.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    sign = float((np.sign(sx) == np.sign(my)).mean())
    mae = float(np.abs(my - yhat).mean())
    return dict(r=r, spearman=rho, slope=float(b1), intercept=float(b0), r2=r2,
               sign_acc=sign, mae=mae)


def _cluster_boot(series, i, j, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(series)
    rs, sl, sg = [], [], []
    for _ in range(B):
        pick = rng.integers(0, n, n)
        sx = np.concatenate([series[k][i] for k in pick]); my = np.concatenate([series[k][j] for k in pick])
        if sx.std() < 1e-9:
            continue
        rs.append(np.corrcoef(sx, my)[0, 1]); sl.append(np.polyfit(sx, my, 1)[0])
        sg.append((np.sign(sx) == np.sign(my)).mean())
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return ci(rs), ci(sl), ci(sg)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", default="data/eval2/calib_v15-pregame-hook-r500_arrays.npz")
    ap.add_argument("--tag", default=None,
                    help="report suffix; the default keeps the original validation_stats.txt")
    ap.add_argument("--season", type=int, default=2024, choices=sorted(ODDS))
    args = ap.parse_args()
    arr = args.arrays
    R = int(np.load(arr)["sim_home"].shape[1])
    series, n_games, n_teams = _series_pairs(arr, ODDS[args.season], args.season)
    n_series = len(series)
    gsz = np.array([len(s[0]) for s in series])

    L = [f"WITHIN-SERIES VALIDATION STATISTICS ({args.season}, R={R}, market forecast changes)",
         f"  arrays: {arr}", ""]
    L.append(f"  {n_games} games in {n_series} series across {n_teams} teams; "
             f"games/series: mean {gsz.mean():.1f}, median {int(np.median(gsz))}, max {gsz.max()}")
    L.append("")
    for scale, (i, j) in (("probability", (0, 1)), ("logit", (2, 3))):
        sx, my = _flat(series, i, j)
        st = _stats(sx, my)
        rci, slci, sgci = _cluster_boot(series, i, j)
        L.append(f"  [{scale} scale]  n_dev={len(sx)}")
        L.append(f"    Pearson r     {st['r']:.3f}  95% CI [{rci[0]:.3f}, {rci[1]:.3f}]  (cluster bootstrap by series)")
        L.append(f"    Spearman rho  {st['spearman']:.3f}")
        L.append(f"    OLS slope     {st['slope']:.3f}  95% CI [{slci[0]:.3f}, {slci[1]:.3f}]   intercept {st['intercept']:+.4f}")
        L.append(f"    R^2           {st['r2']:.3f}")
        L.append(f"    sign accuracy {st['sign_acc']:.3f}  95% CI [{sgci[0]:.3f}, {sgci[1]:.3f}]")
        L.append(f"    MAE of forecast change (residual) {st['mae']:.4f}")
        L.append("")

    # ---- calibration/evaluation split: fit slope on half the series, freeze, eval on the rest ----
    rng = np.random.default_rng(1)
    order = rng.permutation(n_series)
    fit_idx, ev_idx = order[:n_series // 2], order[n_series // 2:]
    fx = np.concatenate([series[k][0] for k in fit_idx]); fy = np.concatenate([series[k][1] for k in fit_idx])
    slope = float(np.polyfit(fx, fy, 1)[0])
    ex = np.concatenate([series[k][0] for k in ev_idx]); ey = np.concatenate([series[k][1] for k in ev_idx])
    pred = slope * ex
    mae_ev = float(np.abs(ey - pred).mean()); mae_null = float(np.abs(ey).mean())
    sign_ev = float((np.sign(ex) == np.sign(ey)).mean())
    r_ev = float(np.corrcoef(ex, ey)[0, 1])
    # held-out magnitude calibration: (a) the held-out half's own OLS slope/intercept, and
    # (b) the frozen-calibration check -- regress held-out market on frozen-calibrated sim, which
    # should have slope ~1 and intercept ~0 if the magnitude calibration transfers out-of-sample.
    slope_ev, int_ev = (float(v) for v in np.polyfit(ex, ey, 1))
    cal_slope, cal_int = (float(v) for v in np.polyfit(pred, ey, 1))
    L.append("  CALIBRATION / EVALUATION SPLIT (fit slope on a random half of series, freeze, eval on rest):")
    L.append(f"    slope fit on {len(fit_idx)} series = {slope:.3f}")
    L.append(f"    held-out {len(ev_idx)} series own OLS: slope {slope_ev:.3f}, intercept {int_ev:+.4f}, "
             f"corr {r_ev:.3f}, sign acc {sign_ev:.3f}")
    L.append(f"    held-out MAE {mae_ev:.4f} vs {mae_null:.4f} (predict-zero) = {(1-mae_ev/mae_null)*100:.0f}% better")
    L.append(f"    frozen-calibration check (market on frozen-calibrated sim): slope {cal_slope:.3f} "
             f"(want ~1), intercept {cal_int:+.4f} (want ~0)")
    rep = "\n".join(L)
    print(rep)
    Path("data/eval2").mkdir(parents=True, exist_ok=True)
    out = f"data/eval2/validation_stats_{args.tag}.txt" if args.tag else "data/eval2/validation_stats.txt"
    Path(out).write_text(rep + "\n")


if __name__ == "__main__":
    main()
