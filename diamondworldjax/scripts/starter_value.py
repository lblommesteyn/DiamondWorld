"""What is a starting pitcher worth per start? The simulator's answer vs the market's.

The within-series validation (validation_stats.py) checks the simulator's game-level
forecast changes against the market's. A front office asks a player-level question
instead: how many win-probability points does this starter add when he takes the ball,
relative to the rest of his team's rotation, and would the market price him the same?
That number is what a trade or a rotation decision turns on.

For every series (same home and away team) each game's forecast is expressed as a
deviation from the series mean, which removes both teams' strength and home field. A
home starter is credited +dev, an away starter -dev, so the credit is always from the
pitcher's own team's point of view. A pitcher's value is the mean credit over his
starts, computed separately from the simulator and from the de-vigged closing market.

Two agreement tests:
  in-sample   corr over pitchers of sim value vs market value on the same starts;
  cross-fit   sim value from one half of a pitcher's starts vs market value from the
              OTHER half, so no game contributes to both sides. This is the test a
              decision needs: does the simulator's valuation predict how the market
              prices the pitcher in starts it has not seen?
The sim is then put in market units with the slope fit on a random half of pitchers
and evaluated frozen on the other half.

  python -m diamondworldjax.scripts.starter_value --arrays data/eval2/calib_ens_post6_arrays.npz
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.paths import processed_root
from diamondworldjax.scripts.simulator_benchmarks import american_implied, team_rates_2024


def starters_2024():
    """game_pk -> {half: starting pitcher id}: the first pitcher of each half."""
    df = load_seasons([2024], data_root=processed_root()).filter(pl.col("pa_terminal"))
    first = (df.sort(["game_pk", "at_bat_number"])
               .group_by(["game_pk", "half"], maintain_order=True)
               .agg(pl.col("pitcher_id").first()))
    out = defaultdict(dict)
    for r in first.iter_rows(named=True):
        out[int(r["game_pk"])][r["half"]] = int(r["pitcher_id"])
    return out


def _corr_ci(x, y, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    rs = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if x[i].std() > 0 and y[i].std() > 0:
            rs.append(np.corrcoef(x[i], y[i])[0, 1])
    return float(np.corrcoef(x, y)[0, 1]), np.percentile(rs, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", default="data/eval2/calib_ens_post6_arrays.npz")
    ap.add_argument("--odds", default="data/eval2/odds_2023_2024.csv")
    ap.add_argument("--min-starts", type=int, default=10)
    ap.add_argument("--top-half", default=None,
                    help="value of the `half` column for the top of an inning (home pitches); "
                         "auto-detected when omitted")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or Path(args.arrays).stem.removeprefix("calib_").removesuffix("_arrays")

    d = np.load(args.arrays)
    sh, sa, pk = d["sim_home"], d["sim_away"], d["game_pk"].astype(int)
    sim = (sh > sa).mean(1)
    _, pkt = team_rates_2024()
    od = pl.read_csv(args.odds)
    om = {int(r["game_pk"]): (r["ml_home"], r["ml_away"]) for r in od.iter_rows(named=True)
          if r["ml_home"] is not None and r["ml_away"] is not None}
    st = starters_2024()
    halves = sorted({h for v in st.values() for h in v})
    top = args.top_half
    if top is None:
        top = next((h for h in halves if str(h).lower() in ("top", "t", "0")), halves[0])
    bot = [h for h in halves if h != top][0]

    rows = []
    for g in range(len(pk)):
        p = int(pk[g])
        if p not in pkt or p not in om or top not in st.get(p, {}) or bot not in st.get(p, {}):
            continue
        ih, ia = american_implied(om[p][0]), american_implied(om[p][1])
        # the home team pitches the top of the inning
        rows.append((p, pkt[p][0], pkt[p][1], st[p][top], st[p][bot], sim[g], ih / (ih + ia)))
    grp = defaultdict(list)
    for r in rows:
        grp[(r[1], r[2])].append(r)
    credit = defaultdict(list)          # pitcher -> [(game_pk, sim_credit, mkt_credit)]
    for ser in grp.values():
        if len(ser) < 2:
            continue
        s = np.array([r[5] for r in ser])
        m = np.array([r[6] for r in ser])
        s, m = s - s.mean(), m - m.mean()
        for r, ds, dm in zip(ser, s, m):
            credit[r[3]].append((r[0], ds, dm))        # home starter
            credit[r[4]].append((r[0], -ds, -dm))      # away starter, own-team view

    names = {}
    nf = Path("data/cache/projections/mlb_names.json")
    if nf.exists():
        names = {int(k): v for k, v in json.loads(nf.read_text()).items()}
    P = [(pid, c) for pid, c in credit.items() if len(c) >= args.min_starts]
    ids = np.array([pid for pid, _ in P])
    sv = np.array([np.mean([x[1] for x in c]) for _, c in P])
    mv = np.array([np.mean([x[2] for x in c]) for _, c in P])
    n_starts = np.array([len(c) for _, c in P])

    # cross-fit: sim from one half of the starts, market from the other, both directions
    rng = np.random.default_rng(0)
    sA, mB, sB, mA = [], [], [], []
    for _, c in P:
        c = np.array([(x[1], x[2]) for x in c])
        idx = rng.permutation(len(c))
        h = len(c) // 2
        a, b = c[idx[:h]], c[idx[h:]]
        sA.append(a[:, 0].mean()); mB.append(b[:, 1].mean())
        sB.append(b[:, 0].mean()); mA.append(a[:, 1].mean())
    sA, mB, sB, mA = map(np.array, (sA, mB, sB, mA))

    # frozen calibration over pitchers
    order = rng.permutation(len(P))
    fit, ev = order[: len(P) // 2], order[len(P) // 2:]
    slope = float(np.polyfit(sv[fit], mv[fit], 1)[0])
    cal_slope, cal_int = (float(v) for v in np.polyfit(slope * sv[ev], mv[ev], 1))
    mae = float(np.abs(mv[ev] - slope * sv[ev]).mean())
    mae0 = float(np.abs(mv[ev]).mean())

    r_in, ci_in = _corr_ci(sv, mv)
    r_x1, ci_x1 = _corr_ci(sA, mB, seed=1)
    r_x2, ci_x2 = _corr_ci(sB, mA, seed=2)
    # the market's own split-half agreement: the ceiling any predictor can reach cross-fit
    r_mm, _ = _corr_ci(mA, mB, seed=3)
    L = [f"STARTER VALUE PER START: simulator vs market ({tag})",
         f"  {len(P)} starters with >= {args.min_starts} starts ({n_starts.sum()} starts); "
         f"credit = within-series WP deviation, own-team view (top half = {top!r})", "",
         f"  in-sample   corr(sim value, market value)   {r_in:.3f}  95% CI [{ci_in[0]:.3f}, {ci_in[1]:.3f}]",
         f"  cross-fit   sim(half A) vs market(half B)    {r_x1:.3f}  95% CI [{ci_x1[0]:.3f}, {ci_x1[1]:.3f}]",
         f"              sim(half B) vs market(half A)    {r_x2:.3f}  95% CI [{ci_x2[0]:.3f}, {ci_x2[1]:.3f}]",
         f"  ceiling     market(half A) vs market(half B) {r_mm:.3f}  (market split-half agreement)",
         "",
         f"  calibration fit on {len(fit)} pitchers: market = {slope:.3f} x sim",
         f"  frozen on the other {len(ev)}: slope {cal_slope:.3f} (want 1), "
         f"intercept {cal_int * 100:+.2f} WP pts (want 0),",
         f"  MAE {mae * 100:.2f} vs {mae0 * 100:.2f} WP pts predict-zero = {(1 - mae / mae0) * 100:.0f}% better",
         "",
         f"  {'pitcher':24s} {'starts':>6s} {'sim (mkt units)':>16s} {'market':>8s}   "
         f"WP points per start vs own rotation"]
    rank = np.argsort(-sv)
    for k in list(rank[:12]) + [None] + list(rank[-8:]):
        if k is None:
            L.append("  ...")
            continue
        L.append(f"  {names.get(int(ids[k]), str(ids[k]))[:24]:24s} {n_starts[k]:6d} "
                 f"{slope * sv[k] * 100:+16.2f} {mv[k] * 100:+8.2f}")
    rep = "\n".join(L)
    print(rep)
    Path(f"data/eval2/starter_value_{tag}.txt").write_text(rep + "\n")
    np.savez(f"data/eval2/starter_value_{tag}.npz", ids=ids, sim=sv, mkt=mv, n=n_starts, slope=slope)


if __name__ == "__main__":
    main()
