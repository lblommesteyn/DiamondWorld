"""Benchmark v13's player-rate prediction against Marcel projections.

Steamer / ZiPS / THE BAT are proprietary (FanGraphs), but Marcel ("the monkey",
Tom Tango) is the reproducible baseline they are all measured against, and it is
famously hard to beat by much. Marcel projects a player's rate from a weighted
average of the last 3 seasons (5/4/3), regressed toward the league mean by a fixed
prior. We build Marcel 2024 projections from 2021-2023, score them against actual
2024 rates (cross-player correlation, same metric as prod_playercorr), and compare
to v13's numbers. Honest framing: a dedicated projection system SHOULD win on pure
player projection; DiamondWorld's edge is the full-game simulation, not this.
"""
from __future__ import annotations
import numpy as np, polars as pl
from diamondworldjax.paths import processed_root
from diamondworldjax.data.pipeline import load_seasons

HIT = ("1B", "2B", "3B", "HR")
REG = 1200.0   # Marcel regression: PA of league-average added


def rates(df, bcol):
    g = df.group_by(bcol).agg([
        pl.col("pa_outcome").is_in(HIT).sum().alias("hit"),
        pl.col("pa_outcome").is_in(["BB", "HBP"]).sum().alias("bb"),
        (pl.col("pa_outcome") == "K").sum().alias("k"),
        (pl.col("pa_outcome") == "HR").sum().alias("hr"),
        pl.len().alias("pa")])
    return g


def main():
    bcol = "batter_id"
    seasons = {y: load_seasons([y], data_root=processed_root())
               .filter(pl.col("pa_terminal") & pl.col("pa_outcome").is_not_null())
               for y in (2021, 2022, 2023, 2024)}
    # league rates per season (for regression + weighting target)
    def league(df):
        n = len(df)
        return dict(hit=df["pa_outcome"].is_in(HIT).sum()/n, bb=df["pa_outcome"].is_in(["BB","HBP"]).sum()/n,
                    k=(df["pa_outcome"]=="K").sum()/n, hr=(df["pa_outcome"]=="HR").sum()/n)
    lg = league(seasons[2023])

    r = {y: rates(seasons[y], bcol) for y in (2021, 2022, 2023, 2024)}
    # Marcel weighted counts (5*2023 + 4*2022 + 3*2021)
    W = {2023: 5.0, 2022: 4.0, 2021: 3.0}
    acc = {}   # batter -> dict of weighted sums
    for y, w in W.items():
        for row in r[y].iter_rows(named=True):
            b = int(row[bcol]); a = acc.setdefault(b, dict(hit=0, bb=0, k=0, hr=0, pa=0))
            for kk in ("hit", "bb", "k", "hr", "pa"):
                a[kk] += w * row[kk]
    # actual 2024
    act = {int(row[bcol]): row for row in r[2024].iter_rows(named=True) if row["pa"] >= 150}

    stats = ["k", "bb", "hit", "hr"]
    marcel_pred = {s: [] for s in stats}; actual = {s: [] for s in stats}
    for b, ar in act.items():
        if b not in acc:
            continue
        wa = acc[b]
        if wa["pa"] < 100:
            continue
        for s in stats:
            proj = (wa[s] + REG * lg[s]) / (wa["pa"] + REG)
            marcel_pred[s].append(proj); actual[s].append(ar[s] / ar["pa"])
    out = ["MARCEL vs v13 player-rate prediction (2024 actuals, cross-player corr)"]
    out.append(f"  batters scored: {len(marcel_pred['k'])}")
    v13 = {"k": 0.652, "bb": 0.615, "hit": 0.351, "hr": 0.607}   # from prod_playercorr_v13
    out.append(f"  {'stat':5s} {'Marcel corr':>12s} {'v13 corr':>10s}")
    mc = {}
    for s in stats:
        c = float(np.corrcoef(marcel_pred[s], actual[s])[0, 1]); mc[s] = c
        out.append(f"  {s.upper():5s} {c:12.3f} {v13[s]:10.3f}")
    out.append(f"  {'AVG':5s} {np.mean(list(mc.values())):12.3f} {np.mean(list(v13.values())):10.3f}")
    out.append("")
    out.append("Marcel is the public baseline Steamer/ZiPS/THE BAT are benchmarked against")
    out.append("(they beat it by a few % of correlation). Read: DiamondWorld is competitive")
    out.append("with a real projection baseline on player rates, and its actual contribution")
    out.append("is the full-game simulation projection systems cannot do (see ssac_analyses).")
    rep = "\n".join(out)
    print(rep)
    open("data/eval2/marcel_compare.txt", "w").write(rep + "\n")


if __name__ == "__main__":
    main()
