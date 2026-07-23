"""Why Steamer beats us, measured lever by lever.

projection_headtohead.py established the gap: Steamer 0.671, Marcel 0.626,
v15 0.594 (avg cross-player rate correlation, 2024). This script asks WHERE the
gap comes from, using only inputs we already have on disk, so each answer is a
number instead of an opinion.

v15's entire player prior is five numbers per batter (recency-weighted K/BB/hit/HR
rates plus a PA count). That is Marcel's information set, minus Marcel's tuning.
Steamer's public description adds three things we do not have:

  L1  per-stat regression. Different rates stabilize at wildly different sample
      sizes (K fast, BABIP slow), so a single regression constant (our REG=1200
      for everything) is wrong for every stat at once. Tuned on a HOLDOUT season
      (project 2023 from 2020-2022), then applied to 2024, so the test set never
      sees the tuning.
  L2  batted-ball quality. A hit rate built from outcomes contains a season of
      BABIP luck. Steamer projects off contact quality instead. We have raw
      launch_speed / launch_angle, which is the same information xBA is built
      from: bin (EV, LA), take the league hit/HR frequency in each bin, and score
      each batter by the quality of contact they made rather than what fell in.
  L3  (not tested here) age curves and minor-league translation for everyone
      rather than rookies only; neither is derivable from the pitch data alone.

  python -m diamondworldjax.scripts.projection_levers
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.paths import processed_root

STATS = ["k", "bb", "hit", "hr"]
HIT = ("1B", "2B", "3B", "HR")
MIN_PA = 150
WEIGHTS = {0: 5.0, 1: 4.0, 2: 3.0}          # most recent, next, next (Marcel's 5/4/3)
REG_GRID = [100, 200, 400, 700, 1000, 1200, 1600, 2200, 3000, 4000, 6000]
STEAMER = {"k": 0.820, "bb": 0.702, "hit": 0.510, "hr": 0.651, "avg": 0.671}
V15 = {"k": 0.741, "bb": 0.645, "hit": 0.411, "hr": 0.580, "avg": 0.594}

EV_BIN, LA_BIN = 2.0, 3.0                   # mph, degrees


def _load(years):
    return {y: load_seasons([y], data_root=processed_root())
            .filter(pl.col("pa_terminal") & pl.col("pa_outcome").is_not_null())
            for y in years}


def _flags(df):
    return df.with_columns([
        pl.col("pa_outcome").is_in(HIT).cast(pl.Float64).alias("f_hit"),
        pl.col("pa_outcome").is_in(["BB", "HBP"]).cast(pl.Float64).alias("f_bb"),
        (pl.col("pa_outcome") == "K").cast(pl.Float64).alias("f_k"),
        (pl.col("pa_outcome") == "HR").cast(pl.Float64).alias("f_hr"),
    ])


def counts(df) -> pl.DataFrame:
    return (_flags(df).group_by("batter_id").agg([
        pl.col("f_hit").sum().alias("hit"), pl.col("f_bb").sum().alias("bb"),
        pl.col("f_k").sum().alias("k"), pl.col("f_hr").sum().alias("hr"),
        pl.len().alias("pa")]))


def league_rates(df) -> dict:
    n = len(df)
    return {s: float(_flags(df)[f"f_{s}"].sum()) / n for s in STATS}


def contact_quality_table(dfs: list) -> pl.DataFrame:
    """League P(hit) and P(HR) for each (exit velo, launch angle) bin.

    This is the xBA idea: a ball hit at 105 mph / 12 degrees is a hit far more
    often than one at 78 mph / 45 degrees, regardless of where the fielders were
    standing that day. Pooling several seasons keeps the bins well populated.
    """
    df = pl.concat([d.select(["launch_speed", "launch_angle", "pa_outcome"]) for d in dfs])
    df = _flags(df.filter(pl.col("launch_speed").is_not_null()
                          & pl.col("launch_angle").is_not_null()))
    return (df.with_columns([
        (pl.col("launch_speed") / EV_BIN).floor().alias("ev_b"),
        (pl.col("launch_angle") / LA_BIN).floor().alias("la_b")])
        .group_by(["ev_b", "la_b"])
        .agg([pl.col("f_hit").mean().alias("p_hit"), pl.col("f_hr").mean().alias("p_hr"),
              pl.len().alias("n")])
        .filter(pl.col("n") >= 25))          # drop bins too thin to trust


def expected_counts(df, tbl: pl.DataFrame) -> pl.DataFrame:
    """Per batter: expected hits/HR from contact quality, K and BB as observed.

    A PA that ends in a strikeout or a walk contributes zero expected hits, so
    summing over all PAs gives an expected hit-per-PA directly comparable to the
    observed one.
    """
    d = (_flags(df).with_columns([
        (pl.col("launch_speed") / EV_BIN).floor().alias("ev_b"),
        (pl.col("launch_angle") / LA_BIN).floor().alias("la_b")])
        .join(tbl.select(["ev_b", "la_b", "p_hit", "p_hr"]), on=["ev_b", "la_b"], how="left"))
    # No launch data (or an unpopulated bin): fall back to what actually happened,
    # so those PAs are neither credited nor penalised by the model.
    d = d.with_columns([
        pl.col("p_hit").fill_null(pl.col("f_hit")).alias("x_hit"),
        pl.col("p_hr").fill_null(pl.col("f_hr")).alias("x_hr")])
    return (d.group_by("batter_id").agg([
        pl.col("x_hit").sum().alias("hit"), pl.col("f_bb").sum().alias("bb"),
        pl.col("f_k").sum().alias("k"), pl.col("x_hr").sum().alias("hr"),
        pl.len().alias("pa")]))


def build_projection(hist: list[pl.DataFrame], lg: dict, reg: dict) -> dict:
    """Marcel: weighted recent counts, regressed toward the league mean."""
    acc: dict[int, dict] = {}
    for i, c in enumerate(hist):
        w = WEIGHTS[i]
        for row in c.iter_rows(named=True):
            a = acc.setdefault(int(row["batter_id"]), dict(k=0.0, bb=0.0, hit=0.0, hr=0.0, pa=0.0))
            for kk in ("k", "bb", "hit", "hr", "pa"):
                a[kk] += w * row[kk]
    return {s: {b: (a[s] + reg[s] * lg[s]) / (a["pa"] + reg[s])
                for b, a in acc.items() if a["pa"] >= 100} for s in STATS}


def score(pred: dict, act: dict) -> dict:
    out = {}
    for s in STATS:
        keys = [b for b in act[s] if b in pred[s]]
        out[s] = float(np.corrcoef([pred[s][b] for b in keys], [act[s][b] for b in keys])[0, 1])
    out["avg"] = float(np.mean([out[s] for s in STATS]))
    return out


def actuals(c: pl.DataFrame) -> dict:
    a = {s: {} for s in STATS}
    for row in c.filter(pl.col("pa") >= MIN_PA).iter_rows(named=True):
        for s in STATS:
            a[s][int(row["batter_id"])] = row[s] / row["pa"]
    return a


def main():
    years = [2019, 2020, 2021, 2022, 2023, 2024]
    print("loading seasons ...", flush=True)
    S = _load(years)
    C = {y: counts(S[y]) for y in years}
    lg23 = league_rates(S[2023])
    lg22 = league_rates(S[2022])
    act24, act23 = actuals(C[2024]), actuals(C[2023])

    # ---- L1: tune regression per stat on a holdout season (2023), never on 2024
    hist_tune = [C[2022], C[2021], C[2020]]
    best = {}
    for s in STATS:
        scores = []
        for reg in REG_GRID:
            p = build_projection(hist_tune, lg22, {x: reg for x in STATS})
            scores.append((score(p, act23)[s], reg))
        best[s] = max(scores)[1]
    print(f"tuned regression (PA of league average added): {best}")

    hist = [C[2023], C[2022], C[2021]]
    base = build_projection(hist, lg23, {s: 1200 for s in STATS})
    l1 = build_projection(hist, lg23, best)

    # ---- L2: rebuild the same projection off contact quality instead of outcomes
    print("building contact-quality table ...", flush=True)
    tbl = contact_quality_table([S[y] for y in (2021, 2022, 2023)])
    XC = {y: expected_counts(S[y], tbl) for y in (2021, 2022, 2023)}
    lg23x = {**lg23}
    l2 = build_projection([XC[2023], XC[2022], XC[2021]], lg23x, best)

    rows = [("Marcel (REG=1200, as shipped)", score(base, act24)),
            ("L1  + per-stat tuned regression", score(l1, act24)),
            ("L2  + contact quality (xBA-style)", score(l2, act24))]

    L = ["CLOSING THE STEAMER GAP: which lever buys what (2024, cross-player rate corr)",
         f"  batters scored: {len(act24['k'])}", ""]
    L.append(f"  {'variant':34s} {'K':>7s} {'BB':>7s} {'HIT':>7s} {'HR':>7s} {'AVG':>7s}")
    for name, sc in rows:
        L.append(f"  {name:34s} {sc['k']:7.3f} {sc['bb']:7.3f} {sc['hit']:7.3f} {sc['hr']:7.3f} {sc['avg']:7.3f}")
    L.append(f"  {'-- Steamer (measured)':34s} {STEAMER['k']:7.3f} {STEAMER['bb']:7.3f} "
             f"{STEAMER['hit']:7.3f} {STEAMER['hr']:7.3f} {STEAMER['avg']:7.3f}")
    L.append(f"  {'-- DiamondWorld v15':34s} {V15['k']:7.3f} {V15['bb']:7.3f} "
             f"{V15['hit']:7.3f} {V15['hr']:7.3f} {V15['avg']:7.3f}")
    L.append("")
    b, top = rows[0][1]["avg"], rows[-1][1]["avg"]
    L.append(f"  levers recover {top - b:+.3f} of the {STEAMER['avg'] - b:+.3f} Marcel-to-Steamer gap "
             f"({(top - b) / max(STEAMER['avg'] - b, 1e-9):.0%})")
    L.append(f"  remaining to Steamer: {STEAMER['avg'] - top:+.3f}")
    rep = "\n".join(L)
    print(rep)
    Path("data/eval2").mkdir(parents=True, exist_ok=True)
    Path("data/eval2/projection_levers.txt").write_text(rep + "\n")


if __name__ == "__main__":
    main()
