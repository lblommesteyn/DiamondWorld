"""Davenport/MLE-style minor-league translations, and whether they predict rookie hitters.

Translation factors are estimated the classic way, from players seen at both levels: for every
hitter with a minor-league season t (level L, >= MIN_MILB PA) and an MLB season t+1 (>= MIN_MLB PA),
the factor for stat s at level L is  sum(MLB count t+1) / sum(PA_MLB * rate_L,t), i.e. the
PA-weighted ratio of what they did next year in the majors to what they did this year in the minors.
It folds in a year of development, which is what a projection needs. Only pairs whose MLB season is
before the test season enter the fit, so the test season never informs its own factors.

Test: hitters with at most ROOKIE_MAX_PA career MLB PA before the test season and >= 150 PA in it
(the same cohort rule as the player metric). Their translated prior-season minor-league lines,
shrunk toward the league rate with the same per-stat constants the model uses, are correlated with
their realized test-season rates, next to the no-information baseline (league average for all),
which is effectively what the model sees for them today.

  python -m diamondworldjax.scripts.milb_translation --test-season 2024
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.paths import processed_root

STATS = ["hit", "bb", "k", "hr"]
SHRINK = {"hit": 2200.0, "bb": 400.0, "k": 200.0, "hr": 2200.0}   # model's per-stat constants
MIN_MILB, MIN_MLB, ROOKIE_MAX_PA = 200, 100, 130


def milb_rates(path="data/cache/milb/milb_hitting.csv"):
    m = pl.read_csv(path)
    m = (m.group_by(["player_id", "season", "level"])
          .agg([pl.col(c).sum() for c in ("pa", "h", "bb", "hbp", "so", "hr")]))
    return m.with_columns([(pl.col("h") / pl.col("pa")).alias("hit"),
                           ((pl.col("bb") + pl.col("hbp")) / pl.col("pa")).alias("bb_r"),
                           (pl.col("so") / pl.col("pa")).alias("k"),
                           (pl.col("hr") / pl.col("pa")).alias("hr_r")]).rename(
        {"bb_r": "bb_rate", "hr_r": "hr_rate"})


def mlb_rates(seasons):
    t = (load_seasons(seasons, data_root=processed_root())
         .filter(pl.col("pa_terminal") & pl.col("pa_outcome").is_not_null())
         .select(["batter_id", "season", "pa_outcome"]))
    o = pl.col("pa_outcome")
    return (t.group_by(["batter_id", "season"]).agg([
        pl.len().alias("pa"),
        o.is_in(["1B", "2B", "3B", "HR"]).sum().alias("n_hit"),
        o.is_in(["BB", "HBP"]).sum().alias("n_bb"),
        (o == "K").sum().alias("n_k"),
        (o == "HR").sum().alias("n_hr")]).rename({"batter_id": "player_id"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-season", type=int, default=2024)
    args = ap.parse_args()
    T = args.test_season

    milb = milb_rates()
    mlb = mlb_rates(list(range(2015, T + 1)))
    rate_cols = {"hit": "hit", "bb": "bb_rate", "k": "k", "hr": "hr_rate"}
    count_cols = {"hit": "n_hit", "bb": "n_bb", "k": "n_k", "hr": "n_hr"}

    # --- factors: minor season t -> MLB season t+1, MLB season strictly before T ---
    pairs = (milb.filter(pl.col("pa") >= MIN_MILB).with_columns((pl.col("season") + 1).alias("next"))
             .join(mlb.filter(pl.col("pa") >= MIN_MLB), left_on=["player_id", "next"],
                   right_on=["player_id", "season"], suffix="_mlb")
             .filter(pl.col("next") < T))
    factors = {}
    L = [f"MINOR-LEAGUE TRANSLATION (hitters), factors fit on MLB seasons < {T}", ""]
    for lvl in ("AAA", "AA"):
        p = pairs.filter(pl.col("level") == lvl)
        f = {}
        for s in STATS:
            expect = (p["pa_mlb"] * p[rate_cols[s]]).sum()
            f[s] = float(p[count_cols[s]].sum() / max(expect, 1e-9))
        factors[lvl] = f
        L.append(f"  {lvl}: {len(p)} player-season pairs   " +
                 "  ".join(f"{s} x{f[s]:.2f}" for s in STATS))
    L.append("  (factor = MLB rate next season / minor-league rate; <1 means the stat shrinks in MLB)")

    # --- league rates for shrinkage (MLB, seasons before T) ---
    prev = mlb.filter(pl.col("season") < T)
    league = {s: float(prev[count_cols[s]].sum() / prev["pa"].sum()) for s in STATS}

    # --- test cohort: rookies of season T ---
    career = prev.group_by("player_id").agg(pl.col("pa").sum().alias("career_pa"))
    test = (mlb.filter((pl.col("season") == T) & (pl.col("pa") >= 150))
            .join(career, on="player_id", how="left").with_columns(pl.col("career_pa").fill_null(0))
            .filter(pl.col("career_pa") <= ROOKIE_MAX_PA))
    # prior-season minor lines (T-1, and T-2 at half weight), translated
    mm = milb.filter(pl.col("season").is_in([T - 1, T - 2]))
    rows = []
    for r in test.iter_rows(named=True):
        lines = mm.filter(pl.col("player_id") == r["player_id"])
        pa_eff, acc = 0.0, {s: 0.0 for s in STATS}
        for ln in lines.iter_rows(named=True):
            w = 1.0 if ln["season"] == T - 1 else 0.5
            for s in STATS:
                acc[s] += w * ln["pa"] * ln[rate_cols[s]] * factors[ln["level"]][s]
            pa_eff += w * ln["pa"]
        pred = {s: (acc[s] + SHRINK[s] * league[s]) / (pa_eff + SHRINK[s]) for s in STATS}
        real = {s: r[count_cols[s]] / r["pa"] for s in STATS}
        rows.append((r["player_id"], pa_eff, pred, real))

    covered = [x for x in rows if x[1] > 0]
    L += ["", f"  test season {T}: {len(rows)} rookie hitters (<= {ROOKIE_MAX_PA} career MLB PA before, "
              f">= 150 PA in {T}); {len(covered)} have AAA/AA lines in {T - 2}-{T - 1}", ""]
    L.append(f"  {'stat':5s} {'corr(translated prior, realized)':>34s} {'league-average baseline':>24s}")
    cs = []
    for s in STATS:
        x = np.array([c[2][s] for c in covered]); y = np.array([c[3][s] for c in covered])
        r_ = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else float("nan")
        cs.append(r_)
        L.append(f"  {s:5s} {r_:34.3f} {'0 (no spread)':>24s}")
    L.append(f"  {'AVG':5s} {np.nanmean(cs):34.3f}")
    rep = "\n".join(L)
    print(rep)
    Path(f"data/eval2/milb_translation_{T}.txt").write_text(rep + "\n")
    np.savez(f"data/eval2/milb_factors_{T}.npz", **{f"{l}_{s}": v for l, f in factors.items() for s, v in f.items()})


if __name__ == "__main__":
    main()
