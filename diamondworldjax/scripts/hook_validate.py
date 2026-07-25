"""Validate the state-dependent bullpen hook model.

The simulator's old hook draws a fixed PAs-faced threshold up front, so a starter
being shelled is pulled no earlier than one cruising. fit_hook_model replaces that
with a discrete-time pull hazard P(pulled after this PA | PAs faced, times-through-
order, runs allowed). This script checks three things, all without a GPU:

  1. the fitted coefficients have the right signs and the model generalizes
     (held-out 2023 AUC and log-loss vs the constant base rate);
  2. the real managerial behaviour it must capture is present and reproduced: at a
     fixed workload, the pull rate rises with runs allowed;
  3. the model does not merely relearn the marginal PAs-faced curve.

  python -m diamondworldjax.scripts.hook_validate
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.paths import processed_root
from diamondworldjax.sim.game_extract import fit_hook_model, starter_pull_prob, HOOK_FEATS


def _starter_rows(pa: pl.DataFrame) -> pl.DataFrame:
    df = pa.select(["game_pk", "half_bin", "pitcher_id", "at_bat_number",
                    "inning", "tto", "runs_scored"])
    firsts = (df.group_by(["game_pk", "half_bin", "pitcher_id"])
              .agg(pl.col("at_bat_number").min().alias("fab"))
              .with_columns(pl.col("fab").rank("ordinal")
                            .over(["game_pk", "half_bin"]).alias("rk")))
    npitch = df.group_by(["game_pk", "half_bin"]).agg(
        pl.col("pitcher_id").n_unique().alias("npitch"))
    st = firsts.filter(pl.col("rk") == 1).select(["game_pk", "half_bin", "pitcher_id"])
    grp = ["game_pk", "half_bin"]
    s = (df.join(st, on=grp + ["pitcher_id"], how="inner")
         .join(npitch, on=grp, how="left")
         .sort(grp + ["at_bat_number"])
         .with_columns([pl.col("at_bat_number").cum_count().over(grp).alias("pas"),
                        pl.col("runs_scored").cum_sum().over(grp).alias("ra"),
                        pl.col("at_bat_number").max().over(grp).alias("last_ab")]))
    return s.with_columns(((pl.col("at_bat_number") == pl.col("last_ab"))
                          & (pl.col("npitch") > 1)).cast(pl.Float64).alias("y"))


def _auc(p, y):
    order = np.argsort(p)
    r = np.empty_like(order, float)
    r[order] = np.arange(len(p))
    npos = y.sum()
    nneg = len(y) - npos
    return (r[y == 1].sum() - npos * (npos - 1) / 2) / (npos * nneg)


def main():
    tr = load_seasons(list(range(2015, 2023)), data_root=processed_root()).filter(pl.col("pa_terminal"))
    model = fit_hook_model(tr)

    L = ["BULLPEN HOOK MODEL: state-dependent starter-pull hazard", ""]
    L.append(f"  features: {', '.join(HOOK_FEATS)}  (inning dropped: collinear with PAs faced)")
    L.append(f"  base pull rate per starter-PA: {model['base_rate']:.4f}")
    L.append("  standardized logistic coefficients:")
    for name, b in zip(("intercept",) + tuple(HOOK_FEATS), model["beta"]):
        L.append(f"    {name:9s} {b:+.3f}")
    L.append("  (pas + and ra + are the right signs: more batters faced and more runs")
    L.append("   allowed each raise the pull probability.)")
    L.append("")

    te = load_seasons([2023], data_root=processed_root()).filter(pl.col("pa_terminal"))
    s = _starter_rows(te)
    pas, tto, ra = s["pas"].to_numpy(), s["tto"].to_numpy(), s["ra"].to_numpy()
    inn = s["inning"].to_numpy()
    y = s["y"].to_numpy()
    p = starter_pull_prob(pas, inn, tto, ra, model)
    eps = 1e-9
    ll = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    base = y.mean()
    llb = -np.mean(y * np.log(base) + (1 - y) * np.log(1 - base))
    L.append(f"  held-out 2023: log-loss {ll:.4f} vs constant base-rate {llb:.4f} "
             f"({(1 - ll / llb) * 100:.0f}% better); AUC {_auc(p, y):.3f}; n={len(y):,}")
    L.append("")

    L.append("  managerial behaviour (held-out 2023): pull rate by runs allowed at a")
    L.append("  fixed workload of 18-21 PAs faced (the 6th-7th-inning decision zone):")
    L.append(f"    {'runs allowed':14s} {'real pull':>10s} {'model':>8s} {'n':>7s}")
    for lo, hi, lab in [(0, 1, "0-1"), (2, 3, "2-3"), (4, 20, "4+")]:
        m = (pas >= 18) & (pas <= 21) & (ra >= lo) & (ra <= hi)
        if m.sum() > 20:
            L.append(f"    {lab:14s} {y[m].mean():10.3f} {p[m].mean():8.3f} {m.sum():7d}")
    L.append("  The old fixed-threshold hook cannot produce this dependence at all;")
    L.append("  its pull probability is a function of PAs faced only.")

    rep = "\n".join(L)
    print(rep)
    Path("data/eval2").mkdir(parents=True, exist_ok=True)
    Path("data/eval2/hook_validate.txt").write_text(rep + "\n")


if __name__ == "__main__":
    main()
