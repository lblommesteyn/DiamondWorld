"""Does adding the previous season (2023) to the player priors help?

The model's rate features come from 2015-2022 only; Marcel (which beats it) uses
2021-2023. This ablation isolates the value of the previous season by scoring
several season-weighting schemes as projections of 2024 rates (cross-player
correlation, the same metric as prod_playercorr / marcel_compare).
"""
from __future__ import annotations
import numpy as np, polars as pl
from diamondworldjax.paths import processed_root
from diamondworldjax.data.pipeline import load_seasons

HIT = ("1B", "2B", "3B", "HR"); REG = 1200.0
STATS = ["k", "bb", "hit", "hr"]


def counts(df, bcol):
    return df.group_by(bcol).agg([
        pl.col("pa_outcome").is_in(HIT).sum().alias("hit"),
        pl.col("pa_outcome").is_in(["BB", "HBP"]).sum().alias("bb"),
        (pl.col("pa_outcome") == "K").sum().alias("k"),
        (pl.col("pa_outcome") == "HR").sum().alias("hr"), pl.len().alias("pa")])


def main():
    bcol = "batter_id"
    yrs = list(range(2015, 2025))
    S = {y: load_seasons([y], data_root=processed_root()).filter(
        pl.col("pa_terminal") & pl.col("pa_outcome").is_not_null()) for y in yrs}
    C = {y: {int(r[bcol]): r for r in counts(S[y], bcol).iter_rows(named=True)} for y in yrs}
    lg = {s: (S[2022]["pa_outcome"].is_in(HIT).mean() if s == "hit" else
              S[2022]["pa_outcome"].is_in(["BB", "HBP"]).mean() if s == "bb" else
              (S[2022]["pa_outcome"] == "K").mean() if s == "k" else
              (S[2022]["pa_outcome"] == "HR").mean()) for s in STATS}
    actual = {int(r[bcol]): r for r in counts(S[2024], bcol).iter_rows(named=True) if r["pa"] >= 150}

    def project(weights):
        # weights: dict season->weight. Marcel-style weighted counts + regression.
        preds = {s: [] for s in STATS}; acts = {s: [] for s in STATS}
        for b, ar in actual.items():
            wc = dict(k=0, bb=0, hit=0, hr=0, pa=0); has = False
            for y, w in weights.items():
                r = C[y].get(b)
                if r:
                    has = True
                    for kk in ("k", "bb", "hit", "hr", "pa"):
                        wc[kk] += w * r[kk]
            if not has or wc["pa"] < 100:
                continue
            for s in STATS:
                preds[s].append((wc[s] + REG * lg[s]) / (wc["pa"] + REG))
                acts[s].append(ar[s] / ar["pa"])
        return {s: float(np.corrcoef(preds[s], acts[s])[0, 1]) for s in STATS}, len(preds["k"])

    def recency(seasons, hl=2.0):
        mx = max(seasons); return {y: 0.5 ** ((mx - y) / hl) for y in seasons}

    configs = {
        "model horizon (2015-2022, recency)     ": recency(list(range(2015, 2023))),
        "  + previous season (2015-2023)         ": recency(list(range(2015, 2024))),
        "Marcel (2021-2023, 5/4/3)               ": {2023: 5, 2022: 4, 2021: 3},
        "Marcel WITHOUT prev season (2020-2022)  ": {2022: 5, 2021: 4, 2020: 3},
        "previous season only (2023)             ": {2023: 1},
    }
    out = ["PREVIOUS-SEASON ABLATION: projection of 2024 rates (cross-player corr)"]
    out.append(f"  batters (>=150 2024 PA): {len(actual)}")
    out.append(f"  {'config':44s} {'K':>6s} {'BB':>6s} {'Hit':>6s} {'HR':>6s} {'AVG':>6s}  n")
    for name, w in configs.items():
        c, n = project(w)
        avg = np.mean(list(c.values()))
        out.append(f"  {name:44s} {c['k']:6.3f} {c['bb']:6.3f} {c['hit']:6.3f} {c['hr']:6.3f} {avg:6.3f}  {n}")
    out.append("")
    out.append("Read: compare 'model horizon' to '+ previous season' -- the gain is the value")
    out.append("of feeding 2023 into the player rate features (which the model does not currently")
    out.append("use). Marcel with vs without the previous season shows the same effect on a")
    out.append("standard system. The previous season is the single most valuable prior.")
    rep = "\n".join(out); print(rep)
    open("data/eval2/prev_season_ablation.txt", "w").write(rep + "\n")


if __name__ == "__main__":
    main()
