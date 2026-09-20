"""Item 7: fit an aging curve and test whether it improves the projection.

Steamer's edge over Marcel is usually credited to age curves plus minor-league
translation. The project has neither. This fits the first one from our own data
and measures what it buys, using the same cheap feature-level protocol that
correctly predicted the contact-quality shrinkage result in item 5: if an age
adjustment does not improve a Marcel-style projection, there is no reason to
spend GPU putting age into the model.

Method. Aging is estimated WITHIN player, from consecutive-season pairs where the
same batter cleared a PA threshold in both seasons. Taking the mean rate at each
age across players instead would confound aging with survivorship: bad 34-year-olds
are released, so the surviving ones look like they improved. A within-player delta
is immune to that for the players who stay, and only mildly biased by who drops out.

The curve is fitted on TRAINING seasons only (2015 through --train-end), so the
test season never informs it. The adjustment's strength is then tuned on a holdout
season and applied to the test season.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from diamondworldjax.scripts.projection_levers import (
    STATS, MIN_PA, WEIGHTS, REG_GRID, _load, counts, league_rates,
    contact_quality_table, expected_counts, build_projection, score, actuals,
)

BIRTHDATES = Path("data/cache/birthdates.json")


def load_ages():
    raw = json.loads(BIRTHDATES.read_text())
    out = {}
    for k, v in raw.items():
        bd = v["birthDate"] if isinstance(v, dict) else v
        try:
            y, m, d = (int(x) for x in bd.split("-"))
            out[int(k)] = date(y, m, d)
        except Exception:
            continue
    return out


def age_in(season, bd):
    """Age as of June 30 of the season, the usual baseball convention."""
    ref = date(season, 6, 30)
    return (ref - bd).days / 365.25


def fit_curve(C, ages, seasons, min_pa=300):
    """Mean within-player year-over-year rate delta, by integer age bucket."""
    per_season = {}
    for y in seasons:
        c = C[y].filter(pl.col("pa") >= min_pa)
        per_season[y] = {int(r["batter_id"]): r for r in c.iter_rows(named=True)}

    deltas = {s: {} for s in STATS}
    for y in seasons[:-1]:
        nxt = y + 1
        if nxt not in per_season:
            continue
        for b, r0 in per_season[y].items():
            r1 = per_season[nxt].get(b)
            if r1 is None or b not in ages:
                continue
            a0 = age_in(y, ages[b])
            bucket = int(round(a0))
            for s in STATS:
                d = r1[s] / r1["pa"] - r0[s] / r0["pa"]
                deltas[s].setdefault(bucket, []).append(d)

    curve = {}
    for s in STATS:
        curve[s] = {a: float(np.mean(v)) for a, v in deltas[s].items() if len(v) >= 20}
    counts_by_age = {a: len(v) for a, v in deltas[STATS[0]].items()}
    return curve, counts_by_age


def cumulative(curve_s, a_from, a_to):
    """Summed per-year delta walking from a_from to a_to through the curve."""
    lo, hi = int(round(a_from)), int(round(a_to))
    if hi == lo:
        return 0.0
    step = 1 if hi > lo else -1
    total = 0.0
    for a in range(lo, hi, step):
        total += step * curve_s.get(a, 0.0)
    return total


def age_adjust(proj, curve, ages, hist_years, target, strength):
    """Shift each player's projected rate by the aging expected between the
    PA-weighted mean age of their history window and their age in the target
    season."""
    out = {s: {} for s in STATS}
    wsum = sum(WEIGHTS[i] for i in range(len(hist_years)))
    for s in STATS:
        for b, v in proj[s].items():
            if b not in ages:
                out[s][b] = v
                continue
            a_hist = sum(WEIGHTS[i] * age_in(y, ages[b])
                         for i, y in enumerate(hist_years)) / wsum
            a_tgt = age_in(target, ages[b])
            out[s][b] = v + strength * cumulative(curve[s], a_hist, a_tgt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-end", type=int, default=2023)
    ap.add_argument("--holdout", type=int, default=2023)
    ap.add_argument("--test", type=int, default=2024)
    ap.add_argument("--out", default="data/eval2/age_curve.txt")
    args = ap.parse_args()

    ages = load_ages()
    years = sorted(set(range(2015, args.train_end + 1)) | {args.holdout, args.test})
    print("loading seasons " + str(years) + " ...", flush=True)
    S = _load(years)
    C = {y: counts(S[y]) for y in years}

    fit_seasons = [y for y in range(2015, args.train_end + 1)]
    curve, n_by_age = fit_curve(C, ages, fit_seasons)

    L = []
    L.append("# Item 7: aging curve fitted within player, " + str(fit_seasons[0])
             + "-" + str(fit_seasons[-1]))
    L.append("")
    L.append("Consecutive-season pairs with >= 300 PA in both seasons, delta in rate by age.")
    L.append("Positive = the rate goes UP with another year of age.")
    L.append("")
    L.append("| age | pairs |       K |      BB |     hit |      HR |")
    L.append("|-----|-------|---------|---------|---------|---------|")
    for a in sorted(n_by_age):
        if n_by_age[a] < 20:
            continue
        row = "| " + str(a).rjust(3) + " | " + str(n_by_age[a]).rjust(5) + " |"
        for s in STATS:
            v = curve[s].get(a)
            row += (" " + format(v, "+.5f") if v is not None else "       -") + " |"
        L.append(row)
    L.append("")

    # Tune the adjustment strength on the holdout season, never on the test season.
    lgh = league_rates(S[args.holdout - 1])
    hist_h = [args.holdout - 1, args.holdout - 2, args.holdout - 3]
    Ch = [C[y] for y in hist_h]
    act_h = actuals(C[args.holdout])
    tune_reg = {}
    for s in STATS:
        tune_reg[s] = max((score(build_projection(Ch, lgh, {x: r for x in STATS}),
                                 act_h)[s], r) for r in REG_GRID)[1]
    base_h = build_projection(Ch, lgh, tune_reg)

    strengths = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    L.append("tuning the adjustment strength on " + str(args.holdout)
             + " (regression constants " + str(tune_reg) + "):")
    best_str = 0.0
    best_avg = -9.0
    for st in strengths:
        adj = age_adjust(base_h, curve, ages, hist_h, args.holdout, st)
        sc = score(adj, act_h)
        L.append("  strength " + format(st, ".2f") + "  K " + format(sc["k"], ".4f")
                 + "  BB " + format(sc["bb"], ".4f") + "  hit " + format(sc["hit"], ".4f")
                 + "  HR " + format(sc["hr"], ".4f") + "  AVG " + format(sc["avg"], ".4f"))
        if sc["avg"] > best_avg:
            best_avg, best_str = sc["avg"], st
    L.append("  chosen strength: " + format(best_str, ".2f"))
    L.append("")

    # Apply to the test season, on both the raw-count and contact-quality projections.
    lg = league_rates(S[args.test - 1])
    hist = [args.test - 1, args.test - 2, args.test - 3]
    act = actuals(C[args.test])
    tbl = contact_quality_table([S[y] for y in hist])
    XC = {y: expected_counts(S[y], tbl) for y in hist}

    base = build_projection([C[y] for y in hist], lg, tune_reg)
    mcq = build_projection([XC[y] for y in hist], lg, tune_reg)
    rows = [
        ("Marcel (tuned reg)", score(base, act)),
        ("Marcel + age", score(age_adjust(base, curve, ages, hist, args.test, best_str), act)),
        ("Marcel + CQ", score(mcq, act)),
        ("Marcel + CQ + age", score(age_adjust(mcq, curve, ages, hist, args.test, best_str), act)),
    ]
    L.append("applied to " + str(args.test) + " (" + str(len(act["k"])) + " batters):")
    L.append("")
    L.append("| " + "projection".ljust(22) + " |      K |     BB |    hit |     HR |    AVG |")
    L.append("|" + "-" * 24 + "|" + ("-" * 8 + "|") * 5)
    for name, sc in rows:
        L.append("| " + name.ljust(22) + " | " + " | ".join(
            format(sc[s], "6.3f") for s in STATS) + " | " + format(sc["avg"], "6.3f") + " |")
    L.append("")
    d_raw = rows[1][1]["avg"] - rows[0][1]["avg"]
    d_cq = rows[3][1]["avg"] - rows[2][1]["avg"]
    L.append("age adjustment buys " + format(d_raw, "+.4f") + " on Marcel and "
             + format(d_cq, "+.4f") + " on Marcel+CQ.")
    L.append("Steamer is 0.671 on this metric; Marcel+CQ+age is "
             + format(rows[3][1]["avg"], ".3f") + ".")

    rep = "\n".join(L)
    print(rep)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(rep + "\n")
    print("\nwrote " + args.out)


if __name__ == "__main__":
    main()
