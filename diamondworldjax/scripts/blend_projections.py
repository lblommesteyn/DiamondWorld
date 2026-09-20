"""Item 6: per-stat blend of DiamondWorld with a Marcel+contact-quality projection.

Why this exists. RESULTS.md measures a Marcel-style estimator built on our own
contact-quality feature at AVG 0.652 against DiamondWorld's 0.624-0.655, and the
two systems are strong on DIFFERENT stats: the hierarchical model wins BB and HR,
the weighted average wins hit rate by a wide margin. That is the signature of
decorrelated errors, which is exactly when a blend beats both parents.

Honesty rules, because a blend is the easiest place in this project to fool
yourself:
  * Blend weights are fitted OUT OF SAMPLE over batters (K-fold), and every
    reported correlation is computed on out-of-fold predictions only.
  * A fixed 50/50 blend is reported alongside. It fits nothing, so it cannot
    overfit, and it is the number to quote if the fitted one looks too good.
  * Per-stat Marcel regression constants are tuned on 2023 and never on 2024,
    reusing projection_levers' own tuning loop.
  * Steamer is scored on the SAME batters as everything else, which fixes the
    "honest wrinkle" projection_headtohead documents.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.paths import processed_root
from diamondworldjax.scripts.train_pa import _build_player_table
from diamondworldjax.scripts.projection_levers import (
    STATS, MIN_PA, REG_GRID, _load, counts, league_rates, contact_quality_table,
    expected_counts, build_projection, score, actuals,
)

NPZ_STAT = {"k": "K", "bb": "BB", "hit": "Hit", "hr": "HR"}


def dw_rates(npz_path, id_to_idx):
    """(pred, real, cnt) keyed by MLBAM id, from a prod_rates_<tag>.npz."""
    d = np.load(npz_path, allow_pickle=True)
    cnt = d["cnt"]
    pred = {s: {} for s in STATS}
    real = {s: {} for s in STATS}
    n_pa = {}
    for mlbam, idx in id_to_idx.items():
        # Index 0 is the pooled unknown-player sink; it is not a batter.
        if idx == 0 or idx >= len(cnt) or cnt[idx] < MIN_PA:
            continue
        n_pa[mlbam] = float(cnt[idx])
        for s in STATS:
            k = NPZ_STAT[s]
            pred[s][mlbam] = float(d["sum" + k][idx] / cnt[idx])
            real[s][mlbam] = float(d["r" + k][idx] / cnt[idx])
    return pred, real, n_pa


def steamer_rates(csv_path):
    df = pl.read_csv(csv_path).filter(
        (pl.col("system") == "steamer") & (pl.col("group") == "bat"))
    out = {s: {} for s in STATS}
    for row in df.iter_rows(named=True):
        b = int(row["mlbam_id"])
        for s in STATS:
            v = row[s + "_rate"]
            if v is not None:
                out[s][b] = float(v)
    return out


def _z(v):
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else v - v.mean()


def blend_cv(systems, y, folds, seed):
    """Out-of-fold blended prediction + mean fitted weights.

    Weights come from least squares on z-scored predictions inside the training
    folds only. Correlation is scale invariant, so z-scoring first makes the
    weights comparable across systems with different spreads.
    """
    n = len(y)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    fold_of = np.empty(n, dtype=int)
    for i, idx in enumerate(order):
        fold_of[idx] = i % folds
    X = np.stack([_z(s) for s in systems], axis=1)
    oof = np.empty(n)
    ws = []
    for f in range(folds):
        tr, te = fold_of != f, fold_of == f
        A = np.column_stack([X[tr], np.ones(tr.sum())])
        w = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        ws.append(w[:-1])
        oof[te] = X[te] @ w[:-1] + w[-1]
    return oof, np.mean(ws, axis=0)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", required=True, help="prod_rates_<tag>.npz for the DW model")
    ap.add_argument("--dw-name", default="DiamondWorld")
    ap.add_argument("--projections", default="data/projections_2024.csv")
    ap.add_argument("--train-end", type=int, default=2023)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=20000)
    ap.add_argument("--out", default="data/eval2/blend_marcel_cq.txt")
    args = ap.parse_args()

    print("building player index map (must match the checkpoint's table) ...", flush=True)
    train_seasons = list(range(2015, args.train_end + 1))
    trp = load_seasons(train_seasons, data_root=processed_root())
    ptab = _build_player_table(trp, recency_halflife=2.0)
    id_to_idx = ptab["id_to_idx"]

    print("loading DW rates ...", flush=True)
    dw, real, n_pa = dw_rates(args.rates, id_to_idx)

    print("loading seasons for Marcel + contact quality ...", flush=True)
    years = [2020, 2021, 2022, 2023, 2024]
    S = _load(years)
    C = {y: counts(S[y]) for y in years}
    lg23, lg22 = league_rates(S[2023]), league_rates(S[2022])
    act24, act23 = actuals(C[2024]), actuals(C[2023])

    # Sanity check: the rebuilt index map must reproduce the npz's observed rates.
    # If the map were misaligned this would blow up, and every number below with it.
    checked = 0
    worst = 0.0
    for b in list(real["k"])[:400]:
        if b in act24["k"]:
            worst = max(worst, abs(real["k"][b] - act24["k"][b]))
            checked += 1
    print(f"  index-map check: {checked} batters, worst |real_npz - real_parquet| "
          f"on K = {worst:.4f}", flush=True)
    if checked and worst > 0.02:
        raise SystemExit(
            "index map does not reproduce observed rates; the player table used here "
            "does not match the one the npz was scored with, so mapping is unsafe")

    # Per-stat regression constants tuned on 2023, never on 2024.
    hist_tune = [C[2022], C[2021], C[2020]]
    best = {}
    for s in STATS:
        best[s] = max((score(build_projection(hist_tune, lg22, {x: r for x in STATS}),
                             act23)[s], r) for r in REG_GRID)[1]
    print("  tuned regression: " + str(best), flush=True)

    print("building contact-quality table ...", flush=True)
    tbl = contact_quality_table([S[y] for y in (2021, 2022, 2023)])
    XC = {y: expected_counts(S[y], tbl) for y in (2021, 2022, 2023)}
    mcq = build_projection([XC[2023], XC[2022], XC[2021]], lg23, best)

    steamer = steamer_rates(args.projections)

    # Common batters: every system must cover them, and they must clear MIN_PA
    # in BOTH the npz cohort and the parquet cohort.
    common = sorted(set(dw["k"]) & set(mcq["k"]) & set(steamer["k"]) & set(act24["k"]))
    print("  common batters: " + str(len(common)), flush=True)

    L = []
    L.append("# Item 6: per-stat blend of DiamondWorld with Marcel + contact quality")
    L.append("")
    L.append("DW model: " + args.dw_name + "  (" + Path(args.rates).name + ")")
    L.append("Batters scored: " + str(len(common)) +
             " (covered by DW, Marcel+CQ and Steamer, and >= " + str(MIN_PA) + " PA in 2024)")
    L.append("Marcel regression constants tuned on 2023: " + str(best))
    L.append("Blend weights: " + str(args.folds) + "-fold out-of-fold over BATTERS, seed "
             + str(args.seed) + "; reported correlations use out-of-fold predictions only.")
    L.append("")

    per_stat = {}
    oof_store = {}
    for s in STATS:
        y = np.array([act24[s][b] for b in common])
        dwv = np.array([dw[s][b] for b in common])
        mcv = np.array([mcq[s][b] for b in common])
        stv = np.array([steamer[s][b] for b in common])

        oof2, w2 = blend_cv([dwv, mcv], y, args.folds, args.seed)
        fixed = _z(dwv) * 0.5 + _z(mcv) * 0.5
        oof3, w3 = blend_cv([dwv, mcv, stv], y, args.folds, args.seed)

        per_stat[s] = {
            "dw": corr(dwv, y), "mcq": corr(mcv, y), "steamer": corr(stv, y),
            "blend_cv": corr(oof2, y), "blend_5050": corr(fixed, y),
            "blend3_cv": corr(oof3, y),
            "w_dw": float(w2[0]), "w_mcq": float(w2[1]),
        }
        oof_store[s] = {"y": y, "dw": dwv, "mcq": mcv, "steamer": stv,
                        "blend_cv": oof2, "blend_5050": fixed, "blend3_cv": oof3}

    cols = ["dw", "mcq", "steamer", "blend_5050", "blend_cv", "blend3_cv"]
    names = {"dw": args.dw_name, "mcq": "Marcel+CQ", "steamer": "Steamer",
             "blend_5050": "Blend 50/50 (no fitting)",
             "blend_cv": "Blend fitted (out-of-fold)",
             "blend3_cv": "Blend + Steamer (out-of-fold)"}
    L.append("| " + "system".ljust(30) + " |      K |     BB |    Hit |     HR |    AVG |")
    L.append("|" + "-" * 32 + "|" + ("-" * 8 + "|") * 5)
    avg = {}
    for c in cols:
        vals = [per_stat[s][c] for s in STATS]
        avg[c] = float(np.mean(vals))
        L.append("| " + names[c].ljust(30) + " | "
                 + " | ".join(format(v, "6.3f") for v in vals)
                 + " | " + format(avg[c], "6.3f") + " |")
    L.append("")
    L.append("Fitted blend weights (mean over folds, on z-scored predictions):")
    for s in STATS:
        L.append("  " + s.ljust(4) + "  DW " + format(per_stat[s]["w_dw"], "+.2f")
                 + "   Marcel+CQ " + format(per_stat[s]["w_mcq"], "+.2f"))
    L.append("")

    # Paired bootstrap over batters: blend vs each parent and vs Steamer.
    rng = np.random.default_rng(args.seed)
    n = len(common)
    idx = rng.integers(0, n, size=(args.reps, n))

    def boot_avg(key):
        acc = np.zeros(args.reps)
        for s in STATS:
            y, p = oof_store[s]["y"], oof_store[s][key]
            ys, ps = y[idx], p[idx]
            yc = ys - ys.mean(1, keepdims=True)
            pc = ps - ps.mean(1, keepdims=True)
            acc += (yc * pc).sum(1) / np.sqrt((yc ** 2).sum(1) * (pc ** 2).sum(1))
        return acc / len(STATS)

    L.append("Paired bootstrap over batters (" + str(args.reps) + " reps, 95% CI) on AVG:")
    base_keys = ["dw", "mcq", "steamer"]
    boots = {k: boot_avg(k) for k in set(base_keys + ["blend_5050", "blend_cv", "blend3_cv"])}
    for cand in ["blend_5050", "blend_cv", "blend3_cv"]:
        for base in base_keys:
            d = boots[cand] - boots[base]
            lo, hi = np.percentile(d, [2.5, 97.5])
            p = 2 * min((d <= 0).mean(), (d >= 0).mean())
            star = "*" if (lo > 0 or hi < 0) else " "
            L.append("  " + names[cand].ljust(30) + " - " + names[base].ljust(12) + "  "
                     + format(avg[cand] - avg[base], "+.3f")
                     + " [" + format(lo, "+.3f") + ", " + format(hi, "+.3f") + "]"
                     + " p=" + format(p, ".3f") + " " + star)
    L.append("")
    L.append("`*` marks a difference whose CI excludes zero.")

    rep = "\n".join(L)
    print(rep)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(rep + "\n")
    print("\nwrote " + args.out)


if __name__ == "__main__":
    main()
