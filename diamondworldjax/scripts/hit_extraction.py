"""Item 5: how much hit-rate signal does the model leave in its own input?

The measured situation. A Marcel-style weighted average built on OUR contact-quality
feature reaches hit-rate correlation 0.497 on 2024. The full hierarchical model,
fed the same feature as player-table columns 5:7, reaches 0.459-0.469. A regularised
average beats the generative model on the one stat the feature was added for.

This script asks whether that is really an extraction failure, by testing the
strong form of the claim: after conditioning on the model's OWN hit prediction,
does the contact-quality feature still predict 2024 hit rate? If it does, the
model is demonstrably not using an input it already receives, and no amount of
new data is needed to fix it.

It also isolates a concrete mechanism. In `_build_player_table`, `--per-stat-shrink`
shrinks columns 0..3 (raw hit/bb/k/hr rates) with each stat's own measured
stabilisation constant, and THEN contact quality overwrites columns 5:7 with
expected hit/HR rates that are never shrunk at all. So for hit rate the model is
handed a properly regularised noisy estimate and an unregularised better estimate
side by side, and has to infer the trust weighting itself. The v17-v21 series is
eight pieces of evidence that this model does not infer such things.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.paths import processed_root
from diamondworldjax.scripts.train_pa import _build_player_table
from diamondworldjax.scripts.projection_levers import STATS, MIN_PA, _load, counts, actuals

NPZ_STAT = {"k": "K", "bb": "BB", "hit": "Hit", "hr": "HR"}


def ols(X, y):
    """Least squares with intercept; returns (coefs, r2)."""
    A = np.column_stack([X, np.ones(len(y))])
    w = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = y - A @ w
    ss_tot = ((y - y.mean()) ** 2).sum()
    return w[:-1], 1.0 - (resid ** 2).sum() / ss_tot


def partial_corr(a, b, given):
    """corr(a, b) after linearly removing `given` (2-D) from both."""
    G = np.column_stack([given, np.ones(len(a))])
    ra = a - G @ np.linalg.lstsq(G, a, rcond=None)[0]
    rb = b - G @ np.linalg.lstsq(G, b, rcond=None)[0]
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", action="append", required=True,
                    help="tag=path/to/prod_rates_<tag>.npz (repeatable)")
    ap.add_argument("--train-end", type=int, default=2023)
    ap.add_argument("--out", default="data/eval2/hit_extraction.txt")
    args = ap.parse_args()

    print("building player tables (raw and shrunk) ...", flush=True)
    trp = load_seasons(list(range(2015, args.train_end + 1)), data_root=processed_root())
    ptab = _build_player_table(trp, recency_halflife=2.0, contact_quality=True)
    id_to_idx = ptab["id_to_idx"]
    stats = ptab["stats"]

    print("loading 2024 actuals ...", flush=True)
    S = _load([2024])
    act = actuals(counts(S[2024]))

    models = {}
    for spec in args.rates:
        tag, path = spec.split("=", 1)
        d = np.load(path, allow_pickle=True)
        models[tag] = d

    L = []
    L.append("# Item 5: hit-rate extraction loss")
    L.append("")
    L.append("Player table rebuilt from " + str(2015) + "-" + str(args.train_end)
             + ", recency halflife 2, contact_quality on (cols 5:7 = expected hit/HR).")
    L.append("")

    for tag, d in models.items():
        cnt = d["cnt"]
        rows = []
        for mlbam, idx in id_to_idx.items():
            if idx == 0 or idx >= len(cnt) or cnt[idx] < MIN_PA:
                continue
            if mlbam not in act["hit"]:
                continue
            rows.append((mlbam, idx))
        ids = [r[0] for r in rows]
        ix = np.array([r[1] for r in rows])

        y = np.array([act["hit"][b] for b in ids])              # realized 2024 hit rate
        pred = d["sumHit"][ix] / cnt[ix]                        # model's hit prediction
        raw_hit = stats[ix, 0].astype(float)                    # col 0: observed hit rate
        xhit = stats[ix, 5].astype(float)                       # col 5: expected hit rate (CQ)
        pa_w = stats[ix, 4].astype(float)                       # col 4: weighted PA

        L.append("## " + tag + "   (" + str(len(ids)) + " batters)")
        L.append("")
        L.append("  corr with realized 2024 hit rate:")
        L.append("    model prediction        " + format(float(np.corrcoef(pred, y)[0, 1]), ".3f"))
        L.append("    raw hit rate (col 0)    " + format(float(np.corrcoef(raw_hit, y)[0, 1]), ".3f"))
        L.append("    expected hit (col 5)    " + format(float(np.corrcoef(xhit, y)[0, 1]), ".3f"))
        L.append("")

        _, r2_m = ols(pred[:, None], y)
        _, r2_x = ols(xhit[:, None], y)
        w_mx, r2_mx = ols(np.column_stack([pred, xhit]), y)
        pc = partial_corr(xhit, y, pred[:, None])
        pc_rev = partial_corr(pred, y, xhit[:, None])

        L.append("  nested regressions on realized hit rate (R^2):")
        L.append("    model only                    " + format(r2_m, ".4f"))
        L.append("    expected-hit only             " + format(r2_x, ".4f"))
        L.append("    model + expected-hit          " + format(r2_mx, ".4f")
                 + "   (+" + format(r2_mx - r2_m, ".4f") + " over model alone)")
        L.append("    coefs: model " + format(w_mx[0], "+.3f")
                 + ", expected-hit " + format(w_mx[1], "+.3f"))
        L.append("")
        L.append("  partial corr(expected-hit, actual | model pred) = " + format(pc, "+.3f"))
        L.append("  partial corr(model pred, actual | expected-hit) = " + format(pc_rev, "+.3f"))
        L.append("")
        # How much of the CQ feature's spread does the model's prediction track?
        b_track, r2_track = ols(xhit[:, None], pred)
        L.append("  how well the model's own prediction tracks the feature it was given:")
        L.append("    regress model pred on expected-hit: slope " + format(b_track[0], "+.3f")
                 + ", R^2 " + format(r2_track, ".3f"))
        # Is the unshrunk feature noisier for low-PA batters?
        lo = pa_w < np.median(pa_w)
        L.append("    corr(expected-hit, actual)  low-PA half "
                 + format(float(np.corrcoef(xhit[lo], y[lo])[0, 1]), ".3f")
                 + "   high-PA half "
                 + format(float(np.corrcoef(xhit[~lo], y[~lo])[0, 1]), ".3f"))
        L.append("")

    L.append("Reading it. A positive partial correlation of the expected-hit feature with the")
    L.append("realized hit rate, AFTER conditioning on the model's own prediction, means the")
    L.append("model is not extracting information it was handed. That is a fixable modelling")
    L.append("defect, not a data ceiling, and it is the opposite of the 'we are at the")
    L.append("information limit of the feature set' reading in RESULTS.md.")
    L.append("")
    L.append("The low-PA / high-PA split is the mechanism test: columns 5:7 are filled AFTER")
    L.append("per-stat shrinkage runs on columns 0..3 and are never shrunk, so if the feature")
    L.append("is much weaker on the low-PA half then it is arriving unregularised and the fix")
    L.append("is to shrink it with the same 2200-PA constant the raw hit column already gets.")

    rep = "\n".join(L)
    print(rep)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(rep + "\n")
    print("\nwrote " + args.out)


if __name__ == "__main__":
    main()
