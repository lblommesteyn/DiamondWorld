"""Paired bootstrap confidence intervals for the cross-player rate-correlation metric.

The project's headline metric is the cross-player correlation between predicted and
real K/BB/Hit/HR rates over batters with >= 150 test PAs. Every model comparison in
RESULTS.md reports that number as a point estimate, so a difference like v15 -> v16
(+0.017 AVG) cannot currently be distinguished from a lucky test season.

This script fixes that. It resamples BATTERS with replacement (the unit of the
metric), recomputes each model's correlation on the resampled set, and reports:

  * a percentile CI on each model's own correlation, and
  * a PAIRED CI on the difference between a model and the baseline.

The paired CI is the one that matters. Because both models are scored on the same
batters in the same season, the two correlations are strongly positively correlated
across bootstrap replicates, so the CI on their difference is far tighter than the
CIs on the individual correlations. Reading the individual CIs and noting that they
overlap is the classic error here and it would wrongly declare every result null.

Inputs are the `prod_rates_<tag>.npz` files written by prod_playercorr.py.

Usage:
  python -m diamondworldjax.scripts.bootstrap_playercorr \
      --rates v15=data/eval2/prod_rates_v15_2024.npz \
      --rates v16=data/eval2/prod_rates_v16.npz \
      --baseline v15 --reps 20000
"""
from __future__ import annotations

import argparse
import json
import numpy as np

STATS = ("K", "BB", "Hit", "HR")


def _load(path: str) -> dict:
    d = np.load(path)
    return {k: d[k].astype(np.float64) for k in d.files}


def _rates(d: dict, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(predicted, real) rate matrices of shape (n_batters, 4), column order STATS."""
    cnt = d["cnt"][keep]
    pred = np.stack([d[f"sum{s}"][keep] / cnt for s in STATS], axis=1)
    real = np.stack([d[f"r{s}"][keep] / cnt for s in STATS], axis=1)
    return pred, real


def _corr_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson r along axis 1 for stacked replicates.

    x, y have shape (reps, n). Returns (reps,). Computed directly rather than via
    np.corrcoef so all replicates are done in one vectorized pass.
    """
    xm = x - x.mean(axis=1, keepdims=True)
    ym = y - y.mean(axis=1, keepdims=True)
    num = (xm * ym).sum(axis=1)
    den = np.sqrt((xm * xm).sum(axis=1) * (ym * ym).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _fisher_se(r: float, n: int) -> float:
    """Analytic SE on a single correlation, converted back to r units.

    Reference point only. It is the SE of ONE correlation, not of a paired
    difference, so it is the wrong yardstick for model-vs-model and is reported
    purely to show how misleading the unpaired view is.
    """
    if n <= 3:
        return float("nan")
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se_z = 1.0 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - se_z), np.tanh(z + se_z)
    return float((hi - lo) / 2.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", action="append", required=True,
                    help="tag=path/to/prod_rates_<tag>.npz (repeatable)")
    ap.add_argument("--baseline", default=None,
                    help="tag to difference the others against (default: first --rates)")
    ap.add_argument("--min-pa", type=float, default=150.0)
    ap.add_argument("--reps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05, help="two-sided level (0.05 -> 95% CI)")
    ap.add_argument("--out", default="data/eval2/bootstrap_playercorr.txt")
    ap.add_argument("--json-out", default="data/eval2/bootstrap_playercorr.json")
    args = ap.parse_args()

    models: dict[str, dict] = {}
    for spec in args.rates:
        tag, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--rates expects tag=path, got {spec!r}")
        models[tag] = _load(path)
    tags = list(models)
    baseline = args.baseline or tags[0]
    if baseline not in models:
        raise SystemExit(f"--baseline {baseline!r} not among {tags}")

    # A paired comparison requires the SAME batters in every model. Player-table
    # indexing is shared across checkpoints trained on the same seasons, but a
    # variant that appends rows (e.g. --mle rookies) lengthens the arrays, so
    # intersect on the common prefix and require identical PA counts there.
    n_common = min(len(m["cnt"]) for m in models.values())
    ref_cnt = models[baseline]["cnt"][:n_common]
    # Index 0 was the old pooled unknown-player sink. It is deliberately
    # excluded from the metric below, so an older baseline that still contains
    # its pooled PA total is compatible with corrected rate files everywhere
    # that actually enters the paired comparison.
    compare_counts = np.ones(n_common, dtype=bool)
    if n_common:
        compare_counts[0] = False
    for tag, m in models.items():
        if not np.allclose(m["cnt"][:n_common][compare_counts], ref_cnt[compare_counts]):
            n_bad = int((m["cnt"][:n_common][compare_counts] != ref_cnt[compare_counts]).sum())
            raise SystemExit(
                f"PA counts differ between {baseline!r} and {tag!r} on {n_bad} players; "
                "these were not scored on the same test set, so a paired comparison is invalid."
            )
    keep = np.zeros(n_common, dtype=bool)
    keep[:] = ref_cnt >= args.min_pa
    # Drop the index-0 unknown-player sink. There is no reserved sentinel in the
    # player table, so slot 0 is a real batter onto whom every unseen player is
    # folded; the pooled row has ~11.8k PA and so passes any min-pa filter. It is
    # excluded here as well as in prod_playercorr so that .npz files written
    # before that fix are scored correctly on read.
    keep[0] = False
    n = int(keep.sum())

    pred: dict[str, np.ndarray] = {}
    real_ref = None
    for tag, m in models.items():
        p, r = _rates({k: v[:n_common] for k, v in m.items()}, keep)
        pred[tag] = p
        if real_ref is None:
            real_ref = r
        elif not np.allclose(r, real_ref):
            raise SystemExit(f"real (observed) rates differ for {tag!r}; test sets are not identical")
    assert real_ref is not None

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n, size=(args.reps, n))

    # Point estimates and bootstrap distributions, per stat and for the AVG.
    point: dict[str, dict[str, float]] = {t: {} for t in tags}
    boot: dict[str, dict[str, np.ndarray]] = {t: {} for t in tags}
    for tag in tags:
        avg_acc = np.zeros(args.reps)
        for j, s in enumerate(STATS):
            x_all, y_all = pred[tag][:, j], real_ref[:, j]
            point[tag][s] = float(np.corrcoef(x_all, y_all)[0, 1])
            b = _corr_rows(x_all[idx], y_all[idx])
            boot[tag][s] = b
            avg_acc += b
        point[tag]["AVG"] = float(np.mean([point[tag][s] for s in STATS]))
        boot[tag]["AVG"] = avg_acc / len(STATS)

    lo_q, hi_q = 100 * args.alpha / 2, 100 * (1 - args.alpha / 2)
    conf = int(round(100 * (1 - args.alpha)))
    cols = list(STATS) + ["AVG"]
    lines: list[str] = []
    W = f"paired bootstrap, {args.reps} reps, {n} batters with >= {int(args.min_pa)} PA, {conf}% CI"
    lines.append(f"# Cross-player rate correlation with confidence intervals ({W})")
    lines.append("")

    lines.append("## Per-model correlation (unpaired CI)")
    lines.append("")
    lines.append("| model | " + " | ".join(cols) + " |")
    lines.append("|---" * (len(cols) + 1) + "|")
    for tag in tags:
        cells = []
        for s in cols:
            b = boot[tag][s]
            lo, hi = np.nanpercentile(b, [lo_q, hi_q])
            cells.append(f"{point[tag][s]:.3f} [{lo:.3f}, {hi:.3f}]")
        lines.append(f"| {tag} | " + " | ".join(cells) + " |")
    lines.append("")

    others = [t for t in tags if t != baseline]
    results_json: dict = {
        "n_batters": n, "reps": args.reps, "min_pa": args.min_pa,
        "baseline": baseline, "alpha": args.alpha,
        "point": point, "paired": {},
    }
    if others:
        lines.append(f"## Paired difference vs {baseline} (the decisive test)")
        lines.append("")
        lines.append("| comparison | " + " | ".join(cols) + " |")
        lines.append("|---" * (len(cols) + 1) + "|")
        for tag in others:
            cells = []
            results_json["paired"][tag] = {}
            for s in cols:
                d = boot[tag][s] - boot[baseline][s]
                dp = point[tag][s] - point[baseline][s]
                lo, hi = np.nanpercentile(d, [lo_q, hi_q])
                # Two-sided bootstrap p-value: how often the resampled difference
                # lands on the opposite side of zero from the point estimate.
                frac = float(np.mean(d <= 0)) if dp > 0 else float(np.mean(d >= 0))
                p = min(1.0, 2 * max(frac, 1.0 / args.reps))
                star = "*" if (lo > 0 or hi < 0) else " "
                cells.append(f"{dp:+.3f} [{lo:+.3f}, {hi:+.3f}] p={p:.3f}{star}")
                results_json["paired"][tag][s] = {
                    "delta": dp, "lo": float(lo), "hi": float(hi), "p": p,
                    "excludes_zero": bool(lo > 0 or hi < 0),
                }
            lines.append(f"| {tag} - {baseline} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("`*` marks a difference whose CI excludes zero.")
        lines.append("")

    lines.append("## Why the paired CI is the right one")
    lines.append("")
    lines.append(
        f"For reference, the analytic (Fisher-z) SE on a SINGLE correlation at n={n} is "
        f"about +/-{_fisher_se(point[baseline]['AVG'], n):.3f} in r units around the baseline AVG. "
        "That is the width of the unpaired view above and it is far too wide to resolve "
        "the differences this project cares about. The paired CI is narrower because both "
        "models see the same batters, so shared batter-level noise cancels in the difference."
    )
    if others:
        for tag in others:
            b_un = np.nanpercentile(boot[tag]["AVG"], [lo_q, hi_q])
            d = boot[tag]["AVG"] - boot[baseline]["AVG"]
            b_pa = np.nanpercentile(d, [lo_q, hi_q])
            lines.append("")
            lines.append(
                f"- {tag} AVG: unpaired CI width {b_un[1] - b_un[0]:.3f}, "
                f"paired-vs-{baseline} CI width {b_pa[1] - b_pa[0]:.3f} "
                f"({(b_un[1] - b_un[0]) / max(b_pa[1] - b_pa[0], 1e-9):.1f}x tighter)."
            )

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        open(args.out, "w").write(text)
    if args.json_out:
        json.dump(results_json, open(args.json_out, "w"), indent=2)


if __name__ == "__main__":
    main()
