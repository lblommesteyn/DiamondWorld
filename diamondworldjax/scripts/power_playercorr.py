"""Power analysis for the cross-player rate-correlation gate.

Ten variants (v17a through v21b) have been rejected by the paired-bootstrap gate in
`bootstrap_playercorr.py`. That record has two very different explanations and the
project has never distinguished them:

  (a) the interventions really are null, or
  (b) the gate cannot resolve effects of the size these interventions produce,
      because 383 batters observed for a few hundred plate appearances each is not
      enough data, and every variant was underpowered from the start.

This script answers that by simulating the test season under a KNOWN ground truth.
Nothing here touches a GPU or a checkpoint; it is pure resampling arithmetic on the
observed PA counts.

Two questions, in order.

1. THE ATTENUATION CEILING. A batter's observed rate is a binomial draw around his
   true rate, so even a model that predicts every true rate exactly cannot correlate
   1.0 with the observed rates. Method of moments on the observed spread gives the
   reliability var(true) / var(observed), whose square root is the highest
   correlation any model can score. Comparing that ceiling to v16's actual score
   says how much of the remaining gap is real headroom versus measurement noise.

2. MINIMUM DETECTABLE EFFECT. Given a true improvement of size delta, how often does
   the paired gate actually fire? Two models are simulated with correlated errors
   (variants of one recipe make similar mistakes, and that correlation is what makes
   the paired CI tight, so it is estimated from the real v16/v21 predictions rather
   than assumed). Sweeping delta gives a power curve and the effect size the gate
   can find 80% of the time.

Usage:
  python -m diamondworldjax.scripts.power_playercorr \
      --rates data/eval2/prod_rates_v16.npz \
      --alt   data/eval2/prod_rates_v21.npz \
      --sims 400 --reps 2000
"""
from __future__ import annotations

import argparse
import json
import numpy as np

STATS = ("K", "BB", "Hit", "HR")


def _load(path: str, min_pa: float):
    d = np.load(path)
    cnt = d["cnt"].astype(np.float64)
    keep = cnt >= min_pa
    keep[0] = False  # index-0 unknown-player sink, see bootstrap_playercorr
    n = cnt[keep]
    pred = np.stack([d["sum" + s][keep] / n for s in STATS], axis=1)
    real = np.stack([d["r" + s][keep] / n for s in STATS], axis=1)
    return n, pred, real


def _corr_rows(x, y):
    xm = x - x.mean(axis=-1, keepdims=True)
    ym = y - y.mean(axis=-1, keepdims=True)
    num = (xm * ym).sum(axis=-1)
    den = np.sqrt((xm * xm).sum(axis=-1) * (ym * ym).sum(axis=-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def reliability(y: np.ndarray, n: np.ndarray):
    """Method-of-moments split of observed rate variance into signal and binomial noise.

    var(observed) = var(true) + E[p(1-p)/n]. The noise term is estimated per batter
    from his own observed rate; y(1-y)/(n-1) is the unbiased form.
    """
    noise = float(np.mean(y * (1.0 - y) / (n - 1.0)))
    var_obs = float(np.var(y, ddof=1))
    var_true = max(var_obs - noise, 1e-12)
    return var_true, noise, var_obs, var_true / var_obs


def shrink(y: np.ndarray, n: np.ndarray, var_true: float):
    """Empirical-Bayes posterior mean per batter: the stand-in for the true rate.

    Weight is per batter because PA counts range over an order of magnitude, so a
    single pooled shrinkage factor would systematically over-shrink the regulars.
    """
    mu = float(np.average(y, weights=n))
    noise_i = np.clip(y * (1.0 - y), 1e-6, None) / n
    w = var_true / (var_true + noise_i)
    return np.clip(w * y + (1.0 - w) * mu, 1e-4, 1 - 1e-4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default="data/eval2/prod_rates_v16.npz",
                    help="incumbent, defines the achieved-correlation calibration target")
    ap.add_argument("--alt", default="data/eval2/prod_rates_v21.npz",
                    help="a variant, used only to estimate how correlated two models' errors are")
    ap.add_argument("--min-pa", type=float, default=150.0)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--deltas", default="0.00,0.01,0.02,0.025,0.03,0.04,0.05,0.06,0.08")
    ap.add_argument("--out", default="data/eval2/power_playercorr.txt")
    ap.add_argument("--json-out", default="data/eval2/power_playercorr.json")
    args = ap.parse_args()

    n, pred, real = _load(args.rates, args.min_pa)
    n_alt, pred_alt, real_alt = _load(args.alt, args.min_pa)
    if not np.allclose(n, n_alt) or not np.allclose(real, real_alt):
        raise SystemExit("--rates and --alt were not scored on the same test set")
    nb = len(n)
    rng = np.random.default_rng(args.seed)

    lines: list[str] = []
    lines.append("# Power analysis of the player-corr gate")
    lines.append("")
    lines.append("%d batters with >= %d PA, %d simulated seasons x %d bootstrap reps, "
                 "%d%% CI, incumbent %s"
                 % (nb, int(args.min_pa), args.sims, args.reps,
                    int(round(100 * (1 - args.alpha))), args.rates))
    lines.append("")

    # ---- 1. attenuation ceiling -------------------------------------------------
    truth = np.empty_like(pred)
    ceil: dict[str, float] = {}
    achieved: dict[str, float] = {}
    err_rho: dict[str, float] = {}
    var_true_s: dict[str, float] = {}
    lines.append("## 1. The attenuation ceiling")
    lines.append("")
    lines.append("| stat | var(true) | binomial noise | reliability | max attainable corr | "
                 "incumbent corr | headroom |")
    lines.append("|---|---|---|---|---|---|---|")
    for j, s in enumerate(STATS):
        y = real[:, j]
        var_true, noise, var_obs, rel = reliability(y, n)
        var_true_s[s] = var_true
        truth[:, j] = shrink(y, n, var_true)
        c = float(np.sqrt(rel))
        a = float(np.corrcoef(pred[:, j], y)[0, 1])
        ceil[s], achieved[s] = c, a
        e_a = pred[:, j] - truth[:, j]
        e_b = pred_alt[:, j] - truth[:, j]
        err_rho[s] = float(np.corrcoef(e_a, e_b)[0, 1])
        lines.append("| %s | %.5f | %.5f | %.3f | %.3f | %.3f | %+.3f |"
                     % (s, var_true, noise, rel, c, a, c - a))
    ceil["AVG"] = float(np.mean([ceil[s] for s in STATS]))
    achieved["AVG"] = float(np.mean([achieved[s] for s in STATS]))
    lines.append("| **AVG** | | | | **%.3f** | **%.3f** | **%+.3f** |"
                 % (ceil["AVG"], achieved["AVG"], ceil["AVG"] - achieved["AVG"]))
    lines.append("")
    lines.append("Reliability is the share of the observed spread in batter rates that is real "
                 "skill rather than binomial luck; its square root is the correlation a model "
                 "that knew every true rate exactly would score.")
    lines.append("")
    rho_avg = float(np.mean([err_rho[s] for s in STATS]))
    lines.append("Error correlation between the two supplied models, used to pair the simulated "
                 "pair: " + ", ".join("%s %.3f" % (s, err_rho[s]) for s in STATS)
                 + " (mean %.3f)." % rho_avg)
    lines.append("")

    # ---- 2. minimum detectable effect ------------------------------------------
    # Calibrate each stat's model-error scale so the simulated incumbent reproduces
    # v16's achieved correlation. For pred = truth + e with e independent of truth,
    # corr(pred, observed) = sqrt(rel) / sqrt(1 + var_e/var_true), which inverts.
    sigma: dict[str, float] = {}
    for s in STATS:
        ratio = max((ceil[s] ** 2 / max(achieved[s], 1e-6) ** 2) - 1.0, 1e-9)
        sigma[s] = float(np.sqrt(ratio * var_true_s[s]))

    idx = rng.integers(0, nb, size=(args.reps, nb))
    lo_q, hi_q = 100 * args.alpha / 2, 100 * (1 - args.alpha / 2)
    deltas = [float(x) for x in args.deltas.split(",")]
    nn = np.rint(n).astype(np.int64)
    power: dict[str, float] = {}

    for delta in deltas:
        # A true AVG improvement of `delta` is applied evenly across the four stats
        # by shrinking the challenger's error scale until its correlation rises by
        # delta, capped just under that stat's ceiling.
        sigma_b: dict[str, float] = {}
        for s in STATS:
            target = min(achieved[s] + delta, ceil[s] - 1e-4)
            ratio = max((ceil[s] ** 2 / max(target, 1e-6) ** 2) - 1.0, 1e-12)
            sigma_b[s] = float(np.sqrt(ratio * var_true_s[s]))

        fires = 0
        for _ in range(args.sims):
            d_avg = np.zeros(args.reps)
            for j, s in enumerate(STATS):
                p = truth[:, j]
                y = rng.binomial(nn, p) / n
                r = abs(err_rho[s])
                z = rng.standard_normal(nb)
                ea = sigma[s] * (np.sqrt(r) * z + np.sqrt(1 - r) * rng.standard_normal(nb))
                eb = sigma_b[s] * (np.sqrt(r) * z + np.sqrt(1 - r) * rng.standard_normal(nb))
                pa, pb = p + ea, p + eb
                d_avg += (_corr_rows(pb[idx], y[idx]) - _corr_rows(pa[idx], y[idx])) / len(STATS)
            lo, hi = np.nanpercentile(d_avg, [lo_q, hi_q])
            if lo > 0 or hi < 0:
                fires += 1
        power["%.3f" % delta] = fires / args.sims

    lines.append("## 2. Power of the gate against a true AVG improvement")
    lines.append("")
    lines.append("| true AVG delta | P(CI excludes zero) |")
    lines.append("|---|---|")
    for delta in deltas:
        lines.append("| %+.3f | %.2f |" % (delta, power["%.3f" % delta]))
    lines.append("")

    mde = None
    for delta in deltas:
        if delta > 0 and power["%.3f" % delta] >= 0.80:
            mde = delta
            break
    if mde is not None:
        lines.append("Minimum detectable effect at 80%% power: **%+.3f AVG**." % mde)
    else:
        lines.append("Minimum detectable effect at 80%% power: beyond the swept range "
                     "(largest tested %+.3f)." % max(deltas))
    lines.append("")
    lines.append("False-positive rate at delta = 0 is %.3f, which should sit near %.2f if the "
                 "gate is correctly calibrated." % (power["0.000"], args.alpha))
    lines.append("")

    txt = "\n".join(lines) + "\n"
    with open(args.out, "w") as f:
        f.write(txt)
    with open(args.json_out, "w") as f:
        json.dump({"n_batters": nb, "sims": args.sims, "reps": args.reps,
                   "ceiling": ceil, "achieved": achieved, "err_rho": err_rho,
                   "sigma_incumbent": sigma, "power": power, "mde_80": mde}, f, indent=2)
    print(txt)


if __name__ == "__main__":
    main()
