"""Item 4: diagnose and correct the simulator's run-total mean bias.

RESULTS.md records the leak-free simulator at mean 9.08 against a real 8.63, a
+0.45 run-per-game overproduction, and a randomized-PIT KS of 0.058 against a
league negative binomial's 0.012. The NB knows nothing about teams, parks or
rosters, so losing to it on calibration is the weakest point in the project's one
genuinely differentiated claim.

This script asks a narrow question: how much of that calibration gap is just the
mean? It fits a one-parameter location correction on HALF the games and scores
the corrected distribution on the other half, so the reported numbers are out of
sample and the correction cannot be credited with fitting noise.

Three corrections are compared, all of which keep run totals integers:

  shift    subtract 1 run with probability q (a randomized location shift). Moves
           the mean by -q and ADDS q(1-q) to the variance.
  thin     keep each simulated run with probability p (binomial thinning). Moves
           the mean to p*mu but SHRINKS the variance to p^2*var + p(1-p)*mu,
           which is the wrong direction here because the sim is already slightly
           under-dispersed against reality.
  both     thin, then shift, with the pair chosen to match mean AND variance.

The point of showing all three is that matching the mean is easy and matching the
mean without wrecking the variance is the actual constraint.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def randomized_pit(samples, y, rng):
    below = (samples < y[:, None]).mean(1)
    at = (samples == y[:, None]).mean(1)
    return below + rng.random(len(y)) * at


def ks_uniform(samples, y, rng):
    n = len(y)
    u = np.sort(randomized_pit(samples, y, rng))
    return float(np.max(np.abs(u - (np.arange(1, n + 1) / n))))


def logscore(samples, y, R):
    ls = 0.0
    for i in range(len(y)):
        p = (samples[i] == y[i]).mean()
        ls += -np.log((p * R + 1) / (R + 20))
    return ls / len(y)


def cover(samples, y, lo, hi):
    ql = np.quantile(samples, lo, axis=1)
    qh = np.quantile(samples, hi, axis=1)
    return float(((y >= ql) & (y <= qh)).mean())


def metrics(samples, y, rng_seed=0):
    R = samples.shape[1]
    rng = np.random.default_rng(rng_seed)
    return {
        "mean": float(samples.mean()),
        "var": float(samples.var()),
        "P>=10": float((samples >= 10).mean()),
        "P<=5": float((samples <= 5).mean()),
        "logscore": logscore(samples, y, R),
        "KS": ks_uniform(samples, y, rng),
        "c50": cover(samples, y, 0.25, 0.75),
        "c80": cover(samples, y, 0.10, 0.90),
        "c90": cover(samples, y, 0.05, 0.95),
    }


def apply_shift(samples, q, rng):
    """Subtract 1 with probability q, floored at zero."""
    drop = rng.random(samples.shape) < q
    return np.maximum(samples - drop.astype(samples.dtype), 0)


def apply_thin(samples, p, rng):
    """Keep each run with probability p."""
    return rng.binomial(samples.astype(int), p).astype(samples.dtype)


def nb_baseline(y, R, seed=0):
    rng = np.random.default_rng(seed)
    mu, var = y.mean(), y.var()
    rr = mu ** 2 / (var - mu) if var > mu else 1e6
    pp = rr / (rr + mu)
    return rng.negative_binomial(rr, pp, (len(y), R)).astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", default="data/eval2/calib_v16-pregame-leakfree_arrays.npz")
    ap.add_argument("--tag", default="v16-leakfree")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = np.load(args.arrays, allow_pickle=True)
    sim = d["sim_total"].astype(float)
    real = d["real_total"].astype(float)
    n, R = sim.shape

    rng = np.random.default_rng(args.seed)
    # Split games in half: fit the correction on A, report everything on B.
    perm = rng.permutation(n)
    A, B = perm[: n // 2], perm[n // 2:]

    L = []
    L.append("# Item 4: run-total mean bias, fitted out of sample")
    L.append("")
    L.append("arrays: " + Path(args.arrays).name + "   games " + str(n) + "   replicas " + str(R))
    L.append("correction fitted on " + str(len(A)) + " games, scored on the held-out "
             + str(len(B)) + ".")
    L.append("")

    # --- fit on A
    mu_sim_A, mu_real_A = sim[A].mean(), real[A].mean()
    var_sim_A, var_real_A = sim[A].var(), real[A].var()
    bias_A = mu_sim_A - mu_real_A
    q = float(np.clip(bias_A, 0.0, 1.0))                 # shift probability
    p = float(np.clip(mu_real_A / mu_sim_A, 0.0, 1.0))   # thinning keep-prob

    # "both": thin by p2 then shift by q2 so that mean AND variance both land.
    # mean:  p2*mu - q2 = mu_real
    # var:   p2^2*var + p2(1-p2)*mu + q2(1-q2) = var_real
    best = None
    for p2 in np.linspace(0.90, 1.0, 201):
        q2 = p2 * mu_sim_A - mu_real_A
        if not (0.0 <= q2 <= 1.0):
            continue
        v = p2 ** 2 * var_sim_A + p2 * (1 - p2) * mu_sim_A + q2 * (1 - q2)
        err = abs(v - var_real_A)
        if best is None or err < best[0]:
            best = (err, float(p2), float(q2))
    _, p2, q2 = best

    L.append("fitted on the training half:")
    L.append("  sim mean " + format(mu_sim_A, ".3f") + "  real mean " + format(mu_real_A, ".3f")
             + "   bias " + format(bias_A, "+.3f"))
    L.append("  sim var  " + format(var_sim_A, ".2f") + "   real var  " + format(var_real_A, ".2f"))
    L.append("  shift q = " + format(q, ".4f"))
    L.append("  thin  p = " + format(p, ".4f"))
    L.append("  both  p = " + format(p2, ".4f") + ", q = " + format(q2, ".4f"))
    L.append("")

    # --- score on B
    r2 = np.random.default_rng(args.seed + 1)
    cands = {
        "sim, uncorrected": sim[B],
        "sim + shift": apply_shift(sim[B], q, r2),
        "sim + thin": apply_thin(sim[B], p, r2),
        "sim + thin&shift": apply_shift(apply_thin(sim[B], p2, r2), q2, r2),
        "league neg-binomial": nb_baseline(real[B], R, seed=args.seed),
    }

    hdr = ("| " + "model".ljust(22) + " |  mean |    var | P>=10 |  P<=5 | logscore |     KS "
           "|   c50 |   c80 |   c90 |")
    L.append("held-out half (" + str(len(B)) + " games), real mean "
             + format(real[B].mean(), ".2f") + "  var " + format(real[B].var(), ".2f") + ":")
    L.append("")
    L.append(hdr)
    L.append("|" + "-" * 24 + "|" + ("-" * 7 + "|") * 4 + "-" * 10 + "|"
             + ("-" * 7 + "|") * 4)
    rows = {}
    for name, S in cands.items():
        m = metrics(S, real[B], rng_seed=args.seed)
        rows[name] = m
        L.append("| " + name.ljust(22) + " | " + format(m["mean"], "5.2f") + " | "
                 + format(m["var"], "6.2f") + " | " + format(m["P>=10"], "5.3f") + " | "
                 + format(m["P<=5"], "5.3f") + " | " + format(m["logscore"], "8.3f") + " | "
                 + format(m["KS"], "6.4f") + " | " + format(m["c50"], "5.3f") + " | "
                 + format(m["c80"], "5.3f") + " | " + format(m["c90"], "5.3f") + " |")
    L.append("")

    base = rows["sim, uncorrected"]
    nb = rows["league neg-binomial"]
    bestname = min((k for k in rows if k.startswith("sim")), key=lambda k: rows[k]["KS"])
    L.append("Reading it:")
    L.append("  uncorrected sim KS " + format(base["KS"], ".4f")
             + " vs league NB " + format(nb["KS"], ".4f")
             + "  (gap " + format(base["KS"] - nb["KS"], "+.4f") + ")")
    L.append("  best correction (" + bestname + ") KS " + format(rows[bestname]["KS"], ".4f")
             + "  (gap to NB " + format(rows[bestname]["KS"] - nb["KS"], "+.4f") + ")")
    closed = (base["KS"] - rows[bestname]["KS"]) / max(base["KS"] - nb["KS"], 1e-9)
    L.append("  a one-parameter location fix closes " + format(closed, ".0%")
             + " of the calibration gap to a baseline that uses no team, park or roster information.")
    L.append("  logscore: sim " + format(base["logscore"], ".3f") + " -> "
             + format(rows[bestname]["logscore"], ".3f") + ", NB "
             + format(nb["logscore"], ".3f"))
    L.append("")
    L.append("Caveat kept in view: this is a post-hoc calibration layer on sim output, in the")
    L.append("same family as the b_heur/recal scale the project already ships. It is fitted on")
    L.append("a disjoint half of the games, so it is honest, but it does not explain WHY the")
    L.append("sim overproduces by half a run. That mechanism is a separate question.")

    rep = "\n".join(L)
    print(rep)
    out = args.out or ("data/eval2/rundist_bias_" + args.tag + ".txt")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(rep + "\n")
    print("\nwrote " + out)


if __name__ == "__main__":
    main()
