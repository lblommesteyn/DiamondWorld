"""Learned calibration: fit a per-class bias + temperature on held-out logits.

The recal used in the simulator is a per-class logit shift log(real/model) with a
single hand-tuned global scale. That is a heuristic. This tool fits, by proper
maximum likelihood on a held-out split, the calibration map

    p = softmax((logits + b) / T)

with per-class bias b (9-vector, E pinned to 0) and a scalar temperature T > 0.
It reports held-out negative log-likelihood and the marginal outcome fit for:
  - raw model,
  - the current heuristic recal (b = log(real/model), no temperature),
  - the learned (b, T),
so we can see whether principled calibration beats the hand-tuned knob out of
sample. Consumes the npz from `diag_outcomes --dump-logits`.

Usage:
  python -m diamondworldjax.scripts.fit_calibration --logits data/eval2/v11_logits.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PA_OUTCOMES = ["K", "BB", "HBP", "1B", "2B", "3B", "HR", "out", "E"]
E_IDX = 8


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits, y, b, logT):
    T = np.exp(logT)
    p = softmax((logits + b) / T)
    p = np.clip(p[np.arange(len(y)), y], 1e-9, 1.0)
    return -np.log(p).mean()


def marginal_fit(logits, y, b, logT):
    """Mean |sim_marginal - real_marginal| across classes (softmax-expected)."""
    T = np.exp(logT)
    p = softmax((logits + b) / T)
    sim = p.mean(axis=0)
    real = np.bincount(y, minlength=9) / len(y)
    return np.abs(sim - real).sum(), sim, real


def fit(logits, y, iters=4000, lr=0.2, seed=0):
    """Gradient descent on (b, logT) minimizing NLL. b[E] pinned to 0."""
    rng = np.random.default_rng(seed)
    b = np.zeros(9)
    logT = 0.0
    n = len(y)
    onehot = np.zeros((n, 9)); onehot[np.arange(n), y] = 1.0
    for it in range(iters):
        T = np.exp(logT)
        z = (logits + b) / T
        p = softmax(z)
        # dNLL/dz = (p - onehot)/n ... averaged
        g = (p - onehot) / n
        grad_b = g.sum(axis=0) / T
        # dz/dlogT = -(logits + b)/T ; chain through
        grad_logT = (g * (-(logits + b) / T)).sum()
        grad_b[E_IDX] = 0.0  # pin E
        b -= lr * grad_b
        logT -= lr * grad_logT
    return b, logT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    d = np.load(args.logits)
    logits, y = d["logits"].astype(np.float64), d["real"].astype(np.int64)
    n = len(y)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    nval = int(n * args.val_frac)
    val, tr = perm[:nval], perm[nval:]
    Ltr, Ytr = logits[tr], y[tr]
    Lva, Yva = logits[val], y[val]

    lines = [f"LEARNED CALIBRATION  logits={args.logits.name}  n={n} (fit {len(tr)}, val {len(val)})", ""]

    # raw
    b0, logT0 = np.zeros(9), 0.0
    # heuristic recal: b = log(real/model marginal on fit set), T=1
    p_raw = softmax(Ltr)
    sim_m = p_raw.mean(axis=0)
    real_m = np.bincount(Ytr, minlength=9) / len(Ytr)
    b_heur = np.zeros(9)
    for j in range(9):
        if j != E_IDX and sim_m[j] > 0 and real_m[j] > 0:
            b_heur[j] = np.log(real_m[j] / sim_m[j])
    # learned
    b_fit, logT_fit = fit(Ltr, Ytr)

    def report(name, b, logT):
        v = nll(Lva, Yva, b, logT)
        mfit, sim, real = marginal_fit(Lva, Yva, b, logT)
        lines.append(f"{name:16s} val-NLL={v:.4f}  marg-L1={mfit:.4f}  T={np.exp(logT):.3f}")
        return v

    lines.append("held-out (val) scores, lower better:")
    report("raw model", b0, logT0)
    report("heuristic recal", b_heur, 0.0)
    report("learned (b,T)", b_fit, logT_fit)
    lines.append("")
    lines.append(f"learned bias b (order {','.join(PA_OUTCOMES)}):")
    lines.append("  RECAL = np.array([" + ", ".join(f"{v:+.4f}" for v in b_fit) + "], dtype=np.float64)")
    lines.append(f"  temperature T = {np.exp(logT_fit):.4f}")

    report_s = "\n".join(lines)
    print(report_s)
    if args.out:
        args.out.write_text(report_s)
        np.savez(str(args.out).replace(".txt", "_params.npz"),
                 b=b_fit, logT=logT_fit, T=float(np.exp(logT_fit)), b_heur=b_heur)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
