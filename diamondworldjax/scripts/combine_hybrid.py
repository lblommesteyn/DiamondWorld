"""Hybrid of the SVI (v12) and the best discriminative MLP, at the player level.

The sweep showed the two are complementary: SVI is best at K, the discriminative
MLP at BB/HR. This averages their per-batter predicted rates (both indexed by
player idx) and reports cross-player correlation for a 50/50 blend, an
outcome-optimal blend, and each model alone. Consumes prod_rates.npz + mlp_rates.npz.
"""
import numpy as np

a = np.load("data/eval2/prod_rates.npz")   # SVI v12
b = np.load("data/eval2/mlp_rates.npz")     # best discriminative MLP
keep = (a["cnt"] >= 150) & (b["cnt"] >= 150)


def rates(d, key):
    return d[key][keep] / d["cnt"][keep]

def corr(pred, real):
    return float(np.corrcoef(pred, real)[0, 1])

out = ["HYBRID: SVI (v12) + discriminative MLP, per-batter rate correlation",
       f"  batters (>=150 PA in both): {int(keep.sum())}", ""]
outcomes = [("K", "sumK", "rK"), ("BB", "sumBB", "rBB"), ("Hit", "sumHit", "rHit"), ("HR", "sumHR", "rHR")]
hdr = f"  {'outcome':7s} {'SVI':>7s} {'MLP':>7s} {'50/50':>7s} {'best':>7s}"
out.append(hdr)
svi_avg = mlp_avg = blend_avg = best_avg = 0.0
for name, sk, rk in outcomes:
    real = rates({k: a[k] for k in a}, rk)  # real is same in both; use a
    ps, pm = rates(a, sk), rates(b, sk)
    cs, cm = corr(ps, real), corr(pm, real)
    cblend = corr(0.5 * ps + 0.5 * pm, real)
    cbest = max(cs, cm)
    out.append(f"  {name:7s} {cs:7.3f} {cm:7.3f} {cblend:7.3f} {cbest:7.3f}")
    svi_avg += cs; mlp_avg += cm; blend_avg += cblend; best_avg += cbest
n = len(outcomes)
out.append(f"  {'AVG':7s} {svi_avg/n:7.3f} {mlp_avg/n:7.3f} {blend_avg/n:7.3f} {best_avg/n:7.3f}")
out.append("")
out.append("50/50 = simple average of the two models' predicted rates.")
out.append("best  = per-outcome oracle (upper bound: pick the better model each outcome).")
rep = "\n".join(out)
print(rep)
open("data/eval2/hybrid.txt", "w").write(rep + "\n")
