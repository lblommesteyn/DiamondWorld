"""Validate common random numbers in the scenario simulator.

CRN pairs the random stream by replica across scenarios, so the causal delta of a
counterfactual (scenario minus baseline) is estimated from paired differences
whose shared game noise cancels. This measures the payoff: pick a real 2024 game,
apply one intervention (swap the home starter for a strong arm), and compare the
standard error of the estimated win-probability and run deltas with CRN off vs on,
at the same R. A large SE reduction means the same precision for far fewer sims.

  python -m diamondworldjax.scripts.crn_validate
"""
from __future__ import annotations

import numpy as np

from diamondworldjax.scripts.scenario_sim import Sim


def delta_stats(H, A, R):
    """Per-replica home-run and win deltas (scenario - baseline), and their SEs.

    With CRN the two rows are paired replica-by-replica, so the SE of the mean
    delta is the SE of the paired differences. Without CRN the rows are
    independent, and the honest SE combines both variances. We report the paired
    SE in both cases (that is what an analyst reads off the two runs), so the
    numbers are directly comparable and the CRN gain is visible.
    """
    base_runs, cf_runs = H[0], H[1]
    d_runs = cf_runs - base_runs
    base_win = (H[0] > A[0]).astype(float)
    cf_win = (H[1] > A[1]).astype(float)
    d_win = cf_win - base_win
    return {
        "mean_run_delta": d_runs.mean(),
        "se_run_delta": d_runs.std(ddof=1) / np.sqrt(R),
        "mean_win_delta": d_win.mean(),
        "se_win_delta": d_win.std(ddof=1) / np.sqrt(R),
        "corr_runs": np.corrcoef(base_runs, cf_runs)[0, 1],
    }


def main():
    R = 400
    s = Sim()
    # a real 2024 game with known rosters
    games = s.real_games(2024, limit=200)
    def known(g):
        return (g["park"] != 0 and g["home_staff"] and g["home_staff"][0] != 0
                and g["away_staff"] and g["away_staff"][0] != 0
                and sum(1 for x in g["home_lineup"] + g["away_lineup"] if x != 0) >= 17)
    game = next(g for g in games if known(g))

    # a strong arm to swap in: among players with a real workload (weighted PA in
    # the stat table), the one with the best K-minus-hits allowed rate.
    workload = s.stats[:, 4]
    dom = np.where(workload >= 300, s.stats[:, 2] - s.stats[:, 0], -np.inf)
    ace = int(np.argmax(dom))

    base = dict(away_lineup=list(game["away_lineup"]), home_lineup=list(game["home_lineup"]),
                away_staff=list(game["away_staff"]), home_staff=list(game["home_staff"]),
                park=game["park"])
    cf = dict(base, home_staff=[ace] + list(game["home_staff"][1:]))
    specs = [base, cf]

    out = ["CRN VALIDATION: baseline vs one-starter-swap, R=%d, same seed" % R, ""]
    rows = {}
    for tag, use in [("CRN off", False), ("CRN on", True)]:
        H, A = s.run(specs, R=R, seed=7, crn=use)
        rows[tag] = delta_stats(H, A, R)
        st = rows[tag]
        out.append(f"  {tag:8s}: run delta {st['mean_run_delta']:+.3f} "
                   f"(SE {st['se_run_delta']:.3f}), win delta {st['mean_win_delta']:+.3f} "
                   f"(SE {st['se_win_delta']:.3f}), corr(base,cf) {st['corr_runs']:+.3f}")
    out.append("")
    rr = rows["CRN off"]["se_run_delta"] / max(rows["CRN on"]["se_run_delta"], 1e-9)
    wr = rows["CRN off"]["se_win_delta"] / max(rows["CRN on"]["se_win_delta"], 1e-9)
    out.append(f"  SE reduction: run delta {rr:.2f}x, win delta {wr:.2f}x")
    out.append(f"  equivalent sim saving: run delta ~{rr**2:.1f}x fewer replicas for the same SE")
    out.append("")
    out.append("  (CRN raises corr(base,cf) toward 1; Var(delta)=Var(base)+Var(cf)-2Cov,")
    out.append("   so a higher covariance is exactly what shrinks the delta's SE.)")
    rep = "\n".join(out)
    print(rep)
    from pathlib import Path
    Path("data/eval2").mkdir(parents=True, exist_ok=True)
    Path("data/eval2/crn_validate.txt").write_text(rep + "\n")


if __name__ == "__main__":
    main()
