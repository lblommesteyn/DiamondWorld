"""Pre-game win probability WITH full uncertainty.

Most win-prob models give a point estimate. A generative model gives a
distribution: aleatoric (game randomness, from replicas) AND epistemic (how sure
we are about the rosters, from the player-skill posterior). We draw K skill
realizations from the fitted posterior (skill-mode sample), simulate each with R
replicas, and report P(home win) as a point estimate plus a credible interval, then
check that the point estimates are calibrated against real outcomes.
"""
from __future__ import annotations
import numpy as np, polars as pl
from diamondworldjax.scripts.scenario_sim import Sim

K_DRAWS = 4      # posterior skill draws (epistemic)
R = 100          # replicas per draw (aleatoric)
NGAMES = 30


def main():
    s = Sim()
    games = [g for g in s.real_games(2024, limit=500) if g["park"] != 0][120:120 + NGAMES]
    # point estimate (posterior mean) + K posterior draws
    Hm, Am = s.run(games, R=R, seed=0, skill_mode="mean")
    wp_point = (Hm > Am).mean(1)
    draws = []
    for k in range(K_DRAWS):
        Hk, Ak = s.run(games, R=R, seed=100 + k, skill_mode="sample")
        draws.append((Hk > Ak).mean(1))
    draws = np.stack(draws)                       # (K, n)
    lo, hi = draws.min(0), draws.max(0)
    epi = draws.std(0)                            # epistemic spread across draws
    ale = np.sqrt(wp_point * (1 - wp_point) / R)  # aleatoric SE of the point estimate

    # real outcomes for calibration
    arr = np.load("data/eval2/calib_v13_bt_arrays.npz")
    real = {int(pk): int(h > a) for pk, h, a in zip(arr["game_pk"], arr["real_home"], arr["real_away"])}
    y = np.array([real.get(int(g["game_pk"]), -1) for g in games])
    m = y >= 0

    out = ["PRE-GAME WIN PROBABILITY WITH FULL UNCERTAINTY (v13)",
           f"  {NGAMES} games, {K_DRAWS} posterior skill draws x {R} replicas each", ""]
    out.append(f"  mean epistemic spread (roster uncertainty) = {epi.mean()*100:.1f}% WP")
    out.append(f"  mean aleatoric SE (game randomness at R={R}) = {ale.mean()*100:.1f}% WP")
    out.append(f"  => roster uncertainty is {'larger' if epi.mean()>ale.mean() else 'smaller'} than "
               f"single-game sampling noise; a point WP hides it.")
    out.append("")
    out.append("  sample games (home win prob, 90%-ish credible band from posterior draws):")
    order = np.argsort(-wp_point)
    for i in list(order[:5]) + list(order[-5:]):
        out.append(f"    game {int(games[i]['game_pk'])}: WP {wp_point[i]:.2f}  [{lo[i]:.2f}, {hi[i]:.2f}]"
                   + (f"   (actual: {'HOME' if y[i]==1 else 'away'})" if m[i] else ""))
    out.append("")
    # calibration of the point estimate
    if m.sum() > 0:
        edges = [0, .4, .5, .6, 1.01]; out.append("  calibration of the point WP:")
        for a, b in zip(edges[:-1], edges[1:]):
            sel = m & (wp_point >= a) & (wp_point < b)
            if sel.sum():
                out.append(f"    WP in [{a:.1f},{b:.1f}): n={int(sel.sum())}  pred {wp_point[sel].mean():.2f}  real {y[sel].mean():.2f}")
    rep = "\n".join(out)
    print(rep)
    open("data/eval2/wp_uncertainty.txt", "w").write(rep + "\n")


if __name__ == "__main__":
    main()
