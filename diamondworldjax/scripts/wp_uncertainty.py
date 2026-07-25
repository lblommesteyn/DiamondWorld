"""Pre-game win probability WITH full uncertainty, and a full-season calibration pass.

Two things a point win-probability throws away, both delivered here:

  1. Uncertainty decomposition (a subset of games): a generative model separates
     aleatoric noise (the game could play out many ways, from replicas) from
     epistemic uncertainty (how sure we are about the rosters, from the player-skill
     posterior). We draw K skill realizations (skill-mode sample) x R replicas.

  2. Full-season calibration (ALL 2024 games): the SSAC writeup flagged the point
     WP as over-confident, but only on a 30-game slice that was small and
     home-unlucky. This runs the point estimate (skill-mode mean) over every 2024
     game with a known result and builds a proper reliability curve (predicted vs
     observed home-win rate by bin), with ECE, Brier, and the base-rate check, so
     the flag is confirmed or quantified rather than left as a caveat. Saves a
     reliability-curve PNG.

Real outcomes come straight from the processed 2024 data (runs by half per game),
so there is no dependence on any earlier backtest artifact.

  python -m diamondworldjax.scripts.wp_uncertainty            # full season
  python -m diamondworldjax.scripts.wp_uncertainty --limit 300
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.paths import processed_root
from diamondworldjax.scripts.scenario_sim import Sim


def real_home_win(season=2024) -> dict[int, int]:
    """Per game_pk: 1 if the home team outscored the away team, else 0.

    half_bin 1 is the bottom half (home batting), 0 the top (away batting).
    """
    te = (load_seasons([season], data_root=processed_root())
          .filter(pl.col("pa_terminal"))
          .group_by(["game_pk", "half_bin"]).agg(pl.col("runs_scored").sum().alias("r")))
    home = {int(r["game_pk"]): r["r"] for r in te.filter(pl.col("half_bin") == 1).iter_rows(named=True)}
    away = {int(r["game_pk"]): r["r"] for r in te.filter(pl.col("half_bin") == 0).iter_rows(named=True)}
    out = {}
    for g in home:
        if g in away and home[g] != away[g]:      # drop the rare unresolved/tie row
            out[g] = int(home[g] > away[g])
    return out


def reliability(pred, y, nbins=10):
    """Binned reliability table + ECE + Brier."""
    edges = np.linspace(0, 1, nbins + 1)
    rows, ece = [], 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (pred >= a) & (pred < b) if b < 1.0 else (pred >= a) & (pred <= b)
        if sel.sum() == 0:
            continue
        p, o, nb = pred[sel].mean(), y[sel].mean(), int(sel.sum())
        rows.append((a, b, nb, p, o))
        ece += nb / len(pred) * abs(p - o)
    brier = float(np.mean((pred - y) ** 2))
    return rows, ece, brier, edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Max games (default: all with a result).")
    ap.add_argument("--r", type=int, default=100, help="Replicas for the point estimate.")
    ap.add_argument("--epi-games", type=int, default=200, help="Games for the epistemic decomposition.")
    ap.add_argument("--k-draws", type=int, default=4, help="Posterior skill draws (epistemic).")
    args = ap.parse_args()

    s = Sim()
    outcomes = real_home_win(2024)
    games = [g for g in s.real_games(2024, limit=10000)
             if g["park"] != 0 and int(g["game_pk"]) in outcomes]
    if args.limit:
        games = games[:args.limit]
    y = np.array([outcomes[int(g["game_pk"])] for g in games])
    print(f"full-season calibration on {len(games)} games (home-win base rate {y.mean():.3f})", flush=True)

    # Per-game WP is independent across games, so CRN (which pairs replicas across
    # scenarios) buys nothing here and would build one RNG per game. Chunk to bound
    # memory at half-GPU, and run without CRN.
    def wp_over(gs, seed, mode, chunk=250):
        out = []
        for i in range(0, len(gs), chunk):
            H, A = s.run(gs[i:i + chunk], R=args.r, seed=seed, skill_mode=mode, crn=False)
            out.append((H > A).mean(1))
            print(f"    {mode} {min(i + chunk, len(gs))}/{len(gs)}", flush=True)
        return np.concatenate(out)

    # ---- point estimate over all games (aleatoric only, skill-mode mean) ----
    wp = wp_over(games, seed=0, mode="mean")
    rows, ece, brier, edges = reliability(wp, y)

    L = ["FULL-SEASON WIN-PROBABILITY CALIBRATION (v15, skill-mode mean)",
         f"  {len(games)} games x R={args.r} replicas; home-win base rate {y.mean():.3f}, "
         f"model mean WP {wp.mean():.3f}", ""]
    L.append("  reliability (predicted vs observed home-win rate):")
    L.append(f"    {'bin':12s} {'n':>5s} {'pred':>7s} {'obs':>7s} {'gap':>7s}")
    for a, b, nb, p, o in rows:
        L.append(f"    [{a:.1f},{b:.1f})     {nb:5d} {p:7.3f} {o:7.3f} {p-o:+7.3f}")
    L.append("")
    L.append(f"  ECE {ece:.3f}   Brier {brier:.3f}   "
             f"(reference: always-base-rate Brier {y.mean()*(1-y.mean()):.3f})")
    # verdict on the SSAC over-confidence flag
    fav = wp > 0.5
    fav_gap = wp[fav].mean() - y[fav].mean() if fav.sum() else 0.0
    L.append(f"  home-favorite check: when model WP>0.5 (n={int(fav.sum())}), "
             f"pred {wp[fav].mean():.3f} vs real {y[fav].mean():.3f} (gap {fav_gap:+.3f})")
    if ece < 0.04:
        L.append("  VERDICT: well calibrated over the full season; the SSAC over-confidence was")
        L.append("  the 30-game slice being small and home-unlucky, not a model problem.")
    else:
        L.append(f"  VERDICT: a real miscalibration of ECE {ece:.3f} survives the full season "
                 f"(over-confident toward {'home' if fav_gap>0 else 'away'}).")
    L.append("")

    # ---- epistemic vs aleatoric on a subset ----
    sub = games[:args.epi_games]
    wp_sub = wp_over(sub, seed=0, mode="mean")
    draws = [wp_over(sub, seed=100 + k, mode="sample") for k in range(args.k_draws)]
    draws = np.stack(draws)
    epi = draws.std(0).mean()
    ale = np.sqrt(wp_sub * (1 - wp_sub) / args.r).mean()
    L.append(f"  uncertainty decomposition ({len(sub)} games, {args.k_draws} skill draws x R={args.r}):")
    L.append(f"    epistemic (roster) spread {epi*100:.1f}% WP  vs  aleatoric SE {ale*100:.1f}% WP")
    L.append(f"    => roster uncertainty is {'comparable to' if abs(epi-ale)<0.01 else ('larger than' if epi>ale else 'smaller than')} "
             f"single-game noise; a point WP hides it.")

    rep = "\n".join(L)
    print(rep)
    Path("data/eval2").mkdir(parents=True, exist_ok=True)
    Path("data/eval2/wp_uncertainty.txt").write_text(rep + "\n")
    np.savez("data/eval2/wp_calibration.npz", wp=wp, y=y,
             bin_pred=np.array([r[3] for r in rows]), bin_obs=np.array([r[4] for r in rows]),
             bin_n=np.array([r[2] for r in rows]))

    # ---- reliability-curve PNG ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        bp = np.array([r[3] for r in rows]); bo = np.array([r[4] for r in rows])
        bn = np.array([r[2] for r in rows])
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "--", color="#888", lw=1, label="perfect")
        ax.scatter(bp, bo, s=20 + bn / bn.max() * 180, color="#AF4A2C", zorder=3, label="v15 (bin)")
        ax.plot(bp, bo, color="#AF4A2C", lw=1, alpha=.6)
        ax.set_xlabel("predicted home-win probability")
        ax.set_ylabel("observed home-win rate")
        ax.set_title(f"Full-season WP reliability (v15)\nECE {ece:.3f}, Brier {brier:.3f}, n={len(games)}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig("data/eval2/wp_reliability.png", dpi=130)
        print("saved -> data/eval2/wp_reliability.png")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
