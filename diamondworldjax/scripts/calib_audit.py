"""Conditional-calibration audit.

Aggregate KL/Wasserstein (RESULTS.md) are MARGINAL metrics: a model can match
the leaguewide run distribution while being miscalibrated per matchup. For
betting, conditional calibration is what matters. This script builds a per-game
predictive distribution by replicating each real game R times through the true
simulator (one big batched simulate() call), then scores it against the real
outcome with proper calibration diagnostics:

  - PIT (probability integral transform) of real total runs, randomized to
    handle discreteness; should be Uniform(0,1). Reported as a 10-bin histogram
    plus a chi-square uniformity statistic.
  - CRPS on total runs (ensemble form), the proper score for the full dist.
  - Moneyline: predicted P(home win) vs realized, as a reliability table with
    ECE, Brier, and log-loss.
  - Totals over/under at a grid of lines: log-loss, Brier, ECE per line.

Baselines (B0/B1) are run-only and produce no per-matchup distribution, so this
audit is about whether THIS model's conditional probabilities are trustworthy.

Usage:
  python -m diamondworldjax.scripts.calib_audit --ckpt <ckpt> --outcome-only \
    --fatigue --use-park --recal --recal-version v10 --recal-scale 0.35 \
    --limit-games 1000 --replicas 120 --out data/eval2/calib_v10.txt
"""
from __future__ import annotations

import argparse
import pickle
import time
from functools import partial
from pathlib import Path

import numpy as np
import polars as pl

from diamondworldjax.paths import processed_root
from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.model.pa_model import pa_model
from diamondworldjax.sim.rules_engine import EmpiricalEngine
from diamondworldjax.sim.game_extract import extract_games, fit_hook_dists
from diamondworldjax.scripts.train_pa import _build_player_table, _build_park_index
from diamondworldjax.scripts.simulate_games import simulate, RECAL_VECS, TRAIN, TEST


def ensemble_crps(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS per game from an ensemble. samples (G,R), y (G,). Returns (G,)."""
    R = samples.shape[1]
    term1 = np.abs(samples - y[:, None]).mean(axis=1)
    # E|X - X'| via sorted-sample identity: (2/R^2) * sum_i (2i - R + 1) x_(i)
    s = np.sort(samples, axis=1)
    idx = np.arange(1, R + 1)
    term2 = (2.0 / (R * R)) * (s * (2 * idx - R - 1)).sum(axis=1)
    return term1 - 0.5 * term2


def randomized_pit(samples: np.ndarray, y: np.ndarray, rng) -> np.ndarray:
    """Randomized PIT for discrete y. samples (G,R), y (G,). Returns (G,) in [0,1]."""
    lt = (samples < y[:, None]).mean(axis=1)
    eq = (samples == y[:, None]).mean(axis=1)
    u = rng.random(len(y))
    return lt + u * eq


def reliability(p: np.ndarray, y: np.ndarray, nbins: int = 10):
    """Reliability table for probabilistic binary forecast p vs outcome y (0/1).
    Returns (rows, ece) where rows = list of (lo, hi, n, mean_p, emp_freq)."""
    edges = np.linspace(0, 1, nbins + 1)
    rows, ece, N = [], 0.0, len(p)
    for i in range(nbins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < nbins - 1 else (p >= lo) & (p <= hi)
        n = int(m.sum())
        if n == 0:
            rows.append((lo, hi, 0, np.nan, np.nan))
            continue
        mp, ef = p[m].mean(), y[m].mean()
        ece += (n / N) * abs(mp - ef)
        rows.append((lo, hi, n, mp, ef))
    return rows, ece


def logloss(p: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(((p - y) ** 2).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--outcome-only", action="store_true")
    ap.add_argument("--fatigue", action="store_true")
    ap.add_argument("--use-park", action="store_true")
    ap.add_argument("--platoon", action="store_true")
    ap.add_argument("--recal", action="store_true")
    ap.add_argument("--recal-version", choices=list(RECAL_VECS), default="v10")
    ap.add_argument("--recal-scale", type=float, default=0.35)
    ap.add_argument("--limit-games", type=int, default=1000)
    ap.add_argument("--replicas", type=int, default=120)
    ap.add_argument("--chunk-games", type=int, default=250,
                    help="Games per batched simulate() call (games*replicas rows).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lines", type=str, default="6.5,7.5,8.5,9.5,10.5")
    ap.add_argument("--recal-file", type=Path, default=None,
                    help="Load fitted recal vector from npz (overrides --recal-version).")
    ap.add_argument("--recal-key", type=str, default="b")
    ap.add_argument("--recal-temp", type=float, default=1.0)
    ap.add_argument("--recency-halflife", type=float, default=None)
    ap.add_argument("--skill-mode", choices=["prior", "mean", "sample"], default="prior")
    ap.add_argument("--no-bullpen", action="store_true",
                    help="Starter pitches all game (no reliever info) - pre-game-only betting eval.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    with open(args.ckpt, "rb") as f:
        params = pickle.load(f)["params"]
    train_pitches = load_seasons(TRAIN, data_root=processed_root())
    ptab = _build_player_table(train_pitches, recency_halflife=args.recency_halflife)
    park_map = _build_park_index(train_pitches) if args.use_park else None
    train_pa = train_pitches.filter(pl.col("pa_terminal"))
    engine = EmpiricalEngine().fit(train_pa)
    hook_dists = fit_hook_dists(train_pa)
    del train_pitches, train_pa

    test_pa = load_seasons(TEST, data_root=processed_root()).filter(pl.col("pa_terminal"))
    keep = test_pa["game_pk"].unique().sort().to_numpy()[:args.limit_games]
    test_pa = test_pa.filter(pl.col("game_pk").is_in(keep.tolist()))

    # Real per-game outcomes (total, home, away). half_bin: top=away bats, bot=home.
    rg = (test_pa.group_by(["game_pk", "half_bin"])
          .agg(pl.col("runs_scored").sum().alias("r")))
    pv = rg.pivot(values="r", index="game_pk", on="half_bin").fill_null(0)
    hb_cols = [c for c in pv.columns if c != "game_pk"]
    # identify which half_bin label is the home side (bottom). Values seen: 0/1 or top/bot.
    print(f"half_bin columns: {hb_cols}", flush=True)

    games = extract_games(test_pa, ptab["id_to_idx"], park_map=park_map)
    game_pks = [g["game_pk"] for g in games]
    real_map = {row["game_pk"]: row for row in pv.iter_rows(named=True)}

    # Map columns -> away/home. half_bin bottom (home) is the larger label.
    def col_for(side_is_home):
        labs = sorted(hb_cols, key=lambda c: str(c))
        return labs[1] if side_is_home else labs[0]
    home_col, away_col = col_for(True), col_for(False)

    real_home = np.array([real_map[pk][home_col] for pk in game_pks], dtype=float)
    real_away = np.array([real_map[pk][away_col] for pk in game_pks], dtype=float)
    real_total = real_home + real_away
    real_homewin = (real_home > real_away).astype(float)
    decided = real_home != real_away  # drop ties for moneyline scoring

    pt = {"stats": jnp.array(ptab["stats"]), "league": jnp.array(ptab["league"]),
          "hand": jnp.array(ptab["hand"]),
          "bat_hand": np.asarray(ptab.get("bat_hand", np.full(len(ptab["hand"]), 0.5, np.float32))),
          "pit_hand": np.asarray(ptab.get("pit_hand", np.full(len(ptab["hand"]), 0.5, np.float32))),
          "_engine": engine, "_hook_dists": hook_dists}
    mkw = {}
    if args.outcome_only:
        mkw["outcome_only"] = True
    if args.fatigue:
        mkw["fatigue"] = True
    if args.platoon:
        mkw["platoon"] = True
    model_fn = partial(pa_model, **mkw) if mkw else pa_model
    if args.recal_file is not None:
        recal_vec = np.load(args.recal_file)[args.recal_key].astype(np.float64)
    else:
        recal_vec = RECAL_VECS[args.recal_version]

    R = args.replicas
    G = len(games)
    sim_total = np.zeros((G, R))
    sim_home = np.zeros((G, R))
    sim_away = np.zeros((G, R))

    t0 = time.time()
    for c0 in range(0, G, args.chunk_games):
        c1 = min(c0 + args.chunk_games, G)
        chunk = games[c0:c1]
        rep = [g for g in chunk for _ in range(R)]  # each game's R replicas contiguous
        res = simulate(
            model_fn, params, pt, rep, jax.random.PRNGKey(args.seed + c0 + 1),
            recal=args.recal, recal_scale=args.recal_scale, recal_vec=recal_vec,
            seed=args.seed + c0 + 1, platoon=args.platoon, recal_temp=args.recal_temp,
            skill_mode=args.skill_mode, no_bullpen=args.no_bullpen,
        )
        a = res["away"].reshape(c1 - c0, R)
        h = res["home"].reshape(c1 - c0, R)
        sim_away[c0:c1] = a
        sim_home[c0:c1] = h
        sim_total[c0:c1] = a + h
        print(f"  chunk {c0}-{c1}/{G}  elapsed={time.time()-t0:.0f}s", flush=True)

    # ---------- metrics ----------
    rng = np.random.default_rng(args.seed)
    L = []
    L.append(f"CONDITIONAL-CALIBRATION AUDIT  ckpt={args.ckpt.name}")
    L.append(f"recal={args.recal_version}@{args.recal_scale}  games={G}  replicas={R}  "
             f"elapsed={time.time()-t0:.0f}s")
    L.append(f"real: total mean={real_total.mean():.2f}  sim total mean={sim_total.mean():.2f}")
    L.append("")

    # PIT on total runs
    pit = randomized_pit(sim_total, real_total, rng)
    hist, _ = np.histogram(pit, bins=10, range=(0, 1))
    exp = len(pit) / 10
    chi2 = float(((hist - exp) ** 2 / exp).sum())
    L.append("PIT (total runs), 10 bins  [uniform => calibrated]")
    L.append("  counts: " + " ".join(f"{c:4d}" for c in hist))
    L.append(f"  expected/bin={exp:.1f}   chi2={chi2:.1f} (df=9; >16.9 = miscalibrated at .05)")
    L.append("")

    # CRPS
    crps = ensemble_crps(sim_total, real_total).mean()
    L.append(f"CRPS (total runs, lower better): {crps:.4f}")
    L.append("")

    # Moneyline reliability
    p_home = sim_home.__gt__(sim_away).mean(axis=1)
    pd_, yd_ = p_home[decided], real_homewin[decided]
    rows, ece = reliability(pd_, yd_)
    L.append(f"MONEYLINE  P(home win)  (n={int(decided.sum())} decided games)")
    L.append(f"  Brier={brier(pd_, yd_):.4f}  LogLoss={logloss(pd_, yd_):.4f}  ECE={ece:.4f}")
    L.append(f"  base rate real home-win={yd_.mean():.3f}  mean pred={pd_.mean():.3f}")
    L.append("  bin        n   pred   real")
    for lo, hi, n, mp, ef in rows:
        if n:
            L.append(f"  {lo:.1f}-{hi:.1f}  {n:5d}  {mp:.3f}  {ef:.3f}")
    L.append("")

    # Totals over/under
    lines = [float(x) for x in args.lines.split(",")]
    L.append("TOTALS over/under")
    L.append("  line   p_over  real_over  Brier  LogLoss   ECE")
    for ln in lines:
        p_over = (sim_total > ln).mean(axis=1)
        y_over = (real_total > ln).astype(float)
        _, e = reliability(p_over, y_over)
        L.append(f"  {ln:5.1f}  {p_over.mean():.3f}   {y_over.mean():.3f}   "
                 f"{brier(p_over,y_over):.4f} {logloss(p_over,y_over):.4f}  {e:.4f}")
    L.append("")

    report = "\n".join(L)
    print(report, flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        # also save arrays for the backtest stage
        np.savez(str(args.out).replace(".txt", "_arrays.npz"),
                 sim_total=sim_total, sim_home=sim_home, sim_away=sim_away,
                 real_total=real_total, real_home=real_home, real_away=real_away,
                 game_pk=np.array(game_pks))
        print(f"saved -> {args.out} (+ arrays)", flush=True)


if __name__ == "__main__":
    main()
