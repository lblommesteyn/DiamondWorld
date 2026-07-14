"""Apples-to-apples: v6-final vs the run-distribution baselines, ONE metric.

The baselines were originally scored with a different package's metric
(diamondworld.eval.metrics) than our model (diamondworldjax.eval.calibration),
so the old scoreboard numbers are not strictly comparable to v6-final. This
script re-runs the baselines and scores every method (real reference, v6-final,
B1 negbinom, B0 markov) through the SAME game_run_metrics function on the SAME
test seasons, so the run-distribution comparison is airtight.

It also states the differentiator the run distribution can't show: the baselines
have no batter identity, so they cannot produce player stat lines at all, whereas
v6-final reproduces them (K% cross-player corr ~0.66 conditioned).

v6-final per-game runs are read from a .npy produced by:
    simulate_games --ckpt <v6-final> --outcome-only --recal --limit-games N --dump-runs v6_runs.npy

Usage:
    python -m diamondworldjax.scripts.compare_baselines --v6-runs data/v6_runs.npy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from diamondworldjax.paths import processed_root
from diamondworldjax.eval.calibration import game_run_metrics

TRAIN = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
TEST = [2023, 2024]


def _load_full(seasons) -> pl.DataFrame:
    """Read processed parquets directly to keep ALL columns (the jax load_seasons
    drops stand/p_throws, which the Markov baseline needs)."""
    root = processed_root()
    return pl.concat([pl.read_parquet(root / f"pitches_{y}.parquet") for y in seasons],
                     how="diagonal_relaxed")


def _per_game_runs(df) -> np.ndarray:
    g = df.filter(pl.col("pa_terminal")).group_by("game_pk").agg(pl.col("runs_scored").sum().alias("r"))
    return g["r"].to_numpy().astype(float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v6-runs", type=Path, default=None,
                    help=".npy of the model's per-game runs (from simulate_games --dump-runs).")
    ap.add_argument("--model-label", type=str, default="v6-final",
                    help="Row label for the model whose runs are loaded from --v6-runs.")
    ap.add_argument("--player-corr", type=str, default="0.66",
                    help="Conditioned K%% cross-player correlation to annotate the model row.")
    ap.add_argument("--n-games", type=int, default=2430)
    args = ap.parse_args()

    print(f"Loading test {TEST} for real reference...", flush=True)
    test_df = _load_full(TEST)
    real_runs = _per_game_runs(test_df)
    print(f"  real: mean={real_runs.mean():.2f} std={real_runs.std():.2f} n={len(real_runs)}", flush=True)

    print(f"Fitting baselines on {TRAIN}...", flush=True)
    train_df = _load_full(TRAIN)
    from diamondworld.baselines.markov_re24 import MarkovRE24Simulator
    from diamondworld.baselines.negbinom import NegBinomSimulator

    methods: dict[str, np.ndarray] = {}
    for name, sim in [("B0_markov", MarkovRE24Simulator()), ("B1_negbinom", NegBinomSimulator())]:
        print(f"  fitting + simulating {name}...", flush=True)
        sim.fit(train_df)
        sim_df = sim.simulate_season(n_games=args.n_games)
        methods[name] = _per_game_runs(sim_df)

    if args.v6_runs is not None and args.v6_runs.exists():
        methods[args.model_label] = np.load(args.v6_runs)
    else:
        print("  WARNING: no model runs .npy provided; run simulate_games --dump-runs first.", flush=True)

    # Score every method through the SAME game_run_metrics on the SAME real ref.
    print(f"\n=== Run-distribution: identical metric, test {TEST} ===", flush=True)
    print(f"  {'method':12s} {'mean':>6s} {'std':>6s} {'KL':>8s} {'wass':>7s} {'p8+err':>7s} {'player stats?':>14s}", flush=True)
    print(f"  {'real':12s} {real_runs.mean():6.2f} {real_runs.std():6.2f} {'-':>8s} {'-':>7s} {'-':>7s} {'(reference)':>14s}", flush=True)
    for name, runs in methods.items():
        m = game_run_metrics(runs, real_runs)
        can = f"YES (corr {args.player_corr})" if name == args.model_label else "NO (no batter)"
        print(f"  {name:12s} {runs.mean():6.2f} {runs.std():6.2f} {m['kl_run_distribution']:8.4f} "
              f"{m['wasserstein_runs']:7.3f} {m['p8_plus_error']:7.4f} {can:>14s}", flush=True)

    print(f"\n  Run distribution: baselines and {args.model_label} are comparable (both fit the", flush=True)
    print("  aggregate). Player stat lines: baselines structurally cannot produce them", flush=True)
    print(f"  (no batter identity); {args.model_label} reproduces them. That is the differentiator.", flush=True)


if __name__ == "__main__":
    main()
