"""Game-level evaluation using generated, rather than replayed, game state.

The simulator receives observed lineups/parks as evaluation covariates, but it
generates every plate appearance, inning transition, walk-off, and extra inning
itself. The score distribution therefore remains a model output.
"""
from __future__ import annotations

import argparse
import pickle
import time
from functools import partial
from pathlib import Path

import numpy as np
import polars as pl

from diamondworldjax.paths import processed_root, checkpoints_root
from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.eval.calibration import game_run_metrics
from diamondworldjax.sim.game_evaluation import simulate_score_draws
from diamondworldjax.sim.game_extract import extract_games, fit_hook_dists
from diamondworldjax.sim.rules_engine import EmpiricalEngine
from diamondworldjax.scripts.train_pa import _build_player_table, _build_park_index, apply_park_idx

TRAIN_SEASONS = list(range(2015, 2023))
TEST_SEASONS = [2023, 2024]


def _real_scores_by_game(test_pa) -> dict[int, tuple[float, float]]:
    """Return game_pk -> (away, home), where top halves are away batting."""
    grouped = (
        test_pa.group_by(["game_pk", "half_bin"])
        .agg(pl.col("runs_scored").sum().alias("runs"))
    )
    out: dict[int, list[float]] = {}
    for row in grouped.iter_rows(named=True):
        game_pk = int(row["game_pk"])
        scores = out.setdefault(game_pk, [0.0, 0.0])
        scores[int(row["half_bin"])] = float(row["runs"])
    return {game_pk: (scores[0], scores[1]) for game_pk, scores in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path,
                        default=checkpoints_root() / "dwjax_pa_v6" / "dwjax_step_0050000.pkl")
    parser.add_argument("--batch", type=int, default=64,
                        help="Number of game specs per simulator batch.")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--limit-games", type=int, default=0)
    parser.add_argument("--outcome-only", action="store_true")
    parser.add_argument("--fatigue", action="store_true")
    parser.add_argument("--platoon", action="store_true")
    parser.add_argument("--pitchformer", action="store_true")
    parser.add_argument("--pitchformer-dim", type=int, default=128)
    parser.add_argument("--pitchformer-layers", type=int, default=2)
    parser.add_argument("--pitchformer-heads", type=int, default=4)
    parser.add_argument("--pitchformer-dropout", type=float, default=0.0)
    parser.add_argument("--use-park", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    from diamondworldjax.model.pa_model import pa_model

    with open(args.ckpt, "rb") as f:
        params = pickle.load(f)["params"]

    train_pitches = load_seasons(TRAIN_SEASONS, data_root=processed_root())
    ptab = _build_player_table(train_pitches)
    park_map = _build_park_index(train_pitches)
    train_pa = train_pitches.filter(pl.col("pa_terminal"))
    engine = EmpiricalEngine().fit(train_pa)
    hook_dists = fit_hook_dists(train_pa)
    del train_pitches, train_pa

    test_pa = load_seasons(TEST_SEASONS, data_root=processed_root()).filter(pl.col("pa_terminal"))
    if args.limit_games > 0:
        keep = test_pa["game_pk"].unique().sort().to_numpy()[:args.limit_games]
        test_pa = test_pa.filter(pl.col("game_pk").is_in(keep.tolist()))
    if args.use_park:
        test_pa = apply_park_idx(test_pa, park_map)

    real_by_game = _real_scores_by_game(test_pa)
    games = extract_games(test_pa, ptab["id_to_idx"], park_map=park_map if args.use_park else None,
                          unknown_idx=ptab["unknown_index"])
    games = [game for game in games if game["game_pk"] in real_by_game]
    if not games:
        raise SystemExit("No complete games could be extracted from the requested data.")
    real_away = np.array([real_by_game[game["game_pk"]][0] for game in games])
    real_home = np.array([real_by_game[game["game_pk"]][1] for game in games])

    pt = {
        "stats": jnp.array(ptab["stats"]), "league": jnp.array(ptab["league"]),
        "hand": jnp.array(ptab["hand"]), "unknown_index": ptab["unknown_index"],
        "bat_hand": np.asarray(ptab["bat_hand"]), "pit_hand": np.asarray(ptab["pit_hand"]),
        "_engine": engine, "_hook_dists": hook_dists,
    }
    model_kwargs = {name: True for name, value in {
        "outcome_only": args.outcome_only, "fatigue": args.fatigue, "platoon": args.platoon,
    }.items() if value}
    if args.pitchformer:
        model_kwargs.update(
            pitchformer=True,
            pitchformer_dim=args.pitchformer_dim,
            pitchformer_layers=args.pitchformer_layers,
            pitchformer_heads=args.pitchformer_heads,
            pitchformer_dropout=args.pitchformer_dropout,
        )
    model_fn = partial(pa_model, **model_kwargs) if model_kwargs else pa_model

    away_draws, home_draws = [], []
    master_key = jax.random.PRNGKey(args.seed)
    t0 = time.time()
    chunks = [games[i:i + args.batch] for i in range(0, len(games), args.batch)]
    for batch_number, game_chunk in enumerate(chunks):
        away, home = simulate_score_draws(
            model_fn, params, pt, game_chunk, jax.random.fold_in(master_key, batch_number),
            args.samples, seed=args.seed + batch_number, platoon=args.platoon,
            pitchformer=args.pitchformer,
        )
        away_draws.append(away)
        home_draws.append(home)
        if batch_number % 10 == 0 or batch_number + 1 == len(chunks):
            print(f"  batch {batch_number + 1}/{len(chunks)} elapsed={time.time() - t0:.0f}s", flush=True)

    sim_away = np.concatenate(away_draws, axis=1)  # (samples, games)
    sim_home = np.concatenate(home_draws, axis=1)
    assert sim_home.shape == (args.samples, len(games))

    sim_total = (sim_home + sim_away).reshape(-1)
    real_total = np.tile(real_home + real_away, args.samples)
    metrics = game_run_metrics(sim_total, real_total)

    print(f"\n=== Game-level metrics ({len(games)} games × {args.samples} generated draws) ===", flush=True)
    print(f"  {'':12s} {'real':>8s} {'sim':>8s}", flush=True)
    print(f"  {'home runs':12s} {real_home.mean():8.2f} {sim_home.mean():8.2f}", flush=True)
    print(f"  {'away runs':12s} {real_away.mean():8.2f} {sim_away.mean():8.2f}", flush=True)
    print(f"  {'home-win %':12s} {np.mean(real_home > real_away) * 100:8.1f} "
          f"{np.mean(sim_home > sim_away) * 100:8.1f}", flush=True)
    print(f"  {'margin std':12s} {np.std(real_home - real_away):8.2f} "
          f"{np.std((sim_home - sim_away).reshape(-1)):8.2f}", flush=True)
    print(f"  {'total mean':12s} {(real_home + real_away).mean():8.2f} {sim_total.mean():8.2f}", flush=True)
    print(f"\n  total-runs KL = {metrics['kl_run_distribution']:.5f}  "
          f"margin |bias| = {abs(np.mean(sim_home - sim_away) - np.mean(real_home - real_away)):.3f}", flush=True)


if __name__ == "__main__":
    main()
