"""DiamondWorldJAX v0 evaluation.

Loads a legacy pitch-level checkpoint for posterior-predictive diagnostics.

This script is intentionally *not* a valid game simulator: its old
posterior-predictive path retains real test-game pitch slots and terminal masks.
It therefore refuses to report game-simulation metrics unless the caller opts
into the explicitly labelled legacy diagnostic mode. Use ``eval_games.py`` for
the causal PA outcome simulator.

Usage
-----
    python -m diamondworldjax.scripts.eval_v0 \
        --ckpt checkpoints/dwjax_v0/dwjax_step_0005000.pkl
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from diamondworldjax.paths import processed_root, checkpoints_root, results_root

_DATA_ROOT    = processed_root()
_RESULTS_DIR  = results_root()
_DEFAULT_CKPT = checkpoints_root() / "dwjax_v0" / "dwjax_step_0005000.pkl"

TEST_SEASONS    = [2023, 2024]
SAMPLER_SEASONS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]

# Re-use train_v0's player-table builder so the registry matches training.
from diamondworldjax.scripts.train_v0 import (
    _build_player_table,
    _make_batch,
)


def _build_chunks(pitches, games_per_batch: int) -> list[np.ndarray]:
    """Split unique game_pks into fixed-size chunks (no shuffle: stable for eval)."""
    game_ids = pitches["game_pk"].unique().to_numpy()
    return [
        game_ids[i:i + games_per_batch]
        for i in range(0, len(game_ids), games_per_batch)
    ]


def _observed_runs_per_game(pitches) -> np.ndarray:
    """Real run total per game from the test data (home + away combined)."""
    import polars as pl
    df = (
        pitches.filter(pl.col("pa_terminal"))
        .group_by("game_pk")
        .agg(pl.col("runs_scored").sum().alias("runs"))
        .sort("game_pk")
    )
    return df["runs"].to_numpy().astype(float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",  type=Path, default=_DEFAULT_CKPT)
    parser.add_argument("--batch", type=int,  default=32, help="games per batch")
    parser.add_argument("--samples", type=int, default=8, help="MC samples per batch")
    parser.add_argument("--seed",  type=int,  default=0)
    parser.add_argument("--out",   type=Path,
                        default=_RESULTS_DIR / "dwjax_v0.json")
    parser.add_argument("--limit-games", type=int, default=0,
                        help="If >0, evaluate only this many games (debug)")
    parser.add_argument("--legacy-hybrid", action="store_true",
                        help="Removed: retained only for CLI compatibility.")
    args = parser.parse_args()

    print("Importing JAX + NumPyro...", flush=True)
    import jax
    import jax.numpy as jnp
    print(f"  JAX devices: {jax.devices()}", flush=True)

    import polars as pl
    from diamondworldjax.data.pipeline import load_seasons
    from diamondworldjax.model.joint import diamondworld_model
    from diamondworldjax.simulate.rollout import autoregressive_joint_rollout_samples
    from diamondworldjax.eval.calibration import game_run_metrics
    from diamondworldjax.train.svi import make_player_skills_guide

    print(f"Loading checkpoint: {args.ckpt}", flush=True)
    with open(args.ckpt, "rb") as f:
        ckpt = pickle.load(f)
    params = ckpt["params"]
    step   = ckpt.get("step", "?")
    print(f"  Loaded params from step {step} ({len(params)} param entries)", flush=True)

    # Player registry must come from training-era seasons so ids line up with the
    # checkpoint's learned embeddings.
    print(f"Building player table from training seasons {SAMPLER_SEASONS}...", flush=True)
    train_pitches = load_seasons(SAMPLER_SEASONS, data_root=_DATA_ROOT)
    player_table_np = _build_player_table(train_pitches)
    P = len(player_table_np["all_ids"])
    print(f"  {P:,} players in registry.", flush=True)
    guide = make_player_skills_guide(P)
    del train_pitches  # free memory; we only need the id_to_idx map going forward

    print(f"Loading test seasons {TEST_SEASONS}...", flush=True)
    test_pitches = load_seasons(TEST_SEASONS, data_root=_DATA_ROOT)
    print(f"  {len(test_pitches):,} test pitches.", flush=True)

    # Trim for debug runs.
    if args.limit_games > 0:
        keep = test_pitches["game_pk"].unique().to_numpy()[: args.limit_games]
        test_pitches = test_pitches.filter(pl.col("game_pk").is_in(keep.tolist()))
        print(f"  Limited to first {args.limit_games} games "
              f"→ {len(test_pitches):,} pitches.", flush=True)

    obs_runs = _observed_runs_per_game(test_pitches)
    print(f"  Observed runs/game: mean={obs_runs.mean():.2f}, "
          f"std={obs_runs.std():.2f}, n_games={len(obs_runs)}", flush=True)

    chunks = _build_chunks(test_pitches, args.batch)
    print(f"  {len(chunks)} batches of up to {args.batch} games each.", flush=True)

    rng = jax.random.PRNGKey(args.seed)
    sim_runs_per_game: list[np.ndarray] = []

    t0 = time.time()
    for b_idx, chunk in enumerate(chunks):
        batch, player_table = _make_batch(
            test_pitches, chunk, player_table_np["id_to_idx"], player_table_np
        )
        if batch is None:
            continue

        rng, key = jax.random.split(rng)
        samples = autoregressive_joint_rollout_samples(
            diamondworld_model, guide, params,
            batch, player_table, key, num_samples=args.samples,
        )
        per_game = samples["runs_scored"].sum(axis=-1)  # (samples, games)
        per_game_np = np.asarray(per_game).mean(axis=0)                # (B,)
        sim_runs_per_game.append(per_game_np)

        if b_idx % 5 == 0 or b_idx + 1 == len(chunks):
            elapsed = time.time() - t0
            done = sum(len(x) for x in sim_runs_per_game)
            print(f"  batch {b_idx+1:4d}/{len(chunks)}  "
                  f"games_simulated={done}  elapsed={elapsed:.0f}s", flush=True)

    sim_runs = np.concatenate(sim_runs_per_game, axis=0).astype(float)
    print(f"\nSimulated runs/game: mean={sim_runs.mean():.2f}, "
          f"std={sim_runs.std():.2f}, n={len(sim_runs)}", flush=True)

    # Truncate to the smaller of obs / sim (game alignment isn't required since
    # we compare distributions, but matching n is cleaner).
    n = min(len(obs_runs), len(sim_runs))
    metrics = game_run_metrics(sim_runs[:n], obs_runs[:n])

    print("\n=== JAX dwjax_v0 game-level metrics ===")
    for k, v in metrics.items():
        print(f"  {k:24s} = {v:.5f}")

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved → {args.out}", flush=True)

    # Also print the comparison row in the same format the PyTorch eval uses.
    print("\n=== comparison row (paste into the leaderboard) ===")
    print(
        f"dwjax_v0_step{step}       | "
        f"{metrics['kl_run_distribution']:8.5f}        | "
        f"{metrics['wasserstein_runs']:8.5f}         | "
        f"{metrics['mean_rg_error']:8.5f}         | "
        f"{metrics['variance_error']:8.5f}         | "
        f"{metrics['p0_error']:8.5f}         | "
        f"{metrics['p5_plus_error']:8.5f}         | "
        f"{metrics['p8_plus_error']:8.5f}"
    )


if __name__ == "__main__":
    main()
