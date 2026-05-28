"""DiamondWorldJAX v0 training entry point.

Usage
-----
    python -m diamondworldjax.scripts.train_v0 [--seasons 2015..2022] [--steps 50000]

Trains the joint DiamondWorldJAX model via SVI with an
AutoLowRankMultivariateNormal guide and saves checkpoints to
checkpoints/dwjax_v0/.
"""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np

_DATA_ROOT   = Path("/scratch/lblommes/diamondworld/data/processed")
_CKPT_DIR    = Path("/scratch/lblommes/diamondworld/checkpoints/dwjax_v0")
_LOG_PATH    = Path("/scratch/lblommes/diamondworld/eval/results/dwjax_v0_elbo.json")

TRAIN_SEASONS = list(range(2015, 2023))
TEST_SEASONS  = [2023, 2024]

# Default player stats dimension — updated once player table is built
DEFAULT_F_PLAYER = 16


def _build_player_table(pitches) -> dict:
    """
    Build a minimal player stats table from pitch data.

    Computes per-player seasonal stats (batting average proxy, walk rate,
    K rate, HR rate) from terminal PA rows.
    """
    import polars as pl

    terminal = pitches.filter(pl.col("pa_terminal"))

    # Unique player IDs from both pitcher and batter columns
    pitcher_col = "pitcher_id" if "pitcher_id" in pitches.columns else "pitcher_idx"
    batter_col  = "batter_id"  if "batter_id"  in pitches.columns else "batter_idx"

    all_ids = np.unique(np.concatenate([
        pitches[pitcher_col].to_numpy(),
        pitches[batter_col].to_numpy(),
    ])).astype(np.int32)
    P = len(all_ids)
    id_to_idx = {int(pid): i for i, pid in enumerate(all_ids)}

    # Compute per-player stats (log counts as features)
    stats = np.zeros((P, DEFAULT_F_PLAYER), dtype=np.float32)
    league = np.zeros(P, dtype=np.int32)   # all MLB for now
    hand   = np.zeros(P, dtype=np.int32)   # default R

    if "pa_outcome" in terminal.columns:
        for row in terminal.filter(pl.col("pa_outcome").is_not_null()).iter_rows(named=True):
            bid  = row.get(batter_col, 0)
            bidx = id_to_idx.get(int(bid), 0)
            outcome = row.get("pa_outcome", "")
            if outcome in ("1B", "2B", "3B", "HR"):
                stats[bidx, 0] += 1
            elif outcome in ("BB", "HBP"):
                stats[bidx, 1] += 1
            elif outcome in ("K",):
                stats[bidx, 2] += 1
            elif outcome == "HR":
                stats[bidx, 3] += 1
            stats[bidx, 4] += 1  # total PA

    # Normalise counts to rates
    pa_count = np.maximum(stats[:, 4:5], 1)
    stats[:, :4] /= pa_count
    # Fill unused dims with zeros (they will be learned)

    return {
        "stats":   stats,
        "league":  league,
        "hand":    hand,
        "id_to_idx": id_to_idx,
        "all_ids": all_ids,
    }


def _map_player_ids(batch_raw: dict, id_to_idx: dict) -> dict:
    """Remap raw player IDs in batch to player-table indices."""
    import jax.numpy as jnp

    def remap(arr):
        arr_np = np.array(arr)
        out    = np.vectorize(lambda x: id_to_idx.get(int(x), 0))(arr_np)
        return jnp.array(out.astype(np.int32))

    batch_raw["pitcher_ids"] = remap(batch_raw["pitcher_ids"])
    batch_raw["batter_ids"]  = remap(batch_raw["batter_ids"])
    return batch_raw


def _build_batches(pitches, id_to_idx: dict, games_per_batch: int = 32):
    """Yield (batch, player_table) tuples by splitting pitches into game groups."""
    import polars as pl
    from diamondworldjax.data.batching import build_batch
    import jax.numpy as jnp

    game_ids = pitches["game_pk"].unique().to_numpy()
    np.random.shuffle(game_ids)

    chunks = [
        game_ids[i:i + games_per_batch]
        for i in range(0, len(game_ids), games_per_batch)
    ]
    return chunks


def _make_batch(pitches, game_id_chunk, id_to_idx, player_table_np):
    """Build a single batch dict from a chunk of game IDs."""
    import polars as pl
    import jax.numpy as jnp
    from diamondworldjax.data.batching import build_batch

    chunk_df = pitches.filter(pl.col("game_pk").is_in(game_id_chunk.tolist()))
    if len(chunk_df) == 0:
        return None, None

    batch = build_batch(chunk_df)
    batch = _map_player_ids(batch, id_to_idx)

    pt = {
        "stats":  jnp.array(player_table_np["stats"]),
        "league": jnp.array(player_table_np["league"]),
        "hand":   jnp.array(player_table_np["hand"]),
    }
    return batch, pt


def _infinite_batch_iter(pitches, chunks, id_to_idx, player_table_np):
    """Cycle infinitely over shuffled game chunks, yielding (batch, player_table)."""
    for chunk in itertools.cycle(chunks):
        result = _make_batch(pitches, chunk, id_to_idx, player_table_np)
        if result[0] is not None:
            yield result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",   type=int, default=50_000)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--rank",    type=int, default=20)
    parser.add_argument("--seed",    type=int, default=0)
    parser.add_argument("--batch",   type=int, default=32,
                        help="Games per mini-batch")
    parser.add_argument("--resume",  type=str, default=None,
                        help="Path to checkpoint .pkl to resume from")
    args = parser.parse_args()

    print("Importing JAX + NumPyro...", flush=True)
    import jax
    print(f"  JAX devices: {jax.devices()}", flush=True)

    import polars as pl
    from diamondworldjax.model.joint import diamondworld_model
    from diamondworldjax.train.svi import train
    from diamondworldjax.data.pipeline import load_seasons

    print(f"Loading training seasons {TRAIN_SEASONS}...", flush=True)
    pitches = load_seasons(TRAIN_SEASONS, data_root=_DATA_ROOT)
    print(f"  {len(pitches):,} pitches loaded.", flush=True)

    print("Building player table...", flush=True)
    player_table_np = _build_player_table(pitches)
    P = len(player_table_np["all_ids"])
    print(f"  {P:,} unique players.", flush=True)

    print(f"Building batch iterator (batch={args.batch} games)...", flush=True)
    chunks = _build_batches(pitches, player_table_np["id_to_idx"], args.batch)
    batch_iter = _infinite_batch_iter(
        pitches, chunks, player_table_np["id_to_idx"], player_table_np
    )

    print(f"Starting SVI: {args.steps} steps, lr={args.lr}, rank={args.rank}", flush=True)
    t0 = time.time()

    svi_state, guide, losses = train(
        model       = diamondworld_model,
        batch_iter  = batch_iter,
        n_steps     = args.steps,
        rank        = args.rank,
        lr          = args.lr,
        seed        = args.seed,
        ckpt_dir    = _CKPT_DIR,
        log_path    = _LOG_PATH,
        resume_path = args.resume,
    )

    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed/3600:.2f}h.", flush=True)
    print(f"Final ELBO = {-losses[-1]:.2f}", flush=True)
    print(f"Checkpoint dir: {_CKPT_DIR}", flush=True)


if __name__ == "__main__":
    main()
