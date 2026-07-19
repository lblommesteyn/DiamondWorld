"""DiamondWorldJAX PA-level model training entry point.

Usage
-----
    python -m diamondworldjax.scripts.train_pa [--steps 50000] [--batch 64]

Trains the PA-level model via SVI and saves checkpoints to
checkpoints/dwjax_pa/.
"""
from __future__ import annotations

import argparse
import itertools
import time
from functools import partial
from pathlib import Path

import numpy as np

from diamondworldjax.paths import processed_root, checkpoints_root, results_root

_DATA_ROOT = processed_root()
_CKPT_DIR  = checkpoints_root() / "dwjax_pa_v5"
_LOG_PATH  = results_root() / "dwjax_pa_v5_elbo.json"

TRAIN_SEASONS = list(range(2015, 2023))
DEFAULT_F_PLAYER = 16


def _build_player_table(pitches) -> dict:
    import polars as pl

    terminal = pitches.filter(pl.col("pa_terminal"))
    pitcher_col = "pitcher_id" if "pitcher_id" in pitches.columns else "pitcher_idx"
    batter_col  = "batter_id"  if "batter_id"  in pitches.columns else "batter_idx"

    all_ids = np.unique(np.concatenate([
        pitches[pitcher_col].to_numpy(),
        pitches[batter_col].to_numpy(),
    ])).astype(np.int32)
    P = len(all_ids)
    id_to_idx = {int(pid): i for i, pid in enumerate(all_ids)}

    stats  = np.zeros((P, DEFAULT_F_PLAYER), dtype=np.float32)
    league = np.zeros(P, dtype=np.int32)
    hand   = np.zeros(P, dtype=np.int32)

    if "pa_outcome" in terminal.columns:
        for row in terminal.filter(pl.col("pa_outcome").is_not_null()).iter_rows(named=True):
            bid  = row.get(batter_col, 0)
            bidx = id_to_idx.get(int(bid), 0)
            outcome = row.get("pa_outcome", "")
            if outcome in ("1B", "2B", "3B", "HR"):
                stats[bidx, 0] += 1
            if outcome in ("BB", "HBP"):
                stats[bidx, 1] += 1
            if outcome == "K":
                stats[bidx, 2] += 1
            if outcome == "HR":
                stats[bidx, 3] += 1
            stats[bidx, 4] += 1

    pa_count = np.maximum(stats[:, 4:5], 1)
    stats[:, :4] /= pa_count

    # Per-player modal handedness for the simulator + the hand embedding.
    # bat_hand: modal batting side (stand); pit_hand: modal throw hand (p_throws).
    # R=1, L=0, unknown=0.5 (a batter who never appears keeps 0.5).
    bat_hand = np.full(P, 0.5, dtype=np.float32)
    pit_hand = np.full(P, 0.5, dtype=np.float32)
    bat_hand_col = "batter_hand" if "batter_hand" in terminal.columns else "stand"
    pit_hand_col = "pitcher_hand" if "pitcher_hand" in terminal.columns else "p_throws"
    if bat_hand_col in terminal.columns:
        m = (terminal.group_by(batter_col)
             .agg((pl.col(bat_hand_col) == "R").mean().alias("r")))
        for row in m.iter_rows(named=True):
            i = id_to_idx.get(int(row[batter_col]), None)
            if i is not None and row["r"] is not None:
                bat_hand[i] = 1.0 if row["r"] >= 0.5 else 0.0
    if pit_hand_col in terminal.columns:
        m = (terminal.group_by(pitcher_col)
             .agg((pl.col(pit_hand_col) == "R").mean().alias("r")))
        for row in m.iter_rows(named=True):
            i = id_to_idx.get(int(row[pitcher_col]), None)
            if i is not None and row["r"] is not None:
                pit_hand[i] = 1.0 if row["r"] >= 0.5 else 0.0
    # hand embedding input (int 0/1): pitchers use throw hand, else batting side.
    hand = np.where(pit_hand != 0.5, pit_hand, bat_hand)
    hand = (hand >= 0.5).astype(np.int32)

    return {"stats": stats, "league": league, "hand": hand,
            "bat_hand": bat_hand, "pit_hand": pit_hand,
            "id_to_idx": id_to_idx, "all_ids": all_ids}


def _build_park_index(pitches) -> dict:
    """Dense park_id -> park_idx mapping fit on the training seasons.

    Index 0 is reserved for unknown parks (test-era venues never seen in
    training), matching the unknown-player convention. Deterministic given the
    training seasons (sorted unique park_id), so eval/sim scripts can rebuild
    the identical mapping instead of persisting it in checkpoints.

    NOTE: before v9 this mapping did not exist anywhere in the pipeline, so
    `pa_batching._col("park_idx", 0.0)` silently filled 0 for every PA and the
    park embedding trained as a constant. Apply with `apply_park_idx` before
    building batches.
    """
    import polars as pl

    # park_id is a string venue code (e.g. "HOU") in the processed parquets.
    ids = sorted(pitches["park_id"].drop_nulls().unique().to_list())
    return {p: i + 1 for i, p in enumerate(ids)}


def apply_park_idx(df, park_map: dict):
    """Materialise the park_idx column pa_batching expects."""
    import polars as pl

    return df.with_columns(
        pl.col("park_id")
        .replace_strict(park_map, default=0, return_dtype=pl.Int32)
        .alias("park_idx")
    )


def _map_player_ids(batch: dict, id_to_idx: dict) -> dict:
    import jax.numpy as jnp

    def remap(arr):
        arr_np = np.array(arr)
        out = np.vectorize(lambda x: id_to_idx.get(int(x), 0))(arr_np)
        return jnp.array(out.astype(np.int32))

    batch["pitcher_ids"] = remap(batch["pitcher_ids"])
    batch["batter_ids"]  = remap(batch["batter_ids"])
    return batch


def _make_pa_batch(pa_df, game_id_chunk, id_to_idx, player_table_np):
    import polars as pl
    import jax.numpy as jnp
    from diamondworldjax.data.pa_batching import build_pa_batch

    chunk_df = pa_df.filter(pl.col("game_pk").is_in(game_id_chunk.tolist()))
    if len(chunk_df) == 0:
        return None, None

    batch = build_pa_batch(chunk_df)
    batch = _map_player_ids(batch, id_to_idx)

    pt = {
        "stats":  jnp.array(player_table_np["stats"]),
        "league": jnp.array(player_table_np["league"]),
        "hand":   jnp.array(player_table_np["hand"]),
    }
    return batch, pt


def _infinite_batch_iter(pa_df, chunks, id_to_idx, player_table_np):
    for chunk in itertools.cycle(chunks):
        result = _make_pa_batch(pa_df, chunk, id_to_idx, player_table_np)
        if result[0] is not None:
            yield result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",  type=int,   default=50_000)
    parser.add_argument("--lr",     type=float, default=3e-4)
    parser.add_argument("--seed",   type=int,   default=0)
    parser.add_argument("--batch",  type=int,   default=64,
                        help="Games per mini-batch (PA model is lightweight — 64 fits easily)")
    parser.add_argument("--resume",    type=str,   default=None)
    parser.add_argument("--ss-rate",   type=float, default=0.5,
                        help="Max scheduled sampling rate (0=disabled)")
    parser.add_argument("--ss-warmup", type=int,   default=20_000,
                        help="Steps to ramp ss_rate from 0 to ss_max_rate")
    parser.add_argument("--cosine-alpha", type=float, default=0.0,
                        help="Cosine LR floor as fraction of init LR (0=decay to 0, 0.1=decay to 10%%)")
    parser.add_argument("--outcome-only", action="store_true",
                        help="Train the outcome-only model (v6): single pa_outcome head, "
                             "runs + base_state handled by the rules engine at eval.")
    parser.add_argument("--engine-ss", action="store_true",
                        help="Phase-2 DAgger: anneal in self-generated base states derived "
                             "from the rules engine (legal by construction). Use with "
                             "--ss-rate (e.g. 0.25) and --outcome-only, resuming a v6 ckpt.")
    parser.add_argument("--fatigue", action="store_true",
                        help="Phase-4: add pitcher cumulative game pitch count to the model "
                             "context (STATE_DIM 8 -> 9). Fresh train (changes context dim).")
    parser.add_argument("--platoon", action="store_true",
                        help="Add batter side + pitcher throw hand (real per-PA stand/p_throws) "
                             "to the context (+2 dims). Fresh train (changes context dim).")
    parser.add_argument("--tag", type=str, default=None,
                        help="Checkpoint/log dir tag override (e.g. v6).")
    args = parser.parse_args()

    global _CKPT_DIR, _LOG_PATH
    if args.tag:
        _CKPT_DIR = checkpoints_root() / f"dwjax_pa_{args.tag}"
        _LOG_PATH = results_root() / f"dwjax_pa_{args.tag}_elbo.json"
    elif args.outcome_only:
        _CKPT_DIR = checkpoints_root() / "dwjax_pa_v6"
        _LOG_PATH = results_root() / "dwjax_pa_v6_elbo.json"

    print("Importing JAX + NumPyro...", flush=True)
    import jax
    print(f"  JAX devices: {jax.devices()}", flush=True)

    import polars as pl
    from diamondworldjax.data.pipeline import load_seasons
    from diamondworldjax.model.pa_model import pa_model
    from diamondworldjax.train.svi import train

    print(f"Loading training seasons {TRAIN_SEASONS}...", flush=True)
    pitches = load_seasons(TRAIN_SEASONS, data_root=_DATA_ROOT)
    print(f"  {len(pitches):,} pitches loaded.", flush=True)

    print("Building player table...", flush=True)
    player_table_np = _build_player_table(pitches)
    print(f"  {len(player_table_np['all_ids']):,} unique players.", flush=True)

    print("Filtering to PA-terminal rows...", flush=True)
    pa_df = pitches.filter(pl.col("pa_terminal"))
    park_map = _build_park_index(pitches)
    pa_df = apply_park_idx(pa_df, park_map)
    print(f"  {len(pa_df):,} plate appearances, {len(park_map)} parks.", flush=True)

    game_ids = pa_df["game_pk"].unique().to_numpy()
    np.random.shuffle(game_ids)
    chunks = [game_ids[i:i + args.batch] for i in range(0, len(game_ids), args.batch)]
    batch_iter = _infinite_batch_iter(pa_df, chunks, player_table_np["id_to_idx"], player_table_np)

    print(f"Starting SVI: {args.steps} steps, lr={args.lr}"
          f"{'  [outcome-only v6]' if args.outcome_only else ''}", flush=True)
    t0 = time.time()

    _mkw = {}
    if args.outcome_only:
        _mkw["outcome_only"] = True
    if args.fatigue:
        _mkw["fatigue"] = True
    if args.platoon:
        _mkw["platoon"] = True
    model_fn = partial(pa_model, **_mkw) if _mkw else pa_model

    svi_state, guide, losses = train(
        model            = model_fn,
        batch_iter       = batch_iter,
        n_steps          = args.steps,
        lr               = args.lr,
        seed             = args.seed,
        ckpt_dir         = _CKPT_DIR,
        log_path         = _LOG_PATH,
        resume_path      = args.resume,
        cosine_decay     = True,
        cosine_alpha     = args.cosine_alpha,
        ss_max_rate      = args.ss_rate,
        ss_warmup_steps  = args.ss_warmup,
        ss_start_step    = 0 if args.resume else 5_000,
        engine_ss        = args.engine_ss,
    )

    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed/3600:.2f}h.", flush=True)
    print(f"Final ELBO = {-losses[-1]:.2f}", flush=True)
    print(f"Checkpoint dir: {_CKPT_DIR}", flush=True)


if __name__ == "__main__":
    main()
