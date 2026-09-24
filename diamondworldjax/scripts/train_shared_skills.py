"""Train PA and combined pitch models against one shared player hierarchy.

The model learns ``shared_player_skills`` plus ``pa_skill_residual`` and
``pitch_skill_residual``.  Both batches must use the same player-index table;
this script builds them from identical game chunks to enforce that invariant.
"""
from __future__ import annotations

import argparse
import itertools
from functools import partial

import numpy as np

from diamondworldjax.paths import checkpoints_root, processed_root, results_root
from diamondworldjax.scripts.train_pa import (
    _build_park_index,
    _build_player_table,
    _map_player_ids,
    apply_park_idx,
)



def _batch_iterator(pitches, pa_rows, chunks, id_to_idx, player_table_np):
    import jax.numpy as jnp
    import polars as pl
    from diamondworldjax.data.batching import build_batch
    from diamondworldjax.data.pa_batching import build_pa_batch

    player_table = {
        "stats": jnp.array(player_table_np["stats"]),
        "league": jnp.array(player_table_np["league"]),
        "hand": jnp.array(player_table_np["hand"]),
    }
    for game_ids in itertools.cycle(chunks):
        pitch_rows = pitches.filter(pl.col("game_pk").is_in(game_ids.tolist()))
        pa_chunk = pa_rows.filter(pl.col("game_pk").is_in(game_ids.tolist()))
        if not len(pitch_rows) or not len(pa_chunk):
            continue
        pitch_batch = _map_player_ids(build_batch(pitch_rows), id_to_idx)
        pa_batch = _map_player_ids(build_pa_batch(pa_chunk), id_to_idx)
        yield {"pa": pa_batch, "pitch": pitch_batch}, player_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=16,
                        help="Games per shared PA/pitch minibatch.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--residual-scale", type=float, default=0.35,
                        help="Prior standard deviation for PA/pitch-specific skill residuals.")
    parser.add_argument("--tag", default="shared_skills")
    parser.add_argument("--pa-pitchformer", action="store_true",
                        help="Use the causal PA transformer in the PA likelihood.")
    parser.add_argument("--pa-pitchformer-dim", type=int, default=128)
    parser.add_argument("--pa-pitchformer-layers", type=int, default=2)
    parser.add_argument("--pa-pitchformer-heads", type=int, default=4)
    # Player-table and head flags, mirrored from train_pa so a shared-skills run
    # can reproduce the v22 recipe exactly (ladder R3 must differ from R2 by the
    # shared latent only).
    parser.add_argument("--train-end", type=int, default=2022,
                        help="Last season included in training (inclusive); v22 uses 2023.")
    parser.add_argument("--contact-quality", action="store_true",
                        help="xBA-style stat columns 5-6 (see train_pa). Eval must match.")
    parser.add_argument("--recency-halflife", type=float, default=None,
                        help="Recency half-life in seasons for the player-table rates.")
    parser.add_argument("--per-stat-shrink", action="store_true",
                        help="Per-stat shrinkage in the player table (v22). Eval must match.")
    parser.add_argument("--fatigue", action="store_true",
                        help="Pitcher fatigue proxy in the PA head (v9+). Eval assumes it.")
    parser.add_argument("--outcome-only", action="store_true",
                        help="Single pa_outcome head (v6+); runs/bases from the rules "
                             "engine at eval. Production checkpoints v16/v22 use this.")
    args = parser.parse_args()
    if args.residual_scale <= 0:
        parser.error("--residual-scale must be positive")

    import polars as pl
    from diamondworldjax.data.pipeline import load_seasons
    from diamondworldjax.model.multitask import multitask_model
    from diamondworldjax.train.svi import train

    train_seasons = list(range(2015, args.train_end + 1))
    print(f"Loading training seasons {train_seasons}...", flush=True)
    pitches = load_seasons(train_seasons, data_root=processed_root())
    player_table_np = _build_player_table(pitches, recency_halflife=args.recency_halflife,
                                          contact_quality=args.contact_quality,
                                          per_stat_shrink=args.per_stat_shrink)
    park_map = _build_park_index(pitches)
    pa_rows = apply_park_idx(pitches.filter(pl.col("pa_terminal")), park_map)
    game_ids = pa_rows["game_pk"].unique().to_numpy()
    np.random.default_rng(args.seed).shuffle(game_ids)
    chunks = [game_ids[i:i + args.batch] for i in range(0, len(game_ids), args.batch)]
    batch_iter = _batch_iterator(
        pitches, pa_rows, chunks, player_table_np["id_to_idx"], player_table_np
    )
    kl_scale = args.batch / max(len(game_ids), 1)
    pa_kwargs = {}
    if args.outcome_only:
        pa_kwargs["outcome_only"] = True
    if args.fatigue:
        pa_kwargs["fatigue"] = True
    if args.pa_pitchformer:
        pa_kwargs.update(
            pitchformer=True,
            pitchformer_dim=args.pa_pitchformer_dim,
            pitchformer_layers=args.pa_pitchformer_layers,
            pitchformer_heads=args.pa_pitchformer_heads,
        )
    model = partial(
        multitask_model,
        kl_scale=kl_scale,
        residual_scale=args.residual_scale,
        pa_model_kwargs=pa_kwargs,
    )
    destination = checkpoints_root() / f"dwjax_{args.tag}"
    log_path = results_root() / f"dwjax_{args.tag}_elbo.json"
    print(f"Training shared hierarchy on {len(game_ids):,} games; KL scale={kl_scale:.6g}"
          f"{'  [outcome-only v6]' if args.outcome_only else ''}"
          f"{'  [fatigue]' if args.fatigue else ''}"
          f"  train_end={args.train_end} contact_quality={args.contact_quality}"
          f" recency_halflife={args.recency_halflife}", flush=True)
    train(
        model=model,
        batch_iter=batch_iter,
        n_steps=args.steps,
        lr=args.lr,
        seed=args.seed,
        ckpt_dir=destination,
        log_path=log_path,
        cosine_decay=True,
        shared_task_skills=True,
        kl_scale=kl_scale,
    )


if __name__ == "__main__":
    main()
