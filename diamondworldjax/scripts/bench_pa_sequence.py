"""Benchmark: GRU vs Transformer sequence inference for the PA model.

Compares wall-clock time for a simulated game loop (~70 PAs) using:

  A) PASequenceInference with pa_arch="transformer" (history-buffer, no NumPyro)
  B) PASequenceInference with pa_arch="gru" (O(1) step, no NumPyro)
  C) Baseline: full NumPyro model call per PA (current pitchformer path)

All three use the same synthetic weights so the comparison is apples-to-apples
on forward-pass and overhead cost.  No real checkpoint is required.

Usage:
    python -m diamondworldjax.scripts.bench_pa_sequence [--games 64] [--pas 70]
"""
from __future__ import annotations

import argparse
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpyro.handlers as nh

from diamondworldjax.model.embeddings import PlayerSeasonEncoder, SkillFusionLayer
from diamondworldjax.model.pa_model import (
    N_PA_OUTCOMES,
    PAOutcomeHeadV6,
    ParkEmbedding,
    pa_model,
)
from diamondworldjax.model.pa_transformer import (
    PAGRU, PATransformer,
)
from diamondworldjax.model.pa_inference import (
    PASequenceInference,
    build_pa_sequence_inference,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_params(
    key: jax.Array,
    n_players: int = 100,
    d_model: int = 128,
    n_layers: int = 2,
    n_heads: int = 4,
    fatigue: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build synthetic params dict and player_table matching the PA model."""
    keys = dict(zip(
        ["enc", "fus", "park", "head", "gru", "xfm"],
        jax.random.split(key, 6),
    ))

    f_player = 8
    stats = jax.random.normal(keys["enc"], (n_players, f_player))
    league = jnp.zeros(n_players, dtype=jnp.int32)
    hand = jnp.zeros(n_players, dtype=jnp.int32)
    skills = jax.random.normal(keys["enc"], (n_players, 32))

    enc = PlayerSeasonEncoder(f_player=f_player)
    enc_params = enc.init(keys["enc"], stats, league, hand)["params"]
    det = enc.apply({"params": enc_params}, stats, league, hand)
    fusion = SkillFusionLayer()
    fus_params = fusion.init(keys["fus"], det, skills)["params"]
    park = ParkEmbedding()
    park_params = park.init(keys["park"], jnp.arange(30, dtype=jnp.int32))["params"]

    # Context dim: 8 base + 1 fatigue + 64 pitcher + 64 batter + 8 park = 145
    context_dim = 8 + (1 if fatigue else 0) + 64 + 64 + 8
    head = PAOutcomeHeadV6()
    # For pitchformer models, the head sees d_model, not context_dim
    head_params = head.init(keys["head"], jnp.zeros((2, d_model)))["params"]

    # GRU params
    gru = PAGRU(d_model=d_model, n_layers=n_layers)
    dummy_ctx = jnp.zeros((2, 10, context_dim))
    dummy_valid = jnp.ones((2, 10), dtype=jnp.bool_)
    gru_params = gru.init(keys["gru"], dummy_ctx, dummy_valid)["params"]

    # Transformer params
    xfm = PATransformer(d_model=d_model, n_layers=n_layers, n_heads=n_heads)
    xfm_params = xfm.init(keys["xfm"], dummy_ctx, dummy_valid)["params"]

    params = {
        "player_skills": skills,
        "player_encoder$params": enc_params,
        "player_encoder_skill_fusion$params": fus_params,
        "park_embedding$params": park_params,
        "pa_outcome_head_v6$params": head_params,
        "pa_gru$params": gru_params,
        "pa_transformer$params": xfm_params,
    }
    table = {"stats": stats, "league": league, "hand": hand}
    return params, table


def _make_batch(key: jax.Array, B: int, T: int) -> dict[str, jax.Array]:
    """Synthetic PA batch for the NumPyro baseline."""
    keys = jax.random.split(key, 4)
    return {
        "pa_valid": jnp.ones((B, T), dtype=jnp.bool_),
        "inning": jax.random.uniform(keys[0], (B, T)),
        "half": jnp.zeros((B, T)),
        "outs": jax.random.uniform(keys[1], (B, T)),
        "base_state": jnp.zeros((B, T)),
        "score_diff": jnp.zeros((B, T)),
        "tto": jax.random.uniform(keys[2], (B, T)),
        "shift_restricted": jnp.ones((B, T)),
        "pitch_clock": jnp.ones((B, T)),
        "pitch_count_game": jax.random.uniform(keys[3], (B, T)),
        "pitcher_ids": jnp.zeros((B, T), dtype=jnp.int32),
        "batter_ids": jnp.ones((B, T), dtype=jnp.int32),
        "park_ids": jnp.zeros((B, T), dtype=jnp.int32),
        "bat_side": jnp.zeros((B, T)),
        "pit_hand": jnp.zeros((B, T)),
        "pa_outcome": jnp.zeros((B, T), dtype=jnp.int32),
        "season": jnp.full((B, T), 2024, dtype=jnp.int32),
    }


def _pa_features_at(batch: dict, t: int) -> dict[str, jax.Array]:
    """Extract single-timestep kwargs for PASequenceInference.step."""
    return {
        "inning": batch["inning"][:, t],
        "half": batch["half"][:, t],
        "outs": batch["outs"][:, t],
        "base_state": batch["base_state"][:, t],
        "score_diff": batch["score_diff"][:, t],
        "tto": batch["tto"][:, t],
        "shift_restricted": batch["shift_restricted"][:, t],
        "pitch_clock": batch["pitch_clock"][:, t],
        "pitch_count_game": batch["pitch_count_game"][:, t],
        "pitcher_ids": batch["pitcher_ids"][:, t],
        "batter_ids": batch["batter_ids"][:, t],
        "park_ids": batch["park_ids"][:, t],
        "bat_side": batch["bat_side"][:, t],
        "pit_hand": batch["pit_hand"][:, t],
    }


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def _bench_sequence(
    adapter: PASequenceInference, batch: dict, T: int, label: str,
    warmup: int = 2, repeats: int = 5,
) -> float:
    B = batch["inning"].shape[0]

    for _ in range(warmup):
        carry = adapter.init_carry(B)
        for t in range(T):
            carry, logits = adapter.step(carry, **_pa_features_at(batch, t))
        logits.block_until_ready()

    times = []
    for _ in range(repeats):
        carry = adapter.init_carry(B)
        t0 = time.perf_counter()
        for t in range(T):
            carry, logits = adapter.step(carry, **_pa_features_at(batch, t))
        logits.block_until_ready()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    best = min(times)
    print(f"  {label:30s}  avg={avg*1000:8.1f}ms  best={best*1000:8.1f}ms  ({B} games x {T} PAs)")
    return avg


def _bench_numpyro(
    params: dict, table: dict, batch: dict, T: int,
    warmup: int = 1, repeats: int = 3,
) -> float:
    """Baseline: call full NumPyro model per PA (current pitchformer path)."""
    B = batch["inning"].shape[0]

    def _run_one():
        for t in range(T):
            sub_batch = {k: v[:, :t + 1] for k, v in batch.items()}
            with nh.seed(rng_seed=jax.random.PRNGKey(t)):
                with nh.substitute(data=params):
                    with nh.trace() as tr:
                        pa_model(
                            sub_batch, table,
                            teacher_force=False,
                            outcome_only=True,
                            fatigue=True,
                            pitchformer=True,
                            pa_arch="gru",
                            pitchformer_dim=128,
                            pitchformer_layers=2,
                        )
            logits = tr["pa_outcome"]["fn"].logits[:, t, :]
        logits.block_until_ready()

    for _ in range(warmup):
        _run_one()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _run_one()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    best = min(times)
    print(f"  {'NumPyro baseline (GRU)':30s}  avg={avg*1000:8.1f}ms  best={best*1000:8.1f}ms  ({B} games x {T} PAs)")
    return avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark PA sequence inference")
    parser.add_argument("--games", type=int, default=64, help="batch size (games)")
    parser.add_argument("--pas", type=int, default=70, help="PAs per game")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--skip-numpyro", action="store_true",
                        help="skip the slow NumPyro baseline")
    args = parser.parse_args()

    print(f"Building synthetic params (d_model={args.d_model}, n_layers={args.n_layers})...")
    key = jax.random.PRNGKey(42)
    params, table = _make_params(
        key, d_model=args.d_model, n_layers=args.n_layers,
    )

    batch = _make_batch(jax.random.PRNGKey(99), args.games, args.pas)

    print(f"\nBenchmark: {args.games} games x {args.pas} PAs")
    print("=" * 70)

    gru_adapter = build_pa_sequence_inference(
        params, table,
        pa_arch="gru", d_model=args.d_model, n_layers=args.n_layers,
        outcome_only=True, fatigue=True,
    )
    gru_time = _bench_sequence(gru_adapter, batch, args.pas, "GRU (scan step)")

    xfm_adapter = build_pa_sequence_inference(
        params, table,
        pa_arch="transformer", d_model=args.d_model, n_layers=args.n_layers,
        outcome_only=True, fatigue=True,
    )
    xfm_time = _bench_sequence(xfm_adapter, batch, args.pas, "Transformer (history buf)")

    if not args.skip_numpyro:
        npy_time = _bench_numpyro(params, table, batch, args.pas)
        print()
        print(f"  Speedup vs NumPyro:  GRU {npy_time/gru_time:.1f}x,  Transformer {npy_time/xfm_time:.1f}x")
    else:
        print(f"\n  GRU/Transformer ratio: {xfm_time/gru_time:.2f}x")

    print()


if __name__ == "__main__":
    main()
