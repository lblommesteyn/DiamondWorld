"""DiamondWorldJAX PA-level model evaluation.

Two eval modes:
  conditioned  (default): uses real game states as context each PA — fast,
                           comparable to baselines.
  free-rollout (--free-rollout): feeds model's own base_state_after back in
                           as base_state for the next PA — true autoregressive;
                           slower (~5-10min for full test set at --samples 1).

Usage
-----
    python -m diamondworldjax.scripts.eval_pa \
        --ckpt checkpoints/dwjax_pa_v2/dwjax_step_0050000.pkl

    python -m diamondworldjax.scripts.eval_pa \
        --ckpt checkpoints/dwjax_pa_v2/dwjax_step_0050000.pkl \
        --free-rollout --samples 1
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from diamondworldjax.paths import processed_root, checkpoints_root, results_root

_DATA_ROOT    = processed_root()
_RESULTS_DIR  = results_root()
_DEFAULT_CKPT = checkpoints_root() / "dwjax_pa_v2" / "dwjax_step_0050000.pkl"

TRAIN_SEASONS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
TEST_SEASONS  = [2023, 2024]

from diamondworldjax.scripts.train_pa import (
    _build_player_table,
    _build_park_index,
    _map_player_ids,
    apply_park_idx,
)


def _observed_runs_per_game(pa_df) -> np.ndarray:
    import polars as pl
    df = (
        pa_df.group_by("game_pk")
        .agg(pl.col("runs_scored").sum().alias("runs"))
        .sort("game_pk")
    )
    return df["runs"].to_numpy().astype(float)


def _conditioned_sample(pa_model, params, batch, pt, rng_key, n_samples):
    """Sample runs using real game states as context (fast)."""
    import jax
    import jax.numpy as jnp
    import numpyro.handlers as nh

    B    = batch["pa_valid"].shape[0]
    valid = np.array(batch["pa_valid"])
    total = np.zeros(B)

    for _ in range(n_samples):
        rng_key, key = jax.random.split(rng_key)
        with nh.seed(rng_seed=key):
            with nh.substitute(data=params):
                with nh.trace() as tr:
                    pa_model(batch, pt, teacher_force=False)
        runs = np.array(tr["runs_scored"]["value"])  # (B, T)
        total += (runs * valid).sum(axis=1)

    return total / n_samples, rng_key


def _free_rollout_sample(pa_model, params, batch, pt, rng_key, n_samples):
    """Sample runs feeding base_state_after back as base_state (slow)."""
    import jax
    import jax.numpy as jnp
    import numpyro.handlers as nh

    B, T  = batch["pa_valid"].shape
    valid = np.array(batch["pa_valid"])  # (B, T)
    total = np.zeros(B)

    for _ in range(n_samples):
        # Start from real initial base states
        current_bs = np.array(batch["base_state"])  # (B, T) normalised 0-1

        sample_runs = np.zeros(B)
        for t in range(T):
            if not valid[:, t].any():
                break

            # Single-PA batch slice with model's predicted base_state
            t_batch = {}
            for k, v in batch.items():
                if isinstance(v, (np.ndarray,)) or hasattr(v, "ndim"):
                    arr = np.array(v)
                    if arr.ndim == 2:
                        t_batch[k] = jnp.array(arr[:, t:t+1])
                    else:
                        t_batch[k] = jnp.array(arr)
            t_batch["base_state"] = jnp.array(current_bs[:, t:t+1])

            rng_key, step_key = jax.random.split(rng_key)
            with nh.seed(rng_seed=step_key):
                with nh.substitute(data=params):
                    with nh.trace() as tr:
                        pa_model(t_batch, pt, teacher_force=False)

            runs_t    = np.array(tr["runs_scored"]["value"])[:, 0]       # (B,)
            bs_after_t = np.array(tr["base_state_after"]["value"])[:, 0] # (B,) int 0-7

            sample_runs += runs_t * valid[:, t]

            if t + 1 < T:
                # Feed model's base_state_after back, normalised to 0-1
                current_bs[:, t+1] = np.where(
                    valid[:, t], bs_after_t / 7.0, current_bs[:, t+1]
                )

        total += sample_runs

    return total / n_samples, rng_key


# Per-class logit recalibration = log(real_freq / model_freq), measured by
# diag_outcomes.py on v5@10K. Corrects the pa_outcome head's undersampling of
# extra-base hits. Order: K, BB, HBP, 1B, 2B, 3B, HR, out, E.
RECAL_VECTOR = np.array([
    np.log(22.65 / 27.39),  # K
    np.log(8.39 / 7.34),    # BB
    np.log(1.13 / 0.96),    # HBP
    np.log(14.17 / 11.35),  # 1B
    np.log(4.36 / 3.24),    # 2B
    np.log(0.38 / 0.41),    # 3B
    np.log(3.09 / 2.02),    # HR
    np.log(45.82 / 47.29),  # out
    0.0,                    # E (absent in data)
], dtype=np.float64)


def _engine_rollout_sample(pa_model, params, batch, pt, rng_key, n_samples, engine=None, recal=False):
    """Sample pa_outcome from the model; derive runs + base_state from the rules engine.

    Phase-1 rollout: the model predicts only the 9-way PA outcome. The empirical
    rules engine maps (base_state, outs, outcome) -> (runs, base_state_after),
    which is unbiased in aggregate (validated at -0.45% run bias). Base state is
    rolled out autoregressively and reset to empty at each half-inning boundary
    (detected from the real inning/half sequence).
    """
    import jax
    import jax.numpy as jnp
    import numpyro.handlers as nh

    B, T   = batch["pa_valid"].shape
    valid  = np.array(batch["pa_valid"])                       # (B, T)
    real_outs   = np.clip(np.rint(np.array(batch["outs"]) * 2.0), 0, 2).astype(np.int64)  # (B,T)
    inning_norm = np.array(batch["inning"])                    # (B, T)
    half_arr    = np.rint(np.array(batch["half"])).astype(np.int64)  # (B, T)
    rng_np = np.random.default_rng(0)

    total = np.zeros(B)
    for _ in range(n_samples):
        current_bs = np.array(batch["base_state"])             # (B, T) normalised 0-1
        sample_runs = np.zeros(B)
        for t in range(T):
            if not valid[:, t].any():
                break

            t_batch = {}
            for k, v in batch.items():
                if hasattr(v, "ndim"):
                    arr = np.array(v)
                    t_batch[k] = jnp.array(arr[:, t:t+1] if arr.ndim == 2 else arr)
            t_batch["base_state"] = jnp.array(current_bs[:, t:t+1])

            rng_key, step_key = jax.random.split(rng_key)
            with nh.seed(rng_seed=step_key):
                with nh.substitute(data=params):
                    with nh.trace() as tr:
                        pa_model(t_batch, pt, teacher_force=False)

            if recal:
                # Re-sample outcome from recalibrated logits (Gumbel-max).
                logits = np.array(tr["pa_outcome"]["fn"].logits)[:, 0, :]  # (B, 9)
                logits = logits + RECAL_VECTOR
                gumbel = rng_np.gumbel(size=logits.shape)
                outcome_t = np.argmax(logits + gumbel, axis=-1).astype(np.int64)
            else:
                outcome_t = np.array(tr["pa_outcome"]["value"])[:, 0].astype(np.int64)  # (B,)
            bs_int    = np.clip(np.rint(current_bs[:, t] * 7.0), 0, 7).astype(np.int64)

            eo = engine.sample(bs_int, real_outs[:, t], outcome_t, rng_np)
            sample_runs += eo["runs"] * valid[:, t]

            if t + 1 < T:
                # Reset bases at a half-inning boundary, else carry engine state.
                boundary = (inning_norm[:, t+1] != inning_norm[:, t]) | (half_arr[:, t+1] != half_arr[:, t])
                next_bs = np.where(boundary, 0, eo["bs_after"]).astype(np.float64) / 7.0
                current_bs[:, t+1] = np.where(valid[:, t], next_bs, current_bs[:, t+1])

        total += sample_runs

    return total / n_samples, rng_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",         type=Path,  default=_DEFAULT_CKPT)
    parser.add_argument("--batch",        type=int,   default=64)
    parser.add_argument("--samples",      type=int,   default=8)
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--out",          type=Path,  default=None)
    parser.add_argument("--limit-games",  type=int,   default=0)
    parser.add_argument("--free-rollout", action="store_true",
                        help="Feed base_state_after back autoregressively "
                             "(slower; use --samples 1 for speed)")
    parser.add_argument("--engine-rollout", action="store_true",
                        help="Sample only pa_outcome from the model; derive runs + "
                             "base_state from the empirical rules engine (Phase 1).")
    parser.add_argument("--recal", action="store_true",
                        help="Apply per-class logit recalibration to pa_outcome "
                             "(test whether outcome miscalibration is the gap).")
    parser.add_argument("--fatigue", action="store_true",
                        help="Checkpoint was trained with the pitcher-fatigue feature "
                             "(context dim 145, e.g. v8/v9).")
    parser.add_argument("--use-park", action="store_true",
                        help="Feed real park indices (v9+ checkpoints trained with the "
                             "park_idx fix; pre-v9 park embeddings trained on all-zeros).")
    parser.add_argument("--outcome-only", action="store_true",
                        help="Evaluate a v6 outcome-only checkpoint (single pa_outcome head).")
    args = parser.parse_args()

    mode = ("engine-rollout" if args.engine_rollout
            else "free-rollout" if args.free_rollout else "conditioned")
    if args.out is None:
        args.out = _RESULTS_DIR / f"dwjax_pa_{mode}.json"

    print("Importing JAX + NumPyro...", flush=True)
    import jax
    import jax.numpy as jnp
    print(f"  JAX devices: {jax.devices()}", flush=True)
    print(f"  Eval mode  : {mode}", flush=True)

    import polars as pl
    from diamondworldjax.data.pipeline import load_seasons
    from diamondworldjax.data.pa_batching import build_pa_batch
    from diamondworldjax.model.pa_model import pa_model
    from diamondworldjax.eval.calibration import game_run_metrics

    print(f"Loading checkpoint: {args.ckpt}", flush=True)
    with open(args.ckpt, "rb") as f:
        ckpt = pickle.load(f)
    params = ckpt["params"]
    step   = ckpt.get("step", "?")
    print(f"  Loaded step {step} ({len(params)} param entries)", flush=True)

    print(f"Building player table from {TRAIN_SEASONS}...", flush=True)
    train_pitches = load_seasons(TRAIN_SEASONS, data_root=_DATA_ROOT)
    player_table_np = _build_player_table(train_pitches)
    park_map = _build_park_index(train_pitches)
    P = len(player_table_np["all_ids"])
    print(f"  {P:,} players, {len(park_map)} parks.", flush=True)

    engine = None
    if args.engine_rollout:
        from diamondworldjax.sim.rules_engine import EmpiricalEngine
        print("  Fitting empirical rules engine on training PAs...", flush=True)
        engine = EmpiricalEngine().fit(train_pitches.filter(pl.col("pa_terminal")))
    del train_pitches

    print(f"Loading test seasons {TEST_SEASONS}...", flush=True)
    test_pitches = load_seasons(TEST_SEASONS, data_root=_DATA_ROOT)
    test_pa = test_pitches.filter(pl.col("pa_terminal"))
    if args.use_park:
        test_pa = apply_park_idx(test_pa, park_map)
    print(f"  {len(test_pa):,} test PAs.", flush=True)

    if args.limit_games > 0:
        keep = test_pa["game_pk"].unique().to_numpy()[:args.limit_games]
        test_pa = test_pa.filter(pl.col("game_pk").is_in(keep.tolist()))
        print(f"  Limited to {args.limit_games} games.", flush=True)

    obs_runs = _observed_runs_per_game(test_pa)
    print(f"  Observed: mean={obs_runs.mean():.2f}, std={obs_runs.std():.2f}, "
          f"n={len(obs_runs)}", flush=True)

    game_ids = test_pa["game_pk"].unique().to_numpy()
    chunks   = [game_ids[i:i + args.batch] for i in range(0, len(game_ids), args.batch)]
    print(f"  {len(chunks)} batches of up to {args.batch} games.", flush=True)

    pt = {
        "stats":  jnp.array(player_table_np["stats"]),
        "league": jnp.array(player_table_np["league"]),
        "hand":   jnp.array(player_table_np["hand"]),
    }

    from functools import partial as _partial
    _mkw = {}
    if args.outcome_only:
        _mkw["outcome_only"] = True
    if args.fatigue:
        _mkw["fatigue"] = True
    model_fn = _partial(pa_model, **_mkw) if _mkw else pa_model

    if args.engine_rollout:
        from functools import partial
        sample_fn = partial(_engine_rollout_sample, engine=engine, recal=args.recal)
    elif args.free_rollout:
        sample_fn = _free_rollout_sample
    else:
        sample_fn = _conditioned_sample

    rng = jax.random.PRNGKey(args.seed)
    sim_runs_per_game: list[np.ndarray] = []
    t0 = time.time()

    for b_idx, chunk in enumerate(chunks):
        chunk_df = test_pa.filter(pl.col("game_pk").is_in(chunk.tolist()))
        if len(chunk_df) == 0:
            continue

        batch = build_pa_batch(chunk_df)
        batch = _map_player_ids(batch, player_table_np["id_to_idx"])

        game_runs, rng = sample_fn(model_fn, params, batch, pt, rng, args.samples)
        sim_runs_per_game.append(game_runs)

        if b_idx % 10 == 0 or b_idx + 1 == len(chunks):
            done = sum(len(x) for x in sim_runs_per_game)
            print(f"  batch {b_idx+1:4d}/{len(chunks)}  games={done}  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    sim_runs = np.concatenate(sim_runs_per_game).astype(float)
    print(f"\nSimulated ({mode}): mean={sim_runs.mean():.2f}, "
          f"std={sim_runs.std():.2f}, n={len(sim_runs)}", flush=True)

    n = min(len(obs_runs), len(sim_runs))
    metrics = game_run_metrics(sim_runs[:n], obs_runs[:n])

    print(f"\n=== PA model [{mode}] game-level metrics ===")
    for k, v in metrics.items():
        print(f"  {k:24s} = {v:.5f}")

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved → {args.out}", flush=True)

    print("\n=== comparison row ===")
    print(
        f"dwjax_pa_{mode}_step{step} | "
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
