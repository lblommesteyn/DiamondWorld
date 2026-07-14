"""Phase 3 (game level): score distribution, margins, home-win reproduction.

Follows each game's real PA sequence, samples outcomes from the model, and uses
the rules engine to score each PA. Runs are split by half (top = away batting,
bottom = home batting) to recover per-team scores, enabling game-level metrics
the run-total view cannot see:

  - mean home / away runs (does the model reproduce home-field advantage?)
  - home-win rate
  - run-margin distribution (|home - away|) and its spread
  - total-runs KL (as a cross-check against eval_pa engine-rollout)

Usage
-----
    python -m diamondworldjax.scripts.eval_games \
        --ckpt checkpoints/dwjax_pa_v6/dwjax_step_0050000.pkl --outcome-only --samples 3
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import polars as pl

from diamondworldjax.paths import processed_root, checkpoints_root
from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.data.pa_batching import build_pa_batch
from diamondworldjax.model.pa_model import pa_model
from diamondworldjax.eval.calibration import game_run_metrics
from diamondworldjax.scripts.train_pa import _build_player_table, _map_player_ids

TRAIN_SEASONS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
TEST_SEASONS = [2023, 2024]


def _real_home_away(test_pa) -> tuple[np.ndarray, np.ndarray]:
    """Per-game (away_runs, home_runs) from real data, split by half."""
    g = (
        test_pa.group_by(["game_pk", "half_bin"])
        .agg(pl.col("runs_scored").sum().alias("r"))
        .sort("game_pk")
    )
    games = test_pa["game_pk"].unique().sort().to_numpy()
    gid_to_i = {int(g): i for i, g in enumerate(games)}
    away = np.zeros(len(games))
    home = np.zeros(len(games))
    for row in g.iter_rows(named=True):
        i = gid_to_i[int(row["game_pk"])]
        if int(row["half_bin"]) == 0:
            away[i] = row["r"]
        else:
            home[i] = row["r"]
    return away, home


def _rollout_split(model_fn, params, batch, pt, rng_key, engine, n_samples):
    """Engine rollout that splits runs into away (top) / home (bottom)."""
    import jax
    import jax.numpy as jnp
    import numpyro.handlers as nh

    B, T = batch["pa_valid"].shape
    valid = np.array(batch["pa_valid"])
    real_outs = np.clip(np.rint(np.array(batch["outs"]) * 2.0), 0, 2).astype(np.int64)
    inning_norm = np.array(batch["inning"])
    half_arr = np.rint(np.array(batch["half"])).astype(np.int64)
    rng_np = np.random.default_rng(0)

    away_tot = np.zeros(B)
    home_tot = np.zeros(B)
    for _ in range(n_samples):
        current_bs = np.array(batch["base_state"])
        away = np.zeros(B)
        home = np.zeros(B)
        for t in range(T):
            if not valid[:, t].any():
                break
            t_batch = {}
            for k, v in batch.items():
                if hasattr(v, "ndim"):
                    arr = np.array(v)
                    t_batch[k] = jnp.array(arr[:, t:t+1] if arr.ndim == 2 else arr)
            t_batch["base_state"] = jnp.array(current_bs[:, t:t+1])
            rng_key, sk = jax.random.split(rng_key)
            with nh.seed(rng_seed=sk):
                with nh.substitute(data=params):
                    with nh.trace() as tr:
                        model_fn(t_batch, pt, teacher_force=False)
            outcome_t = np.array(tr["pa_outcome"]["value"])[:, 0].astype(np.int64)
            bs_int = np.clip(np.rint(current_bs[:, t] * 7.0), 0, 7).astype(np.int64)
            eo = engine.sample(bs_int, real_outs[:, t], outcome_t, rng_np)
            r = eo["runs"] * valid[:, t]
            is_home = half_arr[:, t] == 1
            home += np.where(is_home, r, 0)
            away += np.where(is_home, 0, r)
            if t + 1 < T:
                boundary = (inning_norm[:, t+1] != inning_norm[:, t]) | (half_arr[:, t+1] != half_arr[:, t])
                nb = np.where(boundary, 0, eo["bs_after"]).astype(np.float64) / 7.0
                current_bs[:, t+1] = np.where(valid[:, t], nb, current_bs[:, t+1])
        away_tot += away
        home_tot += home
    return away_tot / n_samples, home_tot / n_samples, rng_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path,
                        default=checkpoints_root() / "dwjax_pa_v6" / "dwjax_step_0050000.pkl")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--limit-games", type=int, default=0)
    parser.add_argument("--outcome-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    from functools import partial
    from diamondworldjax.sim.rules_engine import EmpiricalEngine

    with open(args.ckpt, "rb") as f:
        params = pickle.load(f)["params"]

    train_pitches = load_seasons(TRAIN_SEASONS, data_root=processed_root())
    ptab = _build_player_table(train_pitches)
    engine = EmpiricalEngine().fit(train_pitches.filter(pl.col("pa_terminal")))
    del train_pitches

    test_pa = load_seasons(TEST_SEASONS, data_root=processed_root()).filter(pl.col("pa_terminal"))
    if args.limit_games > 0:
        keep = test_pa["game_pk"].unique().sort().to_numpy()[:args.limit_games]
        test_pa = test_pa.filter(pl.col("game_pk").is_in(keep.tolist()))

    real_away, real_home = _real_home_away(test_pa)
    print(f"  Real: home={real_home.mean():.2f}  away={real_away.mean():.2f}  "
          f"home_win={np.mean(real_home > real_away)*100:.1f}%  n={len(real_home)}", flush=True)

    pt = {"stats": jnp.array(ptab["stats"]), "league": jnp.array(ptab["league"]),
          "hand": jnp.array(ptab["hand"])}
    model_fn = partial(pa_model, outcome_only=True) if args.outcome_only else pa_model

    game_ids = test_pa["game_pk"].unique().sort().to_numpy()
    chunks = [game_ids[i:i+args.batch] for i in range(0, len(game_ids), args.batch)]
    rng = jax.random.PRNGKey(args.seed)
    sim_away, sim_home = [], []
    t0 = time.time()
    for bi, chunk in enumerate(chunks):
        df = test_pa.filter(pl.col("game_pk").is_in(chunk.tolist()))
        if len(df) == 0:
            continue
        batch = _map_player_ids(build_pa_batch(df), ptab["id_to_idx"])
        a, h, rng = _rollout_split(model_fn, params, batch, pt, rng, engine, args.samples)
        sim_away.append(a); sim_home.append(h)
        if bi % 10 == 0 or bi+1 == len(chunks):
            print(f"  batch {bi+1}/{len(chunks)} elapsed={time.time()-t0:.0f}s", flush=True)

    sim_away = np.concatenate(sim_away)
    sim_home = np.concatenate(sim_home)
    n = min(len(real_home), len(sim_home))
    sa, sh, ra, rh = sim_away[:n], sim_home[:n], real_away[:n], real_home[:n]

    print(f"\n=== Game-level metrics ({n} games, {args.samples} samples) ===", flush=True)
    print(f"  {'':12s} {'real':>8s} {'sim':>8s}", flush=True)
    print(f"  {'home runs':12s} {rh.mean():8.2f} {sh.mean():8.2f}", flush=True)
    print(f"  {'away runs':12s} {ra.mean():8.2f} {sa.mean():8.2f}", flush=True)
    print(f"  {'home-win %':12s} {np.mean(rh>ra)*100:8.1f} {np.mean(sh>sa)*100:8.1f}", flush=True)
    print(f"  {'margin std':12s} {np.std(rh-ra):8.2f} {np.std(sh-sa):8.2f}", flush=True)
    print(f"  {'total mean':12s} {(rh+ra).mean():8.2f} {(sh+sa).mean():8.2f}", flush=True)

    tot_metrics = game_run_metrics(sh + sa, rh + ra)
    print(f"\n  total-runs KL = {tot_metrics['kl_run_distribution']:.5f}  "
          f"margin |bias| = {abs(np.mean(sh-sa)-np.mean(rh-ra)):.3f}", flush=True)


if __name__ == "__main__":
    main()
