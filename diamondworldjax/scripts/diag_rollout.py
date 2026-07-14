"""Isolate WHY the engine-rollout undercounts runs (v6/v7 stuck at ~5.0 vs 8.86).

Two ways to score runs from the model, both using the empirical engine for the
(state, outcome) -> runs map, differing only in which base_state the model and
engine see:

  A. conditioned-engine : sample pa_outcome on the REAL base_state each PA, and
     score with the REAL base_state. No feedback. Isolates outcome quality.
  B. engine-rollout     : feed the engine's own base_state forward (what eval_pa
     --engine-rollout does). Adds base-state drift on top of A.

Reference: real outcomes through the engine reproduce ~8.86 (-0.45% bias).

If A ~= 8.86  -> outcomes are fine; the gap is base-state rollout drift (DAgger).
If A ~= 5-6   -> the model's per-PA outcomes are too out-heavy even on real
                 states; a calibration fix is needed, DAgger cannot help.

Usage:  python -m diamondworldjax.scripts.diag_rollout --ckpt <ckpt> --outcome-only --limit-games 512
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
from diamondworldjax.data.pa_batching import build_pa_batch
from diamondworldjax.model.pa_model import pa_model
from diamondworldjax.sim.rules_engine import EmpiricalEngine
from diamondworldjax.scripts.train_pa import _build_player_table, _map_player_ids

TRAIN = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
TEST = [2023, 2024]


def main() -> None:
    import jax
    import jax.numpy as jnp
    import numpyro.handlers as nh

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--outcome-only", action="store_true")
    ap.add_argument("--limit-games", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.ckpt, "rb") as f:
        params = pickle.load(f)["params"]
    train_pitches = load_seasons(TRAIN, data_root=processed_root())
    ptab = _build_player_table(train_pitches)
    engine = EmpiricalEngine().fit(train_pitches.filter(pl.col("pa_terminal")))
    del train_pitches

    test_pa = load_seasons(TEST, data_root=processed_root()).filter(pl.col("pa_terminal"))
    keep = test_pa["game_pk"].unique().sort().to_numpy()[:args.limit_games]
    test_pa = test_pa.filter(pl.col("game_pk").is_in(keep.tolist()))

    pt = {"stats": jnp.array(ptab["stats"]), "league": jnp.array(ptab["league"]),
          "hand": jnp.array(ptab["hand"])}
    model_fn = partial(pa_model, outcome_only=True) if args.outcome_only else pa_model

    game_ids = test_pa["game_pk"].unique().sort().to_numpy()
    chunks = [game_ids[i:i+64] for i in range(0, len(game_ids), 64)]
    rng = jax.random.PRNGKey(args.seed)
    rng_np = np.random.default_rng(0)

    real_total = cond_total = roll_total = 0.0
    n_games = 0
    # outcome marginal accumulators
    oc_hist = np.zeros(9)        # conditioned (real states)
    roll_oc_hist = np.zeros(9)   # during rollout (own states)
    # base occupancy: fraction of PAs with >=1 runner on, real vs rollout
    real_occ = real_occ_n = roll_occ = roll_occ_n = 0.0
    t0 = time.time()

    for ci, chunk in enumerate(chunks):
        df = test_pa.filter(pl.col("game_pk").is_in(chunk.tolist()))
        if len(df) == 0:
            continue
        batch = _map_player_ids(build_pa_batch(df), ptab["id_to_idx"])
        valid = np.array(batch["pa_valid"])
        B, T = valid.shape
        real_bs = np.clip(np.rint(np.array(batch["base_state"]) * 7.0), 0, 7).astype(np.int64)
        real_outs = np.clip(np.rint(np.array(batch["outs"]) * 2.0), 0, 2).astype(np.int64)
        inn = np.array(batch["inning"]); half = np.rint(np.array(batch["half"])).astype(np.int64)
        real_runs = np.array(batch["runs_scored"])

        real_total += (real_runs * valid).sum()
        n_games += B
        real_occ += ((real_bs > 0) & valid).sum(); real_occ_n += valid.sum()

        # --- A: conditioned-engine (sample outcomes on real states) ---
        rng, k = jax.random.split(rng)
        with nh.seed(rng_seed=k):
            with nh.substitute(data=params):
                with nh.trace() as tr:
                    model_fn(batch, pt, teacher_force=False)
        oc_all = np.array(tr["pa_outcome"]["value"])  # (B,T)
        for j in range(9):
            oc_hist[j] += ((oc_all == j) & valid).sum()
        eo = engine.sample(real_bs[valid], real_outs[valid], oc_all[valid], rng_np)
        cond_runs = np.zeros((B, T)); cond_runs[valid] = eo["runs"]
        cond_total += (cond_runs * valid).sum()

        # --- B: engine-rollout (feed engine base_state forward) ---
        cur_bs = np.array(batch["base_state"])
        for t in range(T):
            if not valid[:, t].any():
                break
            tb = {kk: (jnp.array(np.array(v)[:, t:t+1]) if hasattr(v, 'ndim') and np.array(v).ndim == 2
                       else jnp.array(np.array(v))) for kk, v in batch.items() if hasattr(v, 'ndim')}
            tb["base_state"] = jnp.array(cur_bs[:, t:t+1])
            rng, k = jax.random.split(rng)
            with nh.seed(rng_seed=k):
                with nh.substitute(data=params):
                    with nh.trace() as tr:
                        model_fn(tb, pt, teacher_force=False)
            oc = np.array(tr["pa_outcome"]["value"])[:, 0].astype(np.int64)
            bsi = np.clip(np.rint(cur_bs[:, t] * 7.0), 0, 7).astype(np.int64)
            vt = valid[:, t]
            for j in range(9):
                roll_oc_hist[j] += ((oc == j) & vt).sum()
            roll_occ += ((bsi > 0) & vt).sum(); roll_occ_n += vt.sum()
            e = engine.sample(bsi, real_outs[:, t], oc, rng_np)
            roll_total += (e["runs"] * vt).sum()
            if t + 1 < T:
                bnd = (inn[:, t+1] != inn[:, t]) | (half[:, t+1] != half[:, t])
                cur_bs[:, t+1] = np.where(valid[:, t], np.where(bnd, 0, e["bs_after"]) / 7.0, cur_bs[:, t+1])
        if ci % 4 == 0:
            print(f"  chunk {ci+1}/{len(chunks)} elapsed={time.time()-t0:.0f}s", flush=True)

    print(f"\n=== runs/game ({n_games} games) ===", flush=True)
    print(f"  real (real outcomes)        : {real_total/n_games:.2f}", flush=True)
    print(f"  A conditioned-engine        : {cond_total/n_games:.2f}", flush=True)
    print(f"  B engine-rollout            : {roll_total/n_games:.2f}", flush=True)
    print(f"\n  If A near real -> base-state drift (DAgger). If A near B/low -> outcome calibration.", flush=True)

    print(f"\n=== base occupancy (fraction of PAs with >=1 runner on) ===", flush=True)
    print(f"  real states   : {real_occ/real_occ_n*100:.1f}%", flush=True)
    print(f"  rollout states: {roll_occ/roll_occ_n*100:.1f}%", flush=True)
    print(f"  (if rollout much lower -> bases drift empty = the run-loss mechanism)", flush=True)

    print(f"\n=== outcome marginals: real vs conditioned vs ROLLOUT ===", flush=True)
    real_m = test_pa["pa_outcome"].value_counts()
    rm = {r["pa_outcome"]: r["count"] for r in real_m.iter_rows(named=True)}
    names = ["K", "BB", "HBP", "1B", "2B", "3B", "HR", "out", "E"]
    tot_real = sum(rm.get(n, 0) for n in names)
    oc_hist /= oc_hist.sum()
    roll_oc_hist /= roll_oc_hist.sum()
    print(f"  {'oc':4s} {'real%':>7s} {'cond%':>7s} {'roll%':>7s}", flush=True)
    for j, nm in enumerate(names):
        rp = rm.get(nm, 0) / tot_real * 100 if tot_real else 0
        print(f"  {nm:4s} {rp:7.2f} {oc_hist[j]*100:7.2f} {roll_oc_hist[j]*100:7.2f}", flush=True)
    print(f"\n  cond~roll but runs differ -> structural (occupancy). roll more out-heavy -> base-state compounding.", flush=True)


if __name__ == "__main__":
    main()
