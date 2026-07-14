"""Phase 3 (season level): emergent structure across many simulated seasons.

True standings (team win totals) need team-identity columns that the processed
data dropped, so this evaluator measures the season-level emergent structure that
does NOT require team identity, using the home/away split available from `half`:

  - mean team runs / game
  - shutout rate    (fraction of team-games scoring 0 runs)
  - no-hitter rate  (fraction of team-games with 0 hits)
  - blowout rate    (team-games scoring >= 10 runs)

It simulates the full slate of test games K times ("alternate histories") and
reports the real value against the mean +/- std across the K simulated seasons,
flagging whether reality falls within the simulated spread (calibration).

Usage
-----
    python -m diamondworldjax.scripts.eval_season \
        --ckpt checkpoints/dwjax_pa_v6/dwjax_step_0050000.pkl --outcome-only --seasons 20
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
from diamondworldjax.sim.rules_engine import PA_OUTCOME_IDX
from diamondworldjax.scripts.train_pa import _build_player_table, _map_player_ids

TRAIN_SEASONS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
TEST_SEASONS = [2023, 2024]
_HIT_IDX = np.array([PA_OUTCOME_IDX[o] for o in ("1B", "2B", "3B", "HR")])


def _real_team_games(test_pa):
    """Per (game, half) real runs and hits -> arrays of team-game runs, hits."""
    hit_expr = pl.col("pa_outcome").is_in(["1B", "2B", "3B", "HR"]).cast(pl.Int32)
    g = (
        test_pa.with_columns(hit_expr.alias("_hit"))
        .group_by(["game_pk", "half_bin"])
        .agg(pl.col("runs_scored").sum().alias("r"), pl.col("_hit").sum().alias("h"))
    )
    return g["r"].to_numpy().astype(float), g["h"].to_numpy().astype(float)


def _season_metrics(runs: np.ndarray, hits: np.ndarray) -> dict[str, float]:
    return {
        "team_runs": float(runs.mean()),
        "shutout%": float(np.mean(runs == 0) * 100),
        "nohit%": float(np.mean(hits == 0) * 100),
        "blowout%": float(np.mean(runs >= 10) * 100),
    }


def _rollout_team_games(model_fn, params, batch, pt, rng_key, engine, n_samples):
    """One engine rollout; returns per (game,half) team runs and hits.

    Splits by half into away (top) / home (bottom) team-games and tracks hits
    (outcome in {1B,2B,3B,HR}) alongside runs.
    """
    import jax
    import jax.numpy as jnp
    import numpyro.handlers as nh

    B, T = batch["pa_valid"].shape
    valid = np.array(batch["pa_valid"])
    real_outs = np.clip(np.rint(np.array(batch["outs"]) * 2.0), 0, 2).astype(np.int64)
    inning_norm = np.array(batch["inning"])
    half_arr = np.rint(np.array(batch["half"])).astype(np.int64)
    rng_np = np.random.default_rng(0)
    hit_mask = np.zeros(9, dtype=bool); hit_mask[_HIT_IDX] = True

    out = []  # collects (runs, hits) per team-game across samples
    for _ in range(n_samples):
        current_bs = np.array(batch["base_state"])
        a_r = np.zeros(B); h_r = np.zeros(B); a_h = np.zeros(B); h_h = np.zeros(B)
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
            oc = np.array(tr["pa_outcome"]["value"])[:, 0].astype(np.int64)
            bs_int = np.clip(np.rint(current_bs[:, t] * 7.0), 0, 7).astype(np.int64)
            eo = engine.sample(bs_int, real_outs[:, t], oc, rng_np)
            r = eo["runs"] * valid[:, t]
            is_hit = hit_mask[oc] & valid[:, t]
            is_home = half_arr[:, t] == 1
            h_r += np.where(is_home, r, 0); a_r += np.where(is_home, 0, r)
            h_h += np.where(is_home, is_hit, 0); a_h += np.where(is_home, 0, is_hit)
            if t + 1 < T:
                boundary = (inning_norm[:, t+1] != inning_norm[:, t]) | (half_arr[:, t+1] != half_arr[:, t])
                nb = np.where(boundary, 0, eo["bs_after"]).astype(np.float64) / 7.0
                current_bs[:, t+1] = np.where(valid[:, t], nb, current_bs[:, t+1])
        out.append((np.concatenate([a_r, h_r]), np.concatenate([a_h, h_h])))
    return out, rng_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path,
                        default=checkpoints_root() / "dwjax_pa_v6" / "dwjax_step_0050000.pkl")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seasons", type=int, default=20,
                        help="Number of full-season simulations (alternate histories).")
    parser.add_argument("--limit-games", type=int, default=0)
    parser.add_argument("--outcome-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    from functools import partial
    from diamondworldjax.model.pa_model import pa_model
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

    real_runs, real_hits = _real_team_games(test_pa)
    real_m = _season_metrics(real_runs, real_hits)

    pt = {"stats": jnp.array(ptab["stats"]), "league": jnp.array(ptab["league"]),
          "hand": jnp.array(ptab["hand"])}
    model_fn = partial(pa_model, outcome_only=True) if args.outcome_only else pa_model
    game_ids = test_pa["game_pk"].unique().sort().to_numpy()
    chunks = [game_ids[i:i+args.batch] for i in range(0, len(game_ids), args.batch)]

    # Pre-build batches once (reused across season sims).
    batches = []
    for chunk in chunks:
        df = test_pa.filter(pl.col("game_pk").is_in(chunk.tolist()))
        if len(df):
            batches.append(_map_player_ids(build_pa_batch(df), ptab["id_to_idx"]))

    rng = jax.random.PRNGKey(args.seed)
    season_metrics = {k: [] for k in real_m}
    t0 = time.time()
    for s in range(args.seasons):
        all_runs, all_hits = [], []
        for batch in batches:
            res, rng = _rollout_team_games(model_fn, params, batch, pt, rng, engine, 1)
            all_runs.append(res[0][0]); all_hits.append(res[0][1])
        runs = np.concatenate(all_runs); hits = np.concatenate(all_hits)
        m = _season_metrics(runs, hits)
        for k in season_metrics:
            season_metrics[k].append(m[k])
        print(f"  season {s+1}/{args.seasons} elapsed={time.time()-t0:.0f}s", flush=True)

    print(f"\n=== Season-level emergent structure ({args.seasons} simulated seasons) ===", flush=True)
    print(f"  {'metric':10s} {'real':>8s} {'sim_mean':>9s} {'sim_std':>8s} {'in_range':>9s}", flush=True)
    for k in real_m:
        arr = np.array(season_metrics[k])
        lo, hi = arr.mean() - 2*arr.std(), arr.mean() + 2*arr.std()
        in_rng = "yes" if lo <= real_m[k] <= hi else "NO"
        print(f"  {k:10s} {real_m[k]:8.3f} {arr.mean():9.3f} {arr.std():8.3f} {in_rng:>9s}", flush=True)
    print("\n  in_range = real value within sim_mean +/- 2 sim_std (alternate-history calibration)", flush=True)


if __name__ == "__main__":
    main()
