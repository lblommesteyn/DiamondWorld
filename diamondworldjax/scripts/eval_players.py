"""Phase 3 (player level): does the model reproduce individual stat lines?

For every batter with enough real plate appearances in the test seasons, compare
the model's simulated rate stats (AVG, OBP, SLG, K%, BB%, HR%) against the
player's real rate stats. This is the capability the negbinom / Markov baselines
fundamentally lack: they have no notion of *who* is batting.

Outcomes are sampled conditioned on the real game states each PA faced (so this
isolates the outcome head's player-awareness from rollout dynamics). Stats are
derived from outcome counts:

  H   = 1B + 2B + 3B + HR
  AB  = PA - BB - HBP            (sac flies/bunts not separated in this dataset)
  AVG = H / AB
  OBP = (H + BB + HBP) / PA
  SLG = (1B + 2*2B + 3*3B + 4*HR) / AB
  K%  = K / PA,   BB% = BB / PA,   HR% = HR / PA

Usage
-----
    python -m diamondworldjax.scripts.eval_players \
        --ckpt checkpoints/dwjax_pa_v6/dwjax_step_0050000.pkl --outcome-only
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import polars as pl

from diamondworldjax.paths import processed_root, checkpoints_root
from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.data.pa_batching import build_pa_batch
from diamondworldjax.model.pa_model import pa_model
from diamondworldjax.sim.rules_engine import PA_OUTCOMES, PA_OUTCOME_IDX
from diamondworldjax.scripts.train_pa import (
    _build_player_table, _map_player_ids, _build_park_index, apply_park_idx,
)

TRAIN_SEASONS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
TEST_SEASONS = [2023, 2024]

# Outcome index shortcuts
_K, _BB, _HBP, _1B, _2B, _3B, _HR, _OUT, _E = range(9)


def _rate_stats(counts: np.ndarray) -> dict[str, float]:
    """counts: length-9 array of outcome counts -> rate-stat dict."""
    pa = counts.sum()
    if pa == 0:
        return {}
    h = counts[_1B] + counts[_2B] + counts[_3B] + counts[_HR]
    ab = pa - counts[_BB] - counts[_HBP]
    ab = max(ab, 1)
    tb = counts[_1B] + 2 * counts[_2B] + 3 * counts[_3B] + 4 * counts[_HR]
    return {
        "AVG": h / ab,
        "OBP": (h + counts[_BB] + counts[_HBP]) / pa,
        "SLG": tb / ab,
        "K%": counts[_K] / pa,
        "BB%": counts[_BB] / pa,
        "HR%": counts[_HR] / pa,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path,
                        default=checkpoints_root() / "dwjax_pa_v6" / "dwjax_step_0050000.pkl")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--min-pa", type=int, default=100,
                        help="Only evaluate batters with at least this many real test PAs.")
    parser.add_argument("--outcome-only", action="store_true")
    parser.add_argument("--fatigue", action="store_true",
                        help="Checkpoint trained with the pitch-count fatigue feature (v8/v9).")
    parser.add_argument("--use-park", action="store_true",
                        help="Rebuild real park indices for the park-aware model (v9+). Leave OFF "
                             "for pre-v9 checkpoints, which trained on park_idx=0 (all-zeros).")
    parser.add_argument("--platoon", action="store_true",
                        help="Model trained with platoon (batter side + pitcher hand), v11+.")
    parser.add_argument("--recency-halflife", type=float, default=None,
                        help="Match a recency-trained model (v12+): same half-life as training.")
    parser.add_argument("--skill-mode", choices=["prior", "mean"], default="prior",
                        help="mean uses the learned player_mu (v13+ with the KL-scale fix).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import numpyro.handlers as nh
    from functools import partial

    print(f"Loading checkpoint {args.ckpt}", flush=True)
    with open(args.ckpt, "rb") as f:
        params = pickle.load(f)["params"]
    if args.skill_mode == "mean" and "player_mu" in params:
        params = {**params, "player_skills": params["player_mu"]}

    train_pitches = load_seasons(TRAIN_SEASONS, data_root=processed_root())
    ptab = _build_player_table(train_pitches, recency_halflife=args.recency_halflife)
    # Real park indices (0 = unknown park is OOD for v9+, which collapses to all-K);
    # the test parquet lacks park_idx, so materialise it from the train park map.
    # Opt-in: pre-v9 checkpoints trained on park_idx=0 and must keep seeing it.
    park_map = _build_park_index(train_pitches) if args.use_park else None
    del train_pitches
    all_ids = ptab["all_ids"]            # index -> real id
    id_to_idx = ptab["id_to_idx"]
    P = len(all_ids)

    test_pa = load_seasons(TEST_SEASONS, data_root=processed_root()).filter(pl.col("pa_terminal"))
    if park_map is not None:
        test_pa = apply_park_idx(test_pa, park_map)

    # --- Real per-batter outcome counts (keyed by training index) ---
    real_counts = np.zeros((P, 9), dtype=np.float64)
    bcol = "batter_id" if "batter_id" in test_pa.columns else "batter_idx"
    for row in test_pa.select([bcol, "pa_outcome"]).iter_rows():
        bid, oc = row
        if oc in PA_OUTCOME_IDX and int(bid) in id_to_idx:
            real_counts[id_to_idx[int(bid)], PA_OUTCOME_IDX[oc]] += 1

    real_pa = real_counts.sum(axis=1)
    keep_idx = np.where(real_pa >= args.min_pa)[0]
    print(f"  {len(keep_idx):,} batters with >= {args.min_pa} test PAs", flush=True)

    # --- Simulated per-batter outcome counts (conditioned sampling) ---
    pt = {"stats": jnp.array(ptab["stats"]), "league": jnp.array(ptab["league"]),
          "hand": jnp.array(ptab["hand"]),
          "bat_hand": jnp.array(ptab.get("bat_hand", np.full(len(ptab["hand"]), 0.5, np.float32))),
          "pit_hand": jnp.array(ptab.get("pit_hand", np.full(len(ptab["hand"]), 0.5, np.float32)))}
    _mkw = {}
    if args.outcome_only:
        _mkw["outcome_only"] = True
    if args.fatigue:
        _mkw["fatigue"] = True
    if args.platoon:
        _mkw["platoon"] = True
    model_fn = partial(pa_model, **_mkw) if _mkw else pa_model

    sim_counts = np.zeros((P, 9), dtype=np.float64)
    game_ids = test_pa["game_pk"].unique().to_numpy()
    chunks = [game_ids[i:i + args.batch] for i in range(0, len(game_ids), args.batch)]
    rng = jax.random.PRNGKey(args.seed)

    import time
    t0 = time.time()
    for bi, chunk in enumerate(chunks):
        df = test_pa.filter(pl.col("game_pk").is_in(chunk.tolist()))
        if len(df) == 0:
            continue
        batch = _map_player_ids(build_pa_batch(df), id_to_idx)
        valid = np.array(batch["pa_valid"])
        bidx = np.array(batch["batter_ids"])     # (B, T) training indices
        for _ in range(args.samples):
            rng, k = jax.random.split(rng)
            with nh.seed(rng_seed=k):
                with nh.substitute(data=params):
                    with nh.trace() as tr:
                        model_fn(batch, pt, teacher_force=False)
            oc = np.array(tr["pa_outcome"]["value"])  # (B, T)
            flat_idx = bidx[valid]
            flat_oc = oc[valid]
            np.add.at(sim_counts, (flat_idx, flat_oc), 1.0)
        if bi % 20 == 0 or bi + 1 == len(chunks):
            print(f"  batch {bi+1}/{len(chunks)}  elapsed={time.time()-t0:.0f}s", flush=True)

    sim_counts /= max(args.samples, 1)

    # --- Compare rate stats across the kept players ---
    metrics = ["AVG", "OBP", "SLG", "K%", "BB%", "HR%"]
    real_vals = {m: [] for m in metrics}
    sim_vals = {m: [] for m in metrics}
    for idx in keep_idx:
        rs = _rate_stats(real_counts[idx])
        ss = _rate_stats(sim_counts[idx])
        if not rs or not ss:
            continue
        for m in metrics:
            real_vals[m].append(rs[m])
            sim_vals[m].append(ss[m])

    print(f"\n=== Player-level stat reproduction ({len(real_vals['AVG'])} batters) ===", flush=True)
    print(f"  {'stat':5s} {'real_mean':>10s} {'sim_mean':>10s} {'MAE':>8s} {'corr':>7s}", flush=True)
    for m in metrics:
        rv = np.array(real_vals[m])
        sv = np.array(sim_vals[m])
        mae = np.abs(rv - sv).mean()
        corr = np.corrcoef(rv, sv)[0, 1] if len(rv) > 1 else float("nan")
        print(f"  {m:5s} {rv.mean():10.4f} {sv.mean():10.4f} {mae:8.4f} {corr:7.3f}", flush=True)

    print("\n  corr = cross-player correlation (does the model rank players correctly?)", flush=True)


if __name__ == "__main__":
    main()
