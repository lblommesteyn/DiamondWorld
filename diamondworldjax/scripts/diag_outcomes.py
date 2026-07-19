"""Diagnostic: compare the model's sampled pa_outcome distribution to real data.

If the engine-rollout undercounts runs, the most likely cause is a miscalibrated
pa_outcome head (the model was trained mainly via the runs_scored/base_state_after
heads, which can predict runs from context almost independently of pa_outcome).
"""
from __future__ import annotations

import argparse
import pickle
from functools import partial
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

TRAIN = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
TEST = [2023, 2024]
_DEFAULT_CKPT = checkpoints_root() / "dwjax_pa_BEST" / "v5_step10000_rollout_kl0.028.pkl"


def main() -> None:
    import jax
    import jax.numpy as jnp
    import numpyro.handlers as nh

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=_DEFAULT_CKPT)
    parser.add_argument("--outcome-only", action="store_true")
    parser.add_argument("--fatigue", action="store_true",
                        help="Model was trained with the pitch-count fatigue feature (STATE_DIM 9).")
    parser.add_argument("--use-park", action="store_true",
                        help="Rebuild real park indices for the park-aware model (v9+). Leave OFF "
                             "for pre-v9 checkpoints, which trained on park_idx=0 (all-zeros).")
    parser.add_argument("--platoon", action="store_true",
                        help="Model trained with platoon (batter side + pitcher hand), v11+.")
    parser.add_argument("--recency-halflife", type=float, default=None,
                        help="Match a recency-trained model (v12+): same half-life as training.")
    parser.add_argument("--dump-logits", type=Path, default=None,
                        help="Save per-PA (logits, real_outcome) over valid PAs to npz for "
                             "learned calibration (fit_calibration.py).")
    args = parser.parse_args()
    mkw = {}
    if args.outcome_only:
        mkw["outcome_only"] = True
    if args.fatigue:
        mkw["fatigue"] = True
    if args.platoon:
        mkw["platoon"] = True
    model_fn = partial(pa_model, **mkw) if mkw else pa_model

    with open(args.ckpt, "rb") as f:
        params = pickle.load(f)["params"]

    train_pitches = load_seasons(TRAIN, data_root=processed_root())
    ptab = _build_player_table(train_pitches, recency_halflife=args.recency_halflife)
    # Rebuild the park_id -> park_idx map from the training seasons. The processed
    # test parquet has no park_idx column, so build_pa_batch would otherwise fill 0
    # for every PA -- and park index 0 ("unknown park") is out-of-distribution for
    # park-aware checkpoints (v9+), which collapse to all-K on it. Materialise real
    # park indices here (opt-in), exactly as train_pa and simulate_games do.
    park_map = _build_park_index(train_pitches) if args.use_park else None
    del train_pitches

    test_pa = load_seasons(TEST, data_root=processed_root()).filter(pl.col("pa_terminal"))
    if park_map is not None:
        test_pa = apply_park_idx(test_pa, park_map)
    # Real outcome histogram
    real_oc = [o for o in test_pa["pa_outcome"].to_list() if o in PA_OUTCOME_IDX]
    real_hist = np.zeros(len(PA_OUTCOMES))
    for o in real_oc:
        real_hist[PA_OUTCOME_IDX[o]] += 1
    real_hist /= real_hist.sum()

    # Sample model outcomes on real (conditioned) states, several batches.
    game_ids = test_pa["game_pk"].unique().to_numpy()[:512]
    pt = {"stats": jnp.array(ptab["stats"]), "league": jnp.array(ptab["league"]),
          "hand": jnp.array(ptab["hand"]),
          "bat_hand": jnp.array(ptab.get("bat_hand", np.full(len(ptab["hand"]), 0.5, np.float32))),
          "pit_hand": jnp.array(ptab.get("pit_hand", np.full(len(ptab["hand"]), 0.5, np.float32)))}
    rng = jax.random.PRNGKey(0)
    model_hist = np.zeros(len(PA_OUTCOMES))
    dump_logits, dump_real = [], []
    for i in range(0, len(game_ids), 64):
        chunk = game_ids[i:i+64]
        df = test_pa.filter(pl.col("game_pk").is_in(chunk.tolist()))
        batch = _map_player_ids(build_pa_batch(df), ptab["id_to_idx"])
        valid = np.array(batch["pa_valid"])
        rng, k = jax.random.split(rng)
        with nh.seed(rng_seed=k):
            with nh.substitute(data=params):
                with nh.trace() as tr:
                    model_fn(batch, pt, teacher_force=False)
        oc = np.array(tr["pa_outcome"]["value"])
        for j in range(len(PA_OUTCOMES)):
            model_hist[j] += ((oc == j) & valid).sum()
        if args.dump_logits is not None:
            lg = np.array(tr["pa_outcome"]["fn"].logits)  # (B, T, 9)
            ro = np.array(batch["pa_outcome"])            # (B, T) real outcome idx, -1 pad
            vm = valid & (ro >= 0)
            dump_logits.append(lg[vm])
            dump_real.append(ro[vm])
    model_hist /= model_hist.sum()

    if args.dump_logits is not None:
        L = np.concatenate(dump_logits, axis=0)
        Y = np.concatenate(dump_real, axis=0).astype(np.int64)
        args.dump_logits.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.dump_logits, logits=L, real=Y)
        print(f"dumped {len(Y)} PA logits -> {args.dump_logits}", flush=True)

    print(f"\n{'outcome':8s} {'real%':>8s} {'model%':>8s} {'ratio':>7s}", flush=True)
    for j, name in enumerate(PA_OUTCOMES):
        r, m = real_hist[j] * 100, model_hist[j] * 100
        ratio = m / r if r > 0 else float("nan")
        print(f"{name:8s} {r:7.2f}% {m:7.2f}% {ratio:6.2f}x", flush=True)

    # Ready-to-paste per-class logit recalibration = log(real/model).
    # E is pinned to 0.0 (errors are injected structurally, not sampled by the head).
    recal = np.zeros(len(PA_OUTCOMES))
    for j, name in enumerate(PA_OUTCOMES):
        if name == "E" or model_hist[j] <= 0 or real_hist[j] <= 0:
            recal[j] = 0.0
        else:
            recal[j] = np.log(real_hist[j] / model_hist[j])
    print(f"\n# recal vector (log(real/model)), order {','.join(PA_OUTCOMES)}", flush=True)
    print("RECAL = np.array([", flush=True)
    print("    " + ", ".join(f"{v:+.4f}" for v in recal) + ",", flush=True)
    print("], dtype=np.float64)", flush=True)


if __name__ == "__main__":
    main()
