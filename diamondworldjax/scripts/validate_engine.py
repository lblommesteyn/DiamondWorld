"""Validate the deterministic rules engine against real data.

The Phase-1 thesis: runs_scored and base_state_after are (almost) determined by
(base_state, pa_outcome). If true, the deterministic engine should reproduce the
real values for the vast majority of PAs, and the neural runs_scored /
base_state_after heads are the wrong tool.

This script loads the test seasons, runs the engine on each real PA's
(base_state, pa_outcome), and reports match rates overall and per outcome.

Usage
-----
    python -m diamondworldjax.scripts.validate_engine
"""
from __future__ import annotations

import numpy as np
import polars as pl

from diamondworldjax.paths import processed_root
from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.sim import rules_engine as eng
from diamondworldjax.sim.rules_engine import EmpiricalEngine

TRAIN_SEASONS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
TEST_SEASONS = [2023, 2024]


def main() -> None:
    print(f"Loading test seasons {TEST_SEASONS}...", flush=True)
    pitches = load_seasons(TEST_SEASONS, data_root=processed_root())
    pa = pitches.filter(pl.col("pa_terminal"))
    print(f"  {len(pa):,} terminal PAs", flush=True)

    # Keep rows with a usable outcome + observed transition.
    pa = pa.filter(
        pl.col("pa_outcome").is_not_null()
        & pl.col("base_state").is_not_null()
        & pl.col("base_state_after").is_not_null()
    )

    outcome_str = pa["pa_outcome"].to_list()
    keep = np.array([o in eng.PA_OUTCOME_IDX for o in outcome_str])
    n_total = len(outcome_str)
    n_drop = int((~keep).sum())
    if n_drop:
        dropped = sorted(set(o for o, k in zip(outcome_str, keep) if not k))
        print(f"  Dropping {n_drop:,} PAs with out-of-vocab outcomes: {dropped}", flush=True)

    bs = pa["base_state"].to_numpy().astype(np.int64)[keep]
    oc = np.array([eng.PA_OUTCOME_IDX[o] for o, k in zip(outcome_str, keep) if k], dtype=np.int64)
    real_runs = pa["runs_scored"].to_numpy().astype(np.int64)[keep]
    real_bsa = pa["base_state_after"].to_numpy().astype(np.int64)[keep]

    out = eng.step(bs, oc)
    pred_runs = out["runs"]
    pred_bsa = out["bs_after"]

    runs_match = pred_runs == real_runs
    bsa_match = pred_bsa == real_bsa
    both_match = runs_match & bsa_match
    n = len(bs)

    print(f"\n=== Engine vs real data ({n:,} PAs) ===", flush=True)
    print(f"  runs_scored match     : {runs_match.mean()*100:6.2f}%", flush=True)
    print(f"  base_state_after match : {bsa_match.mean()*100:6.2f}%", flush=True)
    print(f"  both match            : {both_match.mean()*100:6.2f}%", flush=True)

    # Aggregate run total error (the metric that actually matters).
    print(f"\n  total real runs : {real_runs.sum():,}", flush=True)
    print(f"  total pred runs : {pred_runs.sum():,}", flush=True)
    print(f"  aggregate run bias : {(pred_runs.sum()-real_runs.sum())/real_runs.sum()*100:+.2f}%", flush=True)

    print(f"\n=== Per-outcome breakdown ===", flush=True)
    print(f"  {'outcome':8s} {'n':>9s} {'runs%':>8s} {'bsa%':>8s}", flush=True)
    for oi, name in enumerate(eng.PA_OUTCOMES):
        m = oc == oi
        cnt = int(m.sum())
        if cnt == 0:
            continue
        rm = runs_match[m].mean() * 100
        bm = bsa_match[m].mean() * 100
        print(f"  {name:8s} {cnt:9,d} {rm:7.2f}% {bm:7.2f}%", flush=True)

    # ------------------------------------------------------------------
    # Empirical engine: fit on train, measure aggregate run bias on test.
    # ------------------------------------------------------------------
    print(f"\nFitting empirical engine on {TRAIN_SEASONS}...", flush=True)
    train_pitches = load_seasons(TRAIN_SEASONS, data_root=processed_root())
    train_pa = train_pitches.filter(pl.col("pa_terminal"))
    emp = EmpiricalEngine().fit(train_pa)
    outs_arr = np.clip(pa["outs"].to_numpy().astype(np.int64)[keep], 0, 2)
    exp_runs = emp.expected_runs(bs, outs_arr, oc)

    print(f"\n=== Aggregate run total (the metric that matters) ===", flush=True)
    print(f"  real runs                      : {real_runs.sum():,}", flush=True)
    print(f"  deterministic engine runs      : {pred_runs.sum():,}  "
          f"({(pred_runs.sum()-real_runs.sum())/real_runs.sum()*100:+.2f}%)", flush=True)
    print(f"  empirical engine expected runs : {exp_runs.sum():,.0f}  "
          f"({(exp_runs.sum()-real_runs.sum())/real_runs.sum()*100:+.2f}%)", flush=True)


if __name__ == "__main__":
    main()
