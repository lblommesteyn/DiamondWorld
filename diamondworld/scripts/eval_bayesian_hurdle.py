"""Evaluate B4 Bayesian Stochastic Multi-Hurdle baseline and print full table."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl

from diamondworld.baselines.bayesian_hurdle import BayesianHurdleSimulator
from diamondworld.eval.metrics import evaluate_game_logs

_DATA_ROOT   = Path("/scratch/lblommes/diamondworld/data/processed")
_CKPT_DIR    = Path("/scratch/lblommes/diamondworld/checkpoints/baselines")
_RESULTS_DIR = Path("/scratch/lblommes/diamondworld/eval/results")

TRAIN_SEASONS = list(range(2015, 2023))
TEST_SEASONS  = [2023, 2024]


def main() -> None:
    print(f"Loading training data {TRAIN_SEASONS}...")
    train_df = pl.concat([
        pl.read_parquet(_DATA_ROOT / f"pitches_{yr}.parquet") for yr in TRAIN_SEASONS
    ])
    print(f"  {len(train_df):,} pitches.")

    print("Loading test data (2023-2024)...")
    test_df = pl.concat([
        pl.read_parquet(_DATA_ROOT / f"pitches_{yr}.parquet") for yr in TEST_SEASONS
    ])
    print(f"  {len(test_df):,} pitches.")

    sim = BayesianHurdleSimulator(rng=np.random.default_rng(42))
    print("Fitting B4 Bayesian Hurdle...")
    sim.fit(train_df)

    print("Posterior parameters:")
    for h, (a, b) in sim._posteriors.items():
        mean = a / (a + b)
        print(f"  {h:8s}: Beta({a:.1f}, {b:.1f})  mean={mean:.4f}")

    _CKPT_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CKPT_DIR / "B4_bayesian_hurdle.pkl", "wb") as f:
        pickle.dump(sim, f)
    print("  Checkpoint saved.")

    print("Simulating 2430 games...")
    sim_df = sim.simulate_season(n_games=2430)

    print("Evaluating...")
    report = evaluate_game_logs(test_df, sim_df, skip_pa_length_kl=True)
    metrics = report.as_dict()

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _RESULTS_DIR / "baselines_B4_bayesian_hurdle.json"
    out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Saved → {out_path}")

    # Full comparison table
    all_results: dict[str, dict] = {}
    for p in sorted(_RESULTS_DIR.glob("baselines_*.json")):
        all_results[p.stem.replace("baselines_", "")] = json.loads(p.read_text())
    for stem, fname in [
        ("Phase3_MLP_randpool", "phase3_no_memory_mlp.json"),
        ("Phase3_MLP_lineup",   "phase3_mlp_lineup.json"),
        ("Phase4_GCT_randpool", "phase4_game_context_transformer.json"),
    ]:
        p = _RESULTS_DIR / fname
        if p.exists():
            all_results[stem] = json.loads(p.read_text())

    headers = [
        "name", "kl_run_distribution", "wasserstein_runs", "mean_rg_error",
        "variance_error", "p0_error", "p5_plus_error", "p8_plus_error", "crooked_kl",
    ]
    W = 22
    print("\n=== FULL RESULTS TABLE ===")
    print(" | ".join(h.ljust(W) for h in headers))
    print("-" * (W * len(headers) + 3 * (len(headers) - 1)))
    for name, m in all_results.items():
        row = [name] + [f"{m.get(h, float('nan')):.5f}" for h in headers[1:]]
        print(" | ".join(v.ljust(W) for v in row))


if __name__ == "__main__":
    main()
