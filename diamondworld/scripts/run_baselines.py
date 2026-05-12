"""Run all four baselines (B0-B3), evaluate vs test data, save results.

Usage:
    python -m diamondworld.scripts.run_baselines
    python -m diamondworld.scripts.run_baselines --dry-run
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import polars as pl

from diamondworld.baselines.lgbm_outcome import LGBMOutcomeSimulator
from diamondworld.baselines.markov_re24 import MarkovRE24Simulator
from diamondworld.baselines.naive_pa import NaivePASimulator
from diamondworld.baselines.negbinom import NegBinomSimulator
from diamondworld.eval.metrics import evaluate_game_logs

_DATA_ROOT = Path("/scratch/lblommes/diamondworld/data/processed")
_RESULTS_ROOT = Path("/scratch/lblommes/diamondworld/eval/results")
_CHECKPOINTS_ROOT = Path("/scratch/lblommes/diamondworld/checkpoints/baselines")

TRAIN_SEASONS = list(range(2015, 2023))  # 2015-2022 for baselines
TEST_SEASONS = [2023, 2024]


def load_seasons(seasons: list[int], data_root: Path = _DATA_ROOT) -> pl.DataFrame:
    frames = [pl.read_parquet(data_root / f"pitches_{s}.parquet") for s in seasons]
    return pl.concat(frames)


def print_table(results: dict[str, dict[str, float]]) -> None:
    headers = [
        "name", "kl_run_distribution", "wasserstein_runs", "mean_rg_error",
        "variance_error", "p0_error", "p5_plus_error", "p8_plus_error", "crooked_kl",
    ]
    col_widths = [max(len(h), 16) for h in headers]
    col_widths[0] = 12

    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))

    for name, metrics in results.items():
        row_vals = [name] + [
            f"{metrics.get(h, float('nan')):.5f}"
            for h in headers[1:]
        ]
        print(" | ".join(v.ljust(col_widths[i]) for i, v in enumerate(row_vals)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DiamondWorld baselines B0-B3.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, fit on 2015 only and simulate 10 games.",
    )
    parser.add_argument(
        "--data-root", type=Path, default=_DATA_ROOT,
    )
    parser.add_argument(
        "--results-root", type=Path, default=_RESULTS_ROOT,
    )
    parser.add_argument(
        "--checkpoints-root", type=Path, default=_CHECKPOINTS_ROOT,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_root.mkdir(parents=True, exist_ok=True)
    args.checkpoints_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("[DRY RUN] Fitting on 2015 only, simulating 10 games per baseline.")
        train_seasons = [2015]
        n_games = 10
    else:
        train_seasons = TRAIN_SEASONS
        n_games = 2430

    print(f"Loading training data (seasons: {train_seasons})...")
    train_df = load_seasons(train_seasons, args.data_root)
    print(f"  Loaded {len(train_df):,} pitches.")

    print("Loading test data (2023-2024)...")
    test_df = load_seasons(TEST_SEASONS, args.data_root)
    print(f"  Loaded {len(test_df):,} pitches.")

    baselines = [
        ("B0_markov_re24", MarkovRE24Simulator()),
        ("B1_negbinom", NegBinomSimulator()),
        ("B2_naive_pa", NaivePASimulator()),
        ("B3_lgbm_outcome", LGBMOutcomeSimulator()),
    ]

    all_results: dict[str, dict[str, float]] = {}

    for name, sim in baselines:
        print(f"\n=== {name} ===")
        print(f"  Fitting...")
        sim.fit(train_df)

        # Save fitted simulator
        ckpt_path = args.checkpoints_root / f"{name}.pkl"
        with open(ckpt_path, "wb") as f:
            pickle.dump(sim, f)
        print(f"  Saved checkpoint to {ckpt_path}")

        # Simulate season
        print(f"  Simulating {n_games} games...")
        sim_df = sim.simulate_season(n_games=n_games)

        # Evaluate
        print(f"  Evaluating...")
        report = evaluate_game_logs(test_df, sim_df, skip_pa_length_kl=True)
        metrics = report.as_dict()
        all_results[name] = metrics

        # Save results
        result_path = args.results_root / f"baselines_{name}.json"
        result_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
        print(f"  Results saved to {result_path}")
        print(f"  kl_run_distribution={metrics.get('kl_run_distribution', 'N/A'):.5f}, "
              f"mean_rg_error={metrics.get('mean_rg_error', 'N/A'):.5f}")

    print("\n\n=== BASELINE COMPARISON TABLE ===")
    print_table(all_results)


if __name__ == "__main__":
    main()
