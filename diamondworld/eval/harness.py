from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from diamondworld.eval.metrics import evaluate_game_logs


def evaluate_files(
    empirical_path: Path,
    simulated_path: Path,
    output_path: Path | None = None,
    *,
    skip_pa_length_kl: bool = False,
) -> dict[str, float]:
    empirical = pl.read_parquet(empirical_path)
    simulated = pl.read_parquet(simulated_path)
    report = evaluate_game_logs(empirical, simulated, skip_pa_length_kl=skip_pa_length_kl).as_dict()
    if output_path is not None:
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate simulator game logs against empirical data.")
    parser.add_argument("--empirical", type=Path, required=True)
    parser.add_argument("--simulated", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(evaluate_files(args.empirical, args.simulated, args.output), indent=2))


if __name__ == "__main__":
    main()
