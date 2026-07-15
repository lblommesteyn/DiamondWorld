"""Fast scorer: per-game-runs .npy vs the FULL real test distribution.

No baseline refit and no GPU -- just loads real once and prints the full
game_run_metrics suite for one or more dumped run vectors. Used to tune the
recal scale on the full test set (not a biased sim subset) and to build the
final scoreboard.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import polars as pl
from diamondworldjax.paths import processed_root
from diamondworldjax.eval.calibration import game_run_metrics

TEST = [2023, 2024]


def real_game_runs() -> np.ndarray:
    root = processed_root()
    df = pl.concat([pl.read_parquet(root / f"pitches_{y}.parquet") for y in TEST],
                   how="diagonal_relaxed").filter(pl.col("pa_terminal"))
    g = df.group_by("game_pk").agg(pl.col("runs_scored").sum().alias("r"))
    return g["r"].to_numpy().astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=str, nargs="+", required=True,
                    help="One or more per-game-runs .npy files (label:path or path).")
    args = ap.parse_args()
    real = real_game_runs()
    print(f"real: mean={real.mean():.3f} std={real.std():.3f} n={len(real)}  "
          f"P(>=8)={ (real>=8).mean():.4f}")
    print(f"\n{'label':22s} {'n':>5s} {'mean':>6s} {'std':>6s} {'KL':>8s} "
          f"{'wass':>7s} {'p5+err':>7s} {'p8+err':>7s} {'var_err':>8s}")
    for spec in args.runs:
        if ":" in spec and not spec[1:3] == ":\\":
            label, path = spec.split(":", 1)
        else:
            label, path = Path(spec).stem, spec
        runs = np.load(path)
        m = game_run_metrics(runs, real)
        print(f"{label:22s} {len(runs):5d} {runs.mean():6.2f} {runs.std():6.2f} "
              f"{m['kl_run_distribution']:8.4f} {m['wasserstein_runs']:7.3f} "
              f"{m['p5_plus_error']:7.4f} {m['p8_plus_error']:7.4f} {m['variance_error']:8.3f}")


if __name__ == "__main__":
    main()
