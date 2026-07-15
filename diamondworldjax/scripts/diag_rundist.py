"""Diagnose the per-game run-total distribution shape: sim vs real.

Isolates whether the tail error (p8+, wasserstein) is a recal-scale artifact, an
extras-inflation artifact, or a genuine within-game variance/shape mismatch.
Reads a per-game-runs .npy (from simulate_games --dump-runs) and the real test
distribution, and prints histograms, tail masses, and moments side by side.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import polars as pl
from diamondworldjax.paths import processed_root

TEST = [2023, 2024]


def real_game_runs() -> np.ndarray:
    root = processed_root()
    df = pl.concat([pl.read_parquet(root / f"pitches_{y}.parquet") for y in TEST],
                   how="diagonal_relaxed").filter(pl.col("pa_terminal"))
    g = df.group_by("game_pk").agg(pl.col("runs_scored").sum().alias("r"))
    return g["r"].to_numpy().astype(float)


def real_extras_and_ties() -> tuple[float, float]:
    root = processed_root()
    df = pl.concat([pl.read_parquet(root / f"pitches_{y}.parquet") for y in TEST],
                   how="diagonal_relaxed").filter(pl.col("pa_terminal"))
    per = df.group_by("game_pk").agg(pl.col("inning").max().alias("mi"))
    extras = (per["mi"] >= 10).mean()
    return float(extras), float("nan")


def summarize(name, runs):
    runs = np.asarray(runs, float)
    q = np.percentile(runs, [50, 90, 95, 99])
    print(f"\n{name}  (n={len(runs)})")
    print(f"  mean={runs.mean():.3f} std={runs.std():.3f} var={runs.var():.3f} "
          f"skew={float(((runs-runs.mean())**3).mean()/runs.std()**3):.3f}")
    print(f"  median={q[0]:.1f} p90={q[1]:.1f} p95={q[2]:.1f} p99={q[3]:.1f} max={runs.max():.0f}")
    for t in (0, 3, 5, 8, 11, 14, 17):
        print(f"    P(>= {t:2d}) = {(runs >= t).mean():.4f}", end="")
    print()
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-runs", type=Path, required=True)
    args = ap.parse_args()

    real = real_game_runs()
    sim = np.load(args.sim_runs)
    r = summarize("REAL", real)
    s = summarize(f"SIM ({args.sim_runs.name})", sim)

    print("\n=== histogram (fraction of games) ===")
    print(f"  {'runs':>5s} {'real':>8s} {'sim':>8s} {'diff':>8s}")
    bins = np.arange(0, 21)
    rh, _ = np.histogram(r, bins=np.append(bins, 999), density=True)
    sh, _ = np.histogram(s, bins=np.append(bins, 999), density=True)
    for i, b in enumerate(bins):
        flag = "  <---" if abs(rh[i] - sh[i]) > 0.012 else ""
        print(f"  {b:5d} {rh[i]:8.4f} {sh[i]:8.4f} {sh[i]-rh[i]:+8.4f}{flag}")

    ex, _ = real_extras_and_ties()
    print(f"\nreal extra-inning game rate: {ex:.4f}")


if __name__ == "__main__":
    main()
