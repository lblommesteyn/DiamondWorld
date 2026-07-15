"""Diagnose the sim's extra-inning inflation (sim ~12% vs real ~8.6%).

Hypothesis: positive home/away score correlation (shared park effect) lowers the
score-margin variance, which raises P(tie after 9) and thus the extra-inning rate,
while also inflating total-run variance. Compares the sim's after-9 score
structure (from simulate_games --dump-scores) to the real test data.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import polars as pl
from diamondworldjax.paths import processed_root

TEST = [2023, 2024]


def real_home_away():
    """Per-game (away9, home9) run totals through 9 innings, plus final."""
    root = processed_root()
    df = pl.concat([pl.read_parquet(root / f"pitches_{y}.parquet") for y in TEST],
                   how="diagonal_relaxed").filter(pl.col("pa_terminal"))
    # half_bin: 0 = top (away bats), 1 = bottom (home bats)
    # half: "top" = away bats, "bot" = home bats (same split as the simulator).
    reg = df.filter(pl.col("inning") <= 9)
    g = (reg.group_by("game_pk")
         .agg([(pl.col("runs_scored") * (pl.col("half") == "top")).sum().alias("away9"),
               (pl.col("runs_scored") * (pl.col("half") == "bot")).sum().alias("home9")]))
    return g["away9"].to_numpy().astype(float), g["home9"].to_numpy().astype(float)


def report(tag, away, home):
    away = np.asarray(away, float); home = np.asarray(home, float)
    m = ~np.isnan(away) & ~np.isnan(home)
    away, home = away[m], home[m]
    tie = (away == home).mean()
    margin = home - away
    corr = np.corrcoef(home, away)[0, 1]
    print(f"\n{tag} (n={len(away)})")
    print(f"  team runs: away mean={away.mean():.3f} sd={away.std():.3f} | "
          f"home mean={home.mean():.3f} sd={home.std():.3f}")
    print(f"  home-away corr = {corr:+.4f}")
    print(f"  margin(home-away): mean={margin.mean():+.3f} sd={margin.std():.3f}")
    print(f"  P(tie after 9) = {tie:.4f}   -> extra-inning rate")
    return tie, corr, margin.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True, help="npz from --dump-scores")
    args = ap.parse_args()
    ra, rh = real_home_away()
    rt, rc, rm = report("REAL (thru 9)", ra, rh)
    d = np.load(args.scores)
    st, sc, sm = report(f"SIM ({args.scores.name}, after9)", d["away9"], d["home9"])
    print("\n=== summary ===")
    print(f"  tie-after-9:   real {rt:.4f}   sim {st:.4f}   (sim/real {st/rt:.2f}x)")
    print(f"  home-away corr: real {rc:+.4f}   sim {sc:+.4f}")
    print(f"  margin sd:      real {rm:.3f}    sim {sm:.3f}")


if __name__ == "__main__":
    main()
