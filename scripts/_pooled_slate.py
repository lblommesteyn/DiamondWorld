"""Item 1: what the pooled 2024+2025 test slate does to the detection floor.

The power analysis in RESULTS.md puts the minimum detectable effect at +0.030 AVG
at 80% power on 382 batters, and states that the batter count is the only thing
that moves it: "Nothing about seeds, steps or architecture" does. This measures
the new batter count and the MDE that follows.

Batter-seasons, not batters, are the unit. A batter who clears 150 PA in both
2024 and 2025 contributes two rows, and those two rows are close to independent
draws of season-level luck even though the underlying talent is shared. That is
the sense in which pooling buys power, and it is also the sense in which it does
NOT buy as much as two fully disjoint cohorts would, so the number below is an
upper bound on the gain.
"""
import numpy as np
import polars as pl

from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.paths import processed_root

MIN_PA = 150
HIT = ("1B", "2B", "3B", "HR")


def season_counts(year):
    d = load_seasons([year], data_root=processed_root()).filter(
        pl.col("pa_terminal") & pl.col("pa_outcome").is_not_null())
    return (d.group_by("batter_id").agg([
        pl.col("pa_outcome").is_in(HIT).sum().alias("hit"),
        pl.col("pa_outcome").is_in(["BB", "HBP"]).sum().alias("bb"),
        (pl.col("pa_outcome") == "K").sum().alias("k"),
        (pl.col("pa_outcome") == "HR").sum().alias("hr"),
        pl.len().alias("pa")]).filter(pl.col("pa") >= MIN_PA))


c24, c25 = season_counts(2024), season_counts(2025)
b24 = set(c24["batter_id"].to_list())
b25 = set(c25["batter_id"].to_list())

n24, n25 = len(b24), len(b25)
pooled_rows = n24 + n25
distinct = len(b24 | b25)
overlap = len(b24 & b25)

print("batters clearing " + str(MIN_PA) + " PA:")
print("  2024                  " + str(n24))
print("  2025                  " + str(n25))
print("  distinct across both  " + str(distinct))
print("  in BOTH seasons       " + str(overlap)
      + "  (" + format(100.0 * overlap / max(distinct, 1), ".0f") + "% of distinct)")
print("  pooled batter-seasons " + str(pooled_rows))
print()

# The paired-bootstrap interval width scales as 1/sqrt(rows), so the MDE does too.
MDE_AT_382 = 0.030
for label, rows in (("2024 only", n24), ("2025 only", n25),
                    ("pooled batter-seasons", pooled_rows),
                    ("distinct batters", distinct)):
    mde = MDE_AT_382 * np.sqrt(382.0 / rows)
    print("  MDE at 80% power, " + label.ljust(22) + " n=" + str(rows).rjust(4)
          + "  ~" + format(mde, ".4f"))
print()
print("Effects this project has actually produced, against the pooled floor:")
for name, eff in (("v22 bug fixes vs v16", 0.019), ("v21 corrected", 0.015),
                  ("v27 joint vs v16", 0.026), ("v27 joint vs v22", 0.007),
                  ("blend 50/50 vs v27", 0.013)):
    mde = MDE_AT_382 * np.sqrt(382.0 / pooled_rows)
    verdict = "detectable" if eff >= mde else "still under the floor"
    print("  " + name.ljust(24) + format(eff, "+.3f") + "   " + verdict)
print()

# Sanity: league rates should agree closely between the two seasons.
print("league rates (sanity check that 2025 is not malformed):")
for name, c in (("2024", c24), ("2025", c25)):
    tot = c["pa"].sum()
    print("  " + name + "  K " + format(c["k"].sum() / tot, ".4f")
          + "  BB " + format(c["bb"].sum() / tot, ".4f")
          + "  hit " + format(c["hit"].sum() / tot, ".4f")
          + "  HR " + format(c["hr"].sum() / tot, ".4f"))
