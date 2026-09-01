---
license: other
license_name: mlbam-derived
license_link: https://www.mlb.com/official-information/copyright
task_categories:
  - tabular-regression
  - time-series-forecasting
tags:
  - baseball
  - sports-analytics
  - simulation
  - statcast
pretty_name: DiamondWorld
---

# DiamondWorld

Data and evaluation artifacts for DiamondWorld, a plate-appearance-level baseball
world model. The headline metric is cross-player rate correlation: how well the
model's simulated season reproduces the K, BB, Hit and HR rates of individual
batters with at least 150 plate appearances in a held-out season.

## What is here

| Path | Size | What it is |
|---|---|---|
| `data/processed/pitches_YYYY.parquet` | 183 MB | Pitch-level records, 2015-2024, one file per season. This is what training and evaluation read. |
| `data/processed/validation_YYYY.json` | small | Per-season row counts and schema checks from the ingest. |
| `data/eval2/` | 72 MB | Evaluation outputs: paired bootstrap results, per-model player-correlation arrays, calibration and backtest reports. |
| `data/chunks/` | 1 MB | Per-chunk simulation checkpoints from the pre-game sweep. |
| `data/projections_2024.csv` | 159 KB | Public projection-system baseline used for comparison. |
| `RESULTS.md` | 89 KB | The full experiment log, including refuted ideas and corrections. |

Optional tiers, uploaded separately: `checkpoints/` (630 MB of trained parameters)
and `data/raw/` (32 GB of API and Statcast pulls, packed as archives).

## Read this before using the numbers

`RESULTS.md` opens with a correction banner, and it is not decorative. An external
review in August 2026 found three defects that had been silently inflating results:

1. **A metric that could not fail.** `p0_error` computed `P(runs >= 0)`, which is 1
   by construction, so a shutout-rate gap always scored as zero error.
2. **An unknown-player sink.** Embedding index 0 was not reserved for unseen
   players; it was the first real player in the training table, so every unseen
   player inherited that player's learned representation, and index 0 was then
   scored as if it were a batter with ~11,800 plate appearances.
3. **Bullpen leakage.** The "pre-game" simulator took each team's relievers, and
   their appearance order, from the completed game.

Defects 1 and 2 are fixed and every affected number in `RESULTS.md` has been
recomputed. The corrected headline is **0.624** average cross-player correlation
(K .792, BB .651, Hit .445, HR .610), not the 0.611 reported earlier. Several
previously reported improvements did not survive: the v21 gain fell from +0.025
(p=0.054) to +0.015 (p=0.130), and three ablation verdicts flipped.

Defect 3 affects **game-level** results only; the plate-appearance-level numbers
above are unaffected. The leak-free sweep has now been run, and the result is worth
stating plainly: with the leak removed, the simulator's win probabilities are **worse
than a constant home-field base rate** (log-loss 0.6985 vs 0.6923), where the leaky
arrays had them better (0.6881). Log5, which uses nothing but season win rates, gets
0.6709. The run-total distribution is unaffected by the leak and is where the model
genuinely performs: it reproduces real overdispersion (1.98x independent-Poisson
variance against a real 2.11x), though a league-wide negative binomial with no team
information is better calibrated still.

A power analysis over the same test season puts the minimum detectable effect at
about **+0.030** average correlation at 80% power. Differences smaller than that
are not resolvable with one season of held-out data, whatever their point estimate.

## Attenuation ceilings

Observed rates are noisy, so correlation against them is bounded. Method-of-moments
reliability gives per-rate ceilings of K 0.929, BB 0.852, Hit 0.691, HR 0.794, and
0.816 on average. The model is well short of those, so this is not a saturated
benchmark.

## Provenance and licensing

Derived from MLB Advanced Media game feeds and Statcast. This upload is a
research artifact; the underlying data is MLBAM's and is subject to their terms.
Check those terms before redistributing, especially the raw tier.

## Code

https://github.com/lblommesteyn/DiamondWorld
