# Audit follow-up, 2026-09-20

Eight items, executed in order. Every number below was produced on this branch by
a script committed alongside it, and every comparison is paired or held out.

Summary of what changed:

| # | item | outcome |
|---|---|---|
| 1 | 2025 ingest + pooled slate | **DONE.** 2025 ingested and validated. Pooled MDE 0.029 -> ~0.021 |
| 2 | v22pf leakage | **PARTIAL.** Discrepancy resolved, gain replicated, mechanism narrowed. Blocked on the checkpoint |
| 3 | v27 vs v22 / v25a | **DONE. Decisive null.** v27 - v22 = +0.007 [-0.006, +0.021] |
| 4 | run-total mean bias | **DONE.** One parameter closes 99% of the calibration gap |
| 5 | Hit% extraction | **DONE, hypothesis revised.** Feature-side, not extraction-side. Fix implemented, retrain running |
| 6 | blend with Marcel+CQ | **DONE. Best result of the day.** 0.668, statistically level with Steamer |
| 7 | age curves + MiLB | **DONE. Near-null.** Age worth +0.001, not the +0.005-0.010 assumed |
| 8 | team defence + Log5 blend | **DONE. Route closed.** The simulator's WP signal is fully redundant |

---

## Item 1: the 2025 season is in, and it helps less than hoped

`scripts/run_ingest_2025.sh`, then `scripts/_pooled_slate.py`.

2025 Statcast plus MLB API runner states and umpires, ingested to
`data/processed/pitches_2025.parquet`. Structurally identical to 2024:

| season | pitches | games | terminal PAs | batters | launch-tracked |
|---|---|---|---|---|---|
| 2024 | 711,898 | 2,429 | 182,449 | 651 | 237,595 |
| 2025 | 712,528 | 2,430 | 182,926 | 673 | 235,711 |

League rates agree closely (2024 K .2219 BB .0936 hit .2211 HR .0308; 2025 K .2168
BB .0956 hit .2232 HR .0320), so 2025 is not malformed.

`TRACKING_SEASONS` now runs to 2025 and `restore_config`'s hard-coded 2025 bound is
a named `LAST_SEASON = 2026`. **Note the behaviour change:** a `train_end=2023`
checkpoint now defaults to testing 2024+2025 pooled, not 2024 alone, so any
comparison against an existing 2024-only rate file must pass `--test-seasons 2024`.

What it buys on the detection floor:

| cohort | n | MDE at 80% power |
|---|---|---|
| 2024 only | 410 | ~0.029 |
| 2025 only | 393 | ~0.030 |
| distinct batters across both | 498 | ~0.026 |
| pooled batter-seasons | 803 | ~0.021 |

**This is less than RESULTS.md implies.** 305 of the 498 distinct batters (61%)
appear in both seasons, so the 803 rows are not independent draws and the true
effective floor sits between 0.021 and 0.026. Against that floor:

- v27 joint vs v16, +0.026: detectable
- v22 bug fixes vs v16, +0.019: still under
- blend vs v27, +0.013: still under
- v27 vs v22, +0.007: still under

So 2025 does not rescue the sub-0.02 effects this project produces. Getting there
needs more seasons or a metric with more resolution, not one more season.

---

## Item 2: the v22pf discrepancy is resolved; the leakage question is not

`data/eval2/bootstrap_AUDIT_v22pf.txt`.

There were two different v22pf runs, which is the whole discrepancy:

| tag | K | BB | Hit | HR | AVG |
|---|---|---|---|---|---|
| v22_s42 (no pitchformer) | 0.797 | 0.669 | 0.483 | 0.627 | 0.644 |
| v22pf, earlier run | 0.779 | 0.693 | 0.474 | 0.592 | **0.635** |
| v22pf_s42 | 0.838 | 0.760 | 0.506 | 0.620 | **0.681** |
| v22pf_s97 | 0.836 | 0.756 | 0.500 | 0.613 | **0.676** |

LADDER.md's 0.635 is the earlier run. The 0.681 replicates at 0.676 on a second
seed, so it is not a fluke. Paired against v22_s42 it is +0.037 [+0.022, +0.053],
**concentrated entirely in K (+0.041) and BB (+0.091) with Hit (+0.024, ns) and HR
(-0.007, ns) flat.**

What the code rules out:

- The PA state vector is 8 dimensions (inning, half, outs, base state, score diff,
  TTO, shift, clock). **No count.** So it is not naive ball/strike leakage.
- Fatigue uses `pitch_count_game - pitch_count_pa`, correctly excluding the current
  PA's own pitches.
- `context_raw = concat([game_state, pitcher_z, batter_z, park_emb])` carries **no
  outcome history**, and `causal_mask` is `tril(k=-1)`, strictly past. So a PA
  cannot see its own outcome or any later one.
- The player table is taken from `pa_metadata` inside the checkpoint, built at
  training time from training seasons, so test-season rates cannot enter it.

What remains, and why it matters more than "leak or no leak":

The PA transformer attends over **all previous PAs in the game**, and the state
channels it aggregates (base state, score diff, TTO) are consequences of what
happened earlier in that test-season game. That is legitimate conditioning for a
simulator and it is the same protocol every version has used. But **Steamer gets no
in-game information at all**, so `0.681 > 0.671` is not a like-for-like win, and
the amount of in-game conditioning being exploited is exactly what grew between
v22 and v22pf. A lineup-slot effect would also produce a K/BB-only gain, and would
be entirely legitimate.

**Blocked:** no v22-v27 checkpoint exists locally (only v0-v21). The decisive
experiment needs `checkpoints/dwjax_pa_v22pf*/`:

> Score v22pf twice, once normally and once with the PA-transformer history
> ablated so each PA attends to nothing (the fully-masked path `CausalBlock`
> already handles). If the +0.041 K and +0.091 BB survive, they come from the
> player representation and the number is real. If they collapse toward v22, they
> come from within-game context, and the Steamer comparison must be withdrawn.

Until that runs, **do not quote 0.681 against Steamer.**

---

## Item 3: nothing in v23 through v27 did anything

`data/eval2/bootstrap_AUDIT_vs_v22_s42.txt`. This comparison had never been run;
every table in the handoff was paired against v16 only.

Paired against **v22_s42** (the bug-fix rung), 20,000 reps, 382 batters:

| comparison | K | BB | Hit | HR | AVG |
|---|---|---|---|---|---|
| v16 - v22_s42 | -0.005 | -0.018* | -0.038* | -0.017 | **-0.019 [-0.031, -0.007] p=0.003*** |
| v25a_pa - v22_s42 | -0.009 | +0.013 | -0.019 | -0.001 | -0.004 [-0.015, +0.007] p=0.488 |
| v26a_pa - v22_s42 | -0.018 | +0.031* | -0.053* | +0.019 | -0.005 [-0.020, +0.010] p=0.500 |
| v27a_pa - v22_s42 | -0.008 | +0.024* | -0.024 | +0.035* | **+0.007 [-0.006, +0.021] p=0.321** |

**The bug fixes are real (+0.019, p=0.003). Everything built on top of them is a
null.** The v27 interval is tight (width 0.027), so this is a well-powered null,
not an underpowered one: it rules out a +0.021 or larger gain from joint ABCD
training.

The sub-structure is the interesting part. v27 genuinely moves two cells against
v22 (BB +0.024, HR +0.035, both CIs excluding zero) and gives it all back on Hit
(-0.024) and K (-0.008). **Joint training redistributes accuracy across stats
without adding any.** That is a more specific finding than "no effect" and worth
keeping.

Practical consequence: **build on v22, not v27.** v27 carries the cost of the joint
ABCD machinery for no measured benefit.

---

## Item 4: the calibration gap was entirely the mean

`diamondworldjax/scripts/rundist_bias.py`, fitted on 1,214 games and scored on the
held-out 1,215.

The leak-free simulator overproduces by +0.459 runs per game. Correcting it with a
single parameter, a randomized location shift (subtract one run with probability
0.459):

| model | mean | var | logscore | KS | c50 | c80 | c90 |
|---|---|---|---|---|---|---|---|
| sim, uncorrected | 9.03 | 18.91 | 2.855 | 0.0514 | 0.537 | 0.808 | 0.902 |
| **sim + shift** | 8.57 | 19.15 | 2.876 | **0.0163** | 0.545 | 0.827 | 0.911 |
| sim + thin | 8.57 | 17.53 | 2.851 | 0.0246 | 0.535 | 0.807 | 0.898 |
| league neg-binomial | 8.62 | 18.92 | 2.882 | 0.0159 | 0.550 | 0.829 | 0.919 |

real: mean 8.60, var 18.85.

**A one-parameter fix closes 99% of the calibration gap** (KS 0.0514 -> 0.0163
against the NB's 0.0159).

Two honest caveats. **Parity, not victory:** the corrected simulator matches a
baseline that uses no team, park or roster information; it does not beat it. And
the mean bias was masking a real tension, since the shift costs logscore (2.855 ->
2.876) while thinning fixes logscore and costs KS. Pick the metric before claiming
the win. The +0.46 mechanism is still unexplained and is the actual fix.

**Separately, a reproducibility catch.** Re-running the benchmark on the v16 arrays
with current code reproduces RESULTS.md exactly and deterministically (sim 2.855,
NB 2.875, KS 0.0579/0.0121). The NB is seeded at `default_rng(0)`. So the 2.853 that
Jaden's v27a report gives for the same baseline on the same slate cannot be
sampling noise: **those two reports came from different pipeline states**, and
sim-versus-NB numbers must not be compared across them.

---

## Item 5: the diagnosis revised the hypothesis

`diamondworldjax/scripts/hit_extraction.py`.

I expected to find the model failing to extract its own contact-quality feature.
Mostly it is not:

| model | its hit corr | partial corr(expected-hit, actual \| model pred) | incremental R² |
|---|---|---|---|
| v16 | 0.448 | +0.116 | +0.0108 |
| v22_s42 | 0.484 | +0.042 | +0.0014 |
| v27a | 0.460 | +0.047 | +0.0017 |

v16 does leave real signal unused, but **v22 and v27 leave almost none** (R² +0.0014).
The bug fixes largely closed the extraction gap. And each model's hit prediction
(0.46-0.48) already beats the raw feature alone (0.409).

So Marcel+CQ's 0.497 does not come from extracting the feature better. It comes
from the feature being **a better estimator**: a 3-season Marcel window with the
measured regression constant, against the model's single unshrunk recency-weighted
column. The real defect is upstream of the model.

**Mechanism, confirmed.** In `_build_player_table`, `--per-stat-shrink` shrinks
columns 0..3 and contact quality then overwrites columns 5:7 **afterward, unshrunk**.
Measured consequence: the expected-hit column correlates 0.275 with the realized
2024 rate on the low-PA half of batters and 0.578 on the high-PA half.

**Fix implemented** as `--shrink-contact-quality` (default off, so the baseline path
is provably unchanged). Constants tuned on 2023 with the table built from 2015-2022,
so 2024 never informed the choice:

| column | reg | holdout 2023 | 2024 raw -> shrunk |
|---|---|---|---|
| expected hit (5) | 700 | 0.359 -> 0.464 | 0.409 -> **0.496** |
| expected HR (6) | 100 | 0.599 -> 0.634 | 0.630 -> **0.643** |

Both holdout choices match the 2024 oracle exactly, so the selection costs nothing.
**My first guess of 2200 for both was wrong** and would have made HR worse than raw
(0.614 against 0.630); the constants differ by 7x between the two columns. Same
lesson as "a single REG=1200 is badly wrong at both ends," on a different pair.

The shrunk expected-hit feature at 0.496 is now level with the Marcel+CQ hit
projection at 0.497, which is the whole point.

Refactor safety: shrinkage is now one helper, `shrink_toward_league`, verified
bit-identical against the pre-refactor source from git across all four flag
combinations that existing checkpoints used, plus six unit tests.

**v28 = v22 + this one lever** is training (`scripts/run_v28.sh`, seed 42, 50k
steps). It answers whether the model converts a better input into a better output,
which the v17-v21 series gives real reason to doubt.

---

## Item 6: the blend reaches Steamer, and the model version barely matters

`diamondworldjax/scripts/blend_projections.py`. 376 batters covered by all systems.
Weights fitted out of fold over batters; correlations from out-of-fold predictions.

| system | K | BB | Hit | HR | AVG |
|---|---|---|---|---|---|
| DiamondWorld v27a | 0.795 | 0.690 | 0.469 | 0.663 | 0.654 |
| Marcel+CQ | 0.793 | 0.674 | 0.497 | 0.641 | 0.651 |
| Steamer | 0.820 | 0.702 | 0.510 | 0.651 | 0.671 |
| **Blend 50/50, no fitting** | 0.799 | 0.689 | **0.509** | **0.674** | **0.668** |
| Blend fitted, out-of-fold | 0.794 | 0.687 | 0.497 | 0.669 | 0.662 |
| Blend + Steamer, out-of-fold | 0.819 | 0.709 | 0.501 | 0.672 | 0.675 |

Paired bootstrap on AVG, 20,000 reps:

- Blend 50/50 - v27a: **+0.013 [+0.002, +0.024] p=0.015***
- Blend 50/50 - Marcel+CQ: **+0.016 [+0.006, +0.027] p=0.003***
- Blend 50/50 - Steamer: **-0.003 [-0.021, +0.014] p=0.724** — level with Steamer
- Blend + Steamer - Steamer: +0.005 [-0.005, +0.013] p=0.329 — adding our model to
  Steamer does **not** reliably improve Steamer

**A zero-parameter blend is statistically indistinguishable from Steamer.** Hit goes
0.469 -> 0.509 (Steamer 0.510) and HR 0.663 -> 0.674, beating Steamer. K remains
the deficit (0.799 against 0.820).

And it does not depend on the model:

| DW model fed in | its own AVG | blended AVG |
|---|---|---|
| v16 | 0.631 | 0.664 |
| v22_s42 | 0.648 | 0.668 |
| v27a | 0.654 | 0.668 |

The blend lands at 0.664-0.668 whichever model feeds it, while the models
themselves span 0.631-0.654. **The blend is the win; the model version is close to
irrelevant.** That corroborates item 3 independently.

Note the fitted blend is *worse* than 50/50 (0.662 against 0.668): least squares
minimises squared error on the rate scale, not correlation. **Quote the 50/50
number** — it fits nothing, so it cannot overfit, and it is the better one.

---

## Item 7: age curves are worth +0.001, not +0.005 to +0.010

`diamondworldjax/scripts/fetch_birthdates.py` (4,017 players, zero missing) and
`diamondworldjax/scripts/age_curve.py`.

The curve is fitted within player from consecutive-season pairs with 300+ PA in
both, on training seasons only, which avoids the survivorship bias that a
cross-sectional age profile would have. It is recognisable baseball: K rate rises
from about age 26 (+0.005/yr, reaching +0.015 by 35), hit rate falls through the
30s (-0.004 to -0.009/yr), HR peaks mid-20s.

Adjustment strength tuned on 2023, which chose a heavily damped 0.25. Applied to 2024:

| projection | K | BB | hit | HR | AVG |
|---|---|---|---|---|---|
| Marcel (tuned reg) | 0.794 | 0.677 | 0.428 | 0.608 | 0.627 |
| Marcel + age | 0.796 | 0.677 | 0.432 | 0.606 | 0.628 |
| Marcel + CQ | 0.794 | 0.677 | 0.497 | 0.641 | 0.652 |
| Marcel + CQ + age | 0.796 | 0.677 | 0.503 | 0.637 | 0.653 |

**+0.0014 and +0.0012.** An order of magnitude under the MDE.

The reason is mechanical: a three-season recency-weighted window already carries
most of the aging signal implicitly, so an explicit age term is nearly redundant.
This refutes the folk explanation that age curves are a large part of Steamer's
edge, at least in the Marcel framing. It does not prove age is useless inside a
full regression with minor-league and park terms.

MiLB translation, bounded rather than built (`scripts/_channels_bound.py`): 30.5% of
the 2024 cohort has 500 or fewer prior MLB PA, so a generous +0.10 lift confined to
that slice bounds the pooled gain near +0.031. That is **above** the pooled MDE and
comparable to the remaining Steamer gap, so it is **not ruled out** — but the bound
is loose and optimistic, and it needs a minor-league data fetch before it can be
measured. It ranks behind everything already measured.

---

## Item 8: the win-probability route is closed

`diamondworldjax/scripts/wp_blend.py` and `scripts/_channels_bound.py`. Half-sample
fit, held-out half scored.

| model | logloss | Brier | AUC | ECE |
|---|---|---|---|---|
| DiamondWorld sim | 0.6938 | 0.2501 | 0.549 | 0.048 |
| sim, recalibrated | 0.6888 | 0.2478 | **0.549** | 0.043 |
| Log5 | 0.6609 | 0.2344 | 0.635 | 0.037 |
| Market (devig close) | 0.6599 | 0.2338 | 0.639 | 0.026 |
| BLEND sim + Log5 | 0.6655 | 0.2365 | 0.635 | 0.052 |

Blend coefficients on the logit scale: **sim +0.008, Log5 +0.832.** Given free rein,
logistic regression assigns the simulator essentially no weight, and the blend is
worse than Log5 alone.

**My decorrelated-errors prediction is refuted by measurement.** Confirmed a second
way with the raw team aggregates:

| model | logloss | AUC |
|---|---|---|
| sim alone | 0.6927 | 0.539 |
| Log5 alone | 0.6765 | 0.614 |
| sim + offence diff + defence diff | 0.6768 | 0.611 |
| **offence diff + defence diff only** | **0.6768** | **0.611** |

Two numbers per team, runs scored and runs allowed per game, reproduce Log5 exactly,
and **adding the simulator on top changes nothing to four decimal places.**

So the simulator's win-probability signal is entirely redundant given season-level
team strength. Building a team-defence channel would recover information that two
aggregates already encode, and the market still beats Log5 anyway. Recalibration is
independently dead: AUC is 0.549 before and after, exactly as monotone invariance
requires.

**Recommendation: stop work on the game-prediction axis.** It is not a tuning
problem and it is not a blending problem.

---

## Where this leaves the project

1. **Build on v22.** The bug fixes are the only model-side gain that replicates, and
   the joint ABCD machinery costs complexity for a measured zero.
2. **The blend is the headline.** Statistically level with Steamer, zero fitted
   parameters, reproducible from committed scripts. It is also honest about being an
   ensemble rather than a better model.
3. **The distributional claim needs restating.** After the mean fix the simulator is
   level with a no-information baseline on calibration, not ahead of it. What
   survives is correlated overdispersion plus per-batter stat lines from one
   generative process, which no baseline here does at all.
4. **The game axis is closed.** Two independent tests say so.
5. **Remaining measured-but-unexplained:** the +0.46 run-per-game bias mechanism, and
   the 0.021 K-rate gap to Steamer, which is now the largest single deficit left.
6. **Still blocked:** the v22pf leakage question, on a checkpoint.
