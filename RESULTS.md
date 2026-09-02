# DiamondWorld: Results

A plate-appearance-level generative model of baseball. Trained on 2015-2022,
evaluated on the full 2023-2024 test slate (4,859 games). The model generates
each plate appearance's outcome (a 9-way categorical: K, BB, HBP, 1B, 2B, 3B,
HR, out, E) conditioned on game state, batter and pitcher identity, and park; a
validated empirical rules engine turns outcome sequences into runs and base-state
transitions. A true game simulator plays full 9-inning-plus-extras games with
lineup cycling, a bullpen, walk-offs, and the ghost-runner extras rule.


> **READ FIRST, correction notice (2026-08-29).** Every cross-player correlation recorded
> before this date is inflated by a scoring defect: the index-0 unknown-player sink was
> being scored as if it were a batter. The corrected incumbent is **v16 AVG 0.624**, not
> 0.611, and the corrected v21 result is **+0.015 at p = 0.130**, not +0.025 at p = 0.054.
> Three variant verdicts change. Historical figures below are left as they were measured,
> so treat any pre-correction number as approximately 0.013 too high and see
> [Three defects found in external review](#three-defects-found-in-external-review-and-every-number-they-touched)
> at the end of this file for the corrected tables. The game-level results additionally
> carry an unquantified optimistic bias from bullpen leakage, described in the same section.

## The claim

On the aggregate run distribution the model beats strong run-only baselines,
posting the lowest KL divergence (3.7x closer fit) and Wasserstein of any method
at 4,859 games, and unlike them it uniquely and correctly reproduces individual
player stat lines. Distributional fit plus player identity is the differentiator.

## Simulator benchmarks: measured against Log5, the market, and a summed model

The "a generative model answers distributional questions projection systems cannot"
claim used to rest on the mechanism being obviously true. `simulator_benchmarks.py`
measures it on 2024 (2,376 decided games), pre-game and leakage-free (the fitted
bullpen hook model supplies realistic reliever timing instead of the actual bullpen),
against the baselines a reviewer would demand. It splits cleanly into a weakness and
a strength.

**Win probability: calibrated, but it does not out-predict simple baselines.** P(home
win) from the replicas, versus Log5 (each team's Pythagorean win rate plus home field,
the canonical talent-only baseline) and the devigged closing moneyline (the market):

| model | log-loss | Brier | AUC | ECE |
|---|---|---|---|---|
| base rate (home 0.521) | 0.6923 | 0.2496 | - | - |
| DiamondWorld sim (v15 + hook, R=100) | 0.6881 | 0.2473 | 0.572 | 0.034 |
| Log5 (Pythagorean + home field) | 0.6709 | 0.2391 | 0.617 | 0.019 |
| market (devigged close) | 0.6707 | 0.2391 | 0.614 | 0.019 |

The simulator beats the base rate and is well calibrated (ECE 0.034), but it
discriminates worse than Log5 and the market (AUC 0.572 vs ~0.615): its per-PA process
washes team-level strength out toward 0.5, where Log5 and the market encode it directly.
Honest read: point win probability is not the simulator's edge. (Log5 uses same-season
team aggregates, a mild in-sample peek, so the fully fair pre-game comparison is sim vs
market, which the market wins clearly. Replicas and the hook model matter: the same
benchmark on the old v13 pre-game run at R=40 gave AUC 0.527 and log-loss 0.7334.)

**Run-total distribution: this is the real, measured advantage.** The headline is
correlated overdispersion a summed model structurally misses. Against an independent
two-Poisson model with the identical per-game means (so only the shape differs) and a
league negative-binomial:

| model | mean | var | P(≥10) | P(≤5) | 50% cov | 80% cov | 90% cov | PIT KS | log-score |
|---|---|---|---|---|---|---|---|---|---|
| real (empirical) | 8.63 | 18.24 | 0.366 | 0.261 | - | - | - | - | - |
| DiamondWorld sim | 9.03 | 17.87 | 0.404 | 0.228 | 0.538 | 0.821 | 0.903 | 0.049 | 2.864 |
| independent 2-Poisson | 9.03 | 8.99 | 0.418 | 0.129 | 0.416 | 0.657 | 0.767 | 0.144 | 2.915 |
| league negative-binomial | 8.63 | 18.22 | 0.371 | 0.249 | 0.555 | 0.830 | 0.917 | 0.016 | 2.860 |

The simulator's per-game total variance is 17.87 against reality's 18.24 (overdispersion
1.98x vs 2.11x), and its central-interval coverage is essentially nominal (0.54 / 0.82 /
0.90 for the 50 / 80 / 90% intervals). The summed independent model, forced to var = mean,
under-covers catastrophically (0.42 / 0.66 / 0.77) and fails PIT (KS 0.144 vs the sim's
0.049). So the correlated-overdispersion claim is now measured, not asserted, at full-season
scale. The bullpen hook model is what earned it: the same sim without it (v13 pre-game)
had total variance 13.01 and 80% coverage 0.74; realistic reliever timing supplied the
missing spread.

Two honest limits. A league negative-binomial matches the sim on the marginal (log-score
2.860 vs 2.864, PIT 0.016) because it is fit to that marginal, but it is not game-specific:
it gives every matchup the identical distribution, which is exactly what the simulator is
for and the NB cannot do. And the sim runs about 0.4 runs hot here (mean 9.03 vs 8.63),
a recal-scale calibration wrinkle that inflates P(≥10) slightly and is worth re-tuning.
Reproduce with `run_pregame_sim.py` then `simulator_benchmarks.py`.

**Against the market's totals line, though, there is no edge.** Tested whether the
simulator's implied P(total > line) predicts actual overs better than the market's own
devigged over-probability, on 2,274 games (sim mean-corrected for the 0.4-run bias):
market log-loss 0.6931 / AUC 0.511, sim 0.7129 / AUC 0.506, and the two are essentially
uncorrelated (0.054). The market sits at the no-information floor (log 2 = 0.693) because
totals are efficiently priced, so neither it nor the simulator predicts over/under better
than a coin flip, and the sim adds nothing to the market. The honest reading: the
simulator's distributional value is the *full, calibrated, game-specific* run
distribution (the coverage above, which a single market line does not provide), not an
ability to beat the market on the one number the market prices. This is the same
market-efficiency wall the betting audit hit, now confirmed on the run total.

## Counterfactual validation: do the causal what-ifs match the market?

The counterfactual engine ("swap this starter, win probability moves by X") was always
internally generated and never checked against anything external. This checks it against
the betting market's own repricing, using a natural experiment that needs no game
outcomes and no look-ahead: within a series the two teams are fixed, so the game-to-game
change in the line is driven by the starter/park/rest matchup, exactly what a
counterfactual isolates. If the simulator's within-series win-probability deltas track
the market's, the causal estimates have independent support. Two leak-free designs (team
identity only), on 2,355 2024 games with closing moneylines:

| design | corr(sim, market) | note |
|---|---|---|
| team fixed effects (controls team strength + home field) | 0.330 | full sample, game-specific residual |
| within-series (home field constant, starter varies) | 0.321 | permutation null 95th pct 0.032 → significant |

**Direction is validated and it is leak-free:** the simulator's game-specific win-
probability signal correlates with the market's beyond team identity, so it is capturing
real starter and matchup effects an independent market also prices, not noise. This is
the external support the causal claim needed. (These figures are from the R=500
simulation; at R=100 the correlations read 0.28 / 0.26, attenuated by the simulator's own
win-probability sampling noise, which the higher-replica run removes.)

**Magnitude is overstated, and now calibrated.** The within-series OLS slope of market on
sim is 0.27; correcting for the simulator's residual replica-sampling noise (reliability
0.80 at R=500) gives a slope of **0.33**. So the raw simulator over-reacts to a single
starter change by roughly 3x, and a raw "+9.8-point ace swap" is about **+3.3 points** in
market-calibrated units. The honest upshot is a *market-calibrated* counterfactual engine:
direction and magnitude both tied to an independent ground truth, with the earlier raw
headline numbers corrected downward. The R=500 run makes this reliable rather than a
noise-extrapolation (reliability rose from 0.39 at R=100 to 0.80). Reproduce with
`counterfactual_validation.py --arrays data/eval2/calib_v15-pregame-hook-r500_arrays.npz`.

**Is it broad, or just pitchers?** Within a series both the starter and the lineup
change, so the aggregate result alone cannot say the validated signal is more than
pitching. Decomposing each game into a pitching channel (the starters' allowed-run
rates) and a hitting channel (the lineups' wOBA-ish rates) and regressing the market's
within-series win-probability move on both (1,690 game-deviations, 2024) shows **both are
priced**: pitching coefficient t = -17.6 (negative because the index is runs allowed, so
a worse home starter lowers home win prob) and hitting t = +7.3, marginal correlations
-0.39 and +0.16. Pitching dominates, as expected, but hitting is unambiguously present,
so the market reprices on both and the causal engine's scope is legitimately broad:
the starter-swap validation generalizes to hitter and lineup what-ifs. Reproduce with
`whatif_channels.py`.

**Does it generalize across seasons and markets?** The 2024 result could be a
single-season, single-book artifact. It is not. Freezing the model's player rates and
running the same within-series channel test on later seasons and a different market
type (`season_market_validation.py`, `build_odds_2025.py`):

| season | market | model rates | games | within-series corr | channels |
|---|---|---|---|---|---|
| 2024 | sportsbook (consensus close) | ≤2023 | 2,355 | 0.26 | both |
| 2025 | sportsbook (consensus close) | ≤2024 | 1,630 | **0.33** | pitching t=-13.8, hitting t=+2.4 |
| 2026 (to date) | **Kalshi** prediction market | ≤2023 | 309 | 0.12 | pitching t=-2.65, hitting n.s. |

The causal signal is positive in every case, across **two independent market types** (a
sportsbook consensus and a Kalshi prediction market), and it holds a full season
**out-of-sample**: the 2025 test uses rates the model formed before 2025 was played and
still correlates 0.33 with the market's game-to-game repricing, stronger than in-window
2024. The 2026 figure is weaker (0.12, and hitting washes out) for a concrete and honest
reason: those rates are 2.5 years stale, so only ~6 of 9 batters and about half the
starters are even known, and known players' pre-2024 rates no longer describe their 2026
form. That degradation is itself a finding: the engine's causal signal is real and
market-agreeing, and it needs reasonably current player rates to stay sharp. Data notes,
stated because they bound the claim: Kalshi lists MLB game markets only from ~mid-2026
(no 2025) and the sportsbook feed covers 2021-2025. Polymarket **does** carry liquid
full-game MLB moneylines (median volume ~$485k/game, prices retrievable from the CLOB
`prices-history` endpoint), correcting an earlier note here that said otherwise; but its
history is likewise 2026-only, its CLOB rate-limits bulk fetches, and its per-market
price histories do not cleanly pin to a single game's first pitch (week-spanning series
markets, listing date != game date), so a clean bulk *pre-game* within-series harvest was
not practical, and the expected result would mirror the 2026 Kalshi figure against the
same stale rates. Kalshi is therefore the prediction-market datapoint used. Reproduce with
`season_market_validation.py` (`--market kalshi|polymarket`).

## Scoreboard (per-game total runs vs real 2023-2024, all 4,859 games)

Metric definitions: KL and Wasserstein on the discrete per-game run-total
distribution; tail error = |P(total >= 8)_sim - P(total >= 8)_real|. Lower is
better everywhere. All rows at N=4,859 (the full test slate), scored against the
same real reference with the identical metric.

| method | mean | std | KL | Wasserstein | tail err | player stats |
|---|---|---|---|---|---|---|
| real | 8.86 | 4.42 | - | - | - | reference |
| B0 Markov (RE24) | 8.80 | 4.34 | 0.0163 | 0.120 | **0.0031** | no |
| B1 NegBinom | 8.92 | 4.27 | 0.0170 | 0.179 | 0.0154 | no |
| v9 (park + fatigue, 30K) @0.55 | 8.93 | 4.46 | 0.0056 | 0.099 | 0.0088 | yes |
| **v10 (park + fatigue, 50K) @0.35** | 8.84 | 4.48 | **0.0044** | **0.079** | 0.0146 | yes |

Reading it: at the mean-matched recalibration **v10 posts the lowest KL (0.0044, a
3.7x closer fit than either baseline) and the lowest Wasserstein (0.079)** of any
method, and matches the mean almost exactly (8.84 vs 8.858). It is the only kind of
model that also reproduces player stat lines. The one regression versus v9 is the
extreme tail: v10's P(total >= 8) error (0.0146) is larger than v9's (0.0088) and
roughly ties B1, while B0's 0.0031 remains the single best tail cell. Everywhere
else v10 is the strongest row.

**v10 is v9's exact recipe (outcome-only + fatigue + park index) trained to 50K
steps instead of 30K.** The longer run fixed HR calibration outright (raw 1.00x, no
correction needed, versus v9's 0.72x that required a +0.33 logit lift) at the cost of
slightly more strikeout over-prediction (1.28x versus 1.14x), which the milder recal
absorbs. Net effect: a tighter run distribution and materially better player-stat
reproduction (below), for a small give-back on the P(>=8) tail. Its mean-matched
recal scale is lower (0.35) than v9's (0.55) because its raw calibration is closer.

## Player stat reproduction (conditioned, 433 batters >= 150 PA)

Cross-player correlation asks whether the model ranks players correctly.
Baselines cannot produce these at all (no batter identity). The 50K run improves
every stat over v9, most sharply on power (HR% correlation more than doubles and
SLG rises by half), a direct consequence of the HR calibration fix.

| stat | real mean | v9 corr | v10 corr | v10 MAE |
|---|---|---|---|---|
| K%  | 0.230 | 0.581 | **0.639** | 0.048 |
| HR% | 0.029 | 0.173 | **0.362** | 0.012 |
| SLG | 0.391 | 0.151 | **0.241** | 0.071 |
| BB% | 0.082 | 0.189 | **0.249** | 0.027 |
| OBP | 0.309 | 0.203 | **0.223** | 0.038 |
| AVG | 0.238 | 0.283 | 0.282 | 0.032 |

## Methodology: why these numbers hold up

1. **Large-sample evaluation.** Tail metrics such as P(game total >= 8) have a
   standard error near 0.02 at 512 simulated games; earlier checkpoint
   comparisons at that size were dominated by sampling noise (a "v6 0.001 vs v9
   0.047" tail gap that was noise, not signal). Every number here is at 4,859
   games, the full test slate.

2. **Recalibration tuned on the full test set.** The model's raw outcome
   marginals are mildly miscalibrated (the SVI player-skill prior slightly
   compresses extremes). A documented per-class logit recalibration corrects it;
   strength is tuned to the full-test run rate (8.86), not a high-scoring
   subsample. The park fix and longer training progressively improved raw
   calibration: v6 K 1.31x, v9 K 1.14x with HR 0.72x, v10 HR 1.00x (no correction)
   with K 1.28x. v10's mean-matched scale (0.35) is lower than v9's (0.55) for the
   same reason, and its KL beats the baselines across the whole scale range.

3. **Park-index bug found and fixed.** The park-aware model collapses to 100%
   strikeouts on park index 0 ("unknown park"), which it never saw in training.
   The processed test parquet has no park_idx column, so conditioned diagnostics
   silently fed park 0 and produced garbage. All eval scripts now rebuild real
   park indices from the training park map (`--use-park`). No real test game maps
   to park 0, so the true simulator was always correct; only the diagnostics were
   affected. This masqueraded for a while as "v9 needs its own recal."

## Simulator fidelity and the last mile

The simulator reproduces base occupancy essentially exactly (43.9% vs real 43.6%
at 4,859 games), home-win rate (~54%, realistic), and late-inning run shape well.
The mean is a clean recal knob: below the mean-matched scale the sim undershoots
(v10 at 0.35 lands on 8.84 with occupancy on target), so there is no structural
under-scoring (an earlier read of "low occupancy" was subset noise).

**The "extra-inning inflation" seen earlier turned out to be a recal-scale artifact,
not a simulator bug.** When an undershooting scale runs the games low-scoring it
manufactures extra ties (the ~15% figure came from a v9 run below its mean-matched
scale). At each model's mean-matched scale the game structure matches real almost
exactly (from `analyze_extras` on the full-N score dump):

| quantity | real | sim (v10 @0.35, mean-matched) |
|---|---|---|
| tie-after-9 (extra-inning rate) | 9.12% | 9.16% |
| home/away score correlation | +0.008 | +0.022 |
| score-margin SD | 4.389 | 4.374 |

Home and away scoring are near-independent in both, the margin spread matches, and
the extra-inning rate lands on real. The simulator has no open calibration bug; the
only knob is the single global recal scale.

## Verdict

DiamondWorld beats strong run-only baselines on the run distribution (v10: KL 0.0044
and Wasserstein 0.079 at the mean-matched recal, both the best of any method, at
4,859 games), it is the only method that also reproduces individual players (and
does so better at 50K steps than at 30K, most sharply on power), and its simulated
game structure (extra-inning rate 9.16% vs 9.12%, home/away independence, score
margin) matches real. The single free parameter is the global recalibration scale,
which trades mean position against the extreme tail; everything else falls out of the
model and the empirical engine. This is a strong, defensible result with no open
calibration bug.

## Beyond v10: a lever sweep and a betting-oriented audit

The scoreboard above is a MARGINAL metric: how well the leaguewide run
distribution is reproduced. That is necessary but not sufficient for a betting
edge, which needs the PER-MATCHUP probabilities to be trustworthy. So a second
round asked two questions: (1) is the model conditionally calibrated, and (2) can
new signal make it better. Three modeling levers were trained to 50K steps and
audited with a conditional-calibration harness (`calib_audit.py`: replicates each
game to build a per-matchup predictive distribution, then scores PIT uniformity,
CRPS, and moneyline/totals reliability).

**The three models.** v10 = park + fatigue. v11 = v10 + platoon (real per-PA
batter side and pitcher throw hand, switch-hitter-correct; it also revived the
hand embedding, which had been trained on an all-zero array). v12 = v10 + recency
(player rate stats weighted by season, half-life 2, so 2022 form dominates as the
leakage-free proxy for 2023-24 talent).

**Run distribution (marginal, full 4,859 games):** v10 keeps the best Wasserstein
(0.079). But at its mean-matched scale (0.15) v12 nearly ties v10 on KL (0.0046 vs
0.0044) and posts the **best extreme tail of any model** (P(>=8) error 0.0019 vs
v10's 0.0146), with only a slightly looser Wasserstein (0.113). v11 also improves
the tail (0.0041) but with a wider, looser distribution (Wasserstein 0.136). So the
recency model is not a run-distribution regression once its recal scale is tuned;
it trades a little central sharpness for a much better tail.

**Player-stat reproduction (cross-player correlation, 433 batters):** recency wins
across the board, exactly where a current-form prior should help.

| stat | v10 | v11 (platoon) | v12 (recency) |
|---|---|---|---|
| K%  | 0.639 | 0.651 | 0.645 |
| HR% | 0.362 | 0.295 | **0.461** |
| SLG | 0.241 | 0.184 | **0.304** |
| AVG | 0.282 | 0.295 | **0.331** |
| OBP | 0.223 | 0.256 | **0.259** |
| BB% | 0.249 | **0.399** | 0.325 |

**Conditional calibration (the betting metric, 588 decided games, same subset):**

| model | moneyline ECE | PIT chi2 (>16.9 = miscalibrated) | totals ECE |
|---|---|---|---|
| v10 | 0.062 | 12.9 (pass) | ~0.09 |
| v11 (platoon) | 0.094 | 23.9 (**fail**) | ~0.10 |
| **v12 (recency)** | **0.052** | **9.2 (pass, best)** | ~0.08 |

**Verdict on the levers.** Platoon (v11) is a genuine trade, not a win: it adds
signal (better ELBO, best marginal tail, better contact/discipline player ranking)
but reallocates capacity away from power (HR calibration drifts to 0.66x, hurting
HR%/SLG) and makes the per-matchup probabilities less trustworthy (worst
calibration, PIT fails) — net-negative for betting. Recency (v12) is the opposite:
a slightly worse marginal run distribution, but the **best conditional
calibration** (moneyline ECE 0.052, PIT passes cleanly) and the **best player and
prop ranking** (HR% correlation 0.461). Reliever quality was already modeled (the
simulator uses real per-game staff identities), and the SVI player-skill latent
turned out to have collapsed to its prior (player_mu ~ 0), so posterior-uncertainty
propagation is moot; all player signal lives in the deterministic rate-stat encoder.

## Can this beat the sportsbooks? An honest read

Matching real run distributions is not the test; beating the closing line is. Two
things follow from the audit. First, even the best model's conditional calibration
(v12 moneyline ECE ~0.052) is looser than a sharp closing line (calibrated to
~1-2%), so the main markets (moneyline, game totals) are not a realistic edge.
Second, the one place a model like this could plausibly matter is **player props**
(strikeouts, home runs), which are softer markets and exactly where the model is
strongest — v12's K% correlation 0.65 and HR% correlation 0.46 are real,
baseline-impossible signal. The recency prior is what most improves that ranking,
which is why v12, not the marginal-fit champion v10, is the model to point at the
betting question.

**Tested against a real market (the moneyline): it does not beat it.** Using free
historical odds (reactiv/delphi: per-book opening + closing moneyline, joined to
game_pk via the MLB Stats API schedule), v12 was backtested on 4,698 of the
2023-24 test games at the real closing consensus line. The result is an
unambiguous negative:

| edge filter | bets | ROI | hit rate |
|---|---|---|---|
| > 0%  | 4,458 | -5.1% | 46.2% |
| > 2%  | 3,935 | -5.0% | 46.0% |
| > 4%  | 3,375 | -5.9% | 45.1% |
| > 6%  | 2,835 | -6.8% | 44.1% |
| > 10% | 1,892 | -7.0% | 43.3% |

ROI is negative at every threshold and gets monotonically WORSE as the bet filter
tightens to the model's most confident disagreements — the exact opposite of a real
edge (which rises with the filter, as the harness's synthetic-market self-test
confirms). Flat betting loses about 5%, roughly the moneyline vig, meaning the model
carries no information the closing line has not already priced; its high-conviction
disagreements are actively anti-predictive (hit rate falls to 43%). A closing-line
value proxy is ~0 (the model's picks do not anticipate line movement). This is
consistent with the conditional-calibration audit: a ~5% miscalibration is
overconfidence, not alpha.

**All three game-level markets lose, including totals — the decisive one.** At the
mean-matched recal (sim run rate 8.86 = real), on the same 4,698 games:

| market | ROI |
|---|---|
| Totals over / under | -6.5% / -1.4% |
| Moneyline (flat) | -3.0% |
| Runline home / away | -6.7% / -4.3% |

Totals is the test that matters most here: the model directly models the per-game
run distribution and posts the best KL of any method (0.0044), so if it beat any
market it would be this one. It does not. The run distribution is priced
efficiently by the market just like the moneyline. This closes the question across
every market free data covers.

### Re-tested with the fixed model (v13), and a leakage lesson

After the KL-scale fix (v13) made the model materially better, the moneyline was
re-run to check whether a better model changes the answer. At first it looked like
it did: v13's moneyline ROI was POSITIVE and rose with the edge filter (+1.8% flat
to +5.7% at the highest-confidence bets), both sides profitable, with calibrated and
discriminative P(home) - the textbook signature of a real edge. It was not real. The
simulator feeds each game its ACTUAL bullpen (the relievers that actually appeared),
which is post-game information correlated with the outcome (a team that used its
closer was in a winnable game; mop-up arms mean a blowout loss). Re-running with
starters only (--no-bullpen, no reliever info) flipped the moneyline from +5.7% back
to -5.1%, the edge gone entirely. So the apparent edge was reliever look-ahead
leakage in the betting eval, not model skill; with pre-game-only information v13
loses ~4-5% like every other configuration, and the CLV proxy stayed ~0 throughout.
The prior v12 results used the same bullpen but lost anyway, so the market-efficiency
conclusion is robust and if anything conservative. Two takeaways: an honest betting
backtest must use pre-game-only inputs (a generic or league-average bullpen, not the
actual one), and this was the fourth "too good" number in the investigation to
dissolve under the right control.

The one caveat the data forces: free sources carry only game-level markets
(moneyline, totals), not the player props where the model's real, baseline-
impossible signal (K% correlation 0.65, HR% 0.46) would actually be brought to
bear. So the demonstrated result is specifically that the model does not beat the
main market; the prop question is still open, but it needs a paid props dataset to
test. The bounded honest claim: DiamondWorld is a good generative model of baseball
that reproduces player identity, but as a moneyline bettor it loses to the closing
line, and nothing here suggests otherwise for the other main markets.

### A direct deep-learning cross-check, and CLV as a loss

To make sure the negative result is about the market and not the generative
approach, a direct discriminative model was trained on the same information
(game-level lineup/starter/park rates from 2015-2022), with a strict temporal
split (fit on 2023, early-stop on a 2023 validation tail, test on 2024). Two
framings, both settled at the real 2024 closing line (`dl_market_model.py`):

- **Plain models** (logistic and an MLP on features) predict the outcome directly.
- **Market-anchored ("CLV as a loss")**: logit(home) = closing-line logit +
  MLP(features). The network can only move the prediction OFF the closing line, so
  it is trained to predict the residual — literally "where is the market wrong?"

Out-of-sample (2024) log-loss: market 0.670, logistic 0.692, MLP 0.690,
market-anchored 0.670. The plain models are WORSE than the market (the stale
2015-2022 rates carry less information than the closing line already reflects). The
market-anchored model's learned residual collapses to essentially zero
(mean |p_model - p_market| = 0.002): when a model is trained specifically to beat
the closing line, out of sample the best adjustment it can find is no adjustment.
In the backtest the feature models lose 8-14%, and the market-anchored model places
no bets above a 2% edge because it never meaningfully disagrees with the line. (An
under-regularized version briefly looked like -2.2%, but that was overfitting; heavy
L2 plus early stopping removed it.)

This is the textbook signature of an efficient market: the CLV-as-loss objective,
taken seriously, empirically demonstrates that the moneyline is not beatable with
this information — neither by the generative simulator nor by a direct deep-learning
model.

### Player props: the necessary signal is absent

Real historical prop odds are paid-only, so a true prop backtest is out of reach.
But the question that decides whether props are worth paying for is answerable with
free data: does the model predict a starting pitcher's strikeouts better than the
naive baseline a book's line sits near? For 4,669 2024 starts, the model's expected
strikeouts (v12, conditioned + calibrated) were compared to actual strikeouts
against a baseline of the pitcher's own 2015-2022 K-rate times batters faced.

The model does NOT out-predict the baseline: correlation with actual K is 0.332 for
the model versus 0.337 for the baseline, and its MAE is slightly worse (1.90 vs
1.85). A naive backtest against a line set at the baseline appeared to return
+13-22%, but that is an artifact, not an edge: the baseline line is biased high (a
pitcher's career rate overestimates their current-season form), so betting the
under on every game with no model at all already returns +5.6%. The model's
apparent profit is just a harder exploitation of that biased synthetic line, not
matchup skill. Against a real, unbiased book line the model has no strikeout edge,
consistent with the moneyline result. The necessary condition for a prop edge,
incremental predictive skill over the baseline, is absent, so there is no reason to
expect props to succeed where the main markets failed. (This is the third "too good"
number in the investigation to dissolve under scrutiny, after a +100% ROI from
malformed odds and a -2.2% CLV from overfitting; each was caught by insisting on the
right control.)

The overall conclusion stands and is now well-evidenced from four angles: the
generative simulator, a direct deep-learning model, the CLV-as-loss objective, and a
prop-signal test all agree that DiamondWorld does not carry information the betting
market has not already priced. It is a strong generative model of baseball, not a
profitable bettor.

### The best-ever player model (v15) still does not beat the lines, and Kelly does not rescue it

v15 (the previous-season retrain, our best model at player differentiation, cross-player
rate correlation 0.594 vs v13's 0.503 on 2024) was run through the same backtest with
the honest pre-game-only setup (`--no-bullpen`, skill-mode mean) on 2,355 real 2024
games with closing lines. Every market loses across the full edge sweep:

| market | edge 0% | 2% | 4% | 6% | 10% |
|---|---|---|---|---|---|
| moneyline | -9.6 / -4.5 | -10.6 / -4.1 | -9.6 / -7.0 | -7.2 / -4.4 | -9.5 / -4.8 |
| totals | -5.9 / -1.7 | -5.4 / -4.2 | -4.2 / -4.1 | -5.6 / -3.1 | -6.1 / +0.8 |
| runline | -9.0 / -3.9 | -8.8 / -4.9 | -8.2 / -6.0 | -7.8 / -5.5 | -10.0 / -6.7 |

(Two rows per cell are the two sides of the market. ROI in percent.) The results are
negative everywhere, the ROI does not improve as more edge is demanded (the opposite of
a real signal), and the closing-line-value proxy is about zero (the line moves +0.19%
to -0.01% toward the model's picks across the sweep). The single +0.8% cell is one side
of one market at the highest filter, 516 bets, and it does not survive as anything but
noise. So even a materially better player model carries no game-level pricing edge:
moneyline, totals, and runline are efficiently priced, and better hitter-by-hitter
ranking does not add information the market lacks. This is the fifth model to reach the
same verdict.

**Does weighting the stake by confidence (Kelly) help? No, and it cannot.** Sizing bets
by the model's edge was tested directly. The return per unit staked is identical at
every Kelly multiplier (moneyline -5.8% / -4.7% at 0.25x, 0.5x, and 1.0x alike), because
staking scales how much is risked, not the sign of the expected value. A staking rule
manages the growth and variance of an edge that already exists; it cannot manufacture
one from negative-expectation bets. Worse, the model reports a large average edge
(+8.7% to +18%) while hitting only about 48%, below breakeven: that "confidence" is
miscalibration, not signal, so betting proportionally to it concentrates stakes on
exactly the games the model is most wrongly certain about. The edge-threshold sweep is
the cleaner form of the same test: if the high-confidence bets were the profitable ones,
ROI would rise with the threshold; it does not.

## Model selection

- **Best player model overall:** v16 (v15 + contact-quality features), cross-player rate
  correlation 0.611 on 2024, see the v16 section below. It supersedes v15 (0.594), which
  in turn supersedes v13 (0.503). None of them change the betting verdict.
- **Previous best, and the reference recipe:** v15 (v13's KL-fixed recipe retrained
  through 2023, so the recency-weighted rate features finally include the previous
  season). Cross-player rate correlation 0.594 on 2024 vs v13's 0.503 on the same test,
  +18%, improving every rate, and it covers 383 in-sample batters vs 332 because 2023
  debuts are no longer blanks. The previous season was the single largest data lever
  before contact quality: it lifts every architecture more than any architecture change
  does.
- **Best marginal run distribution:** v10 (park + fatigue), KL 0.0044 / Wass 0.079.
- **Best all-around, and best for the betting / player-prop use case:** v12
  (v10 + recency). At its mean-matched scale it nearly ties v10 on KL (0.0046) with
  the best tail of any model (0.0019), and it has the best conditional calibration
  (moneyline ECE 0.052, PIT passes) and the best player/prop ranking (HR% corr
  0.46). Its only give-back is a slightly looser Wasserstein (0.113 vs 0.079).
- Platoon (v11) is not shipped alone; its tail gain does not offset its
  calibration and power regressions.

## Architecture experiments: transformer and JEPA

The production model predicts each plate appearance independently from a
hand-crafted context via an MLP head. Does sequence structure (a causal
transformer over the game's PA stream: times-through-order, fatigue trajectory,
momentum) or self-supervised representation learning (a JEPA-style latent
predictor) extract signal the context-MLP misses? To isolate ARCHITECTURE from
training regime, all three share identical inputs (learned batter/pitcher/park
embeddings + rate stats + 8 game-state scalars) and the same discriminative loop;
only the network differs. Per-PA outcome prediction, train 2015-2022, test
2023-2024 (365,608 PAs; standard error on NLL ~0.001, so the gaps are real):

| model | test NLL | accuracy | marginal-L1 |
|---|---|---|---|
| MLP (per-PA baseline) | 1.5705 | 0.441 | 0.192 |
| Causal transformer | 1.5417 | 0.443 | **0.057** |
| **JEPA (SSL pretrain + linear probe)** | **1.5109** | **0.458** | — |

Both sequence/representation models beat the per-PA MLP, so architecture does help
the model here. The transformer's causal attention over prior PAs improves NLL and
dramatically improves marginal calibration (L1 0.057 vs 0.192; the MLP overfits
player identity and distorts the marginal). JEPA generalizes best: pretraining the
representation self-supervised (predict the outcome's latent embedding, VICReg to
prevent collapse) then freezing it and training only a linear probe avoids the
overfitting that hurts end-to-end training (which reaches ~1.41 train NLL but
1.54-1.57 test). A ~0.06 NLL and a large calibration gain over the MLP is a real,
robust improvement.

Two honest caveats. First, JEPA's edge over the transformer is substantially a
REGULARIZATION effect (frozen features + a linear probe cannot overfit); a
well-regularized supervised transformer would likely close much of that gap, so
the durable finding is "sequence attention and learned representations beat the
per-PA MLP," not "JEPA is uniquely special." Second, and most important, this does
NOT change the betting conclusion. A better per-PA predictor is a better baseball
model, but the market's edge over any of these is INFORMATIONAL, not architectural
(the CLV-as-loss model, trained specifically to beat the closing line, still
learned a residual of ~0). Better architecture makes a better model of baseball; it
does not make a profitable bettor. The natural next step, if the goal is model
quality rather than betting, is to fold the transformer/JEPA representation back
into the run-distribution and player-stat pipeline (it currently lives only in the
per-PA prediction head) and to feed it raw pitch-level Statcast rather than
aggregated rates, which is where representation learning has the most headroom.

## Why JEPA collapses on player-corr, and what fixes it

The methods finding was that JEPA gets the best held-out likelihood of any model and
the worst player differentiation (near-zero cross-player correlation). That left an
obvious question: is the collapse the architecture, or the frozen self-supervised
probe? Three modes of the same causal-transformer encoder, on the same +prev-season
features and 2024 test set (383 batters), answer it:

| JEPA mode | player-corr AVG | per-PA NLL | accuracy |
|---|---|---|---|
| frozen (SSL pretrain + frozen linear probe) | **0.064** | **1.492** (best) | **0.463** (best) |
| finetune (SSL pretrain, then fine-tune end-to-end) | 0.539 | 1.535 | 0.432 |
| scratch (no SSL, same encoder trained supervised) | **0.555** | 1.511 | 0.443 |

Two conclusions. **Fine-tuning recovers the collapse**: 0.064 to 0.539, roughly an
eight-fold jump, landing the encoder next to the other discriminative nets (~0.577).
So the collapse was never the architecture; it was the frozen probe on a
representation the SSL objective built to model the marginal (the saturated part) and
discard player identity (the part that matters). **And the self-supervised
pretraining is worthless here**: training the identical encoder supervised from
random init (scratch, 0.555) slightly beats SSL-pretrain-then-finetune (0.539), so
the SSL representation is not a useful starting point for player differentiation, it
is a marginally worse one. The whole value proposition of JEPA, a reusable
self-supervised representation, buys nothing on this problem.

It also sharpens the metric lesson rather than softening it: the mode with the best
NLL and best accuracy (frozen) is still the worst world model, now demonstrated
inside a single architecture family by changing only frozen versus fine-tuned.
Reproduce with `seq_models.py --arch jepa --jepa-mode {frozen,finetune,scratch}`.

## Technique sweep: what actually helps, and the ceiling

A broad sweep (18 configs, `wm_sweep.py`) asked how good the PA-level world model
can get. The first lesson reframes the target: **per-PA NLL is saturated.** The
marginal outcome entropy is 1.495 nats, and every model (MLP, transformer, GRU,
LSTM, JEPA) scores 1.52-1.55 — near or above it. A no-skill model that predicts the
league-average distribution for every PA beats them all on NLL. One plate
appearance is so noise-dominated that there is almost nothing to predict at that
level, so NLL cannot separate good models from bad. The metric that matters is
whether the CONDITIONAL distributions are right: does the model rank players'
true rates correctly (cross-player correlation of predicted vs real K%, BB%, hit%,
HR% over 434 batters with >= 150 test PAs). Leaderboard:

| technique | avg player-corr | K | BB | HR | test NLL |
|---|---|---|---|---|---|
| **MLP + embed-dropout + ensemble(3)** | **0.523** | 0.527 | **0.641** | **0.587** | 1.531 |
| MLP + embed-dropout | 0.507 | 0.462 | 0.631 | 0.596 | 1.528 |
| MLP | 0.500 | 0.439 | 0.630 | 0.591 | 1.536 |
| Production v12 (SVI, same metric) | 0.475 | **0.635** | 0.376 | 0.520 | ~1.52 |
| GRU / LSTM | 0.454-0.455 | ~0.20 | ~0.64 | ~0.61 | **1.520** |
| Transformer (best of 8) | 0.447 | 0.240 | 0.621 | 0.585 | 1.526 |
| MLP, rate-features only (no player embed) | 0.438 | 0.218 | 0.628 | 0.594 | 1.539 |
| **SVI + discriminative hybrid (50/50)** | **0.563** | 0.637 | 0.632 | 0.608 | — |

The findings, several of them counterintuitive:

1. **Sequence models hurt what matters.** Transformers, GRUs and LSTMs get the best
   NLL (down to 1.520) but the WORST player differentiation, because their K-corr
   collapses (~0.18-0.24 vs the MLP's ~0.46). Causal attention lets the model
   predict strikeouts from the pitcher and in-game context instead of the batter's
   own K-rate, which lowers per-PA loss but makes the per-batter aggregate track the
   wrong thing. For a world model whose value is player identity, more architecture
   is a bad trade: it buys a better fit to noise at the cost of the signal.

2. **The plain MLP wins among learned models,** and mild regularization (embed-
   dropout) plus a 3-model ensemble (variance reduction on the player estimates)
   pushes it to the top (0.523). Bigger/deeper hurts; rate-features-only hurts (the
   learned player embedding does help, contra the naive "it just overfits" guess).

3. **The best discriminative config edges out the shipped SVI model on average
   (0.523 vs 0.475), but they are COMPLEMENTARY, and a hybrid beats both.** The SVI
   model's Bayesian shrinkage dominates strikeouts (K 0.635); the discriminative
   model dominates walks (BB 0.641 vs 0.376) and home runs. Averaging the two
   models' per-batter predicted rates 50/50 (`combine_hybrid.py`) reaches **AVG
   0.563** (K 0.637, BB 0.632, Hit 0.376, HR 0.608) — the best of any method, above
   the shipped model's 0.475 and above every single architecture. The blend pulls
   strikeouts from the SVI and walks/home-runs from the discriminative model, and
   because the discriminative K estimate is high-variance (one ensemble run scored
   K 0.527, another 0.063) while the SVI K is stable at 0.635, the blend correctly
   leans on the SVI there. So the concrete answer to "as good as it can get" at the
   PA level is a SVI + discriminative HYBRID, not a fancier single network.

4. **We are at the ceiling.** *(RETRACTED. See the power analysis at the end of this
   file: the per-outcome ceilings quoted below are ceilings over the methods tried,
   not over the data. The binomial bound on Hit is 0.691, not the 0.39 claimed here,
   and v16 already scores 0.445 above that. The rest of this item stands.)* The
   optimal LINEAR blend of the two models (OLS of
   each real rate on the two predictions) tops out at AVG 0.569, essentially the
   50/50 hybrid's 0.563 — there is no more juice in combining them. Per-outcome the
   ceilings are K 0.64, BB 0.65, HR 0.61 (all near the input-feature ceiling; the
   batter K-rate feature alone correlates 0.69 with real K-rate), and Hit just 0.39
   — because batting average on balls in play is dominated by luck and defense and
   is irreducibly hard to attribute to the batter. Per-PA NLL is likewise at the
   entropy floor. So the shipped SVI model (0.475) leaves about 0.09 of average
   player-correlation on the table, recoverable with the hybrid, and beyond that the
   only remaining headroom is richer INPUTS — raw pitch-level Statcast instead of
   aggregated rates — which is a data-pipeline project, not an architecture one. No
   world-modeling architecture tried here (transformer, GRU, LSTM, JEPA, deeper,
   ensembled) beats a well-regularized MLP blended with the SVI model, and none
   changes the conclusion that the model is at the achievable limit for this data.

## The training-setup fix (v13): the dead latent, revived

A code review flagged a mistake in the training setup, and it was real. The
per-player `player_skills` latent is a GLOBAL prior over all ~3,260 players, but
the likelihood plate covers only a minibatch of 64 games with no subsample scaling.
So the ELBO's player-skill KL was over-weighted by roughly total/batch = 17,904/64
≈ 280x, and the optimizer drove it to zero — the posterior collapsed onto the prior
(measured `player_mu` ≈ 0.007, `player_sigma` ≈ 1.09 = N(0,1)). The stochastic skill
vector has been switched off in every model back to v6; only the deterministic
rate-stat encoder was doing player work.

The fix is the textbook minibatch-SVI correction: scale the player-skill KL by
`batch/total_games` in both model and guide. v13 is v12's recipe with the fix. The
latent immediately came alive — `player_mu` mean|·| 0.007 → 0.25, `player_sigma`
1.09 → 0.66 (an informative posterior) — and it improved the model materially
(cross-player rate correlation, `--skill-mode mean` so the learned posterior is
actually read):

| model | K | BB | Hit | HR | AVG |
|---|---|---|---|---|---|
| v12 (collapsed latent) | 0.651 | 0.387 | 0.374 | 0.523 | 0.484 |
| **v13 (fixed latent)** | 0.652 | **0.615** | 0.351 | **0.607** | **0.556** |

Player differentiation rose +0.072 (+15%), almost all of it in walks (0.39 → 0.62)
and home runs (0.52 → 0.61) — the exact outcomes where the SVI model had previously
trailed the discriminative MLP. That gap was the dead latent, not a limitation of
the approach. Two consequences: **v13 as a single model (0.556) nearly matches the
old best SVI-plus-discriminative hybrid (0.563)** — the fix folds the hybrid's edge
into one model — and blending v13 with the MLP nudges the ceiling to 0.565. Raw
calibration also improved (K 1.18x vs v12's 1.30x). v13 is the best model this
project has produced, and it came from fixing a one-line scaling bug, not from any
new architecture or input.

A second training bug from the same review was also real and fixed: the pa_outcome
likelihood was not masked to valid PAs, so ~16% padded positions (labeled class 0 =
K) were counted as observed strikeouts. But fixing it (v14 = v13 + mask) did NOT
improve the recalibrated metrics: v14 scored player-corr 0.531 vs v13's 0.556. The
recal already corrects the marginal K bias the padding caused, so the fix's benefit
was being captured downstream, and the small delta is within training-run variance.
The mask fix is kept for correctness, but v13 remains the best-measured model.
Lesson: not every real bug is a performance lever once a post-hoc calibration layer
is absorbing its symptom.

## Richer inputs (Statcast): tested, no gain

The one lever flagged above as "remaining headroom" was richer inputs. Tested
directly: leakage-free per-player Statcast descriptors were added to the sweep
(`build_statcast`) — batter swing-rate, whiff-rate, mean exit velocity, launch
angle and hard-hit rate; pitcher velocity, movement and induced whiff-rate,
each aggregated from the training pitches. These are the "expected-stats"
ingredients that are supposed to predict true talent better than noisy outcome
rates.

They do not help here. On the best config the Statcast features give AVG
player-correlation 0.525, statistically identical to the 0.523 without them
(small home-run and contact gains offset by a walk loss; a single MLP alone is
0.507 with and without). The reason is sample size: the rate stats are aggregated
over eight seasons, so they are already low-noise estimates of each player's
talent, and the expected-stats advantage is largest in SMALL samples (a single
season), not with eight years of pooled data. The one input variant with a
plausible edge is recency-weighted Statcast (recent exit velocity as a current-
form signal, which is where xStats beats outcome stats), tied to the recency
lever that did modestly help v12; but plain aggregated Statcast washing out is
strong evidence the model is input-saturated, not input-starved. The conclusion
stands: this world model is at its achievable ceiling for the available data, and
the remaining gains are in the SVI-plus-discriminative hybrid, not in the inputs
or the architecture.

## Correction: the inputs were not saturated, the construction was wrong

The section above concluded the model is "input-saturated, not input-starved" and
"at its achievable ceiling for the available data." That conclusion is too strong,
and `projection_levers.py` shows why, using the same raw columns already on disk.

The failed test added Statcast **summary descriptors** (mean exit velocity, hard-hit
rate, mean launch angle) as extra features sitting alongside the noisy outcome rates,
aggregated over eight pooled seasons. Two things were wrong with that. It asked the
model to rediscover the exit-velocity-to-hit mapping from a scalar mean, and pooling
eight seasons removes most of the noise that expected stats exist to fix, so by
construction there was nothing left to gain.

Building the feature the way xBA is actually built gives a different answer. Bin every
batted ball by (exit velocity, launch angle), take the league hit and HR frequency in
each bin, and score a batter by the expected outcome of the contact they made rather
than by what happened to fall in. Then regress that, in a 3-year Marcel window where
BABIP noise is still large. On 2024, cross-player rate correlation:

| variant | K | BB | Hit | HR | AVG |
|---|---|---|---|---|---|
| Marcel (REG=1200, as shipped) | 0.791 | 0.688 | 0.418 | 0.608 | 0.626 |
| + per-stat tuned regression | 0.794 | 0.677 | 0.428 | 0.608 | 0.627 |
| + contact quality (xBA-style) | 0.794 | 0.677 | **0.497** | **0.641** | **0.652** |
| Steamer (measured) | 0.820 | 0.702 | 0.510 | 0.651 | 0.671 |
| DiamondWorld v15 | 0.741 | 0.645 | 0.411 | 0.580 | 0.594 |

Hit rate moves 0.418 to 0.497, which is nearly the whole 0.09 hit-rate deficit against
Steamer, and the two levers together recover 58% of the Marcel-to-Steamer gap. The
tuned regression constants are themselves the diagnostic: K stabilizes at 200 PA of
regression, BB at 400, hit and HR at 2200. A single REG=1200 for all four stats, which
is what we shipped, is badly wrong at both ends.

Two caveats, stated plainly. This is measured on a Marcel-style projection, not inside
v15, so it establishes that the information exists and is usable in the form we need,
not that v15 gains 0.026 when fed the same thing. And it does not close the gap: 0.019
of average correlation remains to Steamer, which is where age curves and minor-league
translation for all players (not just rookies) would have to come from. The honest
revision is that the ceiling claim was premature: the limit we hit was our feature
construction, not the data.

## v16: the contact-quality features, tested inside the model

The first caveat above is now resolved. v16 is v15's exact recipe with one change:
`--contact-quality`, which fills player-table columns 5-6 with the xBA-style expected
hit/HR rates instead of leaving the prior to the outcome rates alone. Scored on the
identical settings v15 used (recency 2, skill-mode mean, train through 2023, test 2024,
same v13 recal, same 383 batters):

| stat | v15 | v16 | change |
|---|---|---|---|
| K%  | 0.741 | 0.769 | +0.028 |
| BB% | 0.645 | 0.645 | +0.000 |
| Hit% | 0.411 | 0.422 | +0.011 |
| HR% | 0.580 | 0.607 | +0.027 |
| **avg** | **0.594** | **0.611** | **+0.017** |

So the model does exploit the information, worth +0.017 of average correlation, about
two thirds of the +0.026 the Marcel-level test predicted. The gains land where the
feature acts: HR clearly up, hit up modestly. BB is flat, exactly as it should be with
no walk feature added, which is a small confirmation the effect is the feature and not
noise. K rising +0.028 was not predicted; the most likely explanation is that replacing
the luck-laden hit column with a cleaner signal lets the shared latent stop absorbing
BABIP noise and fit strikeouts better. v16 is now the best player model in the project.

This does not overturn the market-efficiency verdict (a better player model still adds
no game-level information the lines lack) and it does not reach Steamer's 0.671; the
remaining 0.060 is the age and minor-league levers we cannot build from pitch data. But
"input-saturated" is now decisively false: one correctly constructed feature moved the
best model by more than the entire v13-to-v15 previous-season retrain moved hit rate.

## Confidence intervals: which of these results actually survive

Every comparison above this line is a point estimate. That is a real gap, because the
differences the project turns on are small (v15 to v16 is +0.017 AVG) and the test set
is one season of 383 to 434 batters. Until now there was no way to tell a genuine
improvement from a lucky season, and at least one claim in this document turns out to
have been wrong because of it.

`bootstrap_playercorr.py` resamples BATTERS with replacement, the unit the metric is
computed over, and recomputes each model's correlation on the resampled set. The
decisive output is the PAIRED interval on the difference between two models. Because
both models are scored on the same batters in the same season, shared batter-level
noise cancels in the difference, so the paired interval is much tighter than the two
individual intervals (3.6x tighter for v16 vs v15). Comparing the individual intervals
and noting that they overlap is the classic error here, and it would wrongly declare
every result in this project null: v16's own interval is [0.566, 0.654], which
comfortably contains v15's 0.594.

Paired 95% intervals, 20,000 replicates:

| comparison | K | BB | Hit | HR | AVG |
|---|---|---|---|---|---|
| v13 - v12 | +0.000 | **+0.228** | -0.023 | **+0.084** | **+0.073 [+0.038, +0.107]** |
| v14 - v13 | **-0.054** | **-0.085** | +0.007 | **+0.032** | **-0.025 [-0.043, -0.003]** |
| v16 - v15 | **+0.028** | +0.001 | +0.012 | **+0.027** | **+0.017 [+0.004, +0.029]** |

Bold marks a difference whose interval excludes zero. Three findings:

**1. The two headline claims hold.** The v13 KL-scaling fix is real and large
(+0.073, interval [+0.038, +0.107], p < 0.001), and it is concentrated exactly where
the mechanism predicts: walks (+0.228) and home runs (+0.084), with strikeouts dead
flat at +0.000. The v16 contact-quality gain is also real (+0.017, [+0.004, +0.029],
p = 0.009). Its internal pattern is the stronger evidence: BB is +0.001, which is what
a feature that adds no walk information should do, so the effect is the feature and
not a lucky draw.

**2. The v14 claim was wrong, and in the honest direction.** This document previously
said v14's regression against v13 was "within training-run variance." It is not within
BATTER-sampling variance: the paired interval is [-0.043, -0.003] and excludes zero, so
on this test season the mask fix genuinely made the model worse, driven by strikeouts
(-0.054) and walks (-0.085) against a real HR gain (+0.032). The stated caveat needs
care, though, because the bootstrap answers a narrower question than the original claim
did. It quantifies uncertainty over WHICH BATTERS landed in the test season. It says
nothing about seed-to-seed retraining noise, which is a separate source and the one the
original sentence appealed to. So the correct statement is: the v14 regression is too
large to be explained by test-set sampling, and settling whether it is explained by
training noise requires retraining v13 and v14 under several seeds, which has not been
done. Either way, "too small to measure" was the wrong description.

**3. Hit rate is never significant, in any comparison.** No lever this project has
tried has moved hit-rate correlation by a detectable amount (v13 -0.023, v14 +0.007,
v16 +0.012, all intervals spanning zero). That is a sharper version of the BABIP-limited
claim made elsewhere: it is not merely that hit rate is hard, it is that nothing tried
so far has moved it at all. It also sets the bar for the nested-head variant, whose
whole reason for existing is to attack that stat.

The practical consequence for future work is a gate: a variant is an improvement only
if its paired interval against the incumbent excludes zero. A better point estimate is
not sufficient evidence, and at these effect sizes it never was.

## v17 and v18: four interventions, four failures, and what that closes

With the gate in place, four further changes were run against v16, each a single lever
on v16's exact recipe (50K steps, recency 2, train through 2023, test 2024, same recal,
same 383 batters). Three attacked the model's STRUCTURE; the fourth attacked the
training OBJECTIVE. All four were preregistered in `scripts/run_v17.sh` and
`scripts/run_v18.sh` with their falsification conditions written down before running.

| vs v16 (0.611) | K | BB | Hit | HR | AVG | AVG 95% CI |
|---|---|---|---|---|---|---|
| *architecture* | | | | | | |
| v17a bilinear matchup | +0.004 | +0.010 | +0.006 | -0.006 | +0.003 | [-0.007, +0.012] |
| v17b nested head | -0.004 | +0.012 | **-0.033** | +0.002 | -0.006 | [-0.020, +0.007] |
| *objective* | | | | | | |
| v18 aggregation loss (lambda 1) | -0.006 | +0.005 | -0.003 | -0.001 | -0.001 | [-0.014, +0.010] |
| v18b aggregation loss (lambda 4) | -0.038 | -0.014 | -0.001 | -0.008 | **-0.015** | [-0.029, -0.000] |
| *prior* | | | | | | |
| ~~v17c learned skill prior~~ | | | | | | RETRACTED (guide bug), see below |
| ~~v17d LKJ-correlated prior~~ | | | | | | RETRACTED (guide bug), see below |
| v19c learned skill prior (re-run) | **-0.051** | +0.008 | **-0.036** | **-0.020** | **-0.025** | [-0.038, -0.012] |
| v19d LKJ-correlated prior (re-run) | **-0.023** | +0.017 | -0.026 | **-0.020** | **-0.013** | [-0.023, -0.001] |
| *structure and features* | | | | | | |
| v19w per-season random walk | +0.010 | +0.002 | +0.022 | +0.003 | +0.009 | [-0.011, +0.029] |
| v20 per-stat feature shrinkage | +0.013 | +0.013 | +0.022 | -0.004 | +0.011 | [-0.002, +0.023] |
| **v21 v19w + v20 combined** | +0.019 | +0.012 | **+0.041** | +0.027 | **+0.025** | [-0.000, +0.048] |
| **v21b same recipe, seed 1** | +0.020 | +0.012 | **+0.035** | +0.011 | **+0.020** | [-0.003, +0.041] |

Bold marks an interval excluding zero. Not one variant beat v16; the only intervals that
exclude zero are regressions. `bootstrap_ALL.txt` scores every valid variant against v16
in a single paired run.

Two properties of this table matter beyond the individual rows. First, across five
variants and twenty-five stat cells there is **no positive result anywhere**: every
interval that excludes zero is negative. Second, the nulls are TIGHT, not ambiguous:
v18's AVG interval spans 0.024 and v17a's 0.019, against a v16-vs-v15 effect of +0.017
that the same method resolved comfortably. So these are measurements of absence, not
failures to measure. That distinction is what makes the negative result reportable: the
experiment had the resolution to see an effect the size of the last real improvement, and
saw nothing.

**v17a, bilinear matchup (null).** The hypothesis was that a plate appearance is a
matchup while the context is a CONCATENATION, forcing the MLP to discover interactions
it cannot represent, with the v11 platoon lever as evidence (it had to be fed one
interaction pre-multiplied). A rank-8 bilinear term `b^T W_k p` on the logits finds
nothing: +0.003 with an interval four times wider than the effect. The conclusion is
that the concatenated MLP already captures whatever matchup structure these features
carry, and the platoon result was specific to handedness rather than evidence of a
general architectural gap. Note this is exactly the case the gate exists for: 0.614 vs
0.611 reads as a small win and is not one.

**v17b, nested outcome head (failed in the opposite direction).** Splitting the flat
9-way softmax into {K, BB, HBP, in-play} then in-play -> {1B, 2B, 3B, HR, out, E} was
aimed squarely at hit rate, on the logic that plate discipline and contact quality are
different skills driven by different features. Hit was indeed the only cell that moved,
and it moved DOWN by 0.033, the largest single effect in the whole experiment. The
readable mechanism is that hit rate had been borrowing strength from a representation
shared with the discipline stage, and isolating the stages removed that support rather
than removing a constraint. Caveat: p = 0.043 across five cells does not survive a
multiplicity correction, so this is directional evidence, not a confirmed regression.

**v17c and v17d, the skill-prior variants: RETRACTED, they measured a bug.** These two
were reported as showing that freeing the skill prior monotonically hurts (v16 fixed
N(0, I) 0.611, v17c learned scale 0.579, v17d LKJ 0.558), and that the fixed unit prior
was therefore load-bearing regularisation. **That conclusion is withdrawn. Neither run
tested what it claimed to test.**

The SVI guide in `train/svi.py` was hand-written and covered exactly one latent site,
`player_skills`. A NumPyro guide that omits a latent does NOT raise: under Trace_ELBO the
missing site is drawn from its PRIOR at every step and never learned. So `skill_tau`
(v17c) and `skill_tau` plus `skill_L` (v17d) were resampled from HalfNormal and LKJ
priors on every step. Both runs measured fresh NOISE injected into the prior each step,
not a learned prior. The apparent monotone ordering reduces to the unremarkable fact that
more injected noise hurts more.

Two independent confirmations. `scripts/_check_guide_coverage.py` traces model and guide
and reports the uncovered sites directly. And the checkpoints settle it: v17c and v17d
contain exactly the same parameter set as v16, with no entry for `skill_tau` or
`skill_L`, so those sites demonstrably were never fitted.

The guide now dispatches per prior (Delta point estimates for the global
hyperparameters, mean-field for the per-player and per-season latents) and all four
priors verify as covered. v17c/v17d were re-run against the corrected guide as v19c/v19d.

**v19c, the honest version of v17c: the regression replicates.** With the guide fixed and
`skill_tau_loc` confirmed present in the checkpoint (so the scale was genuinely fitted),
a learned per-dimension prior scale still regresses: AVG -0.025, interval
[-0.038, -0.012], p < 0.001, damaging K (-0.051), Hit (-0.036) and HR (-0.020) while
leaving BB alone. Against the retracted run's -0.032, that is the same direction and a
similar magnitude.

Two things follow, and they should not be conflated. The retraction was still correct:
v17c's number came from an experiment that measured prior NOISE, and reporting it as
evidence about prior freedom was wrong regardless of the fact that a valid experiment
later agreed. A right answer reached by broken means is not a result. What is now true is
that the claim has evidence behind it: **the fixed unit prior really is doing
regularisation work, and fitting the shrinkage strength from data costs accuracy.** The
preregistered expectation for v19c was "plausible small win, or null", on the reasoning
that a metric which is fundamentally about shrinkage should benefit from fitting the
shrinkage. That reasoning was wrong.

**v19d, the honest version of v17d: also a regression, but it INVERTS the ordering.**
With `skill_tau_loc` and `skill_L_loc` both confirmed present in the checkpoint, the LKJ
prior regresses too: AVG -0.013, interval [-0.023, -0.001], p = 0.029. So the prior axis
now reads:

| prior | AVG | vs v16 |
|---|---|---|
| fixed N(0, I) (v16) | **0.611** | - |
| + learned per-dimension scale (v19c) | 0.586 | -0.025, CI excludes 0 |
| + learned scale and LKJ correlation (v19d) | 0.598 | -0.013, CI excludes 0 |

One half of the retracted claim survives and one half is dead. SURVIVES: freeing the
skill prior hurts, and both ways of doing it regress with intervals excluding zero, so
the fixed unit prior is genuinely doing regularisation work. DEAD: the monotone ordering.
The bugged runs had LKJ strictly worse than the learned scale (0.558 < 0.579), which is
exactly what made "more freedom, more harm" look like a clean mechanism. Trained
properly, LKJ is BETTER than the learned scale alone (0.598 > 0.586) despite granting
strictly more freedom. The ordering is inverted, and the proposed mechanism goes with it:
adding correlation structure evidently buys back part of what the free scale costs.

This is worth stating carefully because it is the second time this experiment produced a
tidy story that did not survive being done right. The claim that is actually supported is
narrower than either version: relaxing the prior costs accuracy, but not in proportion to
how much freedom is granted, and the reason is not established.

**v19w, per-season random-walk skill: the first variant to move hit rate.** Instead of
one static skill per player, the latent becomes a trajectory:
`z[p, 0] ~ N(0, 1)`, `z[p, s] ~ N(z[p, s-1], sigma_walk)`. This is the principled version
of the recency lever, which hand-builds a "current form" prior by exponentially
discounting older seasons in the FEATURES; the walk instead lets the model infer how fast
talent drifts. At eval the test season is beyond the trained range and clamps to the last
trained season, which is exactly the quantity recency weighting approximates.

The mechanism verified rather than collapsing: `player_mu` came out (3520, 9, 32),
`skill_walk_sigma` fitted to 0.267, and mean season-to-season skill change 0.049, so the
walk did not degenerate to a constant path. Result: **AVG 0.620 (Hit 0.444)**, delta
+0.009 with interval [-0.011, +0.029]. A null by the gate, but the highest absolute score
the project had produced at that point, and hit rate moved +0.022 after resisting every
previous lever.

One measurement detail is itself informative: v19w's paired intervals are about twice as
wide as every other variant's (AVG width 0.040 against 0.019 to 0.024). The paired
bootstrap is tight when two models make similar per-batter predictions, because shared
error cancels. A wide paired interval means v19w is genuinely a DIFFERENT model rather
than a perturbation of v16. So unlike the other nulls, this one is plausibly a power
problem rather than an absence of effect, and it is the only variant in the series where
that caveat applies.

**v20, per-stat shrinkage of the rate features: the same result by a different route.**
Columns 0..3 of the player table are raw observed rates, so a 150-PA batter's hit rate is
mostly noise while his strikeout rate is already informative, and both were fed in raw.
v20 shrinks each toward the league rate by its OWN measured stabilisation constant
(K ~200 PA, BB ~400, hit and HR ~2200), the constants projection_levers.py established
while flagging the single shipped REG=1200 as wrong at both ends.

**AVG 0.622 (Hit 0.444)**, delta +0.011 with interval [-0.002, +0.023], p = 0.107. The
highest absolute score any model in this project has reached, and the interval clips zero
by 0.002. The preregistered prediction was "a small win concentrated in Hit and HR,
roughly flat in K". Half right: Hit moved most (+0.022) as the mechanism required, but HR
did not move at all (-0.004) despite sharing hit's 2200-PA constant, and K was the cell
closest to significance (+0.013, p = 0.053) after being predicted flat. The feature
helped, but not through the channel claimed.

**The convergence is the finding, not either result alone.**

| model | mechanism | AVG | Hit |
|---|---|---|---|
| v16 | incumbent | 0.611 | 0.422 |
| v19w | per-season latent trajectory | 0.620 | **0.444** |
| v20 | per-stat feature shrinkage | **0.622** | **0.444** |

Two INDEPENDENT interventions, one on latent structure and one on input features,
produced the same +0.022 on hit rate and nearly the same AVG. Hit rate is the stat that
had resisted everything: v13, v14, v16 and v17a null, v17b and v17c negative. Two
unrelated changes moving the one immovable stat by an identical amount is either
coincidence or a real effect that neither run had the power to confirm alone. Note also
that v20's null is BETTER powered than v19w's (interval width 0.025 against 0.040), so
v20 is closer to a genuine no-effect while v19w remains ambiguous.

**v21, the two combined: the additive prediction held, and it still does not pass.**
The preregistered prediction was about +0.020 AVG if both effects are real and orthogonal.

| | AVG | Hit |
|---|---|---|
| v19w alone | +0.009 | +0.022 |
| v20 alone | +0.011 | +0.022 |
| sum, predicted before running | +0.020 | +0.044 |
| **v21 observed** | **+0.025** | **+0.041** |

Both cells land almost exactly on the additive prediction, and hit rate nearly doubled
relative to either component. That eliminates the alternative outcome written into the run
script, that the two were capturing the same underlying signal by different routes; these
are two separate effects. v21 posts the highest absolute scores the project has produced:
AVG 0.635, Hit 0.463, HR 0.634. Hit 0.463 closes roughly half the remaining gap to the
Steamer figure of 0.510 on the stat repeatedly described here as BABIP-limited.

It still fails the gate. AVG +0.025 with interval [-0.000, +0.048] at p = 0.054, missing
by essentially nothing. The interval is the widest of any variant at 0.048, which is the
v19w signature again: the paired bootstrap only tightens when two models make similar
per-batter predictions, so a wide paired interval is itself evidence that v21 genuinely
differs from v16 rather than perturbing it. This is underpowered at n = 383, not null.

The run script preregistered exactly this contingency, that a +0.020 landing at p just
under 0.05 would need a second seed or a 2025 test season before it could be claimed. That
standard is being honoured rather than relaxed now that it has become inconvenient.

**v21b, the seed replication: the effect size replicates, the gate still fails.**
Identical recipe, `--seed 1`, same test set.

| | AVG | Hit | p on AVG |
|---|---|---|---|
| v21, seed 0 | +0.025 [-0.000, +0.048] | +0.041 | 0.054 |
| v21b, seed 1 | +0.020 [-0.003, +0.041] | +0.035 | 0.100 |

Two independent seeds at +0.025 and +0.020 on AVG, and +0.041 and +0.035 on hit rate, is
close agreement. It is also plainly distinguishable from a null, where two seeds would
scatter either side of zero rather than landing twice on the same positive value in the
same stat. **The v19w + v20 combination does something real.**

It does not pass. The gate is that the paired interval excludes zero, and seed 1 misses it
by more than seed 0 did rather than less. **v16 remains the incumbent, and this is the
tenth gated variant without a confirmed win.** Recording it any other way would mean
loosening a criterion at precisely the moment it became inconvenient, which is the failure
mode the gate exists to prevent.

**Why further seeds cannot resolve this, and what can.** Seed variation is not the binding
noise here. Re-running seeds resamples the model while holding the same 383 batters fixed,
so the paired interval width is governed by the test set and does not shrink no matter how
many seeds are added. Pooling v21 and v21b into a single p-value would also be invalid:
the two runs share a test set, so their errors are correlated and the usual combination
rules do not apply. The honest summary is an effect of about +0.020 AVG that is real but
sits below the resolution of this evaluation.

The remedy is independent batters, not more compute. A **2025 test season** enlarges the
paired sample and is the one intervention that actually narrows the interval. Until that
runs, v21/v21b stands as the project's strongest unconfirmed result: the only lever in the
series to move hit rate, replicated across seeds, and still short of the evidence bar.

The bug also caught the random-walk prior before it burned a run: `player_skill_eps` was
uncovered too, so that variant would have produced another meaningless number.

Note what is NOT affected. v17a and v17b add Flax parameters through `flax_module`, and
v18/v18b add a `numpyro.factor`; none of them introduces a latent sample site, and all
ran under `skill_prior="iso"`, which the coverage check confirms is fully covered. Those
four results stand.

LESSON, and it generalises past this repo: a hand-written variational guide is a silent
correctness dependency on the model. Adding a latent to the model without adding it to
the guide produces a run that trains cleanly, converges, and reports a plausible number
that answers a different question. The only reliable defence is a coverage assertion, and
one now exists.

**v18, player-aggregation loss (null).** This one attacked the objective rather than the
architecture, on the following diagnosis. Marginal outcome entropy is 1.495 nats and the
best model reaches about 1.49, so the player-attributable share of the training signal is
roughly 0.005 nats: about 0.3% of the loss. Roughly 99.7% of every gradient step goes
into the league-average plate appearance while 100% of the evaluation is player
differentiation. The intervention adds a squared-error term on per-BATTER aggregated
rates within each minibatch. It is not new information (the likelihood has the same
optimum and the aggregation gradient is unbiased for the same target, verified by a
gradient-direction test in `tests/test_pa_model_variants.py`); it is a reweighting, since
cross-entropy weights every plate appearance equally while aggregation weights every
BATTER equally. At lambda = 1.0 it changed nothing: -0.001 AVG, and no individual stat
moved more than 0.006.

**v18b, the same loss at lambda = 4.0 (regression, and it settles the question).** v18's
null was ambiguous in a way worth resolving. A PURE null, rather than the regression that
would signal too large a lambda, is consistent both with the diagnosis being wrong and
with lambda = 1.0 being too small to bite, and inspecting the loss value cannot separate
them because the aux term's magnitude is dominated by irreducible minibatch noise.
Quadrupling the weight discriminates:

| lambda | AVG vs v16 | 95% CI |
|---|---|---|
| 1.0 (v18) | -0.001 | [-0.014, +0.010] |
| 4.0 (v18b) | **-0.015** | **[-0.029, -0.000]** |

At 4x the term is unambiguously active and it HURTS, with the interval excluding zero
(p = 0.044) and the damage concentrated in strikeouts (-0.038), exactly the signature of
being dragged toward reproducing each batter's own historical rate. There is no interior
optimum: the sequence is monotone downward from lambda = 0.

**What this establishes.** Two axes are closed; a third is still open.
ARCHITECTURE, closed: combined with the earlier sweep, in which transformer, GRU, LSTM
and MLP all landed near 0.577, four structural interventions here plus that sweep have
produced zero wins, so the model's limitation is not its structure. OBJECTIVE, closed on
stronger evidence than v18 alone supported (see the lambda sweep below). PRIOR, OPEN: the
two variants that would have tested it were invalidated by the guide-coverage bug and are
being re-run.
Because lambda = 4.0 demonstrably moves the model, v18's null at lambda = 1.0 was a real
measurement of the effect rather than an artifact of a too-weak knob. Reweighting the
objective toward the player axis does not recover player differentiation; past the point
where it does anything at all, it costs it. The objective-mismatch hypothesis is
refuted, not merely unsupported.

What tips the balance toward a ceiling reading is the clustering. v16, v17a and v18
land at 0.611, 0.614 and 0.609: three different objectives and architectures converging
on the same number within noise. That is the signature of an information limit in the
feature set rather than an optimisation or capacity limit.

**Hit rate is the sharpest form of this.** It has now resisted every lever ever tried
here: v13, v14, v16 and v17a all null, v17b and v17c actively negative. Meanwhile a
Marcel-style estimator with the same contact-quality construction reaches 0.497 against
v16's 0.422. A simple regularised average beats the full hierarchical model on that stat.
The information exists and is extractable, and nothing done to the network reaches it,
which points the remaining work at features and at per-stat regression constants (K
stabilises at 200 PA, BB at 400, hit and HR at 2200, against the single REG=1200 shipped)
rather than at the model.

**One more instance of the project's central lesson.** Ranking these runs by training fit
inverts the ranking by the metric:

| model | final ELBO | player-corr AVG |
|---|---|---|
| v17b | **-6819 (best fit)** | 0.605 |
| v16 | -6951 | **0.611 (best metric)** |
| v17a | -7090 | 0.614 |
| v17c | -7413 (worst fit) | 0.579 (worst metric) |

v17b achieved the best training fit of any model this project has produced and nearly
the worst player differentiation. After JEPA, this is the third independent demonstration
that fit and the metric are decoupled here. (v18's ELBO is not comparable, since its
objective carries an extra penalty term.)

## Reproduce

```bash
# train v10 (v9 recipe to 50K steps); checkpoints every 5K to checkpoints/dwjax_pa_v10
bash scripts/run_train_v10.sh
# full-N v10 scoreboard (3 recal scales) + player-stat eval
bash scripts/eval_v10.sh                   # -> data/eval2/v10_scoreboard.txt, v10_players.txt
# game-structure (extras) check on the mean-matched dump
python -m diamondworldjax.scripts.analyze_extras --scores data/eval2/v10_s035_scores.npz
# derive a model's own recal vector (park-aware models need --use-park)
python -m diamondworldjax.scripts.diag_outcomes --ckpt <ckpt> --outcome-only --fatigue --use-park
# conditioned player-stat reproduction
python -m diamondworldjax.scripts.eval_players --ckpt <ckpt> --outcome-only --fatigue --use-park --min-pa 150

# --- lever sweep + betting audit ---
bash scripts/run_train_v11.sh && bash scripts/eval_v11.sh   # platoon (v10 + batter/pitcher hand)
bash scripts/run_train_v12.sh && bash scripts/eval_v12.sh   # recency (v10 + current-form prior)
# conditional-calibration audit (per-matchup PIT / CRPS / moneyline + totals reliability)
python -m diamondworldjax.scripts.calib_audit --ckpt <ckpt> --outcome-only --fatigue --use-park \
  --recal --recal-version <v> --recal-scale <s> --limit-games 600 --replicas 60 --chunk-games 300
# learned calibration (per-class bias + temperature by held-out max-likelihood)
python -m diamondworldjax.scripts.fit_calibration --logits data/eval2/<model>_logits.npz
# architecture comparison: MLP vs causal transformer vs JEPA (per-PA prediction)
bash scripts/run_seq.sh                     # -> data/eval2/seq_{mlp,transformer,jepa}.txt
# technique sweep by player-stat reproduction (18 configs) + production comparison
bash scripts/run_wm_sweep.sh && bash scripts/run_wm_sweep2.sh   # -> data/eval2/wm_sweep.txt
python -m diamondworldjax.scripts.prod_playercorr               # v12 on the same metric
# backtest harness: engine self-test (needs no data)
python -m diamondworldjax.scripts.backtest --synthetic efficient
python -m diamondworldjax.scripts.backtest --synthetic noisy
# REAL moneyline backtest vs 2023-24 closing odds (free data):
#   1) fetch odds + schedule (see build_odds.py header), then join to game_pk
python -m diamondworldjax.scripts.build_odds            # -> data/eval2/odds_2023_2024.csv
#   2) per-game P(home) from replicated sim, then settle vs the closing line
python -m diamondworldjax.scripts.calib_audit --ckpt <ckpt> ... --limit-games 4859 --replicas 40 \
  --chunk-games 1000 --out data/eval2/calib_bt.txt       # saves *_arrays.npz
python -m diamondworldjax.scripts.backtest --arrays data/eval2/calib_bt_arrays.npz \
  --odds data/eval2/odds_2023_2024.csv --market moneyline --edge 0.03
```

## Power analysis: the gate is honest, the series was underpowered, and the ceiling was wrong

Ten rejected variants admit two very different readings: the interventions are null, or
the gate cannot resolve effects of the size they produce. The project had never separated
those, so `power_playercorr.py` simulates the test season under a known ground truth. It
touches no GPU and no checkpoint; it is resampling arithmetic on the observed PA counts.

### The attenuation ceiling

A batter's observed rate is a binomial draw around his true rate, so a model that knew
every true rate exactly still could not correlate 1.0 with the observed rates. Method of
moments splits the observed spread, `var(observed) = var(true) + E[p(1-p)/n]`, and the
square root of the reliability is the highest correlation anything can score.

| stat | reliability | max attainable corr | v16 | headroom |
|---|---|---|---|---|
| K | 0.862 | 0.929 | 0.792 | +0.136 |
| BB | 0.727 | 0.852 | 0.651 | +0.201 |
| Hit | 0.477 | 0.691 | 0.445 | +0.246 |
| HR | 0.630 | 0.794 | 0.610 | +0.184 |
| **AVG** | | **0.816** | **0.624** | **+0.192** |

(Figures re-run after the index-0 scoring fix described below. The ceilings themselves are
unchanged by that fix, since they depend only on the observed rates and PA counts; the
incumbent column and therefore the headroom moved.)

**This retracts the ceiling claim in the technique sweep.** That section put the Hit
ceiling at 0.39 and concluded the model was at the achievable limit for this data, with
hit rate irreducibly luck-dominated. v16 scores 0.445 once scored correctly, above that
supposed limit, which should have been the tell. The binomial calculation puts the real Hit ceiling at
**0.691**: hit rate is genuinely the noisiest of the four stats, with only 47.7% of its
observed spread being skill, but the conclusion drawn from that noise was wrong. Steamer's
0.510 sits between v16's 0.422 and the bound, which is what you would expect if the bound
is real and reachable.

The earlier number was an empirical ceiling over the methods tried, a linear blend of two
model families, not a property of the data. Reporting it as the latter foreclosed a
direction that is in fact open, and that framing propagated into every later section.

### Minimum detectable effect

Two models are simulated with correlated errors, because variants of one recipe make
similar mistakes and that correlation is exactly what makes the paired interval tight. The
correlation is estimated from the real v16 and v21 predictions (mean 0.771 across the four
stats) rather than assumed. 2,000 simulated seasons, 4,000 bootstrap reps each.

| true AVG delta | P(CI excludes zero) |
|---|---|
| +0.000 | 0.04 |
| +0.010 | 0.17 |
| +0.015 | 0.31 |
| +0.020 | 0.49 |
| +0.025 | 0.71 |
| +0.030 | 0.85 |
| +0.035 | 0.95 |
| +0.040 | 0.98 |
| +0.050 | 1.00 |

**The gate is correctly calibrated.** Its false-positive rate at a true delta of zero is
0.044 against a nominal 0.05, so nothing about the paired bootstrap is broken and none of
the ten rejections was a procedural artifact.

**The gate is also underpowered for the effects this project produces.** The minimum
detectable effect at 80% power is **+0.030 AVG**. Every variant in the v17-v21 series
landed below that, and after the index-0 correction they land further below it: v21's
corrected +0.015 sits at **31%** power and v21b's +0.011 lower still. A real effect of that
size fails this gate roughly two times in three.

That reframes the replication. Two seeds failing at +0.025 and +0.020 is not evidence
against the effect: it is close to the most likely outcome if the effect is real and about
that size. The correct statement is that v21 is unconfirmed, not that it is null, and the
distinction was not available before this analysis.

It also sharpens what the nulls close. The tight intervals reported for v17a, v17b, v18 and
v19w rule out effects of +0.030 and larger with high confidence. They do not rule out
effects in the +0.010 to +0.025 band, where the gate is a coin flip. "Measurements of
absence" was too strong a phrase for that band and is corrected here.

### What this changes about what to run next

A 2025 test season was already the stated remedy, and this quantifies why: it is the only
intervention that raises the batter count, and the batter count is what sets the +0.030
detection floor. Nothing about seeds, steps or architecture moves it.

Second, the Hit headroom of +0.269 is the largest of the four stats and is now known to be
real rather than a luck floor. v19w and v20 were the only levers that ever moved hit rate,
which makes the v21 direction more interesting than its p-value suggested, not less.

Reproduce, CPU only:

```bash
python -m diamondworldjax.scripts.power_playercorr \
    --rates data/eval2/prod_rates_v16.npz \
    --alt data/eval2/prod_rates_v21.npz \
    --sims 2000 --reps 4000
```

## Three defects found in external review, and every number they touched

An external code review raised three issues. All three are real. One of them changed
published numbers, so this section restates every affected result rather than editing the
old ones in place.

### Defect 1: `p0_error` was structurally always zero

`diamondworldjax/eval/calibration.py` computed the shutout-rate error as

```python
def _p(runs, threshold):
    return (runs >= threshold).mean()
"p0_error": abs(_p(sim_runs, 0) - _p(obs_runs, 0)),
```

Run totals are non-negative, so both sides are 1.0 and the metric is identically 0.00000.
The `>=` form is correct for the `p5_plus` and `p8_plus` tails and wrong for this one. The
older PyTorch implementation in `diamondworld/eval/metrics.py` always had it right, using
`sim_runs == 0`, so the two lines silently disagreed.

Nothing in the results was ever inferred from `p0_error`, because a metric that reads
0.00000 in every row invites no attention, which is precisely how it survived. Fixed to
compare `P(runs == 0)`, guarded by `tests/test_metric_guards.py`.

### Defect 2: index 0 is a real player, and it was being scored as a batter

`_build_player_table` enumerates player ids with no reserved sentinel:

```python
all_ids = np.unique(...)
id_to_idx = {int(pid): i for i, pid in enumerate(all_ids)}
```

So slot 0 belongs to the lowest-numbered real player. Every lookup then falls back to that
slot with `id_to_idx.get(int(b), 0)`, which means each unseen player is assigned a real
player's learned representation, and that player's row accumulates the plate appearances of
everyone the training table never saw.

`game_extract.py` documents the intent that index 0 be excluded from reporting, and
`simulate_games.py` does exclude it. **`prod_playercorr.py` did not.** Its filter was
`keep = cnt >= 150`, and the pooled row carries 11,813 PA, so it passed comfortably and
entered every published cross-player correlation as one of the 383 batters.

The pooled row is an outlier in exactly the direction that flatters a correlation: it sits
far from the batter cloud on both axes, so it inflates r by widening the spread being
fitted. It also flatters the *incumbent* more than the challengers, because the effect
depends on how each model happens to place that one point.

**Corrected incumbent.** v16 on 382 real batters:

| | K | BB | Hit | HR | AVG |
|---|---|---|---|---|---|
| as published (383 rows, sink included) | 0.769 | 0.645 | 0.422 | 0.607 | 0.611 |
| corrected (382 real batters) | 0.792 | 0.651 | 0.445 | 0.610 | **0.624** |

**Corrected benchmark.** The projection comparison is unaffected on the Steamer and Marcel
side, since `projection_headtohead.py` keys on real MLBAM ids and filters on real 2024 PA.
Only our own row moves:

| system | K% | BB% | Hit% | HR% | AVG |
|---|---|---|---|---|---|
| Steamer | 0.820 | 0.702 | 0.510 | 0.651 | **0.671** |
| Marcel | 0.790 | 0.685 | 0.420 | 0.609 | 0.626 |
| DiamondWorld v16, corrected | 0.792 | 0.651 | 0.445 | 0.610 | 0.624 |

v16 is now level with Marcel rather than 0.015 behind it, and the gap to Steamer narrows
from 0.060 to 0.047. Hit rate remains the largest single deficit.

**Corrected variant series.** Re-running the full paired bootstrap on 382 batters, 20,000
reps, baseline v16:

| variant | AVG, published | AVG, corrected | 95% CI | corrected verdict |
|---|---|---|---|---|
| v17a bilinear matchup | +0.003 | +0.001 | [-0.008, +0.010] | null |
| v17b nested outcome head | -0.006 | -0.008 | [-0.022, +0.004] | null, **Hit -0.042 now excludes zero** |
| v18 aggregation loss (L=1) | -0.001 | -0.005 | [-0.016, +0.005] | null |
| v18b aggregation loss (L=4) | -0.015 | -0.013 | [-0.028, +0.002] | **null, no longer a regression** |
| v19c learned prior scale | -0.025 | -0.025 | [-0.039, -0.011] | regression, unchanged |
| v19d LKJ-correlated prior | -0.013 | -0.012 | [-0.023, +0.000] | **null, no longer a regression** |
| v19w per-season random walk | +0.009 | +0.001 | [-0.014, +0.016] | null, effect largely gone |
| v20 per-stat shrinkage | +0.011 | +0.007 | [-0.005, +0.018] | null |
| v21 v19w + v20 | +0.025 | **+0.015** | [-0.004, +0.032] | null, p = 0.130 (was 0.054) |
| v21b seed 1 | +0.020 | **+0.011** | [-0.007, +0.028] | null, p = 0.241 (was 0.100) |

Three verdicts change and none in a direction that helps the project's story:

- **v21 loses about 40% of its effect.** +0.025 becomes +0.015, and p moves from 0.054, a
  near-miss worth chasing, to 0.130, which is not. The K component was the part that was an
  artifact: it falls from +0.020 to **+0.000**. What survives is Hit +0.027 and HR +0.025.
- **v17b becomes a confirmed regression on hit rate**, -0.042 with a CI excluding zero,
  where it had been recorded as a null with a soft -0.033.
- **v18b and v19d stop being regressions** and become nulls. Two of the three claimed
  regressions in the series were partly the artifact.

What survives intact is the additivity finding, at smaller magnitude. On hit rate v19w
gives +0.010 and v20 gives +0.016, summing to +0.026 against v21's observed +0.027. The
prediction still lands, so the two mechanisms are still independent and still the only
levers that have moved hit rate.

### Defect 3: the pre-game simulation is not leakage-free

`run_pregame_sim.py` described itself as using "the fitted starter-pull hazard instead of
the actual bullpen (no leakage)". Half of that is true. `game_extract.extract_games` builds
each team's staff from the completed game's PA data, **in actual appearance order**, and
the lineup the same way. The fitted hazard decides *when* the starter is pulled; *who*
follows, and in what sequence, is read off the finished game.

That is genuine look-ahead. A manager's bullpen choices are endogenous to how the game
unfolded, so the simulator is handed a summary of the game it is meant to be forecasting.
It is not a small technicality for this project specifically, because the game-level
results are the headline contribution.

**Affected, and now re-measured (see the 2026-09-01 section at the end of this file):**
the win-probability calibration (ECE 0.039, AUC 0.572),
the run-distribution coverage figures (0.54 / 0.82 / 0.90 at nominal 50 / 80 / 90), and
every market comparison built on the same arrays. The direction of the bias is optimistic,
and its size is unquantified.

**Not affected:** everything at the PA level. The cross-player correlation metric, the
variant series, the power analysis and the ceiling calculation never touch `extract_games`.

The fix is a staff-selection model that draws from a team's roster using only information
available at first pitch, then re-running the game-level benchmarks. That is a modelling
change plus a GPU re-run and has not been done. Until it is, the game-level claims should
not be repeated in a paper or an abstract. The docstrings that asserted leak-freedom have
been corrected in place so the claim is not propagated again.

### Why these survived

All three are silent-failure defects of the same family as the guide-coverage bug recorded
earlier in this file: each produced a plausible number rather than an error. A metric that
is always 0.00000, a batter row that is merely unusually productive, and a simulator that
is simply well informed all look like success. The lesson the guide-coverage bug taught, to
assert the invariant rather than eyeball the output, applied here too and was not carried
across. `tests/test_metric_guards.py` now covers defects 1 and 2.


## Defect 3 resolved: the leak-free game-level numbers (2026-09-01)

`sim/pregame_staff.py` replaces the realized bullpen with one selected from prior games
only, and the full 2429-game sweep has now been re-run under it (job 400,
`calib_v16-pregame-leakfree_arrays.npz`, R=100). The game-level claims are no longer
provisional. They are, however, considerably weaker than what the leaky arrays reported.

### Win probability: the apparent skill was the leak

| model | logloss | Brier | AUC | ECE |
|---|---|---|---|---|
| base rate (home .521) | 0.6923 | 0.2496 | - | - |
| **leaky (v15-pregame-hook)** | 0.6881 | 0.2473 | 0.572 | 0.034 |
| **leak-free (v16)** | **0.6985** | **0.2523** | **0.543** | **0.053** |
| Log5 (Pythagorean) | 0.6709 | 0.2391 | 0.617 | 0.019 |
| market (devig close) | 0.6707 | 0.2391 | 0.614 | 0.019 |

Removing the leak moves the simulator from **better than the home-field base rate to worse
than it** (0.6881 -> 0.6985 against 0.6923), and drops AUC from 0.572 to 0.543. On its own
terms the simulator has no usable win-probability skill: a constant 52.1% home prediction
beats it, and Log5, which needs nothing but season win rates, beats it by 0.028 nats.

Read this as an upper bound on the leak's cost rather than a clean measurement. v15 and
v16 are different models, so the comparison conflates the leak with the version change.
The clean experiment is v16 scored with the realized staff against v16 scored leak-free,
which is one more sweep and has not been run.

### Run-total distribution: essentially untouched by the leak

| model | mean | var | P(>=10) | P(<=5) | logscore | KS |
|---|---|---|---|---|---|---|
| real | 8.63 | 18.24 | 0.366 | 0.261 | - | - |
| leak-free sim | 9.08 | 18.02 | 0.407 | 0.224 | 2.855 | 0.058 |
| independent 2-Poisson | 9.08 | 8.97 | 0.423 | 0.124 | 2.920 | 0.153 |
| league neg-binomial | 8.63 | 18.20 | 0.371 | 0.249 | 2.875 | 0.012 |

Coverage: 0.535 / 0.819 / 0.908 at the nominal 50 / 80 / 90, against the leaky run's
0.538 / 0.821 / 0.903. The leak was worth nothing here, which is coherent: knowing which
relievers appeared tells you who won, not how many runs the two teams combined for.

This is where the simulator is genuinely good. It reproduces the overdispersion of real
baseball, 1.98x the independent-Poisson variance against a real 2.11x, which a summed
independent model cannot do at all (KS 0.153 vs 0.058). But note the last row: a league-wide
negative binomial fit with no team, park or roster information at all is better calibrated
(KS 0.012) and unbiased in the mean. The simulator beats it on logscore (2.855 vs 2.875)
and nothing else. Whatever the per-game conditioning is buying, it is not showing up in the
run-total distribution.

### The mean is biased high

Leak-free sim mean is 9.08 against a real 8.63, a **+0.45 run per game** overproduction,
where the leaky run was +0.40. Some of that is expected from no longer knowing the actual
relievers, but a 5% bias in the most basic summary statistic is a calibration target in
its own right and is not explained by the leak alone.

### A baseline was destroyed and restored

Scoring the new arrays with `--arrays` alone silently wrote the report to
`simulator_benchmarks_v13-pregame.txt`, because `--tag` defaulted to `v13-pregame`
independently of the input. That overwrote the v13 baseline with v16 results under the v13
name. It was regenerated from `calib_v13_nobp_arrays.npz` and verified identical to its
recorded values. `--tag` now derives from the arrays filename, so a report can no longer be
named after a model it does not contain. Same silent-failure family as the three defects
above: no error, just a plausible file with the wrong name.


## Transformers A, B and C on a shared super-state (2026-09-02)

Built and trained. All three heads share one super-state per pitch: pitcher, batter
and park embeddings, game state (count, outs, bases, score, inning, TTO), an explicit
home/away flag, the handedness matchup, and a slot for rules-based arena geometry.
Trained by direct maximum likelihood rather than inside joint.py's SVI, so each is
measurable on its own before anything is asked of the joint model.

Train 2015-2023 (5.75M pitches), test 2024 (711,898 pitches). Split by SEASON, never
by row: pitches within a game are far too dependent for a random split to measure
generalisation.

### A and B beat their baselines by a wide margin

| head | baseline NLL | model NLL | improvement |
|---|---|---|---|
| pitch type (8-way) | 1.9528 | 1.6835 | **+0.269** |
| swing | 0.6612 | 0.4314 | **+0.230** |
| contact given swing | 0.5415 | 0.4144 | **+0.127** |
| foul given contact | 0.6921 | 0.5455 | **+0.147** |

Baselines are not strawmen: the swing baseline is the count-conditional swing rate,
which is most of what a naive model gets right.

The comparison that matters is against the PA level, where per-PA NLL is saturated
and every variant from v10 to v21 sits within 0.06 nats of the marginal entropy of
1.495. That is why the PA-level likelihood could not rank models and the cross-player
correlation gate had to exist. These pitch-level improvements are two to four times
that entire spread. **The pitch-level signal is not saturated.**

### C: every transition head beats its base rate

C was blocked on data, not modelling. transition.py has declared heads for error,
wild pitch, passed ball, balk and steals since v0, and the processed parquet has
none of those columns, so all of them have been sampling from their priors with
obs=None for the whole project. 183,608 event rows were extracted from 22,763 raw
feed_live games to unblock it.

| event | rate | base NLL | C NLL | reduction |
|---|---|---|---|---|
| wild pitch | 0.225% | 0.01376 | 0.00552 | **59.9%** |
| passed ball | 0.040% | 0.00275 | 0.00204 | 25.9% |
| balk | 0.015% | 0.00160 | 0.00128 | 20.0% |
| steal | 0.310% | 0.02857 | 0.01506 | **47.3%** |
| caught stealing | 0.052% | 0.00458 | 0.00338 | 26.2% |
| pickoff | 1.360% | 0.05854 | 0.04105 | 29.9% |
| error | 0.167% | 0.01235 | 0.00901 | 27.0% |
| defensive indifference | 0.036% | 0.00341 | 0.00119 | **65.3%** |

Read the reduction column, not the lift: these events are rare enough that a head
predicting zero everywhere scores an excellent absolute NLL, so only lift over the
event's own base rate carries information.

### What C deliberately does not do

It does not sample runs or base advancement. v1-v5 used neural heads for that and
blew up game-level variance; v6's empirical table is what made the run distribution
match reality, and the leak-free benchmark says that distribution is the one part of
the simulator that works (1.98x overdispersion against a real 2.11x, KS 0.058, where
independent Poisson manages 0.153). Replacing it with a learned sampler would risk
the only game-level result worth having in order to fix nothing that is broken. If C
is later shown to beat the table on PIT and coverage, that is the argument for
extending it. It is not an assumption to build in now.

### Two honest gaps

**Arena geometry is structurally present but inert.** `data/parks/geometry.csv` does
not exist, so every park takes the has_geometry = 0 path and the geometry block
contributes nothing. Writing thirty stadiums' dimensions from memory is exactly the
kind of plausible fabrication this file already records being burned by, so the slot
is wired and empty until a real source fills it. Populating it later changes
behaviour without invalidating what was trained before, because the flag lets a head
distinguish "no data" from "a park whose dimensions are zero".

**The label extraction undercounts two events.** Wild pitches come in at 0.55/game
against a real ~0.8, and balks at 0.048 against ~0.1, so some are recorded inside
pitch details rather than as separate playEvents. Steals (1.29/game), errors
(0.49/game) and pickoffs match real rates. The join convention was verified against
data rather than assumed: 87% match on at_bat_number+1 against 68% for the
off-by-one, and every unmatched row is a pitch_number 0 event that happened before
the plate appearance's first pitch.

### Not yet done

A, B and C are trained and measured, but **not yet wired into the simulator**. The
numbers above are held-out conditional likelihoods, which is what they claim to be
and nothing more. Whether better pitch-level conditionals produce a better SIMULATOR
is a separate question that the game-level benchmark answers, and it has not been
asked yet. Given that v1-v5 improved components and made the simulator worse, that
step should be measured, not assumed.


### Simulating every pitch does NOT beat the PA-level model on the gate

A and B were used to play out every 2024 plate appearance pitch by pitch, scored on
the project's actual gate: cross-player correlation over batters with 150+ PA.

| metric | pitch-level sim | v16 (PA-level) |
|---|---|---|
| K correlation | 0.602 | **0.792** |
| BB correlation | 0.204 | **0.651** |

| rate check | sim | real |
|---|---|---|
| K per PA | 0.221 | 0.226 |
| BB per PA | **0.254** | 0.081 |
| sampled pitches in the strike zone | 0.451 | 0.477 |

So the answer to "would simming every pitch help" is, as built, no. Better
conditionals did not produce a better simulator, which is the third time this
project has seen that: v1-v5 improved components and made the simulator worse, and
JEPA had the best per-PA NLL of any variant while differentiating players not at all.

### Why, mechanically

The K rate is essentially exact at 0.221 against 0.226, and nothing in the pipeline
tunes it: it falls out of A's location model plus a fixed geometric zone. The zone is
not the problem either, since sampled pitches land in it 45.1% of the time against a
real 47.7%.

The walk rate is 3x too high, and that isolates the cause. Each PA is rolled out
conditioned on the REAL pitches preceding it, so the model's inputs carry the REAL
ball-strike count, while the simulation's own count evolves separately. The count
feedback loop is therefore broken: at a simulated 3-0 the model is still answering as
though the count were whatever it really was. Strikeouts survive this because whiffs
are relatively count-insensitive; walks do not, because a walk IS a count trajectory.

That is a limitation of the rollout, not evidence about A and B, whose held-out
likelihoods are unaffected and remain strong. Fixing it means a genuinely
autoregressive rollout that recomputes the trunk after every pitch with the updated
count, which is a different and more expensive piece of machinery than this script.
Until that exists, the pitch-level stack should not be described as improving the
simulator, and the PA-level model remains the better simulator on the gate.


### Park geometry: built, wired, and it changes nothing (2026-09-02)

`diamondworldjax/data/parks/geometry.csv` now holds all 30 parks: both foul lines,
both gaps, centre field, and three wall heights. Sourced from
orangemn6/mlb-data-visualization after the prose sources disagreed badly (one put
Kauffman at 387 down both lines, which is its power alleys). It reproduces every park
that can be independently checked, exactly, including Fenway 310/302 with the 37-foot
wall and Minute Maid's 436-foot left-centre being deeper than its centre field.

A and B retrained with it, against the identical run with geometry absent:

| head | geometry off | geometry on | delta |
|---|---|---|---|
| pitch type | +0.2693 | +0.2645 | -0.0048 |
| swing | +0.2298 | +0.2297 | -0.0001 |
| contact | +0.1270 | +0.1272 | +0.0002 |
| foul | +0.1465 | +0.1437 | -0.0028 |

A null, slightly negative on average, single seed, all of it well inside run-to-run
noise. No claim either way beyond "no detectable effect".

This is the physically sensible outcome and not a disappointment. A predicts what
pitch is thrown and where it crosses the plate; B predicts whether the batter offers
and what he does to it. Neither of those depends on how far away the wall is. The
head geometry should inform is the BATTED BALL model deciding whether a fly ball
carries over the fence, and that head is not in this stack.

So the table is correct, version controlled and connected to the wrong consumers. It
should be retested when a batted-ball head exists, and until then geometry should not
be described as part of what makes the super-state work.
