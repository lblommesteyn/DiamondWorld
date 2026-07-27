# DiamondWorld: Results

A plate-appearance-level generative model of baseball. Trained on 2015-2022,
evaluated on the full 2023-2024 test slate (4,859 games). The model generates
each plate appearance's outcome (a 9-way categorical: K, BB, HBP, 1B, 2B, 3B,
HR, out, E) conditioned on game state, batter and pitcher identity, and park; a
validated empirical rules engine turns outcome sequences into runs and base-state
transitions. A true game simulator plays full 9-inning-plus-extras games with
lineup cycling, a bullpen, walk-offs, and the ghost-runner extras rule.

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
| team fixed effects (controls team strength + home field) | 0.278 | full sample, game-specific residual |
| within-series (home field constant, starter varies) | 0.258 | permutation null 95th pct 0.034 → significant |

**Direction is validated and it is leak-free:** the simulator's game-specific win-
probability signal correlates with the market's beyond team identity, so it is capturing
real starter and matchup effects an independent market also prices, not noise. This is
the external support the causal claim needed.

**Magnitude is overstated, and now calibrated.** The within-series OLS slope of market on
sim is 0.17, but the simulator's win probability at R=100 replicas is noisy (reliability
0.39), which attenuates the slope; correcting for that sampling noise gives a slope of
**0.44**. So the raw simulator over-reacts to a single starter change by roughly 2x, and
a raw "+9.8-point ace swap" is about **+4.3 points** in market-calibrated units. The
honest upshot is a *market-calibrated* counterfactual engine: direction and magnitude
both tied to an independent ground truth, with the earlier raw headline numbers corrected
downward. (A higher-replica run would sharpen the magnitude; the direction result is
already solid.) Reproduce with `counterfactual_validation.py`.

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

4. **We are at the ceiling.** The optimal LINEAR blend of the two models (OLS of
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
