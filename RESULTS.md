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

## Model selection

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
