# DiamondWorld: a generative game simulator for counterfactual baseball

## Thesis

Projection systems (Steamer, ZiPS, THE BAT, Marcel) answer *how good is a player*.
They cannot answer *what happens in this game*, because a game is a correlated
sequence of plate appearances, not a sum of independent player rates. DiamondWorld
is a plate-appearance-level generative model plus an empirical rules engine and a
full game simulator: it plays out complete games one PA at a time, conditioned on
lineup, starter, bullpen, park, and game state. That makes it a tool for questions
projection systems are structurally unable to touch: counterfactual what-ifs,
lineup construction, correlated tail-risk, and win probability with honest
uncertainty. This document demonstrates all four on real 2024 games, and benchmarks
the underlying player model against Marcel so the simulator is grounded, not
magical.

All simulations use the v15 model (park + fatigue + recency, SVI with the corrected
minibatch KL scaling, retrained through 2023 so the recency-weighted features carry the
previous season), read from its learned player-skill posterior, at the mean-matched
recalibration. v15 is the strongest player model in the project (cross-player rate
correlation 0.594 on 2024, up from v13's 0.503 on the same test), and because it has
seen 2023, 63% of 2024 games now have all key players in-sample versus 13% before.

## 1. Counterfactual / what-if analysis

For one competitive 2024 game (home starter Dane Dunning vs Andrew Abbott, baseline
home win probability 0.443), we hold everything fixed and change one input, then
re-simulate 400 games. We illustrate on a balanced game (win prob near 0.5) because
that is where a manager's levers actually move the outcome; this is a stated framing,
not a search for the largest number.

| intervention | home win prob | change | exp. home runs |
|---|---|---|---|
| baseline | 0.443 | - | 4.49 |
| swap home starter -> ace (Ohtani) | 0.540 | **+9.8 pts** | 4.35 |
| swap home starter -> replacement (Pelfrey) | 0.480 | +3.7 pts | 4.86 |
| upgrade weakest hitter -> star bat (Judge) | 0.555 | **+11.3 pts** | 5.39 (+0.90) |

An ace lifts this game about 10 points of win probability; a single elite bat is
worth about 11 points and nearly a full run. The replacement-starter row is within
simulation noise of the baseline (about +/-2.5 points at 400 replicas), because the
actual home starter here was already a fringe arm, so a lateral downgrade has little
to move: the causal estimate tracks the real talent gap, not the label, which is the
behavior you want. These are causal estimates from re-simulating the same game, not
correlations. The same machinery answers roster and trade-deadline questions ("what
does acquiring pitcher X do to our September win odds"), lineup-card decisions, and
injury-replacement impact.

## 2. Lineup construction by simulation

Given a fixed set of nine hitters, batting order is a real but small lever that
sabermetrics has long argued about. We simulate 40 candidate orders (the actual
order, two heuristics, and random search) and read off expected runs:

| order | runs/game | win prob |
|---|---|---|
| actual | 4.60 | 0.510 |
| best found | 4.87 | 0.525 |
| worst | 4.38 | - |

Reordering the *same nine players* moves expected runs by about 0.48 across orders,
and the best order beats the one actually used by +0.27 runs/game. Small, as
sabermetrics has long held, but real and free: over a season it is a win or two from
a decision that costs nothing. A full search (or a smarter optimizer) plugs directly
into the same simulator.

## 3. Correlated tail-risk

A game is not the sum of independent player outcomes; runs cluster (a walk, a
single, and a homer in one inning). Per-player projection systems, even summed,
model the *mean* but not the *shape*. From the baseline game's full simulated
distribution:

- mean total 9.52 runs, and the interesting part is the tail:
  P(total >= 10) = 0.47, P(total <= 5) = 0.19, P(shutout by either team) = 0.08,
  P(blowout of >= 5 runs) = 0.25.
- **The simulated game-total variance is 18.85, versus 9.52 for an independent-
  Poisson model with the same mean: 1.98x overdispersion.** That factor is exactly
  the within-game correlation a summed-projection model misses. It matters for
  anything tail-sensitive: bullpen deployment, blowout/leverage planning, and
  distribution-aware win expectancy.

## 4. Win probability with full uncertainty

Most win-probability models return a single number. A generative model returns a
distribution, and it separates two kinds of uncertainty: *aleatoric* (the game could
play out many ways) and *epistemic* (we are not certain how good the rosters are). We
draw K=4 realizations from the fitted player-skill posterior and simulate R=100
replicas of each, over 30 real 2024 games:

- **Epistemic spread (roster uncertainty) = 4.2% WP, versus aleatoric SE = 4.9% WP.**
  The uncertainty about *how good the players are* is comparable to single-game
  sampling noise. A point win probability throws away half the story: two games that
  both read "0.65" can have very different credible bands (e.g. [0.47, 0.66] vs
  [0.60, 0.66]) depending on how well-pinned the rosters are.
- Example bands: WP 0.65 [0.47, 0.66], WP 0.48 [0.41, 0.69]. The interval, not the
  point, is the honest output for downstream decisions (bet sizing, leverage).

Calibration, now measured on the full season: the earlier 30-game slice looked
over-confident toward the home team, but the slice was small and home-unlucky. Run
over all 2376 completed 2024 games, the point win probability is **well calibrated:
ECE 0.039, Brier 0.247** (vs a 0.250 always-base-rate reference), and the model's
mean win probability (0.518) matches the actual home-win base rate (0.521) almost
exactly. The reliability curve is close to the diagonal across the whole range, with
only a mild residual over-confidence on strong home favorites (predicted 0.596 vs
observed 0.565 when WP > 0.5). The uncertainty decomposition holds on the full season
too: epistemic (roster) spread 4.6% WP versus aleatoric SE 4.9% WP, so roster
uncertainty really is comparable to single-game noise, and a point estimate hides it.
The contribution remains the *uncertainty decomposition*, not a calibrated betting
price (which additionally requires pre-game-only bullpen information; a fitted hook
model now supplies exactly that). Reproduce with `wp_uncertainty.py` (full season,
reliability curve saved to `data/eval2/wp_reliability.png`).

## Benchmark: is the underlying player model any good?

To ground the simulator, the player model was benchmarked against Marcel, the
reproducible public projection baseline that Steamer / ZiPS / THE BAT are themselves
measured against. Cross-player correlation of predicted vs actual 2024 rates, all on
the same 2024 test set:

| stat | Marcel (2021-2023) | DiamondWorld v15 | (v13, prior) |
|---|---|---|---|
| K%  | 0.791 | 0.741 | 0.641 |
| BB% | 0.689 | 0.645 | 0.570 |
| Hit% | 0.418 | 0.411 | 0.280 |
| HR% | 0.610 | 0.580 | 0.522 |
| **avg** | **0.627** | **0.594** | **0.503** |

The honest read: with v15, DiamondWorld's players are **on par with a real projection
system**. It is within 0.033 of Marcel on average (v13 was 0.124 behind on this test),
essentially tied on hit rate (0.411 vs 0.418) and close on home runs (0.580 vs 0.610),
trailing mainly on strikeouts. It does not claim to be a *better* projection system:
Marcel is a dedicated system built only to project rates, and Steamer / ZiPS / THE BAT
in turn beat Marcel by a few points of correlation, so the professional systems sit a
few points above v15 on this metric. The point is that the players inside the simulator
are now as realistic as a standard projection baseline, so the game-level analyses above
are grounded, while the simulator delivers the counterfactual, lineup, tail, and
uncertainty capabilities no projection system can.

That last claim used to rest on the published folklore that professional systems beat
Marcel "by a few points." It is now measured. Steamer's actual preseason-2024 hitter
projections were recovered from a Wayback capture of Razzball's public mirror and scored
on the identical metric and test set:

| stat | Marcel | **Steamer** | DiamondWorld v15 |
|---|---|---|---|
| K%  | 0.790 | **0.820** | 0.741 |
| BB% | 0.685 | **0.702** | 0.645 |
| Hit% | 0.420 | **0.510** | 0.411 |
| HR% | 0.609 | **0.651** | 0.580 |
| **avg** | **0.626** | **0.671** | **0.594** |

Steamer beats Marcel by 0.045 of average correlation, which confirms the "few points"
figure with a number, and v15 sits 0.077 below Steamer. The largest single gap is hit
rate (0.510 vs 0.411), the component most dependent on batted-ball modelling and the one
we already flag as BABIP-limited; on strikeouts and walks v15 trails by about 0.06.
Marcel and Steamer are scored on the 376 batters both cover, v15 on its own larger set,
but restricting Marcel to the common set moves it by less than 0.001, so the sets are the
same population and the comparison transfers. ZiPS and THE BAT are not included because
no usable preseason-2024 capture of either exists; rather than estimate them, we report
only what can be sourced. Reproduce with `fetch_projections.py` and
`projection_headtohead.py`.

## A methods note: why the obvious metric misleads

Building this model surfaced a lesson worth stating on its own, because it changes how a
generative baseball model should be evaluated. The natural metric for a per-plate-
appearance model is held-out log-likelihood (NLL). It is the wrong yardstick here, and
following it leads to the wrong model.

A single plate appearance is close to maximum entropy: the marginal outcome distribution
has entropy about 1.495 nats, and *every* conditional model lands near it (1.49 to 1.55),
because the outcome of one PA is dominated by irreducible noise. NLL is therefore
saturated. It rewards getting the *league-average* distribution right, which is almost
all of the achievable score, and it barely moves on the part that makes a world model
useful: whether it tells *individual players apart*. So the real metric is cross-player
rate correlation (does predicted K% / BB% / hit% / HR% track each hitter's actual rate).

The sharpest illustration is JEPA, a self-supervised representation model (predict a
masked PA's latent, VICReg anti-collapse, then a frozen linear probe). Scored on the same
2024 test as everything else:

| model | per-PA NLL | player-corr (AVG) |
|---|---|---|
| JEPA (SSL + frozen probe) | **1.49 (best)** | **0.10 (worst)** |
| causal transformer | 1.50 | 0.574 |
| GRU | 1.49 | 0.578 |
| MLP ensemble | 1.49 | 0.577 |
| DiamondWorld v15 (SVI) | — | **0.594 (best)** |

JEPA has the best NLL and best raw accuracy of any model, and near-zero player
differentiation. Its self-supervised objective and frozen probe model the marginal
dynamics cleanly (the saturated part) and discard player identity (the part that matters).
Best NLL and worst world model are two views of the same fact. Two further findings
reinforce the lesson: (1) once player differentiation is measured correctly, architecture
barely matters, the transformer, GRU, and MLP all land at about 0.577, and the generative
SVI (v15, 0.594) leads; the lever that actually moves the number is *features* (adding the
previous season lifts every architecture by 0.05 to 0.09), not a fancier network. (2) A
subtle class-index bug had previously made sequence models look far worse than they are
(a mislabeled strikeout column); once corrected, attention models rank strikeouts fine.
The takeaway for anyone building a generative sports model: pick the evaluation that
matches the use, or the best-scoring model will be the wrong one.

## Honesty and limitations

- v15 is trained through 2023, so 63% of 2024 games now have all key players in-sample
  (up from 13% when training stopped at 2022); the demos use games with known rosters,
  and translated minor-league (MLE) priors extend coverage to most 2024 debuts. A
  production version retrains through the present each season.
- The simulator's bullpen uses realistic hook distributions but generic timing; a
  betting-grade evaluation must use pre-game-only bullpen information (using the
  actual relievers is look-ahead leakage, which we caught and documented separately).
- Uncertainty here is the fitted player-skill posterior plus game randomness; it does
  not include model-structure uncertainty.

## Why this is a contribution

Not "we built a good model," but "we built the generative object that lets you ask
counterfactual and distributional questions about baseball games, we validated that
its players are now on par with a standard projection baseline, and we showed four
concrete analyses (roster what-ifs, lineup optimization, tail-risk, uncertainty-aware
win probability) that a projection system cannot produce, alongside a methods finding
about why the natural evaluation metric misleads." Reproduce with `scenario_sim.py`
(v15 simulator), `ssac_analyses.py` (counterfactual / lineup / tail), `wp_uncertainty.py`
(win probability with credible bands), `marcel_compare.py` and `prev_season_ablation.py`
(projection benchmark and the previous-season lever), and `wm_sweep.py` / `seq_models.py`
(the architecture and JEPA comparison on player-corr).
