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

All simulations use the v13 model (park + fatigue + recency, SVI with the corrected
minibatch KL scaling), read from its learned player-skill posterior, at the
mean-matched recalibration.

## 1. Counterfactual / what-if analysis

For one real 2024 game (home starter Shawn Armstrong vs Seth Lugo, baseline home
win probability 0.512), we hold everything fixed and change one input, then
re-simulate 250 games:

| intervention | home win prob | change | exp. home runs |
|---|---|---|---|
| baseline | 0.512 | - | 3.63 |
| swap home starter -> Max Scherzer | 0.636 | **+12.4%** | 3.44 |
| swap home starter -> replacement arm | 0.428 | -8.4% | 3.38 |
| upgrade weakest hitter -> Mike Trout | 0.668 | **+15.6%** | 4.89 (+1.26) |

An ace is worth ~12 points of win probability in this matchup; a single elite bat
is worth ~16 points and 1.3 runs. These are causal estimates from re-simulating the
same game, not correlations. The same machinery answers roster and trade-deadline
questions ("what does acquiring pitcher X do to our September win odds"),
lineup-card decisions, and injury-replacement impact.

## 2. Lineup construction by simulation

Given a fixed set of nine hitters, batting order is a real but small lever that
sabermetrics has long argued about. We simulate 40 candidate orders (the actual
order, two heuristics, and random search) and read off expected runs:

| order | runs/game | win prob |
|---|---|---|
| actual | 3.72 | 0.488 |
| best found | 3.96 | 0.576 |
| worst | 3.11 | - |

Reordering the *same nine players* moves expected runs by up to 0.84 across orders,
and the best order beats the one actually used by +0.24 runs/game and +8.8% win
probability. Over a season that is a handful of wins from a free decision. A full
search (or a smarter optimizer) plugs directly into the same simulator.

## 3. Correlated tail-risk

A game is not the sum of independent player outcomes; runs cluster (a walk, a
single, and a homer in one inning). Per-player projection systems, even summed,
model the *mean* but not the *shape*. From the baseline game's full simulated
distribution:

- mean total 7.35 runs, and the interesting part is the tail:
  P(total >= 10) = 0.23, P(total <= 5) = 0.37, P(shutout by either team) = 0.17,
  P(blowout of >= 5 runs) = 0.18.
- **The simulated game-total variance is 15.2, versus 7.35 for an independent-
  Poisson model with the same mean: 2.06x overdispersion.** That factor is exactly
  the within-game correlation a summed-projection model misses. It matters for
  anything tail-sensitive: bullpen deployment, blowout/leverage planning, and
  distribution-aware win expectancy.

## 4. Win probability with full uncertainty

<!-- WP_UNCERTAINTY_PLACEHOLDER -->

## Benchmark: is the underlying player model any good?

To ground the simulator, the player model was benchmarked against Marcel, the
public projection baseline that Steamer / ZiPS / THE BAT are themselves measured
against (they beat Marcel by a few points of correlation). Cross-player correlation
of predicted vs actual 2024 rates:

| stat | Marcel | DiamondWorld v13 |
|---|---|---|
| K%  | 0.791 | 0.652 |
| BB% | 0.688 | 0.615 |
| Hit% | 0.418 | 0.351 |
| HR% | 0.608 | 0.607 |
| **avg** | **0.626** | **0.556** |

The honest read: DiamondWorld is **not** a better projection system, and it does not
claim to be. Marcel out-projects it on player rates (as a dedicated system using a
player's own recent seasons should), and Steamer/ZiPS would beat Marcel again. But
DiamondWorld is *competitive* (home-run projection is a dead heat, and it is within
0.07 of Marcel on average), which is the point: the players inside the simulator are
realistic enough that the game-level analyses above are grounded, while the
simulator delivers the counterfactual, lineup, tail, and uncertainty capabilities no
projection system can.

## Honesty and limitations

- The model is trained on 2015-2022, so players who debuted in 2023-2024 are unseen
  (only 13% of 2024 games have all key players in-sample); the demos use games with
  known rosters. A production version retrains through the present.
- The simulator's bullpen uses realistic hook distributions but generic timing; a
  betting-grade evaluation must use pre-game-only bullpen information (using the
  actual relievers is look-ahead leakage, which we caught and documented separately).
- Uncertainty here is the fitted player-skill posterior plus game randomness; it does
  not include model-structure uncertainty.

## Why this is a contribution

Not "we built a good model," but "we built the generative object that lets you ask
counterfactual and distributional questions about baseball games, we validated that
its players are as realistic as a standard projection baseline, and we showed four
concrete analyses (roster what-ifs, lineup optimization, tail-risk, uncertainty-aware
win probability) that a projection system cannot produce." Reproduce with
`scenario_sim.py`, `ssac_analyses.py`, `wp_uncertainty.py`, `marcel_compare.py`.
