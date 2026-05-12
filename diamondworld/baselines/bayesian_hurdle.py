"""Baseline B4: Bayesian Stochastic Multi-Hurdle PA Outcome Model.

Each PA outcome is decomposed into 5 sequential binary hurdles:
  1. out:    P(K or out)
  2. walk:   P(BB/HBP | not out)
  3. hr:     P(HR | hit)
  4. xbh:    P(2B or 3B | hit, not HR)
  5. triple: P(3B | 2B or 3B)

Each hurdle has a Beta conjugate prior. After observing training PAs the
posterior is Beta(alpha + successes, beta + failures).

At simulation time, game-level hurdle probabilities are drawn once per game
from these posteriors.  This single draw determines all 18 half-innings of
that game, creating within-game correlation in offense (pitcher's duel vs.
slugfest) and a heavier-tailed run distribution than point-estimate baselines.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import polars as pl

from diamondworld.baselines.base import (
    BaseSimulator,
    GameLog,
    PA_OUTCOMES,
    PA_OUTCOME_IDX,
)
from diamondworld.baselines.markov_re24 import MarkovRE24Simulator

# Beta priors (alpha, beta).  concentration=50 pulls each game's draw
# toward the posterior mean, preventing the over-dispersion seen with
# weakly informative priors.  Equivalent to adding 50 pseudo-observations.
_CONCENTRATION: float = 50.0
_PRIORS: dict[str, tuple[float, float]] = {
    "out":    (2.0, 2.0),
    "walk":   (1.0, 3.0),
    "hr":     (1.0, 10.0),
    "xbh":    (1.0, 3.0),
    "triple": (1.0, 10.0),
}


class BayesianHurdleSimulator(BaseSimulator):
    """B4: Bayesian stochastic multi-hurdle PA outcome simulator."""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, pitches: pl.DataFrame) -> None:
        terminal = pitches.filter(pl.col("pa_terminal"))
        self._fit_hurdles(terminal)
        self._fit_transitions(terminal)

    def _fit_hurdles(self, terminal: pl.DataFrame) -> None:
        counts: dict[str, int] = {o: 0 for o in PA_OUTCOMES}
        for (outcome,) in terminal.filter(
            pl.col("pa_outcome").is_not_null()
        ).select("pa_outcome").iter_rows():
            if outcome in counts:
                counts[outcome] += 1

        n_out     = counts["K"] + counts["out"]
        n_not_out = counts["BB"] + counts["HBP"] + counts["1B"] + counts["2B"] + counts["3B"] + counts["HR"]
        n_walk    = counts["BB"] + counts["HBP"]
        n_hit     = counts["1B"] + counts["2B"] + counts["3B"] + counts["HR"]
        n_hr      = counts["HR"]
        n_xbh     = counts["2B"] + counts["3B"]
        n_1b      = counts["1B"]
        n_triple  = counts["3B"]
        n_double  = counts["2B"]

        pairs = {
            "out":    (n_out,    n_not_out),
            "walk":   (n_walk,   n_hit),
            "hr":     (n_hr,     n_hit - n_hr),
            "xbh":    (n_xbh,    n_1b),
            "triple": (n_triple, n_double),
        }
        self._posteriors: dict[str, tuple[float, float]] = {
            h: (a + pairs[h][0], b + pairs[h][1])
            for h, (a, b) in _PRIORS.items()
        }

        # Fixed sub-split proportions (K/out and BB/HBP don't affect run scoring)
        self._k_frac   = counts["K"]   / n_out   if n_out   > 0 else 0.5
        self._hbp_frac = counts["HBP"] / n_walk  if n_walk  > 0 else 0.12

    def _fit_transitions(self, terminal: pl.DataFrame) -> None:
        table: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
        for row in terminal.filter(
            pl.col("pa_outcome").is_not_null() & pl.col("base_state_after").is_not_null()
        ).iter_rows(named=True):
            outcome = row["pa_outcome"]
            if outcome not in PA_OUTCOME_IDX:
                continue
            bs    = int(row["base_state"])    if row["base_state"]    is not None else 0
            outs  = int(row["outs"])          if row["outs"]          is not None else 0
            bs_af = int(row["base_state_after"])
            runs  = int(row["runs_scored"])   if row["runs_scored"]   is not None else 0
            table[(bs, outs, outcome)].append((bs_af, runs))
        self._transitions = dict(table)

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample_game_hurdles(self) -> dict[str, float]:
        """One draw per game from a concentrated Beta posterior.

        The posterior mean is preserved but variance is reduced by
        _CONCENTRATION, preventing extreme game-level draws that cause
        over-dispersion in the run distribution.
        """
        result = {}
        for h, (a, b) in self._posteriors.items():
            mean = a / (a + b)
            # Re-parameterise as Beta(mean*C, (1-mean)*C) around the posterior mean
            a_tight = mean * _CONCENTRATION
            b_tight = (1.0 - mean) * _CONCENTRATION
            result[h] = float(self.rng.beta(a_tight, b_tight))
        return result

    def _sample_outcome(self, h: dict[str, float]) -> str:
        """Walk the hurdle tree to produce a PA outcome."""
        rng = self.rng.random
        if rng() < h["out"]:
            return "K" if rng() < self._k_frac else "out"
        if rng() < h["walk"]:
            return "HBP" if rng() < self._hbp_frac else "BB"
        if rng() < h["hr"]:
            return "HR"
        if rng() < h["xbh"]:
            return "3B" if rng() < h["triple"] else "2B"
        return "1B"

    def _sample_transition(self, bs: int, outs: int, outcome: str) -> tuple[int, int]:
        key = (bs, outs, outcome)
        entries = self._transitions.get(key)
        if entries:
            return entries[int(self.rng.integers(len(entries)))]
        return MarkovRE24Simulator._deterministic_transition(bs, outs, outcome)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _simulate_half_inning(self, h: dict[str, float]) -> int:
        bs, outs, runs = 0, 0, 0
        while outs < 3:
            outcome = self._sample_outcome(h)
            bs, r = self._sample_transition(bs, outs, outcome)
            runs += r
            if outcome in ("K", "out"):
                outs += 1
        return runs

    def simulate_game(self, game_context: dict) -> GameLog:
        game_id = game_context.get("game_id", 0)
        h = self._sample_game_hurdles()   # one draw → all 18 half-innings share this "day"
        inning_runs_away = [self._simulate_half_inning(h) for _ in range(9)]
        inning_runs_home = [self._simulate_half_inning(h) for _ in range(9)]
        return GameLog(
            game_id=game_id,
            home_runs=sum(inning_runs_home),
            away_runs=sum(inning_runs_away),
            inning_runs_home=inning_runs_home,
            inning_runs_away=inning_runs_away,
        )
