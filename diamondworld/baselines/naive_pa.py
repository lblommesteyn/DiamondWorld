from __future__ import annotations

from collections import defaultdict

import numpy as np
import polars as pl

from diamondworld.baselines.markov_re24 import MarkovRE24Simulator
from diamondworld.baselines.base import PA_OUTCOMES, PA_OUTCOME_IDX


class NaivePASimulator(MarkovRE24Simulator):
    """Baseline B1: Like MarkovRE24 but outcome model only conditions on (stand, p_throws).

    Subclasses MarkovRE24Simulator and overrides the outcome-fitting and
    sampling methods. Transitions and simulation loop are inherited unchanged.
    """

    def _fit_outcomes(self, terminal: pl.DataFrame) -> None:
        """Fit P(outcome | stand, p_throws) ignoring base_state and outs."""
        counts_platoon: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(8))
        counts_overall = np.zeros(8)

        for row in terminal.filter(pl.col("pa_outcome").is_not_null()).iter_rows(named=True):
            outcome = row["pa_outcome"]
            if outcome not in PA_OUTCOME_IDX:
                continue
            idx = PA_OUTCOME_IDX[outcome]
            stand = row["stand"] or "R"
            p_throws = row["p_throws"] or "R"

            counts_platoon[(stand, p_throws)][idx] += 1
            counts_overall[idx] += 1

        self._outcome_platoon: dict[tuple, np.ndarray] = {}
        for key, arr in counts_platoon.items():
            total = arr.sum()
            if total > 0:
                self._outcome_platoon[key] = arr / total

        total_overall = counts_overall.sum()
        self._outcome_overall = (
            counts_overall / total_overall if total_overall > 0 else np.ones(8) / 8
        )

        # Keep these attributes to avoid AttributeError if base class code is called
        self._outcome_detailed = {}
        self._outcome_fallback = {}

    def _sample_outcome(
        self, bs: int, outs: int, stand: str, p_throws: str
    ) -> str:
        """Sample outcome using only platoon split (stand, p_throws)."""
        key = (stand, p_throws)
        probs = self._outcome_platoon.get(key, self._outcome_overall)
        idx = int(np.random.choice(8, p=probs))
        return PA_OUTCOMES[idx]
