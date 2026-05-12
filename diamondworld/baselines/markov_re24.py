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

_MIN_OBS = 30


class MarkovRE24Simulator(BaseSimulator):
    """Baseline B0: Markov chain with RE24 run-expectancy transitions.

    Outcome probabilities are conditioned on (base_state, outs, stand, p_throws)
    with fallbacks for sparse combinations. Runner transitions are sampled from
    the empirical distribution observed in training data.
    """

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, pitches: pl.DataFrame) -> None:
        terminal = pitches.filter(pl.col("pa_terminal"))
        self._fit_outcomes(terminal)
        self._fit_transitions(terminal)
        self._fit_stand_p_throws(pitches)
        self._fit_hbp_rate(terminal)

    def _fit_outcomes(self, terminal: pl.DataFrame) -> None:
        """Build outcome probability tables keyed by (bs, outs, stand, p_throws)."""
        # Detailed: (bs, outs, stand, p_throws)
        counts_detailed: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(8))
        counts_fallback: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(8))
        counts_overall = np.zeros(8)

        for row in terminal.filter(pl.col("pa_outcome").is_not_null()).iter_rows(named=True):
            outcome = row["pa_outcome"]
            if outcome not in PA_OUTCOME_IDX:
                continue
            idx = PA_OUTCOME_IDX[outcome]
            bs = int(row["base_state"]) if row["base_state"] is not None else 0
            outs = int(row["outs"]) if row["outs"] is not None else 0
            stand = row["stand"] or "R"
            p_throws = row["p_throws"] or "R"

            counts_detailed[(bs, outs, stand, p_throws)][idx] += 1
            counts_fallback[(bs, outs)][idx] += 1
            counts_overall[idx] += 1

        # Normalize detailed; only keep entries with >= MIN_OBS observations
        self._outcome_detailed: dict[tuple, np.ndarray] = {}
        for key, arr in counts_detailed.items():
            total = arr.sum()
            if total >= _MIN_OBS:
                self._outcome_detailed[key] = arr / total

        # Normalize fallback (no threshold)
        self._outcome_fallback: dict[tuple, np.ndarray] = {}
        for key, arr in counts_fallback.items():
            total = arr.sum()
            if total > 0:
                self._outcome_fallback[key] = arr / total

        # Overall unconditional
        total = counts_overall.sum()
        self._outcome_overall: np.ndarray = (
            counts_overall / total if total > 0 else np.ones(8) / 8
        )

    def _fit_transitions(self, terminal: pl.DataFrame) -> None:
        """Build empirical runner transition table: (bs, outs, outcome) -> list[(bs_after, runs)]."""
        table: dict[tuple, list[tuple[int, int]]] = defaultdict(list)

        for row in terminal.filter(
            pl.col("pa_outcome").is_not_null() & pl.col("base_state_after").is_not_null()
        ).iter_rows(named=True):
            outcome = row["pa_outcome"]
            if outcome not in PA_OUTCOME_IDX:
                continue
            bs = int(row["base_state"]) if row["base_state"] is not None else 0
            outs = int(row["outs"]) if row["outs"] is not None else 0
            bs_after = int(row["base_state_after"])
            runs = int(row["runs_scored"]) if row["runs_scored"] is not None else 0
            table[(bs, outs, outcome)].append((bs_after, runs))

        self._transitions = dict(table)

    def _fit_stand_p_throws(self, pitches: pl.DataFrame) -> None:
        """Estimate marginal distributions of stand and p_throws from all pitches."""
        # Use polars groupby for speed
        stand_vc = pitches.filter(pl.col("stand").is_not_null()).group_by("stand").len()
        p_throws_vc = pitches.filter(pl.col("p_throws").is_not_null()).group_by("p_throws").len()

        stand_dict = {row["stand"]: row["len"] for row in stand_vc.iter_rows(named=True)
                      if row["stand"] in ("L", "R")}
        p_throws_dict = {row["p_throws"]: row["len"] for row in p_throws_vc.iter_rows(named=True)
                         if row["p_throws"] in ("L", "R")}

        total_s = sum(stand_dict.values()) or 1
        total_p = sum(p_throws_dict.values()) or 1
        self._stand_probs: dict[str, float] = {k: v / total_s for k, v in stand_dict.items()}
        self._p_throws_probs: dict[str, float] = {k: v / total_p for k, v in p_throws_dict.items()}

    def _fit_hbp_rate(self, terminal: pl.DataFrame) -> None:
        n_terminal = len(terminal)
        n_hbp = terminal.filter(pl.col("pa_outcome") == "HBP").shape[0]
        self._hbp_rate: float = n_hbp / n_terminal if n_terminal > 0 else 0.01

    # ------------------------------------------------------------------
    # Outcome sampling
    # ------------------------------------------------------------------

    def _sample_outcome(
        self, bs: int, outs: int, stand: str, p_throws: str
    ) -> str:
        """Sample a PA outcome using the richest available probability table."""
        key_detailed = (bs, outs, stand, p_throws)
        key_fallback = (bs, outs)

        probs = self._outcome_detailed.get(key_detailed)
        if probs is None:
            probs = self._outcome_fallback.get(key_fallback)
        if probs is None:
            probs = self._outcome_overall
        idx = int(np.random.choice(8, p=probs))
        return PA_OUTCOMES[idx]

    # ------------------------------------------------------------------
    # Transition sampling
    # ------------------------------------------------------------------

    def _sample_transition(
        self, bs: int, outs: int, outcome: str
    ) -> tuple[int, int]:
        """Sample (base_state_after, runs_scored) for a given PA outcome."""
        key = (bs, outs, outcome)
        if key in self._transitions and self._transitions[key]:
            entry = self._transitions[key]
            chosen = entry[np.random.randint(len(entry))]
            return chosen
        return self._deterministic_transition(bs, outs, outcome)

    @staticmethod
    def _deterministic_transition(
        bs: int, outs: int, outcome: str
    ) -> tuple[int, int]:
        """Apply standard baseball advancement rules deterministically."""
        on_1b = bool(bs & 1)
        on_2b = bool(bs & 2)
        on_3b = bool(bs & 4)
        count_runners = int(on_1b) + int(on_2b) + int(on_3b)

        if outcome == "K":
            return (bs, 0)

        if outcome in ("BB", "HBP"):
            # Force advancement
            if bs == 7:  # bases loaded
                return (7, 1)
            elif bs == 3:  # 1B+2B
                return (7, 0)
            elif bs == 1:  # 1B only
                return (3, 0)
            elif bs == 6:  # 2B+3B
                return (7, 0)
            elif bs == 2:  # 2B only
                return (3, 0)
            elif bs == 4:  # 3B only
                return (5, 0)
            else:  # empty
                return (1, 0)

        if outcome == "HR":
            return (0, count_runners + 1)

        if outcome == "3B":
            # all runners score, batter on 3B
            return (4, count_runners)

        if outcome == "2B":
            runs = 0
            new_bs = 0
            # runners from 2B and 3B score
            if on_2b:
                runs += 1
            if on_3b:
                runs += 1
            # runner from 1B goes to 3B if possible, else scores
            if on_1b:
                if not on_2b and not on_3b:
                    new_bs |= 4  # 1B runner goes to 3B (2B is open for batter)
                else:
                    runs += 1  # 1B runner scores
            # batter on 2B
            new_bs |= 2
            return (new_bs, runs)

        if outcome == "1B":
            runs = 0
            new_bs = 0
            # runner from 3B scores
            if on_3b:
                runs += 1
            # runner from 2B goes to 3B
            if on_2b:
                new_bs |= 4
            # runner from 1B goes to 2B
            if on_1b:
                new_bs |= 2
            # batter to 1B
            new_bs |= 1
            return (new_bs, runs)

        if outcome == "out":
            return (bs, 0)

        return (bs, 0)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _simulate_half_inning(self) -> int:
        """Simulate one half-inning and return runs scored."""
        bs = 0
        outs = 0
        runs = 0
        stand_choices = list(self._stand_probs.keys())
        stand_weights = np.array([self._stand_probs[k] for k in stand_choices])
        p_throws_choices = list(self._p_throws_probs.keys())
        p_throws_weights = np.array([self._p_throws_probs[k] for k in p_throws_choices])

        while outs < 3:
            stand = stand_choices[int(np.random.choice(len(stand_choices), p=stand_weights))]
            p_throws = p_throws_choices[
                int(np.random.choice(len(p_throws_choices), p=p_throws_weights))
            ]
            outcome = self._sample_outcome(bs, outs, stand, p_throws)
            bs_new, r = self._sample_transition(bs, outs, outcome)
            runs += r
            if outcome in ("K", "out"):
                outs += 1
            bs = bs_new

        return runs

    def simulate_game(self, game_context: dict) -> GameLog:
        """Simulate one 9-inning game."""
        game_id = game_context.get("game_id", 0)
        inning_runs_away = []  # top = away bats
        inning_runs_home = []  # bot = home bats

        for _ in range(9):
            inning_runs_away.append(self._simulate_half_inning())
            inning_runs_home.append(self._simulate_half_inning())

        home_runs = sum(inning_runs_home)
        away_runs = sum(inning_runs_away)
        return GameLog(
            game_id=game_id,
            home_runs=home_runs,
            away_runs=away_runs,
            inning_runs_home=inning_runs_home,
            inning_runs_away=inning_runs_away,
        )
