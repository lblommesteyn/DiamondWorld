from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from diamondworld.baselines.base import GameLog, game_logs_to_frame
from diamondworld.simulate.pa_simulator import PASimulator, PitchSampler


class GameSimulator:
    """Simulate full games using the NoMemoryMLP PA simulator."""

    def __init__(
        self,
        pa_sim: PASimulator,
        pitch_sampler: PitchSampler,
        transition_table: dict,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.pa_sim = pa_sim
        self.pitch_sampler = pitch_sampler
        self.transition_table = transition_table
        self.rng = rng if rng is not None else np.random.default_rng()

    def simulate_game(
        self,
        pitcher_pool: list[int],
        batter_pool: list[int],
        umpire_pool: list[int],
        park_idx: int,
        game_id: int = 0,
    ) -> GameLog:
        """Simulate one 9-inning game.

        Args:
            pitcher_pool: List of pitcher indices to sample from.
            batter_pool: List of batter indices to sample from.
            umpire_pool: List of umpire indices to sample from.
            park_idx: Park index for the game.
            game_id: Unique game identifier.

        Returns:
            GameLog with per-inning run totals.
        """
        inning_runs_away: list[int] = []  # top = away bats
        inning_runs_home: list[int] = []  # bot = home bats

        if not pitcher_pool:
            pitcher_pool = [0]
        if not batter_pool:
            batter_pool = [0]
        if not umpire_pool:
            umpire_pool = [0]

        for inning in range(1, 10):
            for half in ["top", "bot"]:
                base_state = 0
                outs = 0
                runs = 0
                ab_num = 0
                score_diff = sum(inning_runs_away) - sum(inning_runs_home)
                pitch_count_game = 0
                tto_tracker: dict[int, int] = {}

                while outs < 3:
                    pitcher_idx = int(self.rng.choice(pitcher_pool))
                    batter_idx = int(self.rng.choice(batter_pool))
                    umpire_idx = int(self.rng.choice(umpire_pool))

                    # Simple TTO tracking
                    tto_tracker[pitcher_idx] = tto_tracker.get(pitcher_idx, 0) + 1
                    pa_seen = tto_tracker[pitcher_idx]
                    tto = min(3, pa_seen // 9 + 1)

                    game_state: dict[str, Any] = {
                        "base_state": base_state,
                        "outs": outs,
                        "inning": inning,
                        "half": half,
                        "score_diff": score_diff,
                        "tto": tto,
                        "balls": 0,
                        "strikes": 0,
                        "pitch_count_game": pitch_count_game,
                        "pitch_count_inning": ab_num,
                        "tracking_era": 1,
                        "stand": "R",   # default; no player-specific stand in pool sim
                        "p_throws": "R",
                    }

                    result = self.pa_sim.simulate_pa(
                        pitcher_idx, batter_idx, umpire_idx, park_idx,
                        game_state, self.pitch_sampler, rng=self.rng,
                    )

                    bs_after = result.get("base_state_after", base_state)
                    pa_runs = result.get("runs_scored", 0)
                    outcome = result.get("pa_outcome", "out")

                    runs += pa_runs
                    score_diff += pa_runs  # simplified: assumes batting team score increases
                    pitch_count_game += result.get("n_pitches", 1)

                    if outcome in ("K", "out"):
                        outs += 1

                    base_state = bs_after
                    ab_num += 1

                if half == "top":
                    inning_runs_away.append(runs)
                else:
                    inning_runs_home.append(runs)

        return GameLog(
            game_id=game_id,
            home_runs=sum(inning_runs_home),
            away_runs=sum(inning_runs_away),
            inning_runs_home=inning_runs_home,
            inning_runs_away=inning_runs_away,
        )

    def simulate_season(
        self,
        pitcher_pool: list[int],
        batter_pool: list[int],
        umpire_pool: list[int],
        park_idx: int,
        n_games: int = 2430,
    ) -> pl.DataFrame:
        """Simulate a full season and return a DataFrame compatible with eval harness.

        Returns one row per PA (pa_terminal=True), matching the eval harness format.
        Inning/half/runs_scored/game_pk columns are set; at_bat_number is unique per PA.
        """
        logs = [
            self.simulate_game(
                pitcher_pool, batter_pool, umpire_pool, park_idx, game_id=i
            )
            for i in range(n_games)
        ]
        return game_logs_to_frame(logs)
