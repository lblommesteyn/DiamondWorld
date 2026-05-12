from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import polars as pl

PA_OUTCOMES = ["K", "BB", "HBP", "1B", "2B", "3B", "HR", "out", "E"]
PA_OUTCOME_IDX = {o: i for i, o in enumerate(PA_OUTCOMES)}
IN_PLAY_OUTCOMES = {"1B", "2B", "3B", "HR", "out", "E"}
# Outcomes where batter reaches base (E = fielding error, batter safe at 1B)
REACH_OUTCOMES = {"BB", "HBP", "1B", "2B", "3B", "HR", "E"}


@dataclass
class GameLog:
    game_id: int
    home_runs: int
    away_runs: int
    inning_runs_home: list[int] = field(default_factory=list)  # len 9, bot of inning (home bats)
    inning_runs_away: list[int] = field(default_factory=list)  # len 9, top of inning (away bats)


def game_logs_to_frame(logs: list[GameLog]) -> pl.DataFrame:
    """Convert list of GameLogs to a DataFrame compatible with eval harness.
    Produces one row per half-inning. The harness filters pa_terminal=True and
    groups by (game_pk, half) for game runs, (game_pk, inning, half) for inning runs.
    PA length KL will be meaningless (set at_bat_number uniquely per half-inning).
    """
    rows = []
    for log in logs:
        # top = away bats
        for i, runs in enumerate(log.inning_runs_away, start=1):
            rows.append({
                "game_pk": log.game_id,
                "inning": i,
                "half": "top",
                "pa_terminal": True,
                "runs_scored": runs,
                "at_bat_number": log.game_id * 200 + i,
            })
        # bot = home bats
        for i, runs in enumerate(log.inning_runs_home, start=1):
            rows.append({
                "game_pk": log.game_id,
                "inning": i,
                "half": "bot",
                "pa_terminal": True,
                "runs_scored": runs,
                "at_bat_number": log.game_id * 200 + 100 + i,
            })
    return pl.DataFrame(rows, schema={
        "game_pk": pl.Int64,
        "inning": pl.Int32,
        "half": pl.Utf8,
        "pa_terminal": pl.Boolean,
        "runs_scored": pl.Int32,
        "at_bat_number": pl.Int64,
    })


class BaseSimulator(ABC):
    @abstractmethod
    def fit(self, pitches: pl.DataFrame) -> None: ...

    @abstractmethod
    def simulate_game(self, game_context: dict) -> GameLog: ...

    def simulate_season(self, n_games: int = 2430) -> pl.DataFrame:
        logs = [self.simulate_game({"game_id": i}) for i in range(n_games)]
        return game_logs_to_frame(logs)
