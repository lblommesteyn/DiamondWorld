"""Lineup-aware game simulator for DiamondWorld.

Key differences from GameSimulator (random-pool):
  - 9-batter order that cycles across innings (lineup slot persists)
  - Starting pitcher + bullpen; pitcher swapped out by pitch count
  - TTO computed from actual batter-pitcher encounter history
  - Umpire sampled once per game (not per PA)

Works with any PASimulator that has a simulate_pa() method
(Phase 3 NoMemoryMLP or Phase 4 GameContextTransformer).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from diamondworld.baselines.base import GameLog
from diamondworld.simulate.pa_simulator import PASimulator, PitchSampler

# Pitch-count thresholds for pitcher changes
STARTER_LIMIT = 70    # pull starter after ~70 pitches (prevents TTO=3 bleed)
RELIEVER_LIMIT = 25   # each reliever gets ~25 pitches


@dataclass
class Lineup:
    """One team's lineup for a game."""
    batters: list[int]   # 9 batter indices in batting order (slot 0 = leadoff)
    starter: int         # starting pitcher index
    bullpen: list[int]   # relief pitchers in order of intended usage


@dataclass
class _PitcherState:
    """Tracks the current pitcher and his pitch count this appearance."""
    current: int
    remaining: list[int]        # bullpen not yet used
    pitches_this_outing: int = 0
    limit: int = STARTER_LIMIT  # current pitcher's pitch limit

    def record_pitches(self, n: int) -> None:
        self.pitches_this_outing += n

    def maybe_change(self, rng: np.random.Generator) -> bool:
        """Return True and rotate to next pitcher if over the pitch limit."""
        if self.pitches_this_outing >= self.limit and self.remaining:
            self.current = self.remaining.pop(0)
            self.pitches_this_outing = 0
            self.limit = RELIEVER_LIMIT
            return True
        return False

    def force_change(self) -> bool:
        """Unconditionally rotate to next pitcher (TTO-triggered hook)."""
        if self.remaining:
            self.current = self.remaining.pop(0)
            self.pitches_this_outing = 0
            self.limit = RELIEVER_LIMIT
            return True
        return False


class LineupSampler:
    """Generates quality-stratified lineups from player index pools.

    Call fit() once with historical pitch data so batter reach-rates can be
    computed. Without fit(), falls back to uniform random sampling.

    Lineup slot assignment mirrors real baseball construction:
      Slots 0-3  (leadoff + middle): top-tier batters
      Slots 4-5  (lower-middle):     mid-tier batters
      Slots 6-8  (bottom of order):  bottom-tier batters
    """

    # Outcomes that count as "reaching base" for quality scoring
    _REACH = {"BB", "HBP", "1B", "2B", "3B", "HR"}

    def __init__(self) -> None:
        self._quality: dict[int, float] = {}   # batter_idx → reach rate
        self._default_quality: float = 0.32    # population mean, updated by fit()

    def fit(self, pitches, registry, min_pa: int = 5) -> "LineupSampler":
        """Compute per-batter reach rate from terminal-pitch rows."""
        import polars as pl
        if "pa_terminal" not in pitches.columns or "pa_outcome" not in pitches.columns:
            return self

        terminal = pitches.filter(pl.col("pa_terminal"))
        from collections import defaultdict
        reach: dict[int, int] = defaultdict(int)
        total: dict[int, int] = defaultdict(int)

        for row in terminal.select(["batter_id", "pa_outcome"]).iter_rows(named=True):
            idx = registry.batter(row["batter_id"])
            if idx == 0:
                continue
            total[idx] += 1
            if row.get("pa_outcome") in self._REACH:
                reach[idx] += 1

        self._quality = {
            idx: reach[idx] / cnt
            for idx, cnt in total.items()
            if cnt >= min_pa
        }
        if self._quality:
            self._default_quality = sum(self._quality.values()) / len(self._quality)
        return self

    def _tier_split(self, pool: list[int]) -> tuple[list[int], list[int], list[int]]:
        """Split pool into (top, mid, bot) thirds by reach rate."""
        ranked = sorted(pool, key=lambda i: self._quality.get(i, self._default_quality), reverse=True)
        n = len(ranked)
        cut1, cut2 = max(1, n // 3), max(2, 2 * n // 3)
        return ranked[:cut1], ranked[cut1:cut2], ranked[cut2:]

    def sample_lineup(
        self,
        batter_pool: list[int],
        pitcher_pool: list[int],
        rng: np.random.Generator,
        bullpen_size: int = 4,
    ) -> Lineup:
        """Sample a uniformly random 9-batter order + starter + bullpen."""
        chosen = rng.choice(
            batter_pool,
            size=min(9, len(batter_pool)),
            replace=False,
        )
        batters = [int(x) for x in chosen]
        while len(batters) < 9:
            batters.append(int(rng.choice(batter_pool)))

        pitchers = list(rng.choice(
            pitcher_pool, size=min(1 + bullpen_size, len(pitcher_pool)), replace=False
        ))
        return Lineup(batters=batters[:9], starter=pitchers[0],
                      bullpen=pitchers[1:] if len(pitchers) > 1 else [pitchers[0]])


class LineupAwareGameSimulator:
    """Simulate full 9-inning games with cycling lineup and pitcher fatigue.

    Compared to the random-pool simulator this correctly captures:
      - Within-inning correlations (same ordered batters face same tired pitcher)
      - Time-through-order (TTO) effects (pitcher degrades as order cycles)
      - Cross-inning lineup continuity (slot persists between half-innings)
    """

    def __init__(
        self,
        pa_sim: PASimulator,
        pitch_sampler: PitchSampler,
        transition_table: dict,
        lineup_sampler: LineupSampler | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.pa_sim = pa_sim
        self.pitch_sampler = pitch_sampler
        self.transition_table = transition_table
        self.lineup_sampler = lineup_sampler or LineupSampler()
        self.rng = rng if rng is not None else np.random.default_rng()

    def simulate_game(
        self,
        away_lineup: Lineup,
        home_lineup: Lineup,
        umpire_pool: list[int],
        park_idx: int,
        game_id: int = 0,
    ) -> GameLog:
        """Simulate one 9-inning game with full lineup structure."""
        inning_runs_away: list[int] = []
        inning_runs_home: list[int] = []

        umpire_idx = int(self.rng.choice(umpire_pool)) if umpire_pool else 0

        # Persistent batting order slots across innings
        away_slot = 0
        home_slot = 0

        # Pitcher states: away pitcher defends home half, home pitcher defends away half
        away_pitcher = _PitcherState(
            current=away_lineup.starter,
            remaining=list(away_lineup.bullpen),
            limit=STARTER_LIMIT,
        )
        home_pitcher = _PitcherState(
            current=home_lineup.starter,
            remaining=list(home_lineup.bullpen),
            limit=STARTER_LIMIT,
        )

        # TTO tracking: {(pitcher_idx, batter_idx): n_encounters}
        tto_log: dict[tuple[int, int], int] = {}

        for inning in range(1, 10):
            for half in ["top", "bot"]:
                # away bats top (faces home pitcher), home bats bottom (faces away pitcher)
                if half == "top":
                    batting_order = away_lineup.batters
                    defending_pitcher_state = home_pitcher
                    slot_ref = [away_slot]
                else:
                    batting_order = home_lineup.batters
                    defending_pitcher_state = away_pitcher
                    slot_ref = [home_slot]

                base_state = 0
                outs = 0
                runs = 0
                score_diff = sum(inning_runs_away) - sum(inning_runs_home)

                while outs < 3:
                    batter_idx = batting_order[slot_ref[0] % 9]
                    pitcher_idx = defending_pitcher_state.current

                    # TTO: how many times has this batter faced this pitcher this game
                    key = (pitcher_idx, batter_idx)
                    tto_log[key] = tto_log.get(key, 0) + 1
                    tto = min(3, tto_log[key])

                    game_state: dict[str, Any] = {
                        "base_state": base_state,
                        "outs": outs,
                        "inning": inning,
                        "half": half,
                        "score_diff": score_diff,
                        "tto": tto,
                        "balls": 0,
                        "strikes": 0,
                        # Use current pitcher's outing count (resets to 0 on change),
                        # matching the semantic of pitch_count_game in training data.
                        "pitch_count_game": defending_pitcher_state.pitches_this_outing,
                        "pitch_count_inning": slot_ref[0] % 9,
                        "tracking_era": 1,
                        "stand": "R",
                        "p_throws": "R",
                    }

                    result = self.pa_sim.simulate_pa(
                        pitcher_idx, batter_idx, umpire_idx, park_idx,
                        game_state, self.pitch_sampler, rng=self.rng,
                    )

                    n_pitches = result.get("n_pitches", 1)
                    pa_runs = result.get("runs_scored", 0)
                    outcome = result.get("pa_outcome", "out")
                    base_state = result.get("base_state_after", base_state)

                    runs += pa_runs
                    score_diff += pa_runs
                    defending_pitcher_state.record_pitches(n_pitches)
                    defending_pitcher_state.maybe_change(self.rng)

                    if outcome in ("K", "out"):
                        outs += 1

                    slot_ref[0] += 1  # advance batting order slot

                # Write back persistent slot and run totals
                if half == "top":
                    away_slot = slot_ref[0]
                    inning_runs_away.append(runs)
                else:
                    home_slot = slot_ref[0]
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
        batter_pool: list[int],
        pitcher_pool: list[int],
        umpire_pool: list[int],
        park_idx: int,
        n_games: int = 2430,
    ) -> list[GameLog]:
        """Simulate a full season, generating fresh lineups for each game."""
        logs = []
        for i in range(n_games):
            away_lineup = self.lineup_sampler.sample_lineup(batter_pool, pitcher_pool, self.rng)
            home_lineup = self.lineup_sampler.sample_lineup(batter_pool, pitcher_pool, self.rng)
            logs.append(self.simulate_game(away_lineup, home_lineup, umpire_pool, park_idx, game_id=i))
        return logs
