"""GameState and PitchRow dataclasses for DiamondWorldJAX."""
from __future__ import annotations
from dataclasses import dataclass, field
import jax.numpy as jnp


@dataclass
class GameState:
    """Full pre-pitch game state (scalar per field, JAX-compatible)."""
    inning: int = 1
    half: int = 0          # 0=top, 1=bot
    balls: int = 0
    strikes: int = 0
    outs: int = 0
    base_state: int = 0    # 3-bit bitmask
    home_score: int = 0
    away_score: int = 0
    pitch_count_game: int = 0
    pitch_count_inning: int = 0
    pitch_count_pa: int = 0
    tto: int = 1
    shift_restricted: int = 0
    pitch_clock: int = 0

    def score_diff(self) -> int:
        return self.away_score - self.home_score if self.half == 0 else self.home_score - self.away_score

    def to_array(self):
        """Flatten to float32 array for model input."""
        return jnp.array([
            (self.inning - 1) / 8.0,
            float(self.half),
            self.balls / 3.0,
            self.strikes / 2.0,
            self.outs / 2.0,
            self.base_state / 7.0,
            self.score_diff() / 10.0,
            self.pitch_count_game / 100.0,
            self.pitch_count_inning / 30.0,
            self.pitch_count_pa / 10.0,
            self.tto / 3.0,
            float(self.shift_restricted),
            float(self.pitch_clock),
        ], dtype=jnp.float32)

    STATE_DIM: int = 13


@dataclass
class PitchRow:
    """One pitch with all observable fields (None = missing/masked)."""
    game_id: str = ""
    pitcher_id: int = 0
    batter_id: int = 0
    park_id: int = 0
    state: GameState = field(default_factory=GameState)

    # Pitch package
    pitch_type: int | None = None
    release_speed: float | None = None
    plate_x: float | None = None
    plate_z: float | None = None
    pfx_x: float | None = None
    pfx_z: float | None = None

    # Hurdle path
    swing: int | None = None
    called_strike: int | None = None
    contact: int | None = None
    foul: int | None = None
    in_play: int | None = None

    # Batted ball
    launch_speed: float | None = None
    launch_angle: float | None = None
    spray_angle: float | None = None
    hit_distance: float | None = None

    # Transition
    pa_outcome: int | None = None
    runs_scored: int | None = None
    base_state_after: int | None = None
    outs_after: int | None = None
    pa_terminal: bool = False

    # Manager
    pitching_change: int | None = None
    steal_attempt: int | None = None
