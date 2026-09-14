"""Canonical DiamondWorldJAX baseball-domain contracts.

This module deliberately has no Flax or NumPyro dependency. Data preparation,
simulation, evaluation, and model code share the vocabulary and rules here.
"""
from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import jax.numpy as jnp


class PAOutcome(IntEnum):
    STRIKEOUT = 0
    WALK = 1
    HIT_BY_PITCH = 2
    SINGLE = 3
    DOUBLE = 4
    TRIPLE = 5
    HOME_RUN = 6
    OUT = 7
    ERROR = 8


PA_OUTCOMES: tuple[str, ...] = ("K", "BB", "HBP", "1B", "2B", "3B", "HR", "out", "E")
PA_OUTCOME_IDX: dict[str, int] = {label: i for i, label in enumerate(PA_OUTCOMES)}
N_PA_OUTCOMES = len(PA_OUTCOMES)


class ArrayGameState(NamedTuple):
    """JAX-pytree game state. Fields may be scalars or matching arrays."""

    inning: jnp.ndarray
    half: jnp.ndarray
    balls: jnp.ndarray
    strikes: jnp.ndarray
    outs: jnp.ndarray
    base_state: jnp.ndarray
    home_score: jnp.ndarray
    away_score: jnp.ndarray
    pitch_count_game: jnp.ndarray
    pitch_count_inning: jnp.ndarray
    pitch_count_pa: jnp.ndarray
    tto: jnp.ndarray


class PitchResult(NamedTuple):
    """Generated variables consumed by the deterministic rule engine."""

    swing: jnp.ndarray
    called_strike: jnp.ndarray
    contact: jnp.ndarray
    foul: jnp.ndarray
    runs_scored: jnp.ndarray
    base_state_after: jnp.ndarray
    outs_added: jnp.ndarray
    pa_outcome: jnp.ndarray


class StateStep(NamedTuple):
    state: ArrayGameState
    pa_terminal: jnp.ndarray
    inning_over: jnp.ndarray
    game_over: jnp.ndarray
    outcome: jnp.ndarray


def batting_score_diff(state: ArrayGameState) -> jnp.ndarray:
    """Score differential from the team-at-bat perspective."""
    return jnp.where(
        state.half == 0,
        state.away_score - state.home_score,
        state.home_score - state.away_score,
    )


def advance_outs(
    outs: jnp.ndarray,
    outcome_type: jnp.ndarray,
    learned_outs_added: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Canonical out-count transition used by model and simulator adapters.

    Outcome codes are 0=strikeout, 1=walk/HBP, 2=in-play, 3=no-op.
    """
    # Recorded strikeouts can include a second out (for example, a caught
    # stealing on strike three).  Preserve it while ensuring every strikeout
    # still contributes at least one out.
    outs_added = jnp.where(outcome_type == 0, jnp.maximum(1, learned_outs_added), 0)
    outs_added = jnp.where(outcome_type == 2, learned_outs_added, outs_added)
    total_outs = outs + outs_added
    return jnp.clip(total_outs, 0, 2), total_outs >= 3, outs_added


def apply_pitch_result(state: ArrayGameState, result: PitchResult) -> StateStep:
    """Advance one pitch while enforcing count, inning, and score invariants."""
    swing = result.swing.astype(bool)
    contact = result.contact.astype(bool)
    foul = result.foul.astype(bool)
    called_strike = result.called_strike.astype(bool)

    hbp = result.pa_outcome.astype(jnp.int32) == int(PAOutcome.HIT_BY_PITCH)
    in_play = swing & contact & ~foul
    ordinary_strike = (swing & ~contact) | (~swing & called_strike)
    next_strikes = state.strikes + ordinary_strike.astype(jnp.int32)
    next_strikes = jnp.where(foul, jnp.minimum(state.strikes + 1, 2), next_strikes)
    next_balls = state.balls + ((~swing) & (~called_strike) & (~hbp)).astype(jnp.int32)

    strikeout = next_strikes >= 3
    walk = next_balls >= 4
    terminal = strikeout | walk | in_play | hbp
    learned_outs = jnp.where(in_play, result.outs_added, 0).astype(jnp.int32)
    outs_added = jnp.where(strikeout, jnp.maximum(1, result.outs_added), learned_outs)
    total_outs = state.outs + outs_added
    inning_over = total_outs >= 3

    walk_base_table = jnp.asarray([1, 3, 3, 7, 5, 7, 7, 7], dtype=jnp.int32)
    walk_base = walk_base_table[jnp.clip(state.base_state, 0, 7)]
    walk_runs = (state.base_state == 7).astype(jnp.int32)
    runs = jnp.where(in_play, result.runs_scored, 0).astype(jnp.int32)
    runs = jnp.where(walk | hbp, walk_runs, runs)
    home_score = state.home_score + jnp.where(state.half == 1, runs, 0)
    away_score = state.away_score + jnp.where(state.half == 0, runs, 0)
    base_after = jnp.where(in_play, result.base_state_after, state.base_state).astype(jnp.int32)
    base_after = jnp.where(walk | hbp, walk_base, base_after)
    base_after = jnp.where(inning_over, 0, base_after)

    next_state = ArrayGameState(
        inning=state.inning + (inning_over & (state.half == 1)).astype(jnp.int32),
        half=jnp.where(inning_over, 1 - state.half, state.half),
        balls=jnp.where(terminal, 0, next_balls),
        strikes=jnp.where(terminal, 0, next_strikes),
        outs=jnp.where(inning_over, 0, total_outs),
        base_state=base_after,
        home_score=home_score,
        away_score=away_score,
        pitch_count_game=state.pitch_count_game + 1,
        pitch_count_inning=jnp.where(inning_over, 0, state.pitch_count_inning + 1),
        pitch_count_pa=jnp.where(terminal, 0, state.pitch_count_pa + 1),
        tto=state.tto,
    )

    outcome = jnp.full_like(state.outs, -1, dtype=jnp.int32)
    outcome = jnp.where(strikeout, int(PAOutcome.STRIKEOUT), outcome)
    outcome = jnp.where(walk, int(PAOutcome.WALK), outcome)
    outcome = jnp.where(hbp, int(PAOutcome.HIT_BY_PITCH), outcome)
    sampled_outcome = result.pa_outcome.astype(jnp.int32)
    valid_in_play_outcome = (sampled_outcome >= int(PAOutcome.SINGLE)) & (
        sampled_outcome <= int(PAOutcome.ERROR)
    )
    sampled_outcome = jnp.where(valid_in_play_outcome, sampled_outcome, int(PAOutcome.OUT))
    outcome = jnp.where(in_play, sampled_outcome, outcome)
    late_inning = state.inning >= 9
    home_ahead = home_score > away_score
    scores_differ = home_score != away_score
    skip_bottom = inning_over & (state.half == 0) & late_inning & home_ahead
    completed_bottom = inning_over & (state.half == 1) & late_inning & scores_differ
    walk_off = (state.half == 1) & late_inning & home_ahead
    game_over = skip_bottom | completed_bottom | walk_off
    return StateStep(next_state, terminal, inning_over, game_over, outcome)
