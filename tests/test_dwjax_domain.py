import jax.numpy as jnp

from diamondworldjax.domain import (
    ArrayGameState,
    PAOutcome,
    PitchResult,
    apply_pitch_result,
    batting_score_diff,
)


def _state(**overrides):
    values = {
        "inning": jnp.array([1]), "half": jnp.array([0]),
        "balls": jnp.array([0]), "strikes": jnp.array([0]),
        "outs": jnp.array([0]), "base_state": jnp.array([0]),
        "home_score": jnp.array([0]), "away_score": jnp.array([0]),
        "pitch_count_game": jnp.array([0]), "pitch_count_inning": jnp.array([0]),
        "pitch_count_pa": jnp.array([0]), "tto": jnp.array([1]),
    }
    values.update({key: jnp.array([value]) for key, value in overrides.items()})
    return ArrayGameState(**values)


def _pitch(**overrides):
    values = {
        "swing": jnp.array([0]), "called_strike": jnp.array([0]),
        "contact": jnp.array([0]), "foul": jnp.array([0]),
        "runs_scored": jnp.array([0]), "base_state_after": jnp.array([0]),
        "outs_added": jnp.array([0]), "pa_outcome": jnp.array([int(PAOutcome.OUT)]),
    }
    values.update({key: jnp.array([value]) for key, value in overrides.items()})
    return PitchResult(**values)


def test_score_diff_is_always_batting_team_perspective():
    assert batting_score_diff(_state(half=0, away_score=4, home_score=2)).item() == 2
    assert batting_score_diff(_state(half=1, away_score=4, home_score=2)).item() == -2


def test_bases_loaded_walk_forces_one_run_and_preserves_loaded_bases():
    stepped = apply_pitch_result(_state(balls=3, base_state=7), _pitch())
    assert stepped.pa_terminal.item()
    assert stepped.state.away_score.item() == 1
    assert stepped.state.base_state.item() == 7
    assert stepped.outcome.item() == int(PAOutcome.WALK)


def test_third_out_resets_half_inning_state():
    stepped = apply_pitch_result(
        _state(strikes=2, outs=2, base_state=5),
        _pitch(swing=1, contact=0),
    )
    assert stepped.inning_over.item()
    assert stepped.state.half.item() == 1
    assert stepped.state.outs.item() == 0
    assert stepped.state.base_state.item() == 0


def test_tied_game_continues_to_extra_innings():
    stepped = apply_pitch_result(
        _state(inning=9, half=1, strikes=2, outs=2, home_score=3, away_score=3),
        _pitch(swing=1, contact=0),
    )
    assert not stepped.game_over.item()
    assert stepped.state.inning.item() == 10


def test_home_run_can_end_game_mid_bottom_ninth():
    stepped = apply_pitch_result(
        _state(inning=9, half=1, home_score=2, away_score=3),
        _pitch(
            swing=1,
            contact=1,
            foul=0,
            runs_scored=2,
            pa_outcome=int(PAOutcome.HOME_RUN),
        ),
    )
    assert stepped.game_over.item()
    assert stepped.state.home_score.item() == 4


def test_hbp_ends_pa_and_advances_batter():
    stepped = apply_pitch_result(
        _state(balls=1, strikes=1),
        _pitch(pa_outcome=int(PAOutcome.HIT_BY_PITCH)),
    )
    assert stepped.pa_terminal.item()
    assert stepped.outcome.item() == int(PAOutcome.HIT_BY_PITCH)
    assert stepped.state.base_state.item() == 1
    assert stepped.state.balls.item() == 0
    assert stepped.state.strikes.item() == 0


def test_hbp_bases_loaded_scores_run():
    stepped = apply_pitch_result(
        _state(base_state=7),
        _pitch(pa_outcome=int(PAOutcome.HIT_BY_PITCH)),
    )
    assert stepped.pa_terminal.item()
    assert stepped.outcome.item() == int(PAOutcome.HIT_BY_PITCH)
    assert stepped.state.base_state.item() == 7
    assert stepped.state.away_score.item() == 1


def test_hbp_does_not_increment_ball_count():
    stepped = apply_pitch_result(
        _state(balls=3),
        _pitch(pa_outcome=int(PAOutcome.HIT_BY_PITCH)),
    )
    assert stepped.pa_terminal.item()
    assert stepped.outcome.item() == int(PAOutcome.HIT_BY_PITCH)
    assert stepped.state.balls.item() == 0
