import numpy as np
import polars as pl

from diamondworldjax.sim.c_transition_engine import CTransitionEngine


def test_c_transition_engine_samples_observed_nonterminal_action():
    pitches = pl.DataFrame({
        "game_pk": [1, 1, 1], "at_bat_number": [1, 1, 1], "pitch_number": [1, 2, 3],
        "base_state": [1, 2, 2], "outs": [0, 0, 0], "pa_terminal": [False, False, False],
        "home_score": [0, 0, 0], "away_score": [0, 0, 0],
    })
    events = pl.DataFrame({"game_pk": [1], "at_bat_number": [1], "pitch_number": [1],
                           "steal": [1]})
    engine = CTransitionEngine().fit(pitches, events)
    got = engine.sample(np.array([3]), np.array([1]), np.array([0]), np.random.default_rng(0))
    assert got["base"].tolist() == [2]
    assert got["outs"].tolist() == [0]
    assert got["runs"].tolist() == [0]


def test_c_transition_engine_unsupported_action_is_a_legal_noop():
    engine = CTransitionEngine()
    got = engine.sample(np.array([0]), np.array([5]), np.array([2]), np.random.default_rng(0))
    assert got["base"].tolist() == [5]
    assert got["outs"].tolist() == [2]
    assert got["runs"].tolist() == [0]
