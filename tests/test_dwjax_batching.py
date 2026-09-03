import polars as pl
import pytest
import numpy as np

from diamondworldjax.data.batching import build_batch


def _frame():
    return pl.DataFrame(
        {
            "game_pk": [10, 10, 10, 10],
            "at_bat_number": [1, 1, 1, 2],
            "pitch_number": [1, 2, 3, 1],
            "inning": [1, 1, 1, 1],
            "half_bin": [0, 0, 0, 0],
            "balls": [0, 1, 1, 0],
            "strikes": [0, 0, 1, 0],
            "outs": [0, 0, 0, 1],
            "base_state": [0, 0, 0, 0],
            "score_diff": [2, 2, 2, 2],
            "pitch_count_game": [1, 2, 3, 4],
            "pitch_count_inning": [1, 2, 3, 4],
            "tto": [1, 1, 1, 1],
            "pitch_type_idx": [0, 0, 3, 0],
            "release_speed": [95.0, 95.0, 85.0, 94.0],
            "plate_x": [1.2, 0.0, 0.1, 0.0],
            "plate_z": [2.5, 2.5, 2.5, 2.5],
            "swing_obs": [0, 1, 1, 0],
            "called_strike_obs": [0, -1, -1, 1],
            "contact_obs": [0, 0, 1, 0],
            "foul_obs": [0, 0, 0, 0],
            "in_play_obs": [0, 0, 1, 0],
            "launch_speed": [None, None, 101.0, None],
            "launch_angle": [None, None, 20.0, None],
            "pa_terminal": [False, False, True, True],
            "pa_outcome_idx": [-1, -1, 3, 0],
            "runs_scored": [0, 0, 1, 0],
            "base_state_after": [0, 0, 1, 1],
            "pitcher_id": [500, 500, 500, 500],
            "batter_id": [600, 600, 600, 601],
            "park_id": [30, 30, 30, 30],
        }
    )


def test_conditional_masks_are_not_terminal_proxies():
    batch = build_batch(_frame(), max_t=8)
    assert batch["terminal_mask"][0, :4].tolist() == [False, False, True, True]
    assert batch["in_play_mask"][0, :4].tolist() == [False, False, True, False]
    assert batch["contact_mask"][0, :4].tolist() == [False, True, True, False]
    assert batch["called_strike_mask"][0, :4].tolist() == [True, False, False, True]
    assert batch["foul_mask"][0, :4].tolist() == [False, False, True, False]
    assert batch["batted_mask"][0, :4].tolist() == [False, False, True, False]
    assert not batch["pitch_valid"][0, 4:].any()


def test_nullable_targets_keep_independent_masks_and_correct_ids():
    batch = build_batch(_frame(), max_t=8)
    assert batch["launch_speed_mask"][0, :4].tolist() == [False, False, True, False]
    assert batch["park_ids"][0, 0].item() == 30
    assert batch["score_diff"][0, 0].item() == pytest.approx(0.2)
    assert np.allclose(batch["pitch_count_pa"][0, :4], [0.0, 0.1, 0.2, 0.0])


def test_padding_length_does_not_change_valid_masks():
    short = build_batch(_frame(), max_t=6)
    long = build_batch(_frame(), max_t=12)
    for name in ("pitch_valid", "in_play_mask", "runs_mask", "launch_speed_mask"):
        assert short[name][0, :4].tolist() == long[name][0, :4].tolist()
        assert not short[name][0, 4:].any()
        assert not long[name][0, 4:].any()
