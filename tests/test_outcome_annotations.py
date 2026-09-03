"""Regression tests for terminal-out and reached-on-error data annotations."""
from __future__ import annotations

from diamondworld.data.schema import OUTCOME_BY_EVENT
from diamondworld.data.transitions import annotate_terminal_outs


def _row(*, game=1, inning=1, half="top", outs=0, terminal=False):
    return {
        "game_pk": game,
        "inning": inning,
        "half": half,
        "outs": outs,
        "pa_terminal": terminal,
    }


def test_errors_and_catcher_interference_are_reached_base_outcomes():
    assert OUTCOME_BY_EVENT["field_error"] == "E"
    assert OUTCOME_BY_EVENT["catcher_interf"] == "E"


def test_terminal_out_annotation_handles_double_and_inning_ending_plays():
    rows = [
        _row(outs=0, terminal=True),       # double play: 0 -> 2
        _row(outs=2),
        _row(outs=2, terminal=True),       # inning-ending single out: 2 -> 3
        _row(half="bot", outs=0),
        _row(game=2, outs=1, terminal=True),  # final game PA: unobserved
    ]

    annotate_terminal_outs(rows)

    assert (rows[0]["outs_added"], rows[0]["outs_after"]) == (2, 2)
    assert (rows[2]["outs_added"], rows[2]["outs_after"]) == (1, 3)
    assert rows[4]["outs_added"] is None
    assert rows[4]["outs_after"] is None
