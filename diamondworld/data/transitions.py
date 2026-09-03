"""Dependency-free annotations for observed baseball state transitions."""
from __future__ import annotations

from typing import Any


def annotate_terminal_outs(rows: list[dict[str, Any]]) -> None:
    """Recover terminal-PA out increments from the next pitch state.

    The raw Statcast feed records outs *before* each pitch.  Comparing a
    terminal pitch with the following pitch gives the actual number of outs on
    the play, including double and triple plays.  At an inning boundary, the
    next pitch has reset to zero outs, so the play must have supplied the
    remainder of the three outs.  The final PA in a game has no successor and
    is intentionally left unlabelled rather than guessed.
    """
    for i, row in enumerate(rows):
        before = int(row["outs"])
        row["outs_added"] = 0
        row["outs_after"] = before
        if not row["pa_terminal"]:
            continue

        if i + 1 >= len(rows) or rows[i + 1]["game_pk"] != row["game_pk"]:
            row["outs_added"] = None
            row["outs_after"] = None
            continue

        nxt = rows[i + 1]
        same_half_inning = (
            nxt["inning"] == row["inning"] and nxt["half"] == row["half"]
        )
        added = max(0, int(nxt["outs"]) - before) if same_half_inning else 3 - before
        row["outs_added"] = added
        row["outs_after"] = before + added
