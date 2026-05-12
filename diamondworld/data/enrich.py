from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


HOW_ON_EVENT = {
    "Single": "hit",
    "Double": "hit",
    "Triple": "hit",
    "Home Run": "hit",
    "Walk": "walk",
    "Intent Walk": "walk",
    "Hit By Pitch": "hbp",
    "Field Error": "error",
    "Fielders Choice": "fc",
    "Fielders Choice Out": "fc",
    "Forceout": "fc",
}


@dataclass
class AtBatRunnerState:
    how_on: dict[int, str | None] = field(default_factory=lambda: {1: None, 2: None, 3: None})
    base_after: int | None = None
    runs_scored: int = 0


def base_name_to_int(base: str | None) -> int | None:
    if base == "1B":
        return 1
    if base == "2B":
        return 2
    if base == "3B":
        return 3
    return None


def bitmask_from_occupied(how_on: dict[int, str | None]) -> int:
    mask = 0
    for base, source in how_on.items():
        if source is not None:
            mask |= 1 << (base - 1)
    return mask


def source_from_runner(runner: dict[str, Any], event: str | None) -> str | None:
    details = runner.get("details", {})
    runner_event = details.get("event") or event
    return HOW_ON_EVENT.get(runner_event)


def build_at_bat_runner_states(play_by_play: dict[str, Any]) -> dict[int, AtBatRunnerState]:
    """Extract post-PA baserunner chains keyed by Statcast at_bat_number.

    MLB's play index aligns with Statcast's at_bat_number for modern game feeds.
    The cache stores full responses so this extraction can improve later without
    refetching a season.
    """
    out: dict[int, AtBatRunnerState] = {}
    for play in play_by_play.get("allPlays", []):
        about = play.get("about", {})
        at_bat_number = about.get("atBatIndex")
        if at_bat_number is None:
            continue

        result = play.get("result", {})
        event = result.get("event")
        state = AtBatRunnerState()

        for runner in play.get("runners", []):
            movement = runner.get("movement", {})
            end_base = movement.get("end")
            is_out = bool(movement.get("isOut"))
            if end_base == "score":
                state.runs_scored += 1
                continue
            base = base_name_to_int(end_base)
            if base is None or is_out:
                continue
            state.how_on[base] = source_from_runner(runner, event) or "hit"

        state.base_after = bitmask_from_occupied(state.how_on)
        out[int(at_bat_number)] = state
    return out
