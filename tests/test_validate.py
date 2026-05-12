import polars as pl

from diamondworld.data.pipeline import cast_to_schema
from diamondworld.data.validate import validate_frame


def test_validate_frame_accepts_schema_and_next_base_state():
    frame = cast_to_schema(
        pl.DataFrame(
            [
                {
                    "game_pk": 1,
                    "at_bat_number": 1,
                    "pitch_number": 1,
                    "base_state": 0,
                    "base_state_after": 1,
                    "pa_terminal": True,
                },
                {
                    "game_pk": 1,
                    "at_bat_number": 2,
                    "pitch_number": 1,
                    "base_state": 1,
                    "base_state_after": 1,
                    "pa_terminal": False,
                },
            ]
        )
    )

    report = validate_frame(frame, season=2024)

    assert report["missing_columns"] == []
    assert report["base_state_after_next_pitch_mismatches"] == 0
