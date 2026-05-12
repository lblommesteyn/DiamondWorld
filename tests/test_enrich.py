from diamondworld.data.enrich import build_at_bat_runner_states


def test_build_at_bat_runner_states_tracks_runner_destinations():
    payload = {
        "allPlays": [
            {
                "about": {"atBatIndex": 12},
                "result": {"event": "Single"},
                "runners": [
                    {
                        "details": {"event": "Single"},
                        "movement": {"end": "1B", "isOut": False},
                    },
                    {
                        "details": {"event": "Single"},
                        "movement": {"end": "score", "isOut": False},
                    },
                ],
            }
        ]
    }

    states = build_at_bat_runner_states(payload)

    assert states[12].base_after == 1
    assert states[12].how_on[1] == "hit"
    assert states[12].runs_scored == 1
