from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from diamondworldjax.sim.rules_engine import PA_OUTCOME_IDX
from diamondworldjax.train.svi import _rollout_pa_game_states


def test_game_scheduled_sampling_rolls_a_complete_coherent_state_history():
    batch = {
        "pa_valid": jnp.array([[True, True, True]]),
        "inning": jnp.zeros((1, 3)),
        "half": jnp.zeros((1, 3)),
        "outs": jnp.zeros((1, 3)),
        "base_state": jnp.zeros((1, 3)),
        "score_diff": jnp.zeros((1, 3)),
    }
    outcomes = np.array([[PA_OUTCOME_IDX["BB"], PA_OUTCOME_IDX["out"], PA_OUTCOME_IDX["out"]]])

    state = _rollout_pa_game_states(outcomes, batch)

    # The walk reaches first; the following outs retain the runner and add
    # exactly one out each. These fields are generated as one state sequence.
    assert state["base_state"][0].tolist() == pytest.approx([0.0, 1.0 / 7.0, 1.0 / 7.0])
    assert state["outs"][0].tolist() == pytest.approx([0.0, 0.0, 0.5])
    assert state["inning"][0].tolist() == [0.0, 0.0, 0.0]
    assert state["half"][0].tolist() == [0.0, 0.0, 0.0]
