from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.handlers as handlers

from diamondworldjax.model.transition import transition_numpyro


def test_transition_scores_the_observed_pa_outcome_and_has_no_orphan_side_sites():
    """The transition's primary target is the canonical PA outcome label."""
    shape = (1, 2)
    trace = handlers.trace(handlers.seed(transition_numpyro, jax.random.PRNGKey(0))).get_trace(
        jnp.zeros((*shape, 128)),
        jnp.zeros((*shape, 4)),
        jnp.zeros(shape, dtype=jnp.int32),
        jnp.zeros(shape, dtype=jnp.int32),
        jnp.array([[True, False]]),
        obs_pa_outcome=jnp.array([[3, 0]], dtype=jnp.int32),
        obs_runs_scored=jnp.zeros(shape, dtype=jnp.int32),
        obs_base_state_after=jnp.zeros(shape, dtype=jnp.int32),
        obs_outs_added=jnp.zeros(shape, dtype=jnp.int32),
        pa_outcome_mask=jnp.array([[True, False]]),
        runs_mask=jnp.array([[True, False]]),
        base_state_after_mask=jnp.array([[True, False]]),
        outs_added_mask=jnp.array([[True, False]]),
    )

    assert trace["pa_outcome"]["value"].tolist() == [[3, 0]]
    assert {"error_flag", "wild_pitch", "passed_ball", "balk"}.isdisjoint(trace)
