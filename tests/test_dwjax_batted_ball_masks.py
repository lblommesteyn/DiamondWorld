import jax.numpy as jnp
import numpyro.handlers as handlers

from diamondworldjax.model.batted_ball import batted_ball_numpyro


def test_batted_ball_likelihood_masks_missing_and_non_in_play_values():
    kwargs = {
        "shared_context": jnp.zeros((1, 3, 128)),
        "pitch_execution": jnp.zeros((1, 3, 8)),
        "batter_z": jnp.zeros((1, 3, 64)),
        "pitcher_z": jnp.zeros((1, 3, 64)),
        "park_id": jnp.zeros((1, 3), dtype=jnp.int32),
        "in_play_mask": jnp.array([[True, True, False]]),
        "obs_launch_speed": jnp.array([[0.9, 0.0, 0.0]]),
        # Second in-play ball has no Statcast speed; third is not in play.
        "launch_speed_mask": jnp.array([[True, False, False]]),
    }
    trace = handlers.trace(handlers.seed(batted_ball_numpyro, 0)).get_trace(**kwargs)
    assert trace["launch_speed"]["fn"]._mask.tolist() == [[True, False, False]]
