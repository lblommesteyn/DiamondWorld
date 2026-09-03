import jax
import jax.numpy as jnp

from diamondworldjax.simulate.rollout import (
    StepDistribution,
    extract_game_runs,
    free_rollout_samples,
    initial_game_state,
)


def _categorical(batch_size, classes, selected):
    logits = jnp.full((batch_size, classes), -100.0)
    return logits.at[:, selected].set(100.0)


def walk_predictor(params, state, history, history_mask, exogenous):
    del params, history, history_mask, exogenous
    batch_size = state.shape[0]
    negative = jnp.full((batch_size,), -100.0)
    return StepDistribution(
        pitch_type_logits=_categorical(batch_size, 8, 0),
        swing_logits=negative,
        called_strike_logits=negative,
        contact_logits=negative,
        foul_logits=negative,
        runs_logits=_categorical(batch_size, 5, 0),
        base_state_logits=_categorical(batch_size, 8, 1),
        outs_added_logits=_categorical(batch_size, 4, 0),
        pa_outcome_logits=_categorical(batch_size, 9, 7),
    )


def test_free_rollout_feeds_generated_count_into_next_pitch():
    samples = free_rollout_samples(
        walk_predictor,
        None,
        initial_game_state(1),
        jax.random.PRNGKey(1),
        num_samples=2,
        max_steps=5,
    )
    assert samples["balls"][0, :, 0].tolist() == [1, 2, 3, 0, 1]
    assert samples["pa_terminal"][0, :, 0].tolist() == [False, False, False, True, False]
    assert samples["base_state"][0, 3, 0].item() == 1
    assert extract_game_runs(samples).shape == (2, 1)

