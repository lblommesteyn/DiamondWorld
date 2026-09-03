import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from diamondworldjax.simulate.rollout import (
    StepDistribution,
    extract_game_runs,
    free_rollout_samples,
    initial_game_state,
    autoregressive_joint_rollout_samples,
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


def test_joint_autoregressive_rollout_replaces_observed_pitch_history():
    """The second model call must see pitch 0's sample, not pitch 0's label."""
    seen_history = []

    def guide(batch, player_table, teacher_force=False):
        del batch, player_table, teacher_force
        numpyro.sample("player_skills", dist.Normal(jnp.zeros((1, 1)), 1).to_event(2))

    def model(batch, player_table, teacher_force=False):
        del player_table
        assert not teacher_force
        seen_history.append(np.asarray(batch["pitch_type"]).copy())
        B, T = batch["pitch_valid"].shape
        zeros = jnp.zeros((B, T))
        pitch_type = jnp.full((B, T), 3, dtype=jnp.int32)
        for name, value in {
            "pitch_type": pitch_type, "plate_x": zeros, "plate_z": zeros,
            "release_speed": zeros, "swing": zeros, "called_strike": zeros,
            "contact": zeros, "foul": zeros, "launch_speed": zeros,
            "launch_angle": zeros, "spray_angle": zeros, "hit_distance": zeros,
            "pa_outcome": jnp.zeros((B, T), dtype=jnp.int32),
            "runs_scored": jnp.zeros((B, T), dtype=jnp.int32),
            "base_state_after": jnp.zeros((B, T), dtype=jnp.int32),
            "outs_added": jnp.zeros((B, T), dtype=jnp.int32),
        }.items():
            numpyro.sample(name, dist.Delta(value).to_event(value.ndim))

    B, T = 1, 2
    f = lambda: jnp.zeros((B, T), dtype=jnp.float32)
    i = lambda: jnp.zeros((B, T), dtype=jnp.int32)
    batch = {
        "pitch_valid": jnp.ones((B, T), dtype=bool),
        "pitcher_ids": i(), "batter_ids": i(), "park_ids": i(),
        "game_ids": jnp.array([1]), "shift_restricted": f(), "pitch_clock": f(),
        "inning": f(), "half": f(), "balls": f(), "strikes": f(), "outs": f(),
        "base_state": f(), "score_diff": f(), "pitch_count_game": f(),
        "pitch_count_inning": f(), "pitch_count_pa": f(), "tto": f(),
        "pitch_type": jnp.array([[7, 6]], dtype=jnp.int32), "plate_x": f(),
        "plate_z": f(), "release_speed": f(), "pfx_x": f(), "pfx_z": f(),
        "obs_swing": i(), "obs_called_strike": i(), "obs_contact": i(), "obs_foul": i(),
        "launch_speed": f(), "launch_angle": f(), "spray_angle": f(), "hit_distance": f(),
        "pa_outcome": i(), "runs_scored": i(), "base_state_after": i(), "outs_added": i(),
    }
    samples = autoregressive_joint_rollout_samples(
        model, guide, {}, batch, {}, jax.random.PRNGKey(4)
    )
    assert len(seen_history) == 2
    assert seen_history[0][0, 0] == 0  # observed 7 was cleared before rollout
    assert seen_history[1][0, 0] == 3  # sampled pitch 0 was fed to pitch 1
    assert samples["pitch_type"].tolist() == [[[3, 3]]]
