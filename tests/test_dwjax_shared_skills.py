"""Unit guards for player conditioning in the shared PA/pitch hierarchy."""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.handlers as handlers

from diamondworldjax.model.multitask import task_checkpoint_params
from diamondworldjax.model.pitch_transformer import PitchTransformer
from diamondworldjax.train.svi import make_shared_task_skills_guide


def test_pitch_transformer_uses_direct_player_context_when_enabled():
    model = PitchTransformer(player_context_dim=8, d_model=32, n_heads=4, n_layers=1)
    B, T, H = 1, 2, 3
    args = (
        jnp.zeros((B, T, H), dtype=jnp.int32),
        jnp.zeros((B, T, H, 4)),
        jnp.zeros((B, T, H), dtype=jnp.int32),
        jnp.zeros((B, T, H, 8)),
        jnp.ones((B, T, H), dtype=bool),
        jnp.zeros((B, T, 16)),
        jnp.zeros((B, T, 4)),
    )
    player_a = jnp.zeros((B, T, 8))
    player_b = jnp.ones((B, T, 8))
    params = model.init(jax.random.PRNGKey(0), *args, player_a)
    out_a = model.apply(params, *args, player_a)
    out_b = model.apply(params, *args, player_b)
    assert not np.allclose(np.asarray(out_a), np.asarray(out_b))


def test_shared_task_guide_covers_every_latent_family():
    guide = make_shared_task_skills_guide(P=3)
    with handlers.seed(rng_seed=jax.random.PRNGKey(1)):
        with handlers.trace() as trace:
            guide({}, {}, teacher_force=True)
    assert {"shared_player_skills", "pa_skill_residual", "pitch_skill_residual"} <= set(trace)


def test_task_checkpoint_export_marginalises_shared_and_residual_skills():
    params = {
        "pa/head$params": "pa-head",
        "pitch/head$params": "pitch-head",
        "shared_player_skills_mu": jnp.array([[1.0]]),
        "shared_player_skills_sigma": jnp.array([[0.3]]),
        "pa_skill_residual_mu": jnp.array([[2.0]]),
        "pa_skill_residual_sigma": jnp.array([[0.4]]),
        "pitch_skill_residual_mu": jnp.array([[3.0]]),
        "pitch_skill_residual_sigma": jnp.array([[0.5]]),
    }
    exported = task_checkpoint_params(params, "pa")
    assert exported["head$params"] == "pa-head"
    np.testing.assert_allclose(exported["player_mu"], [[3.0]])
    np.testing.assert_allclose(exported["player_sigma"], [[0.5]])
