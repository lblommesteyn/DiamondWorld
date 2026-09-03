"""Causality guards for the combined pitch-and-response hurdle model."""
import jax
import jax.numpy as jnp
import numpy as np

from diamondworldjax.model.hurdle import D_MODEL, D_PRE_PITCH, HurdleNet, N_PITCH_TYPES


def test_pitch_generation_does_not_read_realised_pitch():
    """Changing a realised pitch may affect response heads, never its generator."""
    net = HurdleNet(hidden_dim=16)
    context = jnp.ones((1, 2, D_MODEL), dtype=jnp.float32)
    pre_pitch = jnp.ones((1, 2, D_PRE_PITCH), dtype=jnp.float32)
    pitch_a = jnp.zeros((1, 2, N_PITCH_TYPES + 3), dtype=jnp.float32)
    pitch_b = pitch_a.at[..., 0].set(1.0).at[..., -3:].set(99.0)
    params = net.init(jax.random.PRNGKey(0), context, pre_pitch, pitch_a)

    out_a = net.apply(params, context, pre_pitch, pitch_a)
    out_b = net.apply(params, context, pre_pitch, pitch_b)

    for key in ("pitch_type_logits", "plate_x_mu", "plate_z_mu", "speed_mu"):
        np.testing.assert_array_equal(np.asarray(out_a[key]), np.asarray(out_b[key]))
    assert not np.allclose(np.asarray(out_a["swing_logit"]), np.asarray(out_b["swing_logit"]))
