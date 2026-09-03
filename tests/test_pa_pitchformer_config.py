"""Architecture-config tests for the optional PA transformer."""
import jax
import jax.numpy as jnp
import pytest

from diamondworldjax.model.pa_transformer import PATransformer, pa_transformer_numpyro


def test_pa_transformer_supports_nondefault_width():
    model = PATransformer(d_model=48, n_layers=1, n_heads=4)
    context = jnp.ones((2, 5, 17), dtype=jnp.float32)
    valid = jnp.ones((2, 5), dtype=bool)
    params = model.init(jax.random.PRNGKey(0), context, valid)
    assert model.apply(params, context, valid).shape == (2, 5, 48)


def test_pa_transformer_rejects_incompatible_head_width():
    with pytest.raises(ValueError, match="divisible"):
        pa_transformer_numpyro(
            jnp.ones((1, 2, 3)), jnp.ones((1, 2), dtype=bool),
            d_model=31, n_heads=4,
        )
