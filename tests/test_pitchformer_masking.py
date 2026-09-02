"""Guards for the causal mask in the pitch-level trunk.

The bug these exist for: the first pitch of a sequence has no history, so its
attention row is fully masked. Softmax over an all-masked row does not raise, it
returns a UNIFORM mix over every key including padding, and position 1 then attends
to that contaminated vector, so padding propagates through the whole stack. It
trains fine and produces plausible numbers, which is exactly the failure family this
project keeps finding.
"""
import jax
import jax.numpy as jnp
import numpy as np

from diamondworldjax.model.pitchformer import TransformerA, causal_mask
from diamondworldjax.data.pitch_seq import D_CTX
from diamondworldjax.model.superstate import N_GEOMETRY

T = 6
VALID = jnp.array([[1, 1, 1, 0, 0, 0]], dtype=bool)


def _batch(ctx_fill_pad=0.0):
    ctx = np.zeros((1, T, D_CTX), np.float32)
    ctx[:, 3:, :] = ctx_fill_pad
    return {
        "pitcher_idx": jnp.ones((1, T), jnp.int32),
        "batter_idx": jnp.ones((1, T), jnp.int32),
        "park_idx": jnp.ones((1, T), jnp.int32),
        "ctx": jnp.asarray(ctx),
        "geom": jnp.zeros((1, T, N_GEOMETRY)),
        "valid": VALID.astype(jnp.float32),
    }


def test_mask_is_strictly_causal_and_pad_aware():
    m = np.asarray(causal_mask(VALID))[0, 0]
    # Position t must never see itself or anything later: A predicts THIS pitch.
    assert not m.diagonal().any()
    assert not np.triu(m).any()
    # Padded keys are never attendable.
    assert not m[:, 3:].any()
    # The first position genuinely has nothing to attend to.
    assert not m[0].any()


def test_padding_does_not_leak_into_real_positions():
    model = TransformerA(n_pitchers=5, n_batters=5, n_parks=5,
                         d_model=32, n_layers=2, n_heads=4)
    params = model.init(jax.random.PRNGKey(0), _batch(), train=False)

    a = model.apply(params, _batch(0.0), train=False)["type_logits"]
    b = model.apply(params, _batch(99.0), train=False)["type_logits"]

    assert not bool(jnp.isnan(a).any())
    # Changing ONLY padded positions must not move any valid position. Before the
    # fix this was 1.77 at t=0 and 0.81 at t=1.
    delta = np.abs(np.asarray(a - b))[0, :3].max()
    assert delta == 0.0, f"padding leaked into valid positions, max delta {delta}"
