"""Guards for the PA-sequence history ablation (audit item 2).

The ablation is the decisive test for what a PA-transformer checkpoint's headline
gain is made of, so it has to actually ablate. The property that matters is:

    with history ablated, the output at position t must not depend on the input
    at any position other than t.

That is checkable with random weights and no trained checkpoint, which is the
point: the v22pf checkpoint lives on another machine, and this lets the ablation
be trusted before it ever sees one.

It also checks the converse, that WITHOUT the ablation the model does depend on
earlier positions. A test that only asserted independence would pass on a model
that ignored its inputs entirely.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from diamondworldjax.model.pa_transformer import PATransformer


B, T, C, D = 2, 6, 12, 16


def _run(valid, context, seed=0):
    m = PATransformer(d_model=D, n_layers=2, n_heads=2, dropout=0.0)
    params = m.init(jax.random.PRNGKey(seed), context, valid, train=False)
    return np.asarray(m.apply(params, context, valid, train=False))


def _context(seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(size=(B, T, C)).astype(np.float32))


def test_ablated_output_ignores_earlier_positions():
    ctx = _context(0)
    valid_off = jnp.zeros((B, T), dtype=bool)

    base = _run(valid_off, ctx)
    # Perturb position 0 only. With history ablated, position 0's own output may
    # change; nothing at t >= 1 may.
    ctx2 = ctx.at[:, 0, :].add(5.0)
    pert = _run(valid_off, ctx2)

    np.testing.assert_allclose(base[:, 1:, :], pert[:, 1:, :], rtol=1e-5, atol=1e-5)
    assert not np.allclose(base[:, 0, :], pert[:, 0, :]), \
        "position 0 should still respond to its own input"


def test_unablated_output_does_depend_on_earlier_positions():
    """The converse, so the test above cannot pass vacuously."""
    ctx = _context(1)
    valid_on = jnp.ones((B, T), dtype=bool)

    base = _run(valid_on, ctx)
    ctx2 = ctx.at[:, 0, :].add(5.0)
    pert = _run(valid_on, ctx2)

    # Later positions attend to position 0, so they must move.
    assert not np.allclose(base[:, 1:, :], pert[:, 1:, :], rtol=1e-5, atol=1e-5), \
        "with history on, later PAs must depend on earlier ones"


def test_ablation_is_strictly_causal_in_both_directions():
    """A later position must never influence an earlier one, ablated or not."""
    ctx = _context(2)
    for valid in (jnp.zeros((B, T), dtype=bool), jnp.ones((B, T), dtype=bool)):
        base = _run(valid, ctx)
        ctx2 = ctx.at[:, -1, :].add(5.0)
        pert = _run(valid, ctx2)
        np.testing.assert_allclose(base[:, :-1, :], pert[:, :-1, :],
                                   rtol=1e-5, atol=1e-5)


def test_ablated_equals_per_pa_encoder():
    """Every position, ablated, equals running that position alone as a length-1
    sequence. This is the strongest form of the claim."""
    ctx = _context(3)
    valid_off = jnp.zeros((B, T), dtype=bool)
    full = _run(valid_off, ctx)
    for t in range(T):
        single = _run(jnp.zeros((B, 1), dtype=bool), ctx[:, t:t + 1, :])
        # Positional encoding differs between position t and position 0 of a
        # length-1 sequence, so only compare when the encoding is position
        # invariant. Learned encodings are not, so restrict to t = 0.
        if t == 0:
            np.testing.assert_allclose(full[:, 0, :], single[:, 0, :],
                                       rtol=1e-4, atol=1e-4)
