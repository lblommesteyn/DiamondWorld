import numpy as np
import pytest
import jax
import jax.numpy as jnp
import flax.linen as nn

from test_model_review_fixes import batch
from diamondworldjax.eval.calibration import kl_divergence_runs
from diamondworldjax.scripts.train_pitchformer import (
    baselines, c_baselines, d_baselines, c_improvements,
)
from diamondworldjax.model.pa_transformer import (
    PAGRU, PATransformer, transformer_init_carry, transformer_step_fn,
)


def test_kl_retains_runaway_tail_mass():
    assert kl_divergence_runs(np.array([1, 100]), np.array([1, 1])) == pytest.approx(np.log(2), abs=1e-5)
    assert np.isfinite(kl_divergence_runs(np.array([100, 100]), np.array([1, 1])))
    assert kl_divergence_runs(np.array([1, 22]), np.array([1, 22])) == 0


@pytest.mark.parametrize('baseline', [baselines, d_baselines,
    lambda tr, te: c_baselines(tr, te, 'bundles'),
    lambda tr, te: c_baselines(tr, te, 'legacy')])
def test_baselines_ignore_duplicate_warmup(baseline):
    b = batch(4)
    b['swing'][0] = [1, 1, 1, 0]
    b['contact'][0] = [1, 1, 0, 0]
    b['batted_valid'][:] = 1
    b['launch_valid'][:] = 1
    b['c_eligible'] = np.ones((1, 4), bool)
    b['pitch_type'][0] = [0, 1, 2, 3]
    b['batted_out'][0] = [0, 1, 2, 4]
    b['launch'][0] = [[0, 0], [1, 1], [2, 2], [3, 3]]
    expected = baseline(b, b)
    duplicate = {k: np.concatenate([v[:, :2], v], axis=1) for k, v in b.items()}
    duplicate['loss_mask'] = np.array([[0, 0, 1, 1, 1, 1]], np.float32)
    actual = baseline(duplicate, duplicate)
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], equal_nan=True)


def test_bundle_report_and_eligibility():
    b = batch(2)
    b['c_eligible'] = np.array([[True, False]])
    b['events'][:, 1] = 1
    baseline = c_baselines(b, b, 'bundles')
    assert baseline['bundle'] < 1e-4
    assert c_improvements(baseline, {'nll_bundle': 0.5}) == {'bundle': baseline['bundle'] - 0.5}


@pytest.mark.parametrize('position', ['learned', 'sinusoidal'])
def test_kv_decode_async_games_and_growth(position):
    x = jax.random.normal(jax.random.PRNGKey(9), (2, 6, 7))
    model = PATransformer(d_model=12, n_layers=2, n_heads=3, position_encoding=position)
    params = model.init(jax.random.PRNGKey(3), x, jnp.ones((2, 6), bool))
    full = model.apply(params, x, jnp.ones((2, 6), bool))
    carry = transformer_init_carry(2, 3, n_layers=2, d_model=12, n_heads=3)
    step = jax.jit(transformer_step_fn(model, params['params'], 3))
    for game in [0, 0, 1, 0, 1, 1]:
        t = int(carry[2][game])
        one = jax.tree.map(lambda v: v[game:game+1], carry)
        one, value = step(one, x[game:game+1, t])
        np.testing.assert_allclose(value[0], full[game, t], atol=2e-5)
        carry = jax.tree.map(lambda all_, v: all_.at[game:game+1].set(v), carry, one)
    layers, valid, positions = carry
    layers = jax.tree.map(lambda v: jnp.pad(v, ((0, 0), (0, 3), (0, 0), (0, 0))), layers)
    carry = layers, jnp.pad(valid, ((0, 0), (0, 3))), positions
    for t in range(3, 6):
        carry, value = step(carry, x[:, t])
        np.testing.assert_allclose(value, full[:, t], atol=2e-5)


def _unrolled_gru(params, x, valid):
    def norm(p, value):
        return nn.LayerNorm().apply({'params': p}, value)
    x = nn.Dense(8).apply({'params': params['input_proj']}, x)
    x = norm(params['input_ln'], x)
    for i in range(2):
        h = jnp.zeros((x.shape[0], 8))
        outputs = []
        for t in range(x.shape[1]):
            new, _ = nn.GRUCell(8).apply({'params': params['gru_stack'][f'gru_{i}']}, h, x[:, t])
            h = jnp.where(valid[:, t, None], new, h)
            outputs.append(h)
        x = jnp.stack(outputs, 1)
        if i == 0:
            x = norm(params['gru_stack']['ln_0'], x)
    return norm(params['out_norm'], x)


def test_scanned_gru_preserves_masked_outputs_and_gradients():
    x = jax.random.normal(jax.random.PRNGKey(7), (2, 5, 3))
    valid = jnp.array([[True, True, False, True, False], [False]*5])
    model = PAGRU(d_model=8, n_layers=2)
    params = model.init(jax.random.PRNGKey(8), x, valid)['params']
    new = lambda p: model.apply({'params': p}, x, valid)
    old = lambda p: _unrolled_gru(p, x, valid)
    np.testing.assert_allclose(new(params), old(params), atol=2e-5)
    weights = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 8))
    for a, b in zip(jax.tree.leaves(jax.grad(lambda p: (new(p)*weights).sum())(params)),
                    jax.tree.leaves(jax.grad(lambda p: (old(p)*weights).sum())(params))):
        np.testing.assert_allclose(a, b, atol=5e-5, rtol=2e-4)


def test_free_rollout_keeps_prefix_and_resets_half_inning():
    import numpyro
    from diamondworldjax.scripts.eval_pa import _free_rollout_sample
    seen = []
    def fake_model(b, pt, teacher_force):
        seen.append(np.asarray(b['base_state']).copy())
        numpyro.deterministic('runs_scored', jnp.ones_like(b['base_state']))
        numpyro.deterministic('base_state_after', jnp.full_like(b['base_state'], 7))
    b = dict(pa_valid=jnp.ones((1, 4), bool), base_state=jnp.array([[0., 0., 0., 2/7]]),
             inning=jnp.array([[1., 1., 1., 2.]]), half=jnp.array([[0., 0., 1., 0.]]))
    draws, _ = _free_rollout_sample(fake_model, {}, b, {}, jax.random.PRNGKey(0), 1)
    assert [v.shape[1] for v in seen] == [1, 2, 3, 4]
    np.testing.assert_allclose(seen[-1], [[0, 1, 0, 2/7]])
    np.testing.assert_array_equal(draws, [[4]])
