import itertools
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

from test_model_review_fixes import batch
from diamondworldjax.train.runtime import prefetch, bucket_batch, safe_update
from diamondworldjax.model.marginal_pitch_likelihood import marginal_log_likelihood
from diamondworldjax.scripts.train_pitchformer import (
    SharedPitchformer, _init_shared, shared_marginal_prepare,
)


def test_prefetch_order_exception_and_shutdown():
    assert list(prefetch(range(7))) == list(range(7))
    def broken():
        yield 1
        raise ValueError('producer failure')
    iterator = prefetch(broken())
    assert next(iterator) == 1
    with pytest.raises(ValueError, match='producer failure'):
        next(iterator)
    iterator = prefetch(itertools.count())
    assert next(iterator) == 0
    iterator.close()


def test_bucketing_preserves_every_target_and_warmup():
    b = batch(192)
    b['valid'][:, 67:] = 0
    b['loss_mask'] = b['valid'].copy()
    b['loss_mask'][:, :32] = 0
    actual = bucket_batch(b)
    assert actual['valid'].shape == (1, 128)
    for name, value in actual.items():
        np.testing.assert_array_equal(value, b[name][:, :128])
    assert actual['loss_mask'].sum() == b['loss_mask'].sum()


def test_safe_update_rolls_back_nonfinite_state_even_with_finite_loss():
    update = lambda s, x: ({'weight': s['weight'] + x, 'momentum': x}, jnp.array(1.))
    state = {'weight': jnp.array(2.), 'momentum': jnp.array(3.)}
    result, loss = jax.jit(lambda s: safe_update(update, s, jnp.inf))(state)
    assert np.isnan(loss)
    for key in state:
        np.testing.assert_array_equal(result[key], state[key])


@pytest.mark.parametrize('missing', [True, False])
def test_cached_marginal_preserves_likelihood_and_gradients_with_dropout(missing):
    b = {k: jnp.asarray(v) for k, v in batch(2).items()}
    b.update(launch_valid=jnp.ones((1, 2)), batted_valid=jnp.ones((1, 2)),
             stuff_observed=jnp.ones((1, 2, 5), bool), launch_observed=jnp.ones((1, 2, 2), bool))
    if missing:
        b['stuff_observed'] = b['stuff_observed'].at[0, 1, 0].set(False)
        b['launch_observed'] = b['launch_observed'].at[0, 0, 1].set(False)
        b['type_valid'] = b['type_valid'].at[0, 1].set(0)
    model = SharedPitchformer(n_pitchers=3, n_batters=3, n_parks=3,
        d_model=6, n_layers=1, n_heads=2, dropout=.2, heads='abcd', d_residual=2,
        pitch_history=True, observation_masks=True, window_size=1, position_encoding='sinusoidal')
    params = _init_shared(model, jax.random.PRNGKey(1), b, 'abcd')
    key = jax.random.PRNGKey(2)
    def objective(p, cached):
        apply = lambda data: {h: model.apply(p, data, head=h, train=True,
            rngs={'dropout': jax.random.fold_in(key, ord(h))}) for h in 'abcd'}
        if not missing and not cached:
            from diamondworldjax.model.bayesian_pitchformer import likelihood
            return likelihood(apply(b), b)
        prepare = shared_marginal_prepare(model, p, 'abcd', train=True, key=key) if cached else None
        return marginal_log_likelihood(apply, b, key, 1, prepare=prepare)
    expected = jax.jit(jax.value_and_grad(lambda p: objective(p, False)))(params)
    actual = jax.jit(jax.value_and_grad(lambda p: objective(p, True)))(params)
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, e, rtol=3e-4, atol=5e-5)


def test_chunked_svi_matches_single_steps_and_checkpoint_boundaries(tmp_path, monkeypatch):
    import diamondworldjax.train.svi as training
    table = {'stats': jnp.zeros((1, 2))}
    def model(b, pt, teacher_force=True):
        z = numpyro.sample('player_skills', dist.Normal(jnp.zeros((1, 32)), 1).to_event(2))
        weight = numpyro.param('weight', jnp.array(.2))
        numpyro.sample('y', dist.Normal(weight + z.mean(), 1), obs=b['y'])
    monkeypatch.setattr(training, '_eval_metrics', lambda *args: {})
    monkeypatch.setattr(training, 'LOG_INTERVAL', 3)
    monkeypatch.setattr(training, 'CKPT_INTERVAL', 4)
    steps = []
    monkeypatch.setattr(training, '_save_checkpoint', lambda svi, state, path, step, metadata: steps.append(step))
    def batches():
        for i in itertools.count():
            yield {'y': jnp.array((i % 4) * .1)}, table
    s1, _, l1 = training.train(model, batches(), n_steps=9, update_chunk_size=1, prefetch_depth=0)
    s2, _, l2 = training.train(model, batches(), n_steps=9, update_chunk_size=4,
                               ckpt_dir=tmp_path, prefetch_depth=2)
    np.testing.assert_allclose(l1, l2, atol=1e-5)
    for a, e in zip(jax.tree.leaves(s2), jax.tree.leaves(s1)):
        np.testing.assert_allclose(a, e, atol=1e-6)
    assert steps == [4, 8, 9]


def test_fused_pa_sampler_matches_logits():
    from test_pa_inference import _params_and_table
    from diamondworldjax.model.pa_inference import build_pa_inference
    params, table = _params_and_table()
    adapter = build_pa_inference(params, table, outcome_only=True, fatigue=True)
    inputs = {name: jnp.zeros(4) for name in ('inning', 'half', 'outs', 'base_state',
        'score_diff', 'tto', 'shift_restricted', 'pitch_clock', 'pitch_count_game', 'bat_side', 'pit_hand')}
    inputs.update({name: jnp.zeros(4, jnp.int32) for name in ('pitcher_ids', 'batter_ids', 'park_ids')})
    key = jax.random.PRNGKey(3)
    expected = jax.random.categorical(key, adapter.logits(**inputs))
    np.testing.assert_array_equal(adapter.sample(key, **inputs), expected)


def test_chunked_training_keeps_scheduled_sampling_order(monkeypatch):
    import diamondworldjax.train.svi as training
    table = {'stats': jnp.zeros((1, 2))}
    def model(b, pt, teacher_force=True):
        z = numpyro.sample('player_skills', dist.Normal(jnp.zeros((1, 32)), 1).to_event(2))
        numpyro.sample('y', dist.Normal(z.mean(), 1), obs=b['y'])
    calls = []
    def rollout(model, svi, state, b, table, rate, key):
        calls.append((rate, np.asarray(key).copy()))
        return {'y': b['y'] + rate * jax.random.normal(key)}
    monkeypatch.setattr(training, '_apply_game_scheduled_sampling', rollout)
    monkeypatch.setattr(training, '_eval_metrics', lambda *args: {})
    def run(size):
        return training.train(model, itertools.repeat(({'y': jnp.array(.2)}, table)),
            n_steps=6, update_chunk_size=size, ss_max_rate=.5, ss_start_step=2,
            ss_warmup_steps=3)[2]
    expected = run(1)
    first_calls = calls[:]
    calls.clear()
    np.testing.assert_allclose(run(4), expected, atol=1e-5)
    assert len(calls) == 4
    for (r1, k1), (r2, k2) in zip(calls, first_calls):
        assert r1 == r2
        np.testing.assert_array_equal(k1, k2)


def test_bayesian_cached_marginal_preserves_network_and_skill_gradients():
    from diamondworldjax.model.bayesian_pitchformer import BayesianNetwork, bayesian_marginal_prepare
    b = {k: jnp.asarray(v) for k, v in batch(2).items()}
    b['type_valid'] = jnp.zeros((1, 2))
    b['stuff_observed'] = jnp.zeros((1, 2, 5), bool)
    b['batted_valid'] = jnp.ones((1, 2))
    features = (jnp.ones((3, 1)), jnp.zeros(3, jnp.int32), jnp.zeros(3, jnp.int32))
    skills = jax.random.normal(jax.random.PRNGKey(4), (5, 3, 1, 32))
    roles = {'pitcher': jnp.arange(3), 'batter': jnp.arange(3)}
    options = dict(n_pitchers=3, n_batters=3, n_parks=3, d_model=6, n_layers=1,
        n_heads=2, player_mode='pa', skill_seasons=1, pitch_history=True,
        position_encoding='sinusoidal', window_size=1, observation_masks=True)
    model = BayesianNetwork(options, 'abcd', 1, dropout=.2)
    params = model.init(jax.random.PRNGKey(1), b, features, skills, roles)['params']
    key = jax.random.PRNGKey(2)
    def objective(values, cached):
        p, s = values
        apply = lambda data: model.apply({'params': p}, data, features, s, roles,
            train=True, rngs={'dropout': jax.random.fold_in(key, 892)})[0]
        prepare = bayesian_marginal_prepare(model, p, features, s, roles,
                                           train=True, key=key) if cached else None
        return marginal_log_likelihood(apply, b, key, 1, prepare=prepare)
    expected = jax.jit(jax.value_and_grad(lambda x: objective(x, False)))((params, skills))
    actual = jax.jit(jax.value_and_grad(lambda x: objective(x, True)))((params, skills))
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, e, rtol=4e-4, atol=8e-5)


@pytest.mark.parametrize('architecture', ['gru', 'transformer'])
def test_fused_sequence_sampler_and_carry(architecture):
    from diamondworldjax.scripts.bench_pa_sequence import _make_params, _make_batch, _pa_features_at
    from diamondworldjax.model.pa_inference import build_pa_sequence_inference
    p, table = _make_params(jax.random.PRNGKey(0), d_model=16, n_layers=1)
    adapter = build_pa_sequence_inference(p, table, pa_arch=architecture, d_model=16,
        n_layers=1, outcome_only=True, fatigue=True)
    inputs = _pa_features_at(_make_batch(jax.random.PRNGKey(1), 4, 2), 0)
    carry = adapter.init_carry(4)
    expected_carry, logits = adapter.step(carry, **inputs)
    key = jax.random.PRNGKey(3)
    actual_carry, outcomes = adapter.sample_step(carry, key, **inputs)
    np.testing.assert_array_equal(outcomes, jax.random.categorical(key, logits))
    for a, e in zip(jax.tree.leaves(actual_carry), jax.tree.leaves(expected_carry)):
        np.testing.assert_allclose(a, e, atol=1e-6)
