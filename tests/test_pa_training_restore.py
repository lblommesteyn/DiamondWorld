"""PA constrained checkpoint restoration and posterior replay regressions."""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam
import numpyro.handlers as h
import pytest
from diamondworldjax.train.svi import (
    _resume_params, _validate_resume_metadata, _apply_game_scheduled_sampling,
    make_player_skills_guide, make_shared_task_skills_guide,
)


def test_resume_preserves_constrained_values_and_can_update():
    def model(batch, table, teacher_force=True):
        weight = numpyro.param('new_weight', jnp.asarray(.7))
        z = numpyro.sample('z', dist.Normal(0, 1))
        numpyro.sample('y', dist.Normal(z + weight, 1), obs=jnp.asarray(.5))
    def guide(batch, table, teacher_force=True):
        mu = numpyro.param('mu', jnp.asarray(0.), constraint=constraints.interval(-5, 5))
        sigma = numpyro.param('sigma', jnp.asarray(.3), constraint=constraints.interval(.05, 2))
        positive = numpyro.param('positive', jnp.asarray(.4), constraint=constraints.positive)
        numpyro.sample('z', dist.Normal(mu + positive, sigma))
    svi = SVI(model, guide, Adam(.001), Trace_ELBO())
    key = jax.random.PRNGKey(0)
    state = svi.init(key, {}, {})
    saved = {'mu': jnp.asarray(-1.3), 'sigma': jnp.asarray(.3), 'positive': jnp.asarray(.17)}
    restored = _resume_params(svi, state, key, {}, {}, saved)
    params = svi.get_params(restored)
    for name, value in saved.items():
        np.testing.assert_allclose(params[name], value, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(params['new_weight'], .7)
    restored, loss = svi.update(restored, {}, {})
    assert np.isfinite(loss)
    assert all(np.isfinite(v).all() for v in jax.tree.leaves(svi.get_params(restored)))
    with pytest.raises(ValueError, match='shape/tree'):
        _resume_params(svi, state, key, {}, {}, {'mu': jnp.ones(2)})
    with pytest.raises(ValueError, match='incompatible parameter'):
        _resume_params(svi, state, key, {}, {}, {'wrong_model': jnp.asarray(1.)})
    with pytest.raises(ValueError, match='constraints'):
        _resume_params(svi, state, key, {}, {}, {'sigma': jnp.asarray(3.)})


def test_resume_metadata_rejects_changed_registry_or_config():
    meta = {'pa_metadata': {'config': {'pa_arch': 'gru'}, 'train_seasons': [2023],
                           'park_map': {7: 1}, 'player_table': {'all_ids': np.array([10, 20])}}}
    _validate_resume_metadata(meta, meta)
    for field, replacement in [('config', {'pa_arch': 'transformer'}),
                               ('player_table', {'all_ids': np.array([20, 10])}),
                               ('train_seasons', [2022]), ('park_map', {7: 2})]:
        altered = {'pa_metadata': {**meta['pa_metadata'], field: replacement}}
        with pytest.raises(ValueError, match=field):
            _validate_resume_metadata(meta, altered)


@pytest.mark.parametrize('prior,shared', [('iso', False), ('walk', False), ('learned', False),
                                         ('lkj', False), ('walk', True)])
def test_scheduled_sampling_replays_all_guide_sites(prior, shared):
    guide = (make_shared_task_skills_guide(1, skill_prior=prior, n_seasons=2) if shared else
             make_player_skills_guide(1, skill_prior=prior, n_seasons=2))
    pa = {'pa_valid': jnp.ones((1, 2), bool), **{name: jnp.zeros((1, 2)) for name in
          ['inning', 'half', 'outs', 'base_state', 'score_diff']}}
    batch = {'pa': pa, 'pitch': {}} if shared else pa
    initial = h.trace(h.seed(guide, 0)).get_trace(batch, {}, teacher_force=False)
    params = {name: site['value'] for name, site in initial.items() if site['type'] == 'param'}
    for name in params:
        if name.endswith('_mu'):
            params[name] = jnp.full_like(params[name], 2.5)
    key = jax.random.PRNGKey(19)
    guide_key = jax.random.split(key, 3)[0]
    expected_trace = h.trace(h.seed(h.substitute(guide, data=params), guide_key)).get_trace(batch, {}, teacher_force=False)
    expected = {name: site['value'] for name, site in expected_trace.items() if site['type'] == 'sample'}
    seen = {}
    def model(batch, table, teacher_force=False):
        for name, value in expected.items():
            # Deliberately different priors: using them instead of the guide
            # makes the equality checks fail for skills and hierarchy sites.
            seen[name] = numpyro.sample(name, dist.Normal(jnp.zeros_like(value), 1).to_event(value.ndim))
        numpyro.sample('pa/pa_outcome' if shared else 'pa_outcome',
                       dist.Categorical(probs=jnp.ones((1, 2, 9))/9))
    svi = SimpleNamespace(guide=guide, get_params=lambda state: params)
    _apply_game_scheduled_sampling(model, svi, None, batch, {}, 1., key)
    for name in expected:
        np.testing.assert_array_equal(seen[name], expected[name])
    first = dict(seen)
    _apply_game_scheduled_sampling(model, svi, None, batch, {}, 1., key)
    for name in first:
        np.testing.assert_array_equal(seen[name], first[name])
    _apply_game_scheduled_sampling(model, svi, None, batch, {}, 1., jax.random.PRNGKey(20))
    assert any(not np.array_equal(seen[name], first[name]) for name in seen)

def test_training_resume_uses_inverse_transforms(tmp_path):
    import itertools
    import pickle
    from diamondworldjax.train.svi import train
    def model(batch, table, teacher_force=True):
        numpyro.sample('player_skills', dist.Normal(jnp.zeros((1, 32)), 1).to_event(2))
    guide = make_player_skills_guide(1)
    svi = SVI(model, guide, Adam(.001), Trace_ELBO())
    table = {'stats': jnp.zeros((1, 1))}
    state = svi.init(jax.random.PRNGKey(0), {}, table)
    saved = svi.get_params(state)
    saved['player_mu'] = jnp.full((1, 32), -1.2)
    saved['player_sigma'] = jnp.full((1, 32), .3)
    path = tmp_path / 'input.pkl'
    with path.open('wb') as f:
        pickle.dump({'params': saved}, f)
    # A zero-update restart isolates the restoration from subsequent learning.
    train(model, itertools.repeat(({}, table)), n_steps=0, resume_path=path, ckpt_dir=tmp_path)
    with (tmp_path / 'dwjax_step_0000000.pkl').open('rb') as f:
        actual = pickle.load(f)['params']
    for name, value in saved.items():
        np.testing.assert_allclose(actual[name], value, atol=1e-6)
