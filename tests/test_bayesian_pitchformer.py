import argparse
import pickle
import json
import pytest
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from test_model_review_fixes import batch
from diamondworldjax.model.bayesian_pitchformer import (
    BayesianNetwork, draw_skills, skill_kl, likelihood, export_world, run_bayesian)
from diamondworldjax.model.pitchformer import TransformerA, TransformerB
from diamondworldjax.model.transformer_c import TransformerC
from diamondworldjax.model.transformer_d import TransformerD


def test_native_training_residuals_export_and_worlds(tmp_path, monkeypatch):
    b = batch(2)
    b['season'] = np.full((1, 2), 2020, np.int32)
    b['launch_valid'][:] = 1
    b['batted_valid'][:] = 1
    feature_path = tmp_path / 'features.npz'
    np.savez(feature_path, player_ids=[10, 20], stats=[[.1, .2], [.3, .4]],
             league=[0, 0], hand=[0, 1], through_year=2019)
    args = argparse.Namespace(skill_residual_scale=.35, skill_walk_scale=.3,
        skill_prior='walk', skill_features=str(feature_path), d_model=6, layers=1, heads=2,
        stack='abcd', pitch_history=True, seed=1, lr=.001, steps=2, bs=1,
        out=str(tmp_path), tag='native', shared_emb=False, missing_samples=1, window_size=1,
        position_encoding='sinusoidal', c_event_mode='bundles')
    maps = dict(pitcher={10: 1}, batter={20: 1, 10: 2}, n_pitcher=2, n_batter=3, n_park=2)
    c = run_bayesian(args, b, b.copy(), maps, [2020, 2021])
    assert c['posterior']['mu'].shape == (5, 2, 2, 32)
    assert np.isfinite(c['posterior']['rho']).all()
    for i in range(1, 5):
        assert np.any(np.asarray(c['posterior']['mu'][i]) != 0)
    models = dict(a=TransformerA, b=TransformerB, c=TransformerC, d=TransformerD)
    assert c['role_indices']['pitcher'][1] == c['role_indices']['batter'][2]
    native = BayesianNetwork(c['options'], c['heads'], 2)
    skill = draw_skills(c['posterior'], jax.random.PRNGKey(3), .35, 'walk', .3, False)
    expected, tables = native.apply({'params': c['network']}, c['example'], c['features'], skill, c['role_indices'])
    for head, variables in export_world(c).items():
        actual = models[head](**c['options'], dropout=0.).apply(variables, c['example'], train=False)
        for key in actual:
            np.testing.assert_allclose(actual[key], expected[head][key], atol=2e-6)
        for role in ('pitcher', 'batter'):
            assert np.all(np.asarray(tables[head][role][0]) == 0)
    w1, w2, w3 = export_world(c, 3, True), export_world(c, 3, True), export_world(c, 4, True)
    for h in 'abcd':
        p1 = w1[h]['player_data']['trunk']['super_state']['pitcher']
        np.testing.assert_array_equal(p1, w2[h]['player_data']['trunk']['super_state']['pitcher'])
        assert not np.allclose(p1, w3[h]['player_data']['trunk']['super_state']['pitcher'])
    # One head's residual cannot affect any other head's player table.
    altered = dict(c, posterior=dict(c['posterior'], mu=c['posterior']['mu'].at[1, 0, 0, 0].add(5)))
    changed = export_world(altered)
    base = export_world(c)
    for h in 'bcd':
        np.testing.assert_array_equal(changed[h]['player_data']['trunk']['super_state']['pitcher'],
                                      base[h]['player_data']['trunk']['super_state']['pitcher'])
    assert not np.allclose(changed['a']['player_data']['trunk']['super_state']['pitcher'],
                           base['a']['player_data']['trunk']['super_state']['pitcher'])
    with open(tmp_path / 'bayesian_native.pkl', 'rb') as f:
        restored = pickle.load(f)
    assert set(export_world(restored)) == set('abcd')
    with open(tmp_path / 'bayesian_native_report.json') as f:
        report = json.load(f)
    assert report['statistical_covariates'] == 'provided'
    assert report['heldout_target_counts']['outcome'] == 2
    from diamondworldjax.scripts.export_bayesian_pitchformer import main
    monkeypatch.setattr('sys.argv', ['export', '--params-dir', str(tmp_path), '--tag', 'native',
                                    '--out-tag', 'world', '--mode', 'sample', '--seed', '4'])
    main()
    with open(tmp_path / 'A_world_params.pkl', 'rb') as f:
        world = pickle.load(f)
    np.testing.assert_array_equal(world['player_data']['trunk']['super_state']['pitcher'],
                                  w3['a']['player_data']['trunk']['super_state']['pitcher'])
    with pytest.raises(SystemExit):
        main()  # Export never silently replaces another world's files.


def test_kl_and_missing_likelihood():
    scales = jnp.asarray([np.sqrt(1 - .35 ** 2)] + [.35] * 4)[:, None, None, None]
    p = dict(mu=jnp.zeros((5, 2, 3, 32)), rho=jnp.broadcast_to(jnp.log(jnp.expm1(scales)), (5, 2, 3, 32)))
    np.testing.assert_allclose(skill_kl(p, .35), 0, atol=1e-5)
    b = batch(2)
    out = dict(b={label + '_logit': jnp.zeros((1, 2)) for label in ['swing', 'contact', 'foul', 'hbp']})
    np.testing.assert_allclose(likelihood(out, b), -4 * np.log(2), rtol=1e-6)
    doubled_b = {k: np.concatenate([v, v]) for k, v in b.items()}
    doubled_out = {'b': {k: jnp.concatenate([v, v]) for k, v in out['b'].items()}}
    np.testing.assert_allclose(likelihood(doubled_out, doubled_b), 2 * likelihood(out, b))
    b['stuff_valid'][:] = 0
    assert float(likelihood(out, b)) == 0
    p['mu'] = p['mu'].at[0, 0, 0, 0].set(1.)
    assert float(skill_kl(p, .35)) > 0
    skills = draw_skills(p, jax.random.PRNGKey(0), .35, 'walk', .3, False)
    np.testing.assert_array_equal(skills[:, 0], 0)
    np.testing.assert_array_equal(skills[0, 1, :, 0], 1.)


def test_feature_leakage_is_rejected(tmp_path):
    path = tmp_path / 'future.npz'
    np.savez(path, through_year=2021)
    args = argparse.Namespace(skill_residual_scale=.35, skill_walk_scale=.3, skill_features=str(path))
    maps = dict(pitcher={10: 1}, batter={20: 1}, n_pitcher=2, n_batter=2)
    with pytest.raises(ValueError, match='held-out'):
        run_bayesian(args, {}, {}, maps, [2020])


def test_pa_role_features_give_pitchers_their_own_opponent_outcomes():
    from diamondworldjax.model.bayesian_pitchformer import pa_role_skill_features
    pitches = pl.DataFrame({
        'pitcher_id': [10, 10, 11], 'batter_id': [20, 21, 20],
        'season': [2021, 2021, 2021], 'pa_terminal': [True, True, True],
        'pa_outcome': ['HR', 'BB', 'K'],
    })
    args = argparse.Namespace(recency_halflife=None, contact_quality=False,
                              per_stat_shrink=False)
    stats, league, hand = pa_role_skill_features(
        pitches, {10: 1, 11: 2, 20: 3, 21: 4}, args)

    # A pitcher now receives the two outcomes he allowed; before this change,
    # player 10's rate/count vector was all zero because the table grouped only
    # on batter_id.
    assert stats[0, 1, 4] > 0
    assert stats[1, 3, 4] > 0
    assert stats[..., 4].max() <= 1.0
    assert not stats[:, 0].any() and not league[:, 0].any() and not hand[:, 0].any()


def test_bayesian_network_uses_distinct_role_feature_tables():
    b = batch(1)
    features = (
        jnp.zeros((2, 5, 3)).at[0, 1, 0].set(1.).at[1, 1, 0].set(-1.),
        jnp.zeros((2, 5), jnp.int32),
        jnp.zeros((2, 5), jnp.int32),
    )
    options = dict(n_pitchers=3, n_batters=3, n_parks=3, d_model=6,
                   n_layers=1, n_heads=2, player_mode='pa', skill_seasons=1)
    model = BayesianNetwork(options, 'b', 3, dropout=0.)
    skills = jnp.zeros((2, 5, 1, 32))
    roles = {'pitcher': jnp.array([0, 1, 2]), 'batter': jnp.array([0, 3, 1])}
    params = model.init(jax.random.PRNGKey(22), b, features, skills, roles)
    _, tables = model.apply(params, b, features, skills, roles)

    assert tables['b']['pitcher'].shape == tables['b']['batter'].shape == (5, 1, 64)
    assert not np.allclose(tables['b']['pitcher'][1], tables['b']['batter'][1])
