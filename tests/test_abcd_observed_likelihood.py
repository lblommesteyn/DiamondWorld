import jax
import jax.numpy as jnp
import numpy as np
from test_model_review_fixes import batch, KW
from diamondworldjax.model.pitchformer import TransformerA
from diamondworldjax.model.marginal_pitch_likelihood import marginal_log_likelihood
from diamondworldjax.model.transformer_c import TransformerC, loss_c
from diamondworldjax.simulate.pitchformer_rollout import _sample_c_event, _event_flags


def test_strict_window_decode_and_old_history_invariance():
    b = batch(6)
    b['stuff'] = np.arange(30, dtype=np.float32).reshape(1,6,5) / 30
    model = TransformerA(**KW, pitch_history=True, observation_masks=True,
                         window_size=2, position_encoding='sinusoidal')
    p = model.init(jax.random.PRNGKey(0), b, train=False)
    expected = model.apply(p, b, train=False)['type_logits']
    changed = {**b, 'stuff': b['stuff'].copy(), 'ctx': b['ctx'].copy()}
    changed['stuff'][:, :3] += 100
    changed['ctx'][:, :3] += 10
    np.testing.assert_allclose(model.apply(p, changed, train=False)['type_logits'][:, -1], expected[:, -1], atol=1e-6)
    cache = model.init(jax.random.PRNGKey(0), b, train=False, decode=True)['cache']
    values = []
    for t in range(6):
        one = {k:jnp.asarray(v[:,t:t+1]) for k,v in b.items()}
        out, mutable = model.apply({**p, 'cache':cache}, one, train=False, decode=True, mutable=['cache'])
        cache = mutable['cache']
        values.append(out['type_logits'])
    np.testing.assert_allclose(jnp.concatenate(values, 1), expected, atol=2e-5)


def test_marginal_observed_and_missing_fill_invariance():
    b = {k:jnp.asarray(v) for k,v in batch(2).items()}
    b['loss_mask'] = jnp.array([[1.,0.]])
    def apply(data):
        shape = data['valid'].shape
        a = dict(type_logits=jnp.zeros((*shape,8)), stuff_mu=jnp.zeros((*shape,8,5)), stuff_logsigma=jnp.zeros((*shape,8,5)))
        return {'a':a}
    expected = -np.log(8) - 2.5*np.log(2*np.pi)
    np.testing.assert_allclose(marginal_log_likelihood(apply,b,jax.random.PRNGKey(0),2), expected, atol=1e-5)
    b['type_valid'] = jnp.zeros_like(b['valid'])
    b['stuff_observed'] = jnp.zeros_like(b['stuff'],dtype=bool)
    b['stuff'] = jnp.full_like(b['stuff'],jnp.nan)
    np.testing.assert_allclose(marginal_log_likelihood(apply,b,jax.random.PRNGKey(0),2), 0., atol=1e-5)


def test_bundle_loss_sampling_and_support():
    b = batch(2)
    b['events'][...,0] = 1
    b['events'][...,3] = 1
    b['c_eligible'] = np.ones((1,2),bool)
    support = np.zeros((256,24),bool); support[[0,9]] = True
    model = TransformerC(**KW,c_event_mode='bundles',c_support=tuple(support.ravel()))
    p = model.init(jax.random.PRNGKey(1),b,train=False)
    out = model.apply(p,b,train=False)
    assert np.isfinite(loss_c(out,b)[0])
    probs = jax.nn.softmax(out['event_logits'])
    assert float(probs[...,1].sum()) == 0
    logits = jnp.full((10000,256),-1e30).at[:,0].set(np.log(.7)).at[:,9].set(np.log(.3))
    events = _sample_c_event(logits,jax.random.PRNGKey(1))
    flags = np.asarray(_event_flags(events,'bundles'))
    assert abs(flags[:,0].mean()-.3) < .02
    np.testing.assert_array_equal(flags[:,0],flags[:,3])

def test_missing_launch_marginal_and_gradient():
    b = {k:jnp.asarray(v) for k,v in batch(1).items()}
    b['batted_valid'] = jnp.ones((1,1))
    b['launch_observed'] = jnp.zeros((1,1,2),bool)
    b['launch'] = jnp.full((1,1,2), jnp.nan)
    def score(theta):
        def apply(data):
            shape = data['valid'].shape
            return {'a':dict(type_logits=jnp.zeros((*shape,8)), stuff_mu=jnp.zeros((*shape,8,5)), stuff_logsigma=jnp.zeros((*shape,8,5))),
                    'd':dict(launch_mu=jnp.ones((*shape,2))*theta, launch_logsigma=jnp.zeros((*shape,2)),
                             outcome_logits=jnp.stack([data['launch'][...,0],jnp.zeros(shape)],-1))}
        return marginal_log_likelihood(apply,b,jax.random.PRNGKey(2),4)
    value, gradient = jax.value_and_grad(score)(jnp.array(.2))
    assert np.isfinite(value) and np.isfinite(gradient)
    assert abs(float(gradient)) > .01


def test_shared_marginal_training_smoke(tmp_path):
    from diamondworldjax.scripts.train_pitchformer import SharedPitchformer, run_shared
    from diamondworldjax.model.pitchformer import loss_a, loss_b
    b = batch(2)
    b['stuff_observed'] = np.ones_like(b['stuff'],bool)
    b['stuff_observed'][0,0,0] = False
    b['launch_observed'] = np.zeros_like(b['launch'],bool)
    model = SharedPitchformer(**{**KW,'n_layers':1},heads='ab',window_size=1,
        observation_masks=True,position_encoding='sinusoidal',pitch_history=True)
    result = run_shared(model,b,b,steps=2,bs=1,lr=.001,seed=0,out_dir=tmp_path,
        tag='marginal',heads='ab',loss_fns={'a':loss_a,'b':loss_b},base={},missing_samples=1)
    assert np.isfinite(result['joint_marginal']['log_likelihood'])


def test_completion_report_does_not_filter_headlines():
    from diamondworldjax.scripts.eval_pitchformer_games import completion_report
    rep = dict(total=np.array([5,12]), completion_faults=dict(
        regulation_truncated=np.array([False,True]), unresolved_tie=np.array([False,False])))
    result = completion_report([rep],{123:0,456:1})
    assert result['headline_metrics_include_flagged_games']
    assert result['per_rep'][0]['reasons']['regulation_truncated'] == [456]
    assert result['per_rep'][0]['mean_total_all'] == 8.5
    np.testing.assert_array_equal(rep['total'],[5,12])


def test_pa_statistical_features_match_and_unknown_stays_zero():
    import argparse
    import polars as pl
    from diamondworldjax.scripts.train_pa import _build_player_table
    from diamondworldjax.model.bayesian_pitchformer import pa_skill_features
    pitches = pl.DataFrame(dict(pitcher_id=[10,10,10],batter_id=[20,20,20],
        season=[2020,2021,2021],pa_terminal=[True,True,True],pa_outcome=['HR','K','BB']))
    args = argparse.Namespace(recency_halflife=2.,contact_quality=False,per_stat_shrink=True)
    expected = _build_player_table(pitches,recency_halflife=2.,per_stat_shrink=True)
    actual = pa_skill_features(pitches,{20:1,10:2,999:3},args)
    for j, name in enumerate(['stats','league','hand']):
        for pid, target in [(20,1),(10,2)]:
            np.testing.assert_array_equal(actual[j][target],expected[name][expected['id_to_idx'][pid]])
        assert not np.any(actual[j][0]) and not np.any(actual[j][3])


def test_c_bundle_joint_transition_and_third_out():
    import polars as pl
    from diamondworldjax.sim.c_transition_engine import CTransitionEngine
    p = pl.DataFrame(dict(game_pk=[1,1],at_bat_number=[1,2],pitch_number=[1,1],
        base_state=[1,0],outs=[2,0],pa_terminal=[False,True],home_score=[0,0],away_score=[0,0]))
    e = p.select(['game_pk','at_bat_number','pitch_number']).with_columns(
        pl.Series('steal',[1,0]),pl.Series('caught_stealing',[1,0]))
    engine = CTransitionEngine('bundles').fit(p,e)
    bundle = 8+16
    assert np.asarray(engine.support()).reshape(256,24)[bundle,5]
    out = engine.sample(np.array([bundle-1]),np.array([1]),np.array([2]),np.random.default_rng(0))
    assert out['outs'][0] == 3 and out['base'][0] == 0
