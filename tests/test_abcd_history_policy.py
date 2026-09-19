import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from test_model_review_fixes import batch, KW
from diamondworldjax.model.pitchformer import TransformerA
from diamondworldjax.simulate.pitchformer_rollout import _cache_template, _cached_apply, GameHistory, PitchformerHeads


def test_window_continuation_inactive_padding_and_batch_reordering():
    b = batch(3)
    model = TransformerA(**KW, window_size=2, pitch_history=True, observation_masks=True)
    params = model.init(jax.random.PRNGKey(0), b, train=False)
    cache = _cache_template(model,params,b)
    first = {k:jnp.asarray(v[:,:1]) for k,v in b.items()}
    _, cache = _cached_apply(model,params,cache,first)
    invalid = {**first,'valid':jnp.zeros((1,1))}
    _, after = _cached_apply(model,params,cache,invalid)
    for a,z in zip(jax.tree.leaves(cache),jax.tree.leaves(after)):
        np.testing.assert_array_equal(a,z)
    heads = PitchformerHeads(model,model,None,None,params,params,None,None)
    histories = GameHistory('game')
    histories.put([22],1,0,(cache,cache,None,None))
    doubled = {k:np.concatenate([v,v]) for k,v in b.items()}
    restored = histories.get(heads,doubled,[11,22],2,1)
    for old,new in zip(jax.tree.leaves(cache),jax.tree.leaves(restored[0])):
        np.testing.assert_array_equal(old,new[1:])
        assert not np.any(new[:1])
    for policy in ['half_inning','batting_side']:
        h = GameHistory(policy); h.put([22],1,0,(cache,cache,None,None))
        fresh = h.get(heads,b,[22],2,1)
        assert not any(np.any(x) for x in jax.tree.leaves(fresh))


def test_training_reset_policy_and_target_coverage():
    from diamondworldjax.data.pitch_seq import make_sequences, STUFF_COLS
    d = dict(game_pk=[1]*4,pitch_number=[1]*4,at_bat_number=[1,2,3,4],
        half=['top','bot','top','bot'],inning=[1,1,2,2],pitcher_id=[10]*4,batter_id=[20]*4,
        park_id=['X']*4,pitch_type=['FF']*4,stand=['R']*4,p_throws=['R']*4,
        pa_outcome=['out']*4,season=[2020]*4)
    for name in ['balls','strikes','outs','base_state','score_diff','tto','launch_speed','launch_angle',*STUFF_COLS]:
        d[name]=[0.]*4
    for name in ['pa_terminal','in_play','swing','contact','foul']:
        d[name]=[False]*4
    maps=dict(pitcher={10:1},batter={20:1},park={'X':1})
    for policy, count in [('game',1),('batting_side',2),('half_inning',4)]:
        a=make_sequences(pl.DataFrame(d),maps,max_len=8,context_len=2,history_reset=policy,geometry_table={})
        assert len(a['valid']) == count
        assert (a['valid']*a['loss_mask']).sum()==4
    a=make_sequences(pl.DataFrame(d),maps,max_len=3,context_len=1,history_reset='game',geometry_table={},include_game_pk=True)
    assert (a['valid']*a['loss_mask']).sum()==4
    np.testing.assert_array_equal(a['ctx'][0,:2,9],[0,1])
    np.testing.assert_array_equal(a['game_pk'][0, a['valid'][0].astype(bool)], [1, 1])


def test_context_accepts_canonical_hand_columns_without_raw_statcast_names():
    from diamondworldjax.data.pitch_seq import _ctx
    frame = pl.DataFrame({
        "balls": [0], "strikes": [0], "outs": [0], "base_state": [0],
        "score_diff": [0], "inning": [1], "half": ["top"], "tto": [1],
        "batter_hand": ["R"], "pitcher_hand": ["L"],
    })
    context = _ctx(frame)
    np.testing.assert_array_equal(context[0, 13:16], [1.0, 0.0, 0.0])


def test_bayesian_dropout_train_only():
    from diamondworldjax.model.bayesian_pitchformer import BayesianNetwork
    b = batch(3)
    options={**KW,'player_mode':'pa','skill_seasons':1}
    options.pop('dropout')
    model=BayesianNetwork(options,'ab',2,dropout=.5)
    features=(jnp.ones((3,2)),jnp.zeros(3,jnp.int32),jnp.zeros(3,jnp.int32))
    skills=jnp.zeros((3,3,1,32))
    roles={r:jnp.arange(3) for r in ['pitcher','batter']}
    p=model.init(jax.random.PRNGKey(0),b,features,skills,roles)
    def run(seed,train):
        return model.apply(p,b,features,skills,roles,train=train,rngs={'dropout':jax.random.PRNGKey(seed)})[0]['a']['type_logits']
    np.testing.assert_array_equal(run(1,False),run(2,False))
    assert not np.allclose(run(1,True),run(2,True))


def test_unlimited_pa_game_cohort():
    from diamondworldjax.scripts.simulate_games import select_game_ids
    frame=pl.DataFrame({'game_pk':[3,1,2,1]})
    np.testing.assert_array_equal(select_game_ids(frame,0),[1,2,3])
    np.testing.assert_array_equal(select_game_ids(frame,2),[1,2])
