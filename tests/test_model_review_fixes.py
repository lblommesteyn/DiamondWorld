"""Regression coverage for PA evaluation and comparable ABCD experiments."""
import argparse
import pickle

import flax
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.handlers as nh
import optax
import polars as pl
import pytest

from diamondworldjax.model.pitchformer import TransformerA, TransformerB, SuperState, loss_a, loss_b
from diamondworldjax.model.transformer_c import TransformerC
from diamondworldjax.model.transformer_d import TransformerD
from diamondworldjax.model.pitchformer_checkpoint import (
    export_shared_head, install_player_data, trainable_optimizer, transfer_pa_skills,
)
from diamondworldjax.scripts.train_pitchformer import SharedPitchformer, _init_shared


def batch(t=4):
    z = np.zeros((1, t), np.float32)
    return dict(pitcher_idx=np.ones((1, t), np.int32), batter_idx=np.ones((1, t), np.int32),
                park_idx=np.ones((1, t), np.int32), ctx=np.zeros((1, t, 24), np.float32),
                geom=np.zeros((1, t, 10), np.float32), valid=np.ones_like(z),
                pitch_type=np.zeros((1, t), np.int32), stuff=np.zeros((1, t, 5), np.float32),
                swing=z.copy(), contact=z.copy(), foul=z.copy(), hbp=z.copy(),
                called_strike=z.copy(),
                launch=np.zeros((1, t, 2), np.float32), type_valid=np.ones_like(z),
                stuff_valid=np.ones_like(z), launch_valid=z.copy(), batted_valid=z.copy(),
                batted_out=np.zeros((1, t), np.int32), events=np.zeros((1, t, 8), np.float32),
                skill_season=np.zeros((1, t), np.int32))


KW = dict(n_pitchers=3, n_batters=3, n_parks=3, d_model=12,
          n_layers=2, n_heads=3, dropout=0.0)


def test_learned_called_strike_head_is_scored_on_taken_pitches():
    b = batch()
    b["called_strike"][0, 0] = 1.0
    model = TransformerB(**KW, learned_called_strike=True)
    params = model.init(jax.random.PRNGKey(11), b, train=False)
    out = model.apply(params, b, train=False)
    loss, metrics = loss_b(out, b)

    assert "called_strike_logit" in out
    assert np.isfinite(loss)
    assert float(metrics["nll_called_strike"]) > 0.0


def test_posterior_binds_latent_sites_and_walk():
    from diamondworldjax.model.pa_checkpoint import posterior_params
    p = {"player_mu": jnp.ones((2, 3)), "player_sigma": jnp.ones((2, 3)) * .1}
    result = posterior_params(p)
    np.testing.assert_array_equal(result["player_skills"], p["player_mu"])
    assert "player_skills" not in posterior_params(p, mode="prior")
    w = posterior_params({**p, "skill_walk_sigma_loc": .2}, "walk")
    np.testing.assert_array_equal(w["player_skill_eps"], p["player_mu"])
    assert w["skill_walk_sigma"] == .2


def test_checkpoint_restores_split_and_features():
    from diamondworldjax.model.pa_checkpoint import restore_config, model_kwargs
    args = argparse.Namespace(test_seasons=None, train_end=2022, use_park=False)
    ckpt = {"pa_metadata": {"train_seasons": list(range(2015, 2024)), "config": {
        "train_end": 2023, "skill_prior": "walk", "nested": True, "contact_quality": True,
        "pitchformer": True, "pa_arch": "gru"}}}
    train, test = restore_config(ckpt, args)
    assert test == [2024] and args.use_park and args.contact_quality
    assert model_kwargs(args)["n_seasons"] == 9
    args.test_seasons = "2023"
    with pytest.raises(ValueError, match="held-out"):
        restore_config(ckpt, args)


def test_pa_padding_never_truncates_and_fatigue_is_pre_pa():
    from diamondworldjax.data.pa_batching import build_pa_batch
    df = pl.DataFrame({"game_pk": [1] * 101, "at_bat_number": list(range(101)),
                       "pitch_count_game": [10] * 101, "pitch_count_pa": [4] * 101})
    b = build_pa_batch(df)
    assert int(b["pa_valid"].sum()) == 101
    np.testing.assert_allclose(b["pitch_count_game"][0, :101], 6 / 120)
    with pytest.raises(ValueError, match="truncate"):
        build_pa_batch(df, max_pa=90)


def test_numpyro_pa_dropout_train_only():
    from diamondworldjax.model.pa_transformer import pa_transformer_numpyro
    x, valid = jnp.ones((1, 4, 6)), jnp.ones((1, 4), bool)
    def run(seed, train, params=None):
        with nh.seed(rng_seed=seed), nh.substitute(data=params or {}), nh.trace() as trace:
            out = pa_transformer_numpyro(x, valid, d_model=12, n_heads=3,
                                         n_layers=1, dropout=.5, train=train)
        return out, {k: v["value"] for k, v in trace.items() if v["type"] == "param"}
    _, params = run(0, False)
    np.testing.assert_array_equal(run(1, False, params)[0], run(2, False, params)[0])
    assert not np.allclose(run(1, True, params)[0], run(2, True, params)[0])


@pytest.mark.parametrize("mode", ["id", "pa", "none"])
def test_shared_exports_match_all_four_heads_and_neutral_unknowns(mode):
    b = batch()
    b["pitcher_idx"][0, 1] = 0
    b["batter_idx"][0, 1] = 0
    model = SharedPitchformer(**KW, heads="abcd", d_residual=3, player_mode=mode,
                              pitch_history=True)
    params = _init_shared(model, jax.random.PRNGKey(0), b, "abcd")
    if mode == "pa":
        params = install_player_data(params, {role: np.ones((3, 1, 64), np.float32) for role in ("pitcher", "batter")})
    for h, cls in zip("abcd", (TransformerA, TransformerB, TransformerC, TransformerD)):
        expected = model.apply(params, b, head=h, train=False)
        standalone = cls(**KW, residual_dim=3, player_mode=mode, pitch_history=True)
        exported = export_shared_head(params, h)
        got = standalone.apply(exported, b, train=False)
        for key in expected:
            np.testing.assert_allclose(got[key], expected[key], atol=2e-6)


def test_unknown_identity_cannot_use_random_embedding_row():
    model = SuperState(3, 3, 3, d_model=12)
    b = batch(1)
    ids = jnp.zeros((1, 1), jnp.int32)
    inputs = (ids, ids, b["park_idx"], b["ctx"], b["geom"])
    p = model.init(jax.random.PRNGKey(0), *inputs)
    expected = model.apply(p, *inputs)
    changed = flax.core.unfreeze(p)
    for role in ("pitcher_emb", "batter_emb"):
        changed["params"][role]["embedding"] = changed["params"][role]["embedding"].at[0].set(1e6)
    np.testing.assert_array_equal(model.apply(changed, *inputs), expected)


def test_history_changes_future_not_current_and_decode_matches_full():
    from diamondworldjax.simulate.pitchformer_rollout import _cache_template, _cached_apply
    model = TransformerA(**KW, pitch_history=True)
    b = batch()
    p = model.init(jax.random.PRNGKey(0), b, train=False)
    expected = model.apply(p, b, train=False)["type_logits"]
    changed = {**b, "stuff": b["stuff"].copy()}
    changed["stuff"][0, 1] = 20
    actual = model.apply(p, changed, train=False)["type_logits"]
    np.testing.assert_array_equal(actual[:, :2], expected[:, :2])
    assert not np.allclose(actual[:, 2:], expected[:, 2:])
    cache = _cache_template(model, p, changed)
    valid, decoded = jnp.zeros((1, 4), bool), []
    for t in range(4):
        token = {k: jnp.asarray(v[:, t:t+1]) for k, v in changed.items()}
        token.update(_decode_position=jnp.array(t), _cache_valid=valid)
        out, cache = _cached_apply(model, p, cache, token)
        decoded.append(out["type_logits"][:, 0])
        valid = valid.at[:, t].set(True)
    np.testing.assert_allclose(jnp.stack(decoded, 1), actual, atol=2e-5)


def test_d_hr_auxiliary_loss_is_opt_in_and_additive():
    from diamondworldjax.model.transformer_d import loss_d
    b = batch(2)
    b["launch_valid"][:] = 1
    b["batted_valid"][:] = 1
    b["batted_out"][0] = np.array([4, 0])  # HR then a non-HR ball in play
    out = {"launch_mu": jnp.zeros((1, 2, 2)), "launch_logsigma": jnp.zeros((1, 2, 2)),
           "outcome_logits": jnp.zeros((1, 2, 6))}
    base, parts = loss_d(out, b)
    weighted, _ = loss_d(out, b, hr_weight=.25)
    np.testing.assert_allclose(weighted - base, .25 * parts["nll_hr"], atol=1e-6)


def test_frozen_pa_features_survive_optimizer_step():
    model = TransformerA(**KW, player_mode="pa")
    b = batch()
    tables = {role: np.ones((3, 1, 64), np.float32) for role in ("pitcher", "batter")}
    p = install_player_data(model.init(jax.random.PRNGKey(0), b, train=False), tables)
    opt = trainable_optimizer(optax.adamw(.01, weight_decay=.1))
    state = opt.init(p)
    grads = jax.grad(lambda v: loss_a(model.apply(v, b, train=False), b)[0])(p)
    upd, _ = opt.update(grads, state, p)
    new = optax.apply_updates(p, upd)
    for a, z in zip(jax.tree.leaves(p["player_data"]), jax.tree.leaves(new["player_data"])):
        np.testing.assert_array_equal(a, z)


def test_resolution_hbp_tree_and_continuation():
    from diamondworldjax.eval.pitch_calibration import pitch_resolution_probs
    p = pitch_resolution_probs(np.array([.4]), np.array([.5]), np.array([.2]),
        np.array([.1]), np.ones((1, 6)) / 6, np.array([False]), np.array([3]), np.array([2]))
    assert p[0, 2] == pytest.approx(.06)
    assert p[0, 0] == pytest.approx(.2)
    assert p[0, 1] == pytest.approx(.54)
    assert p[0, [7, 3, 4, 5, 6, 8]].sum() == pytest.approx(.16)
    assert p[0, 9] == pytest.approx(.04)
    assert p.sum() == pytest.approx(1)


def test_transfer_matches_pa_fusion_and_latent_ablation(tmp_path):
    from diamondworldjax.model.embeddings import PlayerSeasonEncoder, SkillFusionLayer
    stats = jnp.array([[.1, .2], [.3, .4]])
    league = hand = jnp.zeros(2, jnp.int32)
    encoder = PlayerSeasonEncoder(f_player=2)
    enc = encoder.init(jax.random.PRNGKey(1), stats, league, hand)
    det = encoder.apply(enc, stats, league, hand)
    mu = jnp.arange(64, dtype=jnp.float32).reshape(2, 32) / 30
    fusion = SkillFusionLayer()
    fp = fusion.init(jax.random.PRNGKey(2), det, mu)
    path = tmp_path / "pa.pkl"
    with path.open("wb") as f:
        pickle.dump({"params": {"player_mu": mu, "player_encoder$params": enc["params"],
                                "player_encoder_skill_fusion$params": fp["params"]},
                     "pa_metadata": {"train_seasons": [2022], "config": {"skill_prior": "iso"},
                                     "player_table": {"stats": stats, "league": league, "hand": hand,
                                                      "id_to_idx": {100: 0, 200: 1}}}}, f)
    maps = {"pitcher": {100: 1, 999: 2}, "batter": {200: 1}, "n_pitcher": 3, "n_batter": 2}
    tables, base = transfer_pa_skills(path, maps, "pa", [2022])
    expected = fusion.apply(fp, det, mu)
    np.testing.assert_allclose(tables["pitcher"][1, 0], expected[0])
    np.testing.assert_allclose(tables["batter"][1, 0], expected[1])
    assert not tables["pitcher"][[0, 2]].any()
    ablated, _ = transfer_pa_skills(path, maps, "pa-no-latent", [2022])
    np.testing.assert_allclose(ablated["batter"][1, 0], fusion.apply(fp, det, jnp.zeros_like(mu))[1])
    assert not np.allclose(ablated["batter"][1], tables["batter"][1])
    with pytest.raises(ValueError, match="leak"):
        transfer_pa_skills(path, maps, "pa", [2021])


def test_shared_training_saves_usable_pa_skill_exports(tmp_path):
    from diamondworldjax.scripts.train_pitchformer import run_shared
    model = SharedPitchformer(**{**KW, "n_layers": 1}, heads="ab", d_residual=3,
                              player_mode="pa", pitch_history=True)
    b = batch(2)
    tables = {role: np.ones((3, 1, 64), np.float32) for role in ("pitcher", "batter")}
    from diamondworldjax.model.pitchformer import loss_b
    result = run_shared(model, b, b, steps=2, bs=1, lr=.001, seed=0, out_dir=tmp_path,
                        tag="tiny", heads="ab", loss_fns={"a": loss_a, "b": loss_b},
                        base={}, player_tables=tables)
    assert np.isfinite(result["a"]["loss"])
    with (tmp_path / "shared_tiny_params.pkl").open("rb") as f:
        shared = pickle.load(f)
    with (tmp_path / "A_tiny_params.pkl").open("rb") as f:
        exported = pickle.load(f)
    standalone = TransformerA(**{**KW, "n_layers": 1}, residual_dim=3,
                              player_mode="pa", pitch_history=True)
    expected = model.apply(shared, b, head="a", train=False)["type_logits"]
    np.testing.assert_allclose(standalone.apply(exported, b, train=False)["type_logits"], expected, atol=2e-6)


@pytest.mark.parametrize("architecture", ["transformer", "gru"])
def test_pa_sequence_step_matches_full_sequence(architecture):
    from diamondworldjax.model.pa_transformer import (
        PAGRU, PATransformer, gru_step_fn, gru_init_carry, transformer_step_fn, transformer_init_carry,
    )
    x = jax.random.normal(jax.random.PRNGKey(4), (2, 5, 7))
    mask = jnp.ones((2, 5), bool)
    if architecture == "gru":
        model = PAGRU(d_model=12, n_layers=2)
        params = model.init(jax.random.PRNGKey(2), x, mask)
        step, carry = gru_step_fn(model, params["params"]), gru_init_carry(2, 2, 12)
    else:
        model = PATransformer(d_model=12, n_layers=2, n_heads=3, position_encoding="sinusoidal")
        params = model.init(jax.random.PRNGKey(2), x, mask)
        step, carry = transformer_step_fn(model, params["params"], 8), transformer_init_carry(2, 8, 7, n_layers=model.n_layers, d_model=model.d_model, n_heads=model.n_heads)
    expected = model.apply(params, x, mask)
    out = []
    for t in range(5):
        carry, value = step(carry, x[:, t])
        out.append(value)
    np.testing.assert_allclose(jnp.stack(out, 1), expected, atol=2e-5)


def test_exported_pa_skill_stack_runs_cached_rollout():
    from diamondworldjax.simulate.pitchformer_rollout import PitchformerHeads, rollout_batch
    from diamondworldjax.sim.rules_engine import EmpiricalEngine
    b = batch(2)
    kw = {**KW, "n_layers": 1, "window_size": 1, "position_encoding": "sinusoidal", "observation_masks": True, "c_event_mode": "bundles"}
    shared = SharedPitchformer(**kw, d_residual=3, player_mode="pa", pitch_history=True)
    p = _init_shared(shared, jax.random.PRNGKey(5), b, "abcd")
    p = install_player_data(p, {role: np.ones((3, 1, 64), np.float32) for role in ("pitcher", "batter")})
    models = [cls(**kw, residual_dim=3, player_mode="pa", pitch_history=True)
              for cls in (TransformerA, TransformerB, TransformerC, TransformerD)]
    heads = PitchformerHeads(*models, *[export_shared_head(p, h) for h in "abcd"])
    result = rollout_batch(heads, b, seed=6, engine=EmpiricalEngine(), decode_len=2)
    assert result["active"].sum() == 2
    assert np.isfinite(result["stuff"]).all()


def test_heldout_calibration_scores_observed_labels():
    from diamondworldjax.eval.pitch_calibration import score_heldout
    from diamondworldjax.simulate.pitchformer_rollout import PitchformerHeads
    b = batch(2)
    b["pa_terminal"] = np.ones((1, 2), bool)
    b["pa_outcome"] = np.full((1, 2), 7, np.int32)
    kw = {**KW, "n_layers": 1}
    a, bm, d = TransformerA(**kw), TransformerB(**kw), TransformerD(**kw)
    heads = PitchformerHeads(a, bm, None, d,
        a.init(jax.random.PRNGKey(1), b, train=False),
        bm.init(jax.random.PRNGKey(2), b, train=False), None,
        d.init(jax.random.PRNGKey(3), b, train=False))
    result = score_heldout(heads, b, batch_size=1, launch_samples=2)
    b["pa_outcome"][:] = 0  # strikeout cannot happen at the zero-strike context
    changed = score_heldout(heads, b, batch_size=1, launch_samples=2)
    assert result["n"] == changed["n"] == 2
    assert changed["nll"] > result["nll"]
    assert result["observed_rates"][7] == changed["observed_rates"][0] == 1


def test_fast_pa_adapter_respects_explicit_position_encoding():
    from diamondworldjax.scripts.bench_pa_sequence import _make_params, _make_batch, _pa_features_at
    from diamondworldjax.model.pa_inference import build_pa_sequence_inference
    from diamondworldjax.model.pa_model import pa_model
    params, table = _make_params(jax.random.PRNGKey(2), n_players=3,
                                 d_model=12, n_layers=1, n_heads=3)
    b = _make_batch(jax.random.PRNGKey(3), 1, 3)
    # A resumed checkpoint can retain unused legacy position parameters. The
    # saved configuration, not their mere presence, must control fast inference.
    with nh.seed(rng_seed=0), nh.substitute(data=params), nh.trace() as tr:
        pa_model(b, table, teacher_force=False, outcome_only=True, fatigue=True,
                 pitchformer=True, pitchformer_dim=12, pitchformer_layers=1,
                 pitchformer_heads=3, pitchformer_position="sinusoidal")
    expected = jax.nn.softmax(tr["pa_outcome"]["fn"].logits)
    fast = build_pa_sequence_inference(params, table, pa_arch="transformer", d_model=12,
        n_layers=1, n_heads=3, outcome_only=True, fatigue=True, position_encoding="sinusoidal")
    carry, values = fast.init_carry(1), []
    for t in range(3):
        carry, logits = fast.step(carry, **_pa_features_at(b, t))
        values.append(jax.nn.softmax(logits))
    np.testing.assert_allclose(jnp.stack(values, 1), expected, atol=2e-5)


def test_gru_skip_fast_adapter_matches_training_model():
    """The residual raw context must be identical in training and rollout."""
    from diamondworldjax.scripts.bench_pa_sequence import _make_params, _make_batch, _pa_features_at
    from diamondworldjax.model.pa_inference import build_pa_sequence_inference
    from diamondworldjax.model.pa_model import PAOutcomeHeadV6, pa_model
    params, table = _make_params(jax.random.PRNGKey(8), n_players=3,
                                 d_model=12, n_layers=1, n_heads=3)
    # GRU-skip uses an independent backbone and a wider outcome head.
    params["pa_gru_skip$params"] = params["pa_gru$params"]
    params["pa_outcome_head_v6$params"] = PAOutcomeHeadV6().init(
        jax.random.PRNGKey(9), jnp.zeros((1, 12 + 145)),
    )["params"]
    b = _make_batch(jax.random.PRNGKey(10), 1, 3)
    with nh.seed(rng_seed=0), nh.substitute(data=params), nh.trace() as tr:
        pa_model(b, table, teacher_force=False, outcome_only=True, fatigue=True,
                 pitchformer=True, pa_arch="gru_skip", pitchformer_dim=12,
                 pitchformer_layers=1, pitchformer_heads=3)
    expected = jax.nn.softmax(tr["pa_outcome"]["fn"].logits)
    fast = build_pa_sequence_inference(
        params, table, pa_arch="gru_skip", d_model=12, n_layers=1, n_heads=3,
        outcome_only=True, fatigue=True,
    )
    carry, values = fast.init_carry(1), []
    for t in range(3):
        carry, logits = fast.step(carry, **_pa_features_at(b, t))
        values.append(jax.nn.softmax(logits))
    np.testing.assert_allclose(jnp.stack(values, 1), expected, atol=2e-5)


def test_outcome_only_eval_cli_restores_checkpoint_and_uses_engine(tmp_path, monkeypatch):
    import sys
    import json
    from diamondworldjax.scripts import eval_pa
    from diamondworldjax.data import pipeline
    from diamondworldjax.sim import game_extract, game_evaluation, rules_engine
    from diamondworldjax.eval import calibration
    table = {"all_ids": np.array([100]), "stats": np.ones((1, 8)),
             "hand": np.zeros(1, np.int32), "league": np.zeros(1, np.int32),
             "bat_hand": np.ones(1), "pit_hand": np.ones(1),
             "id_to_idx": {100: 0}, "unknown_index": 1}
    cfg = dict(outcome_only=True, train_end=2023, skill_prior="walk", fatigue=True,
               platoon=False, pitchformer=False, contact_quality=True, per_stat_shrink=True,
               recency_halflife=2.0)
    checkpoint = tmp_path / "pa.pkl"
    with checkpoint.open("wb") as f:
        pickle.dump({"params": {"player_mu": np.ones((1, 9, 32)), "skill_walk_sigma_loc": .2},
                     "pa_metadata": {"train_seasons": list(range(2015, 2024)), "config": cfg,
                                     "player_table": table, "park_map": {"A": 1}}}, f)
    frame = pl.DataFrame({"game_pk": [1], "pa_terminal": [True], "park_id": ["A"], "runs_scored": [2]})
    seasons = []
    def load(years, **kwargs):
        seasons.append(years)
        return frame
    monkeypatch.setattr(pipeline, "load_seasons", load)
    monkeypatch.setattr(rules_engine.EmpiricalEngine, "fit", lambda self, df: self)
    monkeypatch.setattr(game_extract, "fit_hook_dists", lambda df: ([], []))
    monkeypatch.setattr(game_extract, "extract_games", lambda *a, **kw: [{"game_pk": 1}])
    def simulate(model, params, pt, games, rng, samples, **kwargs):
        assert model.keywords["skill_prior"] == "walk"
        assert "player_skill_eps" in params and pt["stats"].shape == (1, 8)
        return np.ones((samples, 1)), np.ones((samples, 1))
    monkeypatch.setattr(game_evaluation, "simulate_score_draws", simulate)
    names = ("kl_run_distribution", "wasserstein_runs", "mean_rg_error", "variance_error", "p0_error", "p5_plus_error", "p8_plus_error")
    monkeypatch.setattr(calibration, "game_run_metrics", lambda *a: dict.fromkeys(names, 0.0))
    output = tmp_path / "eval.json"
    monkeypatch.setattr(eval_pa, "_RESULTS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["eval_pa", "--ckpt", str(checkpoint), "--out", str(output), "--samples", "1"])
    eval_pa.main()
    report = json.loads(output.read_text())
    assert report["mode"] == "engine-rollout" and report["skill_mode"] == "mean"
    assert seasons == [list(range(2015, 2024)), [2024]]
