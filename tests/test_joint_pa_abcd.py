"""Smoke guards for the new PA + current-ABCD joint likelihood."""
from functools import partial
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import SVI, Trace_ELBO

from diamondworldjax.model.joint_pa_abcd import joint_pa_abcd_model
from diamondworldjax.model.multitask import task_checkpoint_params
from diamondworldjax.scripts.train_pa_abcd import _export_artifacts, _joint_task_metrics
from diamondworldjax.model.pitchformer import TransformerA, TransformerB
from diamondworldjax.model.transformer_c import TransformerC
from diamondworldjax.model.transformer_d import TransformerD
from diamondworldjax.train.svi import make_optimizer, make_shared_task_skills_guide


def _inputs():
    pa = {
        "pa_valid": jnp.ones((1, 2), bool),
        "inning": jnp.zeros((1, 2)), "half": jnp.zeros((1, 2)),
        "outs": jnp.zeros((1, 2)), "base_state": jnp.zeros((1, 2)),
        "score_diff": jnp.zeros((1, 2)), "tto": jnp.zeros((1, 2)),
        "shift_restricted": jnp.zeros((1, 2)), "pitch_clock": jnp.zeros((1, 2)),
        "pa_outcome": jnp.zeros((1, 2), jnp.int32),
        "runs_scored": jnp.zeros((1, 2), jnp.int32),
        "base_state_after": jnp.zeros((1, 2), jnp.int32),
        "pitcher_ids": jnp.zeros((1, 2), jnp.int32),
        "batter_ids": jnp.ones((1, 2), jnp.int32),
        "park_ids": jnp.zeros((1, 2), jnp.int32),
        "season": jnp.full((1, 2), 2015, jnp.int32),
    }
    abcd = {
        "pitcher_idx": jnp.ones((1, 2), jnp.int32),
        "batter_idx": jnp.ones((1, 2), jnp.int32),
        "park_idx": jnp.ones((1, 2), jnp.int32),
        "ctx": jnp.zeros((1, 2, 24)), "geom": jnp.zeros((1, 2, 10)),
        "skill_season": jnp.zeros((1, 2), jnp.int32),
        "valid": jnp.ones((1, 2)), "loss_mask": jnp.ones((1, 2)),
        "pitch_type": jnp.zeros((1, 2), jnp.int32),
        "type_valid": jnp.ones((1, 2)), "stuff": jnp.zeros((1, 2, 5)),
        "stuff_valid": jnp.ones((1, 2)), "stuff_observed": jnp.ones((1, 2, 5), bool),
        "swing": jnp.zeros((1, 2)), "contact": jnp.zeros((1, 2)),
        "foul": jnp.zeros((1, 2)), "hbp": jnp.zeros((1, 2)),
        "events": jnp.zeros((1, 2, 8)), "c_eligible": jnp.ones((1, 2), bool),
        "launch": jnp.zeros((1, 2, 2)), "launch_valid": jnp.zeros((1, 2)),
        "launch_observed": jnp.zeros((1, 2, 2), bool),
        "batted_out": jnp.zeros((1, 2), jnp.int32),
        "batted_valid": jnp.zeros((1, 2)), "pa_terminal": jnp.zeros((1, 2), bool),
    }
    table = {
        "stats": jnp.zeros((2, 7)), "league": jnp.zeros(2, jnp.int32),
        "hand": jnp.zeros(2, jnp.int32),
    }
    return {"pa": pa, "abcd": abcd}, table


def test_joint_svi_step_updates_one_shared_hierarchy(tmp_path):
    batch, table = _inputs()
    options = dict(
        n_pitchers=2, n_batters=2, n_parks=2, d_model=8, n_layers=1,
        n_heads=2, d_residual=2, dropout=0.0, heads="abcd", skill_seasons=1,
        pitch_history=False, position_encoding="sinusoidal", window_size=1,
        observation_masks=True, c_event_mode="bundles", c_support=None,
    )
    model = partial(
        joint_pa_abcd_model, abcd_options=options,
        role_global_indices={"pitcher": jnp.array([0, 0]), "batter": jnp.array([0, 1])},
        pa_model_kwargs={"outcome_only": True, "fatigue": False,
                         "season_base": 2015, "n_seasons": 1},
        residual_scale=.35, skill_prior="iso", n_seasons=1, missing_samples=1,
    )
    guide = make_shared_task_skills_guide(2, skill_prior="iso")
    svi = SVI(model, guide, make_optimizer(1e-3), Trace_ELBO())
    state = svi.init(jax.random.PRNGKey(0), batch, table, teacher_force=True)
    state, loss = svi.update(state, batch, table, teacher_force=True)
    params = svi.get_params(state)
    assert np.isfinite(float(loss))
    assert {"shared_player_skills_mu", "pa_skill_residual_mu", "pitch_skill_residual_mu"} <= set(params)
    assert "pa/player_encoder$params" in params
    assert "abcd_network$params" in params
    assert "player_mu" in task_checkpoint_params(params, "pa")

    diagnostics = _joint_task_metrics(model, svi, state, batch, table)
    assert set(diagnostics) == {"pa_outcome_nll", "abcd_nll_per_pitch"}
    assert all(np.isfinite(value) for value in diagnostics.values())

    source = tmp_path / "joint.pkl"
    with source.open("wb") as f:
        pickle.dump({"step": 1, "params": jax.device_get(params)}, f)
    pa_metadata = {
        "version": 1, "train_seasons": [2015],
        "config": {"outcome_only": True, "fatigue": False, "platoon": False,
                   "bilinear_rank": 0, "nested": False, "skill_prior": "iso",
                   "pitchformer": False, "pitchformer_dim": 128,
                   "pitchformer_layers": 2, "pitchformer_heads": 4,
                   "pitchformer_dropout": 0.0, "pa_arch": "transformer",
                   "pitchformer_position": "sinusoidal", "runs_upweight": 0.0},
        "player_table": {k: np.asarray(v) for k, v in table.items()}, "park_map": {},
    }
    _export_artifacts(
        source, tmp_path, "joint", player_table={k: np.asarray(v) for k, v in table.items()},
        maps={"n_pitcher": 2, "n_batter": 2, "n_park": 2},
        role_global_indices={"pitcher": np.array([0, 0]), "batter": np.array([0, 1])},
        train_seasons=[2015], pa_metadata=pa_metadata, abcd_options=options,
        d_residual=2, skill_prior="iso", args_dict={},
    )
    assert (tmp_path / "pa_joint.pkl").exists()
    assert all((tmp_path / f"{head}_joint_params.pkl").exists() for head in "ABCD")
    head_kwargs = dict(
        n_pitchers=2, n_batters=2, n_parks=2, d_model=8, n_layers=1,
        n_heads=2, dropout=0.0, player_mode="pa", skill_seasons=1,
        residual_dim=2, pitch_history=False, position_encoding="sinusoidal",
        window_size=1, observation_masks=True, c_event_mode="bundles",
    )
    for name, cls, output_key in (("A", TransformerA, "type_logits"),
                                  ("B", TransformerB, "swing_logit"),
                                  ("C", TransformerC, "event_logits"),
                                  ("D", TransformerD, "outcome_logits")):
        with (tmp_path / f"{name}_joint_params.pkl").open("rb") as f:
            variables = pickle.load(f)
        assert output_key in cls(**head_kwargs).apply(variables, batch["abcd"], train=False)
