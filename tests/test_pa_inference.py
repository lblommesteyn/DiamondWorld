"""Equivalence guards for the fast ordinary-PA simulator inference path."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.handlers as nh

from diamondworldjax.model.embeddings import PlayerSeasonEncoder, SkillFusionLayer
from diamondworldjax.model.pa_inference import bucket_size, build_pa_inference
from diamondworldjax.model.pa_model import PAOutcomeHeadV6, ParkEmbedding, pa_model


def _params_and_table():
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 4)
    stats = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=jnp.float32)
    league = jnp.array([0, 1, 0], dtype=jnp.int32)
    hand = jnp.array([0, 1, 1], dtype=jnp.int32)
    skills = jnp.arange(3 * 32, dtype=jnp.float32).reshape(3, 32) / 100.0

    enc = PlayerSeasonEncoder(f_player=2)
    enc_params = enc.init(keys[0], stats, league, hand)["params"]
    det = enc.apply({"params": enc_params}, stats, league, hand)
    fusion = SkillFusionLayer()
    fusion_params = fusion.init(keys[1], det, skills)["params"]
    park = ParkEmbedding()
    park_params = park.init(keys[2], jnp.array([0, 1], dtype=jnp.int32))["params"]
    # fatigue adds one state column: 8 state + 2 player embeddings + 8 park = 145.
    head = PAOutcomeHeadV6()
    head_params = head.init(keys[3], jnp.zeros((2, 145), dtype=jnp.float32))["params"]
    params = {
        "player_skills": skills,
        "player_encoder$params": enc_params,
        "player_encoder_skill_fusion$params": fusion_params,
        "park_embedding$params": park_params,
        "pa_outcome_head_v6$params": head_params,
    }
    return params, {"stats": stats, "league": league, "hand": hand}


def test_fast_inference_matches_numpyro_model_logits():
    params, table = _params_and_table()
    batch = {
        "pa_valid": jnp.ones((2, 1), dtype=bool),
        "inning": jnp.array([[0.0], [0.5]], dtype=jnp.float32),
        "half": jnp.array([[0.0], [1.0]], dtype=jnp.float32),
        "outs": jnp.array([[0.0], [0.5]], dtype=jnp.float32),
        "base_state": jnp.array([[0.0], [3 / 7]], dtype=jnp.float32),
        "score_diff": jnp.array([[0.1], [-0.2]], dtype=jnp.float32),
        "tto": jnp.array([[1 / 3], [2 / 3]], dtype=jnp.float32),
        "shift_restricted": jnp.ones((2, 1), dtype=jnp.float32),
        "pitch_clock": jnp.ones((2, 1), dtype=jnp.float32),
        "pitch_count_game": jnp.array([[0.1], [0.4]], dtype=jnp.float32),
        "pitcher_ids": jnp.array([[0], [1]], dtype=jnp.int32),
        "batter_ids": jnp.array([[2], [99]], dtype=jnp.int32),  # unknown stays neutral
        "park_ids": jnp.array([[0], [1]], dtype=jnp.int32),
    }
    with nh.seed(rng_seed=jax.random.PRNGKey(1)), nh.substitute(data=params), nh.trace() as trace:
        pa_model(batch, table, teacher_force=False, outcome_only=True, fatigue=True)
    expected = trace["pa_outcome"]["fn"].logits[:, 0, :]

    fast = build_pa_inference(params, table, outcome_only=True, fatigue=True)
    actual = fast.logits(
        inning=batch["inning"][:, 0], half=batch["half"][:, 0],
        outs=batch["outs"][:, 0], base_state=batch["base_state"][:, 0],
        score_diff=batch["score_diff"][:, 0], tto=batch["tto"][:, 0],
        shift_restricted=batch["shift_restricted"][:, 0],
        pitch_clock=batch["pitch_clock"][:, 0],
        pitch_count_game=batch["pitch_count_game"][:, 0],
        pitcher_ids=batch["pitcher_ids"][:, 0], batter_ids=batch["batter_ids"][:, 0],
        park_ids=batch["park_ids"][:, 0],
        bat_side=jnp.zeros(2), pit_hand=jnp.zeros(2),
    )
    assert jnp.allclose(actual, expected, atol=1e-6)


def test_bucket_size_avoids_shape_per_active_count():
    assert [bucket_size(n) for n in (1, 2, 3, 4, 5, 8, 9)] == [1, 2, 4, 4, 8, 8, 16]
