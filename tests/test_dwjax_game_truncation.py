from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from diamondworldjax.scripts.simulate_games import simulate
from diamondworldjax.sim.rules_engine import EmpiricalEngine, PA_OUTCOME_IDX


def _walk_model(batch, player_table, teacher_force=False):
    del player_table, teacher_force
    logits = jnp.full((*batch["pa_valid"].shape, 9), -100.0)
    logits = logits.at[..., PA_OUTCOME_IDX["BB"]].set(100.0)
    numpyro.sample("pa_outcome", dist.Categorical(logits=logits))


def test_simulation_reports_half_inning_truncation():
    games = [{
        "away_lineup": [0] * 9,
        "home_lineup": [0] * 9,
        "home_staff": [0],
        "away_staff": [0],
        "park": 0,
    }]
    player_table = {
        "stats": jnp.zeros((1, 16)),
        "league": jnp.zeros((1,), dtype=jnp.int32),
        "hand": jnp.zeros((1,), dtype=jnp.int32),
        "_engine": EmpiricalEngine(),
        "_hook_dists": (jnp.array([99]), jnp.array([99])),
    }
    result = simulate(
        _walk_model, {}, player_table, games, jax.random.PRNGKey(0),
        fixed_nine=True, no_bullpen=True, max_pa_per_half=1,
    )
    assert result["truncated"].tolist() == [True]
    assert result["n_truncated"] == 1
    assert result["truncated_half_innings"] > 0
