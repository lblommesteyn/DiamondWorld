"""Teacher-forced and free rollout utilities for DiamondWorldJAX."""
from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import Predictive


# ---------------------------------------------------------------------------
# Teacher-forced rollout
# ---------------------------------------------------------------------------

def teacher_forced_samples(
    model: Callable,
    guide: Any,
    params: dict,
    batch: dict,
    player_table: dict,
    rng_key: jax.Array,
    num_samples: int = 1,
) -> dict:
    """
    Run posterior predictive sampling with observed pitch data as conditioning.

    Useful for calibration: the model sees all real observations (teacher_force=True)
    and we examine what it samples at each site.

    Returns
    -------
    dict mapping site name → array of shape (num_samples, B, T, ...)
    """
    predictive = Predictive(
        model,
        guide      = guide,
        params     = params,
        num_samples= num_samples,
        return_sites = _ALL_OBSERVABLE_SITES,
    )
    return predictive(
        rng_key,
        batch,
        player_table,
        teacher_force = True,
    )


# ---------------------------------------------------------------------------
# Free rollout (game simulation)
# ---------------------------------------------------------------------------

def free_rollout_samples(
    model: Callable,
    guide: Any,
    params: dict,
    batch: dict,
    player_table: dict,
    rng_key: jax.Array,
    num_samples: int = 1,
) -> dict:
    """
    Draw from the posterior predictive without conditioning on observations.

    Samples pitch sequences autoregressively from the learned distributions.

    Returns
    -------
    dict mapping site name → array of shape (num_samples, B, T, ...)
    """
    predictive = Predictive(
        model,
        guide      = guide,
        params     = params,
        num_samples= num_samples,
        return_sites = _ALL_OBSERVABLE_SITES,
    )
    return predictive(
        rng_key,
        batch,
        player_table,
        teacher_force = False,
    )


# ---------------------------------------------------------------------------
# Game-level aggregate: extract run totals from a free rollout
# ---------------------------------------------------------------------------

def extract_game_runs(
    samples: dict,
    terminal_mask: jnp.ndarray,   # (B, T) bool
) -> jnp.ndarray:
    """
    Sum runs_scored at terminal PAs per game per sample.

    Parameters
    ----------
    samples       : output of free_rollout_samples
    terminal_mask : (B, T) — which positions are PA terminals

    Returns
    -------
    game_runs : (num_samples, B) int — total runs per game per sample
    """
    # runs_scored site: (num_samples, B, T) int
    runs = samples.get("runs_scored")
    if runs is None:
        raise KeyError("'runs_scored' not found in samples dict.")

    # Mask out non-terminal positions
    mask = terminal_mask[None, :, :]          # (1, B, T)
    game_runs = (runs * mask).sum(axis=-1)    # (num_samples, B)
    return game_runs


def simulate_season_runs(
    model: Callable,
    guide: Any,
    params: dict,
    season_batches: list[dict],
    player_table: dict,
    seed: int = 0,
) -> jnp.ndarray:
    """
    Simulate full-season run totals by iterating over all game batches.

    Returns
    -------
    all_runs : (total_games,) float  — one total run value per game,
               averaged over draws.
    """
    rng = jax.random.PRNGKey(seed)
    all_runs = []

    for batch in season_batches:
        rng, key = jax.random.split(rng)
        samples = free_rollout_samples(
            model, guide, params, batch, player_table, key, num_samples=8
        )
        terminal_mask = batch["terminal_mask"]
        game_runs = extract_game_runs(samples, terminal_mask)  # (8, B)
        # Average over samples: (B,)
        all_runs.append(game_runs.mean(axis=0))

    return jnp.concatenate(all_runs, axis=0)


# ---------------------------------------------------------------------------
# Site names returned by Predictive
# ---------------------------------------------------------------------------

_ALL_OBSERVABLE_SITES = [
    # Hurdle
    "pitch_type", "plate_x", "plate_z", "release_speed",
    "swing", "called_strike", "contact", "foul",
    # Batted ball
    "launch_speed", "launch_angle", "spray_angle", "hit_distance",
    # Transition
    "runs_scored", "base_state_after", "outs_added",
    "error_flag", "wild_pitch", "passed_ball", "balk",
    # Manager
    "pitching_change", "steal_attempt", "runner_send", "defensive_alignment",
]
