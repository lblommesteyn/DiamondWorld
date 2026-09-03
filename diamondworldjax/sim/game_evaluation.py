"""Helpers for evaluating full generated baseball games.

The returned sample axis is intentionally retained.  Collapsing it to a
per-game mean before computing a distribution metric removes precisely the
variance that Monte Carlo evaluation is meant to measure.
"""
from __future__ import annotations

import numpy as np


def simulate_game_draws(
    model_fn,
    params,
    player_table: dict,
    games: list[dict],
    rng_key,
    num_samples: int,
    *,
    seed: int = 0,
    **simulate_kwargs,
) -> dict[str, np.ndarray]:
    """Return generated game fields, each with shape ``(S, G)``.

    Each game spec is replicated once per Monte Carlo draw and passed to the
    true simulator, which generates its own outs, inning boundaries, walk-offs,
    and extra innings.  Keeping draws separate makes this safe for calibration
    and distributional metrics.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be at least one")
    if not games:
        return {name: np.empty((num_samples, 0))
                for name in ("away", "home", "away_hits", "home_hits")}

    from diamondworldjax.scripts.simulate_games import simulate

    repeated_games = [game for game in games for _ in range(num_samples)]
    result = simulate(
        model_fn, params, player_table, repeated_games, rng_key,
        seed=seed, **simulate_kwargs,
    )
    n_games = len(games)
    return {
        name: np.asarray(result[name], dtype=float).reshape(n_games, num_samples).T
        for name in ("away", "home", "away_hits", "home_hits")
    }


def simulate_score_draws(*args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Return only the away and home score draws (a convenience wrapper)."""
    draws = simulate_game_draws(*args, **kwargs)
    return draws["away"], draws["home"]
