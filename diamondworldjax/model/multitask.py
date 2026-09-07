"""Joint PA/pitch training with shared talent and task-specific residuals.

The two likelihoods are deliberately kept separate: PA outcomes and pitch-level
events contain overlapping, but not identical, evidence about a player.  A global
latent therefore carries transferable talent while small task residuals prevent
the PA task from forcing its representation onto pitch selection (and vice versa).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpyro.handlers as handlers

from .embeddings import SKILL_DIM
from .joint import diamondworld_model
from .pa_model import pa_model


DEFAULT_RESIDUAL_SCALE = 0.35


def task_checkpoint_params(params: Mapping[str, Any], task: str) -> dict[str, Any]:
    """Extract a PA or pitch checkpoint with its marginal player posterior.

    Scoped model parameters are renamed to their standalone names.  The normal
    posterior approximation follows from independence in the mean-field guide:
    ``Var(shared + residual) = Var(shared) + Var(residual)``.
    """
    if task not in {"pa", "pitch"}:
        raise ValueError("task must be 'pa' or 'pitch'")
    prefix = f"{task}/"
    extracted = {key.removeprefix(prefix): value for key, value in params.items()
                 if key.startswith(prefix)}
    if not extracted:
        raise ValueError(f"checkpoint has no {task!r}-scoped parameters")
    try:
        shared_mu = params["shared_player_skills_mu"]
        shared_sigma = params["shared_player_skills_sigma"]
        residual_mu = params[f"{task}_skill_residual_mu"]
        residual_sigma = params[f"{task}_skill_residual_sigma"]
    except KeyError as exc:
        raise ValueError("checkpoint does not contain a shared-task-skills guide") from exc
    extracted["player_mu"] = shared_mu + residual_mu
    extracted["player_sigma"] = jnp.sqrt(shared_sigma ** 2 + residual_sigma ** 2)
    if "skill_walk_sigma_loc" in params:
        extracted["skill_walk_sigma_loc"] = params["skill_walk_sigma_loc"]
    return extracted


def sample_shared_task_skills(
    n_players: int,
    residual_scale: float = DEFAULT_RESIDUAL_SCALE,
    kl_scale: float = 1.0,
    skill_prior: str = "iso",
    n_seasons: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample a global skill and PA/pitch deviations for every player.

    The narrower residual prior identifies the decomposition: shared skill is
    where cross-task evidence accumulates, while deviations only capture effects
    the other task cannot explain.
    """
    if not 0 < residual_scale < 1:
        raise ValueError("residual_scale must be between zero and one")
    if skill_prior not in {"iso", "walk"}:
        raise ValueError("shared task skills support only 'iso' and 'walk'")
    shape = ((n_players, SKILL_DIM) if skill_prior == "iso"
             else (n_players, n_seasons, SKILL_DIM))
    # Keep each task's *combined* innovation at unit variance, matching the
    # R2 walk prior. residual_scale controls only how much of that innovation
    # is task-specific; it must not also change the total regularisation.
    shared_scale = (1.0 - residual_scale ** 2) ** 0.5
    with handlers.scale(scale=kl_scale):
        shared = numpyro.sample(
            "shared_player_skills",
            dist.Normal(jnp.zeros(shape), shared_scale * jnp.ones(shape)).to_event(len(shape)),
        )
        pa_residual = numpyro.sample(
            "pa_skill_residual",
            dist.Normal(jnp.zeros(shape), residual_scale * jnp.ones(shape)).to_event(len(shape)),
        )
        pitch_residual = numpyro.sample(
            "pitch_skill_residual",
            dist.Normal(jnp.zeros(shape), residual_scale * jnp.ones(shape)).to_event(len(shape)),
        )
        if skill_prior == "walk":
            walk_sigma = numpyro.sample("skill_walk_sigma", dist.HalfNormal(0.3))

    if skill_prior == "walk":
        def _walk(innovations: jnp.ndarray) -> jnp.ndarray:
            steps = jnp.concatenate(
                [innovations[:, :1, :], innovations[:, 1:, :] * walk_sigma], axis=1
            )
            return jnp.cumsum(steps, axis=1)

        shared = _walk(shared)
        pa_residual = _walk(pa_residual)
        pitch_residual = _walk(pitch_residual)
    return shared, pa_residual, pitch_residual


def multitask_model(
    batch: Mapping[str, dict],
    player_table: dict,
    teacher_force: bool = True,
    *,
    residual_scale: float = DEFAULT_RESIDUAL_SCALE,
    kl_scale: float = 1.0,
    pa_model_kwargs: Mapping[str, Any] | None = None,
    pitch_model_kwargs: Mapping[str, Any] | None = None,
    skill_prior: str = "iso",
    n_seasons: int = 1,
) -> None:
    """Score PA and pitch batches against one hierarchical player hierarchy.

    ``batch`` must contain ``{"pa": pa_batch, "pitch": pitch_batch}`` made
    from the same player-index mapping. Scopes keep the two likelihoods' sample
    and parameter sites distinct; the three latent skill tensors above remain
    deliberately unscoped and are consequently shared by both tasks.
    """
    if set(batch) < {"pa", "pitch"}:
        raise ValueError("multitask batch must contain 'pa' and 'pitch'")
    n_players = player_table["stats"].shape[0]
    shared, pa_residual, pitch_residual = sample_shared_task_skills(
        n_players,
        residual_scale=residual_scale,
        kl_scale=kl_scale,
        skill_prior=skill_prior,
        n_seasons=n_seasons,
    )

    with handlers.scope(prefix="pa"):
        pa_model(
            batch["pa"],
            player_table,
            teacher_force=teacher_force,
            player_skills_override=shared + pa_residual,
            **dict(pa_model_kwargs or {}),
        )
    with handlers.scope(prefix="pitch"):
        diamondworld_model(
            batch["pitch"],
            player_table,
            teacher_force=teacher_force,
            player_skills_override=shared + pitch_residual,
            direct_player_context=True,
            **dict(pitch_model_kwargs or {}),
        )
