"""Joint PA + ABCD likelihood with one shared player-skill hierarchy.

This module intentionally does not alter either standalone training path.  It
joins the PA NumPyro likelihood to the current ABCD Transformer heads at the
only representation that should be shared: player talent.  The task decoders,
history encoders, and small task residuals remain private so that pitch choice
and plate-appearance outcome evidence do not force one another into the same
conditional representation.
"""
from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpyro
import numpyro.handlers as handlers
from numpyro.contrib.module import flax_module

from .bayesian_pitchformer import likelihood as abcd_log_likelihood
from .embeddings import encode_players_numpyro
from .marginal_pitch_likelihood import marginal_log_likelihood, prepare_hidden
from .multitask import sample_shared_task_skills
from .pa_model import pa_model
from .pitchformer import HeadResidual, SuperState, TransformerA, TransformerB
from .transformer_c import TransformerC
from .transformer_d import TransformerD


class JointABCDNetwork(nn.Module):
    """Current ABCD heads conditioned on externally supplied player vectors.

    ``pitcher_vectors`` and ``batter_vectors`` are generated from the shared
    stochastic hierarchy in :func:`joint_pa_abcd_model`.  Keeping them as
    inputs makes it impossible for the ABCD path to create a private ID table
    that bypasses the joint latent.  The per-head residual is context-only;
    task-specific player variation belongs in the explicit ABCD residual.
    """

    n_pitchers: int
    n_batters: int
    n_parks: int
    d_model: int = 192
    n_layers: int = 4
    n_heads: int = 6
    d_residual: int = 16
    dropout: float = 0.1
    heads: str = "abcd"
    skill_seasons: int = 1
    pitch_history: bool = True
    position_encoding: str = "sinusoidal"
    window_size: int = 32
    observation_masks: bool = True
    c_event_mode: str = "bundles"
    c_support: tuple | None = None
    learned_called_strike: bool = False

    def setup(self):
        # player_mode="none" ensures no private player-ID embeddings are
        # allocated.  ``player_vectors`` below supplies the only identity path.
        self.shared_ss = SuperState(
            self.n_pitchers, self.n_batters, self.n_parks,
            d_model=self.d_model, player_mode="none",
            skill_seasons=self.skill_seasons,
        )
        common = dict(
            n_pitchers=self.n_pitchers, n_batters=self.n_batters,
            n_parks=self.n_parks, d_model=self.d_model,
            n_layers=self.n_layers, n_heads=self.n_heads, dropout=self.dropout,
            player_mode="none", skill_seasons=self.skill_seasons,
            pitch_history=self.pitch_history,
            position_encoding=self.position_encoding,
            window_size=self.window_size,
            observation_masks=self.observation_masks,
            c_event_mode=self.c_event_mode, c_support=self.c_support,
            learned_called_strike=self.learned_called_strike,
        )
        classes = dict(a=TransformerA, b=TransformerB, c=TransformerC, d=TransformerD)
        for head in self.heads:
            setattr(self, f"res_{head}", HeadResidual(
                self.n_pitchers, self.n_batters, self.d_residual,
                self.d_model, "none",
            ))
            setattr(self, f"head_{head}", classes[head](**common))

    def __call__(self, batch, pitcher_vectors, batter_vectors, *, train: bool = False,
                 hidden_override=None, encode_only: bool = False,
                 selected_heads=None):
        selected = tuple(selected_heads or self.heads)
        if hidden_override is not None:
            return {
                head: getattr(self, f"head_{head}")(
                    batch, train=train, hidden_override=hidden_override[head]
                )
                for head in selected
            }

        ss = self.shared_ss(
            batch["pitcher_idx"], batch["batter_idx"], batch["park_idx"],
            batch["ctx"], batch["geom"], batch.get("skill_season"),
            player_vectors=(pitcher_vectors, batter_vectors),
        )
        output = {}
        for head in selected:
            residual = getattr(self, f"res_{head}")(
                batch["pitcher_idx"], batch["batter_idx"], ss
            )
            output[head] = getattr(self, f"head_{head}")(
                batch, train=train, ss_override=ss + residual,
                encode_only=encode_only,
            )
        return output


def _joint_marginal_prepare(apply):
    """Reuse ABCD history representations across measurement integrations."""
    return prepare_hidden(
        lambda data: apply(data, encode_only=True),
        lambda data, hidden, selected: apply(
            data, hidden_override=hidden, selected_heads=selected
        ),
    )


def joint_pa_abcd_model(
    batch: Mapping[str, dict],
    player_table: dict,
    teacher_force: bool = True,
    *,
    abcd_options: Mapping[str, object],
    role_global_indices: Mapping[str, jnp.ndarray],
    pa_model_kwargs: Mapping[str, object],
    pa_weight: float = 1.0,
    abcd_weight: float = 1.0,
    residual_scale: float = 0.35,
    kl_scale: float = 1.0,
    skill_prior: str = "walk",
    n_seasons: int = 1,
    missing_samples: int = 2,
    d_hr_weight: float = 0.0,
) -> None:
    """Score matched PA and ABCD batches with a shared posterior hierarchy.

    The ABCD factor is rescaled from *per valid pitch* to the PA batch's
    effective count.  This makes equal task weights interpretable rather than
    letting its several pitch measurements dominate the PA likelihood merely
    because they occur more often.
    """
    if set(batch) < {"pa", "abcd"}:
        raise ValueError("joint PA/ABCD training requires 'pa' and 'abcd' batches")
    if not pa_weight > 0 or not abcd_weight > 0:
        raise ValueError("joint task weights must be positive")
    if d_hr_weight < 0:
        raise ValueError("d_hr_weight must be nonnegative")

    pa_batch, abcd_batch = batch["pa"], batch["abcd"]
    n_players = player_table["stats"].shape[0]
    shared, pa_residual, abcd_residual = sample_shared_task_skills(
        n_players,
        residual_scale=residual_scale,
        kl_scale=kl_scale,
        skill_prior=skill_prior,
        n_seasons=n_seasons,
    )

    # PA owns its original decoder and (separately named) deterministic player
    # encoder.  Its gradient still updates ``shared`` directly.
    with handlers.scale(scale=pa_weight):
        with handlers.scope(prefix="pa"):
            pa_model(
                pa_batch,
                player_table,
                teacher_force=teacher_force,
                player_skills_override=shared + pa_residual,
                skill_prior=skill_prior,
                **dict(pa_model_kwargs),
            )

    # ABCD uses its own deterministic fusion map, but consumes exactly the
    # same shared latent and all player IDs resolve through the PA registry.
    pitcher_global = role_global_indices["pitcher"][abcd_batch["pitcher_idx"]]
    batter_global = role_global_indices["batter"][abcd_batch["batter_idx"]]
    season_idx = abcd_batch.get("skill_season") if skill_prior == "walk" else None
    pitcher_vectors, batter_vectors = encode_players_numpyro(
        player_table["stats"], player_table["league"], player_table["hand"],
        pitcher_ids=pitcher_global,
        batter_ids=batter_global,
        player_skills=shared + abcd_residual,
        name="abcd_player_encoder",
        season_idx=season_idx,
    )
    # Pitchformer reserves local index zero as unknown.  Seasonal vector lookup
    # cannot carry that sentinel itself, so make its representation explicitly
    # neutral after the shared lookup.
    pitcher_vectors = jnp.where(
        (abcd_batch["pitcher_idx"] > 0)[..., None], pitcher_vectors, 0.0
    )
    batter_vectors = jnp.where(
        (abcd_batch["batter_idx"] > 0)[..., None], batter_vectors, 0.0
    )

    network = flax_module(
        "abcd_network", JointABCDNetwork(**dict(abcd_options)),
        abcd_batch, pitcher_vectors, batter_vectors, train=teacher_force,
    )
    rngs = {"dropout": numpyro.prng_key()} if teacher_force and abcd_options.get("dropout", 0.0) else {}

    def apply(data, **kwargs):
        return network(data, pitcher_vectors, batter_vectors, train=teacher_force,
                       rngs=rngs, **kwargs)

    if missing_samples:
        ll = marginal_log_likelihood(
            apply, abcd_batch, numpyro.prng_key(), missing_samples,
            prepare=_joint_marginal_prepare(apply),
            d_hr_weight=d_hr_weight,
        )
    else:
        ll = abcd_log_likelihood(apply(abcd_batch), abcd_batch,
                                 d_hr_weight=d_hr_weight)

    pa_count = jnp.maximum(jnp.sum(pa_batch["pa_valid"]), 1)
    pitch_count = jnp.maximum(
        jnp.sum(abcd_batch["valid"] * abcd_batch.get("loss_mask", 1)), 1
    )
    # These are diagnostic-only values.  Keep them before the task weight and
    # count-normalisation factor so a training log shows each task in its own
    # natural unit instead of an opaque combined ELBO.
    numpyro.deterministic("joint/abcd_nll_per_pitch", -ll / pitch_count)
    numpyro.deterministic("joint/pa_count", pa_count)
    numpyro.deterministic("joint/pitch_count", pitch_count)
    numpyro.factor("abcd_likelihood", abcd_weight * (pa_count / pitch_count) * ll)
