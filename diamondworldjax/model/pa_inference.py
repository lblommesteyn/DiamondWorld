"""Fast, pure inference for the ordinary PA outcome model.

The NumPyro model is the right training interface, but it is deliberately
expensive to call one plate appearance at a time: each invocation rebuilds the
full player embedding table and enters probabilistic handlers.  Game simulation
has fixed parameters and a fixed player table, so those operations can be done
once before the first PA.  This module exposes the remaining deterministic
``state + matchup -> outcome logits`` calculation as a JIT-able function.

PitchFormer deliberately does not use this adapter; it has a different
history-dependent inference contract.
"""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .embeddings import PlayerSeasonEncoder, SkillFusionLayer
from .pa_model import (
    BilinearMatchup,
    N_PARKS,
    PAOutcomeHead,
    PAOutcomeHeadV6,
    NestedOutcomeHead,
    ParkEmbedding,
)


def bucket_size(n: int) -> int:
    """Smallest power-of-two batch size containing ``n`` active games."""
    if n < 1:
        raise ValueError("batch size must be positive")
    return 1 << (n - 1).bit_length()


class PAInference:
    """Cached inference state for a non-history-dependent PA model.

    The constructor owns immutable simulation state only.  ``logits`` accepts a
    one-dimensional PA batch and is JIT-compiled by shape, which makes the
    power-of-two buckets in the simulator reusable across half innings.
    """

    def __init__(
        self,
        params: dict[str, Any],
        player_table: dict[str, Any],
        *,
        outcome_only: bool,
        fatigue: bool,
        platoon: bool,
        nested: bool,
        bilinear_rank: int,
        skill_prior: str,
        season_base: int,
        n_seasons: int,
        simulation_season: int,
    ) -> None:
        self.fatigue = fatigue
        self.platoon = platoon
        self.bilinear_rank = bilinear_rank

        stats = jnp.asarray(player_table["stats"])
        league = jnp.asarray(player_table["league"])
        hand = jnp.asarray(player_table["hand"])
        player_skills = self._resolve_skills(
            params,
            skill_prior=skill_prior,
            season_base=season_base,
            n_seasons=n_seasons,
            simulation_season=simulation_season,
        )

        encoder = PlayerSeasonEncoder(f_player=stats.shape[-1])
        deterministic = encoder.apply(
            {"params": params["player_encoder$params"]}, stats, league, hand
        )
        fusion = SkillFusionLayer()
        self.player_z = fusion.apply(
            {"params": params["player_encoder_skill_fusion$params"]},
            deterministic,
            player_skills,
        )

        park = ParkEmbedding()
        self.park_z = park.apply(
            {"params": params["park_embedding$params"]},
            jnp.arange(N_PARKS, dtype=jnp.int32),
        )

        if outcome_only:
            if nested:
                head = NestedOutcomeHead()
                head_params = params["pa_outcome_head_nested$params"]
            else:
                head = PAOutcomeHeadV6()
                head_params = params["pa_outcome_head_v6$params"]
        else:
            head = PAOutcomeHead()
            head_params = params["pa_outcome_head$params"]
        self._head = head
        self._head_params = head_params

        self._bilinear = None
        self._bilinear_params = None
        if bilinear_rank > 0:
            self._bilinear = BilinearMatchup(rank=bilinear_rank)
            self._bilinear_params = params["bilinear_matchup$params"]

        # This is intentionally built after the tables and parameter subtrees are
        # fixed.  Only the PA-shaped inputs are dynamic, so a bucket size compiles
        # once and is reused for every matching half-inning batch.
        self._logits_jit = jax.jit(self._logits)

    @staticmethod
    def _resolve_skills(
        params: dict[str, Any],
        *,
        skill_prior: str,
        season_base: int,
        n_seasons: int,
        simulation_season: int,
    ) -> jax.Array:
        if skill_prior == "walk":
            eps = params.get("player_skill_eps")
            if eps is None:
                raise KeyError("player_skill_eps must be materialized before fast simulation")
            sigma = params.get("skill_walk_sigma")
            if sigma is None:
                sigma = params.get("skill_walk_sigma_loc")
            if sigma is None:
                raise KeyError("skill_walk_sigma must be materialized before fast simulation")
            eps = jnp.asarray(eps)
            steps = jnp.concatenate([eps[:, :1, :], eps[:, 1:, :] * sigma], axis=1)
            skills = jnp.cumsum(steps, axis=1)
            season_idx = min(max(simulation_season - season_base, 0), n_seasons - 1)
            return skills[:, season_idx, :]

        skills = params.get("player_skills")
        if skills is None:
            raise KeyError("player_skills must be materialized before fast simulation")
        return jnp.asarray(skills)

    def _lookup_players(self, ids: jax.Array) -> jax.Array:
        n_players = self.player_z.shape[0]
        ids = ids.astype(jnp.int32)
        known = (ids >= 0) & (ids < n_players)
        safe_ids = jnp.clip(ids, 0, n_players - 1)
        value = self.player_z[safe_ids]
        return jnp.where(known[:, None], value, jnp.zeros_like(value))

    def _logits(
        self,
        inning: jax.Array,
        half: jax.Array,
        outs: jax.Array,
        base_state: jax.Array,
        score_diff: jax.Array,
        tto: jax.Array,
        shift_restricted: jax.Array,
        pitch_clock: jax.Array,
        pitch_count_game: jax.Array,
        pitcher_ids: jax.Array,
        batter_ids: jax.Array,
        park_ids: jax.Array,
        bat_side: jax.Array,
        pit_hand: jax.Array,
    ) -> jax.Array:
        pitcher_z = self._lookup_players(pitcher_ids)
        batter_z = self._lookup_players(batter_ids)
        park_z = self.park_z[jnp.clip(park_ids.astype(jnp.int32), 0, N_PARKS - 1)]
        state = [inning, half, outs, base_state, score_diff, tto,
                 shift_restricted, pitch_clock]
        if self.fatigue:
            state.append(pitch_count_game)
        if self.platoon:
            state.extend([bat_side, pit_hand])
        context = jnp.concatenate([jnp.stack(state, axis=-1), pitcher_z, batter_z, park_z], axis=-1)
        logits = self._head.apply({"params": self._head_params}, context)
        if self._bilinear is not None:
            logits = logits + self._bilinear.apply(
                {"params": self._bilinear_params}, batter_z, pitcher_z
            )
        return logits

    def logits(self, **inputs: Any) -> jax.Array:
        """Return outcome logits for one fixed-size, one-PA-per-row batch."""
        return self._logits_jit(**{name: jnp.asarray(value) for name, value in inputs.items()})


def build_pa_inference(
    params: dict[str, Any],
    player_table: dict[str, Any],
    *,
    outcome_only: bool = False,
    fatigue: bool = False,
    platoon: bool = False,
    nested: bool = False,
    bilinear_rank: int = 0,
    skill_prior: str = "iso",
    season_base: int = 2015,
    n_seasons: int = 9,
    simulation_season: int = 2024,
) -> PAInference:
    """Construct the fast path; callers fall back if its checkpoint is unsupported."""
    return PAInference(
        params, player_table,
        outcome_only=outcome_only, fatigue=fatigue, platoon=platoon,
        nested=nested, bilinear_rank=bilinear_rank, skill_prior=skill_prior,
        season_base=season_base, n_seasons=n_seasons,
        simulation_season=simulation_season,
    )
