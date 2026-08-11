from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import flax_module
from typing import Optional

# Dimension of the per-player stochastic skill vector.
SKILL_DIM = 32


# ---------------------------------------------------------------------------
# Flax sub-modules
# ---------------------------------------------------------------------------

class LeagueEmbedding(nn.Module):
    """Embeds league_id ∈ {0=MLB, 1=MiLB} → R^8."""
    embed_dim: int = 8

    @nn.compact
    def __call__(self, league_id: jnp.ndarray) -> jnp.ndarray:
        # league_id: (P,)  int32
        return nn.Embed(num_embeddings=2, features=self.embed_dim)(league_id)


class HandednessEmbedding(nn.Module):
    """Embeds handedness ∈ {0=L, 1=R} → R^4."""
    embed_dim: int = 4

    @nn.compact
    def __call__(self, hand: jnp.ndarray) -> jnp.ndarray:
        # hand: (P,) int32
        return nn.Embed(num_embeddings=2, features=self.embed_dim)(hand)


class SkillFusionLayer(nn.Module):
    """Fuses deterministic player encoding with a stochastic skill vector.

    det_emb  : (P, out_dim)   — output of PlayerSeasonEncoder
    skill_vec: (P, SKILL_DIM) — sampled from variational posterior
    → fused  : (P, out_dim)
    """
    out_dim: int = 64

    @nn.compact
    def __call__(
        self,
        det_emb: jnp.ndarray,   # (P, out_dim)
        skill_vec: jnp.ndarray, # (P, SKILL_DIM)
    ) -> jnp.ndarray:           # (P, out_dim)
        x = jnp.concatenate([det_emb, skill_vec], axis=-1)
        h = nn.Dense(self.out_dim)(x)
        h = nn.LayerNorm()(h)
        return h


class PlayerSeasonEncoder(nn.Module):
    """
    Maps raw player-season stat vectors + categorical covariates to latent
    embeddings.

    Input
    -----
    player_stats : float32  (P, F_player)   — normalised seasonal stats
    league_id    : int32    (P,)            — 0=MLB, 1=MiLB
    handedness   : int32    (P,)            — 0=L, 1=R

    Output
    ------
    player_z : float32  (P, D)  where D = out_dim = 64
    """
    f_player: int          # number of raw stat features
    hidden_dim: int = 128
    out_dim: int = 64

    @nn.compact
    def __call__(
        self,
        player_stats: jnp.ndarray,   # (P, F_player)
        league_id: jnp.ndarray,      # (P,)
        handedness: jnp.ndarray,     # (P,)
    ) -> jnp.ndarray:                # (P, D)

        # --- categorical embeddings ---
        league_emb = LeagueEmbedding(embed_dim=8)(league_id)       # (P, 8)
        hand_emb   = HandednessEmbedding(embed_dim=4)(handedness)  # (P, 4)

        # --- concatenate all inputs ---
        x = jnp.concatenate([player_stats, league_emb, hand_emb], axis=-1)
        # x: (P, F_player + 8 + 4)

        # --- first dense block ---
        h = nn.Dense(self.hidden_dim)(x)
        h = nn.LayerNorm()(h)
        h = nn.relu(h)

        # --- residual block ---
        r = nn.Dense(self.hidden_dim)(h)
        r = nn.LayerNorm()(r)
        r = nn.relu(r + h)          # residual addition before activation

        # --- project to output dim ---
        out = nn.Dense(self.out_dim)(r)
        out = nn.LayerNorm()(out)
        return out                   # (P, D)


# ---------------------------------------------------------------------------
# NumPyro-registered encoder with registry lookup
# ---------------------------------------------------------------------------

class PlayerRegistry:
    """
    Wraps a PlayerSeasonEncoder registered as a NumPyro flax_module and
    provides indexed lookup from (B, T) id tensors to (B, T, D) embeddings.

    Usage inside a NumPyro model function
    --------------------------------------
    registry = PlayerRegistry(player_stats, league_ids, handedness,
                               name="player_encoder")
    batter_emb  = registry.lookup(batter_ids)   # (B, T, D)
    pitcher_emb = registry.lookup(pitcher_ids)  # (B, T, D)
    """

    def __init__(
        self,
        player_stats: jnp.ndarray,    # (P, F_player)  full player table
        league_ids: jnp.ndarray,      # (P,)  int32
        handedness: jnp.ndarray,      # (P,)  int32
        player_skills: jnp.ndarray,   # (P, SKILL_DIM)  stochastic skill vectors
        name: str = "player_encoder",
        hidden_dim: int = 128,
        out_dim: int = 64,
    ):
        f_player = player_stats.shape[-1]

        # Deterministic encoding: raw stats + league + hand → (P, out_dim)
        encoder = flax_module(
            name,
            PlayerSeasonEncoder(
                f_player=f_player,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
            ),
            player_stats,
            league_ids,
            handedness,
        )
        det_emb: jnp.ndarray = encoder(player_stats, league_ids, handedness)

        # Fuse deterministic encoding with stochastic skill vector → (P, out_dim)
        fusion = flax_module(
            f"{name}_skill_fusion",
            SkillFusionLayer(out_dim=out_dim),
            det_emb,
            player_skills,
        )
        self._table: jnp.ndarray = fusion(det_emb, player_skills)
        self.out_dim = out_dim

    def lookup(self, ids: jnp.ndarray) -> jnp.ndarray:
        """
        Index embedding table with arbitrary id tensor.

        Parameters
        ----------
        ids : int32  (...,)  — player indices into the P-row table

        Returns
        -------
        embeddings : float32  (..., D)
        """
        return self._table[ids]  # uses JAX advanced indexing


# ---------------------------------------------------------------------------
# Convenience: standalone NumPyro model fragment (for testing / documentation)
# ---------------------------------------------------------------------------

def encode_players_numpyro(
    player_stats: jnp.ndarray,    # (P, F_player)
    league_ids: jnp.ndarray,      # (P,) int32
    handedness: jnp.ndarray,      # (P,) int32
    pitcher_ids: jnp.ndarray,     # (B, T) int32
    batter_ids: jnp.ndarray,      # (B, T) int32
    player_skills: jnp.ndarray,   # (P, SKILL_DIM), or (P, S, SKILL_DIM) if seasonal
    hidden_dim: int = 128,
    out_dim: int = 64,
    name: str = "player_encoder",
    season_idx: Optional[jnp.ndarray] = None,  # (B, T) int32, required if seasonal
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    NumPyro model fragment.

    When `player_skills` is 3-D the skill latent is per (player, season) and
    `season_idx` selects which season's skill each plate appearance sees. The
    deterministic encoding is season-invariant (the stat table is one row per
    player), so it is broadcast across seasons before fusion; only the stochastic
    part carries time.

    Returns
    -------
    pitcher_z : (B, T, D)
    batter_z  : (B, T, D)
    """
    if player_skills.ndim == 2:
        registry = PlayerRegistry(
            player_stats, league_ids, handedness, player_skills,
            name=name, hidden_dim=hidden_dim, out_dim=out_dim,
        )
        return registry.lookup(pitcher_ids), registry.lookup(batter_ids)

    if season_idx is None:
        raise ValueError("seasonal player_skills requires season_idx")

    P, S, _ = player_skills.shape
    encoder = flax_module(
        name,
        PlayerSeasonEncoder(f_player=player_stats.shape[-1],
                            hidden_dim=hidden_dim, out_dim=out_dim),
        player_stats, league_ids, handedness,
    )
    det_emb = encoder(player_stats, league_ids, handedness)          # (P, D)
    det_rep = jnp.broadcast_to(det_emb[:, None, :], (P, S, out_dim))

    fusion = flax_module(
        f"{name}_skill_fusion", SkillFusionLayer(out_dim=out_dim),
        det_rep.reshape(P * S, out_dim), player_skills.reshape(P * S, -1),
    )
    table = fusion(det_rep.reshape(P * S, out_dim),
                   player_skills.reshape(P * S, -1)).reshape(P, S, out_dim)

    return table[pitcher_ids, season_idx], table[batter_ids, season_idx]
