from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import flax_module
from typing import Optional


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
        player_stats: jnp.ndarray,   # (P, F_player)  full player table
        league_ids: jnp.ndarray,     # (P,)  int32
        handedness: jnp.ndarray,     # (P,)  int32
        name: str = "player_encoder",
        hidden_dim: int = 128,
        out_dim: int = 64,
    ):
        f_player = player_stats.shape[-1]

        # Register the Flax module as a NumPyro site so SVI / MCMC can tune
        # its parameters.
        encoder = flax_module(
            name,
            PlayerSeasonEncoder(
                f_player=f_player,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
            ),
            input_shape=[
                (player_stats.shape[0], f_player),  # player_stats
                (player_stats.shape[0],),            # league_id
                (player_stats.shape[0],),            # handedness
            ],
        )

        # Forward-pass to produce the full embedding table (P, D).
        self._table: jnp.ndarray = encoder(player_stats, league_ids, handedness)
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
    player_stats: jnp.ndarray,   # (P, F_player)
    league_ids: jnp.ndarray,     # (P,) int32
    handedness: jnp.ndarray,     # (P,) int32
    pitcher_ids: jnp.ndarray,    # (B, T) int32
    batter_ids: jnp.ndarray,     # (B, T) int32
    hidden_dim: int = 128,
    out_dim: int = 64,
    name: str = "player_encoder",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    NumPyro model fragment.

    Returns
    -------
    pitcher_z : (B, T, D)
    batter_z  : (B, T, D)
    """
    registry = PlayerRegistry(
        player_stats, league_ids, handedness,
        name=name, hidden_dim=hidden_dim, out_dim=out_dim,
    )
    pitcher_z = registry.lookup(pitcher_ids)  # (B, T, D)
    batter_z  = registry.lookup(batter_ids)   # (B, T, D)
    return pitcher_z, batter_z
