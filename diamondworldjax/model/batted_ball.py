from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpyro
import numpyro.distributions as dist
import numpyro.handlers as handlers
from numpyro.contrib.module import flax_module
from typing import Optional
import dataclasses


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D_SHARED_CONTEXT  = 128   # from PitchTransformer
D_EXECUTION       = 8     # pitch execution features
D_PLAYER_EMB      = 64    # batter_z / pitcher_z
N_PARKS           = 31    # MLB park vocabulary
D_PARK_EMB        = 16

# Input dim = 128 + 8 + 64 + 16 = 216  (plus pitcher_z injected separately)
# Full input: shared_context(128) + execution(8) + batter_z(64) + park_emb(16) = 216
# pitcher_z(64) is concatenated as well → total 280, but documented at 216 per spec;
# we include pitcher_z as the spec says "given (pitch_package, batter_z, pitcher_z, park)".
# Actual input dim = D_SHARED_CONTEXT + D_EXECUTION + D_PLAYER_EMB + D_PLAYER_EMB + D_PARK_EMB
D_INPUT = D_SHARED_CONTEXT + D_EXECUTION + D_PLAYER_EMB + D_PLAYER_EMB + D_PARK_EMB  # 280


# ---------------------------------------------------------------------------
# Flax: BattedBallNet
# ---------------------------------------------------------------------------

class BattedBallNet(nn.Module):
    """
    Two-layer MLP with LayerNorm that maps batted-ball context to (mu, sigma)
    parameters for each batted-ball outcome.

    Inputs
    ------
    shared_context   : float32  (B, T, 128)
    execution        : float32  (B, T, 8)
    batter_z         : float32  (B, T, 64)
    pitcher_z        : float32  (B, T, 64)
    park_emb         : float32  (B, T, 16)   — from ParkEmbedding

    Output (dict of mu/sigma pairs)
    ------
    Each output is (B, T) shaped.
    """
    hidden_dim: int = 256
    n_parks: int = N_PARKS

    @nn.compact
    def __call__(
        self,
        shared_context: jnp.ndarray,   # (B, T, 128)
        execution: jnp.ndarray,         # (B, T, 8)
        batter_z: jnp.ndarray,          # (B, T, 64)
        pitcher_z: jnp.ndarray,         # (B, T, 64)
        park_emb: jnp.ndarray,          # (B, T, 16)
    ) -> dict:

        x = jnp.concatenate(
            [shared_context, execution, batter_z, pitcher_z, park_emb], axis=-1
        )  # (B, T, 280)

        # --- layer 1 ---
        h = nn.Dense(self.hidden_dim)(x)
        h = nn.LayerNorm()(h)
        h = nn.relu(h)

        # --- layer 2 ---
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.LayerNorm()(h)
        h = nn.relu(h)

        # --- output heads: each outputs (mu, log_sigma) → (B, T, 2) ---
        def _head(name: str) -> tuple[jnp.ndarray, jnp.ndarray]:
            params = nn.Dense(2, name=name)(h)           # (B, T, 2)
            mu        = params[..., 0]
            log_sigma = params[..., 1]
            sigma = jax.nn.softplus(log_sigma) + 5e-2
            return mu, sigma

        ls_mu,   ls_sigma   = _head("head_launch_speed")   # launch speed  mph
        la_mu,   la_sigma   = _head("head_launch_angle")   # launch angle  degrees
        sa_mu,   sa_sigma   = _head("head_spray_angle")    # spray angle   degrees
        hd_mu,   hd_sigma   = _head("head_hit_distance")   # hit distance  feet

        return dict(
            launch_speed_mu   = ls_mu,
            launch_speed_sigma= ls_sigma,
            launch_angle_mu   = la_mu,
            launch_angle_sigma= la_sigma,
            spray_angle_mu    = sa_mu,
            spray_angle_sigma = sa_sigma,
            hit_distance_mu   = hd_mu,
            hit_distance_sigma= hd_sigma,
        )


class ParkEmbedding(nn.Module):
    """Embeds park_id ∈ {0..30} → R^16."""
    n_parks: int = N_PARKS
    embed_dim: int = D_PARK_EMB

    @nn.compact
    def __call__(self, park_id: jnp.ndarray) -> jnp.ndarray:
        # park_id: (B, T) int32
        return nn.Embed(num_embeddings=self.n_parks, features=self.embed_dim)(park_id)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class BattedBallOutcomes:
    launch_speed: jnp.ndarray    # (B, T) float  — masked to in_play
    launch_angle: jnp.ndarray    # (B, T) float
    spray_angle: jnp.ndarray     # (B, T) float
    hit_distance: jnp.ndarray    # (B, T) float


# ---------------------------------------------------------------------------
# NumPyro model fragment
# ---------------------------------------------------------------------------

def batted_ball_numpyro(
    shared_context: jnp.ndarray,        # (B, T, 128)
    pitch_execution: jnp.ndarray,       # (B, T, 8)
    batter_z: jnp.ndarray,              # (B, T, 64)
    pitcher_z: jnp.ndarray,             # (B, T, 64)
    park_id: jnp.ndarray,               # (B, T) int32
    in_play_mask: jnp.ndarray,          # (B, T) bool/int  — 1 if ball in play
    # observed values for teacher-forcing (None = free rollout)
    obs_launch_speed: Optional[jnp.ndarray]  = None,   # (B, T)
    obs_launch_angle: Optional[jnp.ndarray]  = None,   # (B, T)
    obs_spray_angle: Optional[jnp.ndarray]   = None,   # (B, T)
    obs_hit_distance: Optional[jnp.ndarray]  = None,   # (B, T)
    launch_speed_mask: Optional[jnp.ndarray] = None,   # (B, T) bool
    launch_angle_mask: Optional[jnp.ndarray] = None,   # (B, T) bool
    spray_angle_mask: Optional[jnp.ndarray]  = None,   # (B, T) bool
    hit_distance_mask: Optional[jnp.ndarray] = None,   # (B, T) bool
    name: str = "batted_ball_net",
    park_emb_name: str = "park_embedding",
) -> BattedBallOutcomes:
    """
    NumPyro model fragment for batted-ball physics outputs.

    All four continuous quantities are sampled via dist.Normal.
    The continuous likelihood is evaluated only where the ball was in play and
    that particular Statcast measurement exists.  The field-specific masks are
    optional for backwards compatibility; when omitted, every in-play value is
    considered observed.  Values at other positions are still sampled in free
    rollout, then zeroed before they reach the transition model.

    Parameters
    ----------
    in_play_mask : (B, T) int {0, 1}  — 1 means the ball was put in play
    """
    B, T = park_id.shape

    # --- park embedding (registered as its own NumPyro site) ---
    park_embedder = flax_module(
        park_emb_name,
        ParkEmbedding(),
        jnp.zeros((B, T), dtype=jnp.int32),
    )
    park_emb = park_embedder(park_id)   # (B, T, 16)

    # --- batted ball net ---
    net = flax_module(
        name,
        BattedBallNet(),
        jnp.ones((B, T, D_SHARED_CONTEXT)),
        jnp.ones((B, T, D_EXECUTION)),
        jnp.ones((B, T, D_PLAYER_EMB)),
        jnp.ones((B, T, D_PLAYER_EMB)),
        jnp.ones((B, T, D_PARK_EMB)),
    )

    raw = net(shared_context, pitch_execution, batter_z, pitcher_z, park_emb)

    ipm = in_play_mask.astype(jnp.float32)    # (B, T)

    def _sample_continuous(site, mu, sigma, obs, field_mask):
        """Sample one Statcast field without scoring placeholder values."""
        observed = in_play_mask.astype(bool)
        if field_mask is not None:
            observed = observed & field_mask.astype(bool)
        # ``mask`` changes only the log density, not the sampled value.  That
        # preserves a dense (B,T) return shape for downstream JAX code while
        # excluding null/out-of-play placeholders from the training objective.
        with handlers.mask(mask=observed):
            return numpyro.sample(site, dist.Normal(mu, sigma), obs=obs)

    # ------------------------------------------------------------------ #
    # 1. launch_speed — Normal                                            #
    # ------------------------------------------------------------------ #
    launch_speed_raw = _sample_continuous(
        "launch_speed", raw["launch_speed_mu"], raw["launch_speed_sigma"],
        obs_launch_speed, launch_speed_mask,
    )  # (B, T)
    launch_speed = launch_speed_raw * ipm   # zero out non-in-play positions

    # ------------------------------------------------------------------ #
    # 2. launch_angle — Normal                                            #
    # ------------------------------------------------------------------ #
    launch_angle_raw = _sample_continuous(
        "launch_angle", raw["launch_angle_mu"], raw["launch_angle_sigma"],
        obs_launch_angle, launch_angle_mask,
    )  # (B, T)
    launch_angle = launch_angle_raw * ipm

    # ------------------------------------------------------------------ #
    # 3. spray_angle — Normal                                             #
    # ------------------------------------------------------------------ #
    spray_angle_raw = _sample_continuous(
        "spray_angle", raw["spray_angle_mu"], raw["spray_angle_sigma"],
        obs_spray_angle, spray_angle_mask,
    )  # (B, T)
    spray_angle = spray_angle_raw * ipm

    # ------------------------------------------------------------------ #
    # 4. hit_distance — Normal                                            #
    # ------------------------------------------------------------------ #
    hit_distance_raw = _sample_continuous(
        "hit_distance", raw["hit_distance_mu"], raw["hit_distance_sigma"],
        obs_hit_distance, hit_distance_mask,
    )  # (B, T)
    hit_distance = hit_distance_raw * ipm

    return BattedBallOutcomes(
        launch_speed = launch_speed,
        launch_angle = launch_angle,
        spray_angle  = spray_angle,
        hit_distance = hit_distance,
    )
