from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import flax_module
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FATIGUE_DIM = 16          # hidden fatigue state dimension
PITCH_PKG_DIM = 12        # pitch_package_features dimensionality expected by cell


# ---------------------------------------------------------------------------
# Flax: GRU-based fatigue cell
# ---------------------------------------------------------------------------

class FatigueCell(nn.Module):
    """
    Single-step fatigue state updater based on a GRU cell.

    Inputs  (all have a leading batch dimension B)
    ------
    prev_state              : float32  (B, FATIGUE_DIM)
    pitcher_z               : float32  (B, D_pitcher)    latent pitcher embedding
    pitch_package_features  : float32  (B, PITCH_PKG_DIM) pitch-level features
                              (e.g. velocity, spin, pitch_type_onehot, release coords)
    game_state_scalar       : float32  (B, 1)             e.g. normalised pitch count

    Output
    ------
    next_state : float32  (B, FATIGUE_DIM)   — deterministic GRU output
                 Stochastic noise is added *outside* this module in the
                 NumPyro model via numpyro.sample.
    """
    fatigue_dim: int = FATIGUE_DIM

    @nn.compact
    def __call__(
        self,
        prev_state: jnp.ndarray,              # (B, fatigue_dim)
        pitcher_z: jnp.ndarray,               # (B, D_pitcher)
        pitch_package_features: jnp.ndarray,  # (B, PITCH_PKG_DIM)
        game_state_scalar: jnp.ndarray,       # (B, 1)
    ) -> jnp.ndarray:                          # (B, fatigue_dim)

        # Fuse all non-state inputs into a single input vector.
        x = jnp.concatenate(
            [pitcher_z, pitch_package_features, game_state_scalar], axis=-1
        )  # (B, D_pitcher + PITCH_PKG_DIM + 1)

        # Project input to fatigue_dim for GRU compatibility.
        x_proj = nn.Dense(self.fatigue_dim, name="input_proj")(x)  # (B, fatigue_dim)

        # ----- GRU equations (manual implementation to keep shapes explicit) -----
        # Combined input+hidden for gate computations.
        xh = jnp.concatenate([x_proj, prev_state], axis=-1)  # (B, 2*fatigue_dim)

        r = nn.sigmoid(nn.Dense(self.fatigue_dim, name="reset_gate")(xh))
        z = nn.sigmoid(nn.Dense(self.fatigue_dim, name="update_gate")(xh))

        xh_reset = jnp.concatenate([x_proj, r * prev_state], axis=-1)
        h_cand   = jnp.tanh(nn.Dense(self.fatigue_dim, name="candidate")(xh_reset))

        next_state = (1.0 - z) * prev_state + z * h_cand  # (B, fatigue_dim)
        return next_state


# ---------------------------------------------------------------------------
# NumPyro: stochastic fatigue rollout
# ---------------------------------------------------------------------------

def fatigue_rollout(
    pitcher_z: jnp.ndarray,               # (B, T, D_pitcher)
    pitch_package_features: jnp.ndarray,  # (B, T, PITCH_PKG_DIM)
    game_state_scalar: jnp.ndarray,       # (B, T, 1)
    pitch_count_plate_appearance: jnp.ndarray,  # (B, T) int — resets fatigue at 0
    obs_fatigue: Optional[jnp.ndarray] = None,  # (B, T, FATIGUE_DIM) or None
    name_prefix: str = "fatigue",
) -> jnp.ndarray:                          # (B, T, FATIGUE_DIM)
    """
    NumPyro model fragment: rolls out the stochastic fatigue process over T
    pitch steps.

    Stochastic sites
    ----------------
    {name_prefix}_sigma   — HalfNormal(0.1), global noise scale
    {name_prefix}_init_0  — Normal(0, 0.3), initial fatigue state per appearance
    {name_prefix}_eps_{t} — Normal(0, sigma), additive noise at each step

    Parameters
    ----------
    pitch_count_plate_appearance : (B, T) int32
        Pitch count within the current plate appearance.  When this equals 0,
        a new plate appearance (and thus pitcher appearance) boundary is
        assumed and the fatigue state is reset to a freshly-sampled initial
        value.

    Returns
    -------
    fatigue_states : float32  (B, T, FATIGUE_DIM)
    """
    B, T, D_pitcher = pitcher_z.shape

    # --- global noise-scale parameter ---
    sigma_fatigue = numpyro.sample(
        f"{name_prefix}_sigma",
        dist.HalfNormal(0.1),
    )  # scalar

    # Register the Flax FatigueCell as a NumPyro parameter site.
    cell = flax_module(
        f"{name_prefix}_cell",
        FatigueCell(fatigue_dim=FATIGUE_DIM),
        input_shape=[
            (B, FATIGUE_DIM),
            (B, D_pitcher),
            (B, PITCH_PKG_DIM),
            (B, 1),
        ],
    )

    # --- initial fatigue state (per pitcher appearance) ---
    # Shape (B, FATIGUE_DIM): one per sequence in the batch.
    fatigue_init = numpyro.sample(
        f"{name_prefix}_init_0",
        dist.Normal(
            jnp.zeros((B, FATIGUE_DIM)),
            0.3 * jnp.ones((B, FATIGUE_DIM)),
        ),
    )  # (B, FATIGUE_DIM)

    fatigue_states = []
    state = fatigue_init  # (B, FATIGUE_DIM)

    for t in range(T):
        # --- reset at appearance boundary (pitch_count_plate_appearance == 0) ---
        is_new_appearance = (pitch_count_plate_appearance[:, t] == 0)  # (B,)
        # Broadcast reset mask over fatigue dim.
        reset_mask = is_new_appearance[:, None]                         # (B, 1)

        # For positions that start a new appearance, sample a new init state.
        new_init = numpyro.sample(
            f"{name_prefix}_init_{t}_reset",
            dist.Normal(
                jnp.zeros((B, FATIGUE_DIM)),
                0.3 * jnp.ones((B, FATIGUE_DIM)),
            ),
        )  # (B, FATIGUE_DIM)

        state = jnp.where(reset_mask, new_init, state)  # (B, FATIGUE_DIM)

        # --- deterministic GRU step ---
        det_next = cell(
            state,
            pitcher_z[:, t, :],               # (B, D_pitcher)
            pitch_package_features[:, t, :],  # (B, PITCH_PKG_DIM)
            game_state_scalar[:, t, :],        # (B, 1)
        )  # (B, FATIGUE_DIM)

        # --- stochastic noise ---
        obs_t = obs_fatigue[:, t, :] if obs_fatigue is not None else None
        eps = numpyro.sample(
            f"{name_prefix}_eps_{t}",
            dist.Normal(
                jnp.zeros((B, FATIGUE_DIM)),
                sigma_fatigue * jnp.ones((B, FATIGUE_DIM)),
            ),
            obs=obs_t,
        )  # (B, FATIGUE_DIM)  — obs=None → sampled freely

        state = det_next + eps             # (B, FATIGUE_DIM)
        fatigue_states.append(state)

    return jnp.stack(fatigue_states, axis=1)  # (B, T, FATIGUE_DIM)
