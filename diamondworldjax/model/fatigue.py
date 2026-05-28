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
    pitch_count_plate_appearance: jnp.ndarray,  # (B, T) int — resets state at 0
    obs_fatigue: Optional[jnp.ndarray] = None,  # ignored for v0 (deterministic)
    name_prefix: str = "fatigue",
) -> jnp.ndarray:                          # (B, T, FATIGUE_DIM)
    """
    Deterministic GRU-based fatigue rollout (v0).

    The FatigueCell weights are registered as a NumPyro param site via
    flax_module so they are optimised by SVI.  No stochastic latents are
    introduced here, avoiding plate-declaration complexity while still
    allowing the GRU to capture pitcher fatigue dynamics.

    Returns
    -------
    fatigue_states : float32  (B, T, FATIGUE_DIM)
    """
    B, T, D_pitcher = pitcher_z.shape

    # Register the Flax FatigueCell as a NumPyro parameter site.
    cell = flax_module(
        f"{name_prefix}_cell",
        FatigueCell(fatigue_dim=FATIGUE_DIM),
        jnp.ones((B, FATIGUE_DIM)),
        jnp.ones((B, D_pitcher)),
        jnp.ones((B, PITCH_PKG_DIM)),
        jnp.ones((B, 1)),
    )

    # Use lax.scan instead of a Python loop so JAX compiles one step and
    # repeats it — O(1) compile time instead of O(T) unrolled graph.
    def step_fn(state, t_inputs):
        pc_pa, pz, ppf, gs = t_inputs          # scalars/slices at time t
        is_new = (pc_pa == 0)[:, None]         # (B, 1) bool
        state  = jnp.where(is_new, jnp.zeros_like(state), state)
        state  = cell(state, pz, ppf, gs)
        return state, state                    # (carry, output)

    init_state = jnp.zeros((B, FATIGUE_DIM))

    # Pack per-timestep inputs as (T, ...) leading axis for lax.scan
    t_inputs = (
        pitch_count_plate_appearance.transpose(1, 0),   # (T, B)
        pitcher_z.transpose(1, 0, 2),                   # (T, B, D)
        pitch_package_features.transpose(1, 0, 2),      # (T, B, Pkg)
        game_state_scalar.transpose(1, 0, 2),           # (T, B, 1)
    )

    _, fatigue_states = jax.lax.scan(step_fn, init_state, t_inputs)
    return fatigue_states.transpose(1, 0, 2)  # (B, T, FATIGUE_DIM)
