from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import flax_module
from typing import Optional, NamedTuple
import dataclasses


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# One-hot base-state dimension: 8 possible runner configurations (3-bit).
BASE_STATE_DIM = 8

# Number of defensive alignment categories: standard / shift / extreme.
N_ALIGNMENT = 3

# Input feature dimensionality
#   inning(1) + outs(1) + score_diff(1) + pitch_count_game(1)
#   + base_state(8) + pitcher_z(D) + batter_z(D)
# D is injected at call time via concatenation.


# ---------------------------------------------------------------------------
# Flax: ManagerNet
# ---------------------------------------------------------------------------

class ManagerNet(nn.Module):
    """
    MLP that maps game-state + player embeddings to per-decision logits.

    Inputs (all concatenated along last axis)
    -----------------------------------------
    inning                : float32  (B, T, 1)   — normalised [0,1]
    outs                  : float32  (B, T, 1)   — 0/1/2 normalised
    score_diff            : float32  (B, T, 1)   — runs_away - runs_home
    pitch_count_game      : float32  (B, T, 1)   — normalised
    base_state_onehot     : float32  (B, T, 8)   — one-hot over 8 configs
    pitcher_z             : float32  (B, T, D)
    batter_z              : float32  (B, T, D)

    Outputs (returned as individual logit tensors)
    -----------------------------------------------
    pitching_change_logit      : (B, T, 1)  — Bernoulli
    steal_logit                : (B, T, 1)  — Bernoulli
    runner_send_logit          : (B, T, 1)  — Bernoulli
    alignment_logits           : (B, T, 3)  — Categorical
    """
    hidden_dim: int = 256

    @nn.compact
    def __call__(
        self,
        inning: jnp.ndarray,            # (B, T, 1)
        outs: jnp.ndarray,              # (B, T, 1)
        score_diff: jnp.ndarray,        # (B, T, 1)
        pitch_count_game: jnp.ndarray,  # (B, T, 1)
        base_state_onehot: jnp.ndarray, # (B, T, 8)
        pitcher_z: jnp.ndarray,         # (B, T, D)
        batter_z: jnp.ndarray,          # (B, T, D)
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

        # --- concatenate all features ---
        x = jnp.concatenate(
            [inning, outs, score_diff, pitch_count_game,
             base_state_onehot, pitcher_z, batter_z],
            axis=-1,
        )  # (B, T, 4 + 8 + 2*D)

        # --- shared trunk ---
        h = nn.Dense(self.hidden_dim)(x)
        h = nn.LayerNorm()(h)
        h = nn.relu(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.LayerNorm()(h)
        h = nn.relu(h)

        # --- decision heads ---
        pitching_change_logit = nn.Dense(1, name="head_pitching")(h)    # (B, T, 1)
        steal_logit           = nn.Dense(1, name="head_steal")(h)       # (B, T, 1)
        runner_send_logit     = nn.Dense(1, name="head_send")(h)        # (B, T, 1)
        alignment_logits      = nn.Dense(N_ALIGNMENT, name="head_align")(h)  # (B, T, 3)

        return pitching_change_logit, steal_logit, runner_send_logit, alignment_logits


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ManagerDecisions:
    pitching_change: jnp.ndarray       # (B, T)  int {0,1}
    steal_attempt: jnp.ndarray         # (B, T)  int {0,1}
    runner_send: jnp.ndarray           # (B, T)  int {0,1}
    defensive_alignment: jnp.ndarray   # (B, T)  int {0,1,2}
    # Soft probabilities — same shape as above but float, safe after enumeration
    pc_prob: jnp.ndarray               # (B, T)  float  P(pitching_change=1)
    st_prob: jnp.ndarray               # (B, T)  float  P(steal=1)
    rs_prob: jnp.ndarray               # (B, T)  float  P(runner_send=1)
    al_probs: jnp.ndarray              # (B, T, 3) float


# ---------------------------------------------------------------------------
# NumPyro model fragment
# ---------------------------------------------------------------------------

def manager_decisions_numpyro(
    inning: jnp.ndarray,               # (B, T, 1)  float
    outs: jnp.ndarray,                 # (B, T, 1)  float
    score_diff: jnp.ndarray,           # (B, T, 1)  float
    pitch_count_game: jnp.ndarray,     # (B, T, 1)  float
    base_state_onehot: jnp.ndarray,    # (B, T, 8)  float one-hot
    pitcher_z: jnp.ndarray,            # (B, T, D)
    batter_z: jnp.ndarray,             # (B, T, D)
    # masks
    steal_valid_mask: jnp.ndarray,     # (B, T)  bool — runner on base
    # observed decisions for teacher-forcing (None = free rollout)
    obs_pitching_change: Optional[jnp.ndarray] = None,  # (B, T) int
    obs_steal: Optional[jnp.ndarray] = None,            # (B, T) int
    obs_runner_send: Optional[jnp.ndarray] = None,      # (B, T) int
    obs_alignment: Optional[jnp.ndarray] = None,        # (B, T) int
    name: str = "manager_net",
    hidden_dim: int = 256,
) -> ManagerDecisions:
    """
    NumPyro model fragment for manager decisions.

    All four decisions are stochastically sampled via numpyro.sample.
    When obs_* is provided (teacher-forcing / SVI training), the site scores
    the likelihood of the observed decision.  When obs_* is None (free
    rollout / posterior predictive), the site draws from the learned
    distribution.

    Masks
    -----
    steal_valid_mask  — steal is only meaningful when a runner is on base.
        When mask=False the steal logit is zeroed so the model learns a
        near-zero probability automatically; no hard masking of the sample
        site is needed.
    """
    B, T, D = pitcher_z.shape

    # --- register Flax module ---
    net = flax_module(
        name,
        ManagerNet(hidden_dim=hidden_dim),
        jnp.ones((B, T, 1)),
        jnp.ones((B, T, 1)),
        jnp.ones((B, T, 1)),
        jnp.ones((B, T, 1)),
        jnp.ones((B, T, 8)),
        jnp.ones((B, T, D)),
        jnp.ones((B, T, D)),
    )

    pc_logit, st_logit, rs_logit, al_logits = net(
        inning, outs, score_diff, pitch_count_game,
        base_state_onehot, pitcher_z, batter_z,
    )
    # pc_logit : (B, T, 1)
    # st_logit : (B, T, 1)
    # rs_logit : (B, T, 1)
    # al_logits: (B, T, 3)

    # --- apply steal validity mask ---
    # Zero the logit when no runner is on base so the network naturally
    # outputs near-zero probability; the sample site still fires every step.
    steal_mask_f = steal_valid_mask[..., None].astype(jnp.float32)  # (B, T, 1)
    # Large negative shift when invalid → effectively forces P(steal)≈0.
    st_logit = st_logit * steal_mask_f + (steal_mask_f - 1.0) * 1e4

    # --- squeeze trailing 1-dim for scalar logits ---
    pc_logit_sq = pc_logit[..., 0]   # (B, T)
    st_logit_sq = st_logit[..., 0]   # (B, T)
    rs_logit_sq = rs_logit[..., 0]   # (B, T)

    # ------------------------------------------------------------------ #
    # Stochastic sample sites                                             #
    # ------------------------------------------------------------------ #

    # 1. pitching_change: Bernoulli
    pitching_change = numpyro.sample(
        "pitching_change",
        dist.Bernoulli(logits=pc_logit_sq),
        obs=obs_pitching_change,
    )  # (B, T)  int

    # 2. steal_attempt: Bernoulli (masked logit)
    steal_attempt = numpyro.sample(
        "steal_attempt",
        dist.Bernoulli(logits=st_logit_sq),
        obs=obs_steal,
    )  # (B, T)  int

    # 3. runner_send: Bernoulli
    runner_send = numpyro.sample(
        "runner_send",
        dist.Bernoulli(logits=rs_logit_sq),
        obs=obs_runner_send,
    )  # (B, T)  int

    # 4. defensive_alignment: Categorical(3) — standard/shift/extreme
    al_probs = jax.nn.softmax(al_logits, axis=-1)  # (B, T, 3)
    defensive_alignment = numpyro.sample(
        "defensive_alignment",
        dist.Categorical(probs=al_probs),
        obs=obs_alignment,
    )  # (B, T)  int in {0,1,2}

    return ManagerDecisions(
        pitching_change=pitching_change,
        steal_attempt=steal_attempt,
        runner_send=runner_send,
        defensive_alignment=defensive_alignment,
        # Soft probabilities — safe to use downstream even with enumeration
        pc_prob  = jax.nn.sigmoid(pc_logit_sq),
        st_prob  = jax.nn.sigmoid(st_logit_sq),
        rs_prob  = jax.nn.sigmoid(rs_logit_sq),
        al_probs = al_probs,
    )
