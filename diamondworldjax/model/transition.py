from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpyro
import numpyro.distributions as dist
import numpyro.handlers as handlers
from numpyro.contrib.module import flax_module
from typing import Optional, NamedTuple
import dataclasses


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_STATE_DIM    = 8     # one-hot dimension for 3-bit base state (0-7)
MAX_OUTS_ADDED    = 3     # categories: 0, 1, 2, 3 outs added on a batted ball
MAX_RUNS          = 5     # categories: 0, 1, 2, 3, 4 runs scored
D_BATTED          = 4     # (launch_speed, launch_angle, spray_angle, hit_distance)
D_SHARED_CONTEXT  = 128


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TransitionState:
    """All post-pitch / post-play state fields."""
    # Stochastically sampled
    pa_outcome: jnp.ndarray         # (B, T) int {0..8}
    runs_scored: jnp.ndarray        # (B, T) int {0..4}
    base_state_after: jnp.ndarray   # (B, T) int {0..7}
    outs_added: jnp.ndarray         # (B, T) int {0..3}

    # Deterministic rule-engine outputs (for downstream convenience)
    outs_after: jnp.ndarray         # (B, T) int {0..2}  — clamped
    inning_over: jnp.ndarray        # (B, T) bool


# ---------------------------------------------------------------------------
# Pure JAX rule engine
# ---------------------------------------------------------------------------

def rule_engine_step(
    base_state: jnp.ndarray,   # (B, T) int {0..7}  — 3-bit encoding
    outs: jnp.ndarray,         # (B, T) int {0,1,2}
    outcome_type: jnp.ndarray, # (B, T) int — see OutcomeType codes below
    runs_scored: jnp.ndarray,  # (B, T) int {0..4}   — from learned model
    outs_added: jnp.ndarray,   # (B, T) int {0..3}   — from learned model
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Deterministic rule-based component of the state transition.

    Outcome type codes
    ------------------
    0 = strikeout     — adds 1 out, no base change
    1 = walk / HBP   — advances all runners if forced (simplified: +1 base)
    2 = in_play       — outs and base state from learned distributions
    3 = no_outcome    — no-op (e.g. non-decisive pitch, foul not ending PA)

    Returns
    -------
    outs_after     : (B, T) int  — clamped [0, 2]
    inning_over    : (B, T) bool — True when outs_after reaches 3+
    det_outs_added : (B, T) int  — deterministic outs added by rule engine
    """
    # Strikeout double plays are possible, so retain an observed/sampled second
    # out while enforcing the ordinary one-out minimum for every strikeout.
    det_outs_added_so = jnp.where(outcome_type == 0, jnp.maximum(1, outs_added), 0)

    # Walk/HBP: adds 0 outs.
    det_outs_added_walk = jnp.where(outcome_type == 1, 0, 0)

    # In-play: outs_added comes from the learned distribution.
    det_outs_added_ip = jnp.where(outcome_type == 2, outs_added, 0)

    # Combine (only one branch fires per position).
    det_outs_added = det_outs_added_so + det_outs_added_walk + det_outs_added_ip
    # For outcome_type==3 (no_outcome) all branches are 0.

    total_outs = outs + det_outs_added          # (B, T)
    inning_over = total_outs >= 3               # (B, T) bool
    outs_after  = jnp.clip(total_outs, 0, 2)   # (B, T) — reset happens externally

    return outs_after, inning_over, det_outs_added


def encode_base_state_onehot(base_state: jnp.ndarray) -> jnp.ndarray:
    """
    Convert 3-bit scalar base state {0..7} to one-hot (B, T, 8).
    """
    return jax.nn.one_hot(base_state, num_classes=8)   # (B, T, 8)


# ---------------------------------------------------------------------------
# Flax: TransitionNet (learned uncertain events)
# ---------------------------------------------------------------------------

class TransitionNet(nn.Module):
    """
    MLP that maps in-play context to logits/params for uncertain events.

    Inputs
    ------
    shared_context       : float32  (B, T, 128)
    batted_ball_features : float32  (B, T, 4)   — (ls, la, sa, hd)
    base_state_onehot    : float32  (B, T, 8)
    outs_onehot          : float32  (B, T, 3)   — one-hot outs {0,1,2}

    Outputs
    -------
    pa_outcome_logits    : (B, T, 9)    Categorical — canonical PA outcome
    runs_logits          : (B, T, 5)    Categorical
    base_after_logits    : (B, T, 8)    Categorical
    outs_added_logits    : (B, T, 4)    Categorical  — {0,1,2,3}
    """
    hidden_dim: int = 256

    @nn.compact
    def __call__(
        self,
        shared_context: jnp.ndarray,        # (B, T, 128)
        batted_ball_features: jnp.ndarray,  # (B, T, 4)
        base_state_onehot: jnp.ndarray,     # (B, T, 8)
        outs_onehot: jnp.ndarray,           # (B, T, 3)
    ) -> dict:

        x = jnp.concatenate(
            [shared_context, batted_ball_features, base_state_onehot, outs_onehot],
            axis=-1,
        )  # (B, T, 128+4+8+3 = 143)

        h = nn.Dense(self.hidden_dim)(x)
        h = nn.LayerNorm()(h)
        h = nn.relu(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.LayerNorm()(h)
        h = nn.relu(h)

        pa_outcome_logits  = nn.Dense(9, name="head_pa_outcome")(h)             # (B,T,9)
        runs_logits        = nn.Dense(MAX_RUNS, name="head_runs")(h)            # (B,T,5)
        base_after_logits  = nn.Dense(BASE_STATE_DIM, name="head_base")(h)     # (B,T,8)
        outs_added_logits  = nn.Dense(MAX_OUTS_ADDED + 1, name="head_outs")(h) # (B,T,4)

        return dict(
            pa_outcome_logits  = pa_outcome_logits,
            runs_logits        = runs_logits,
            base_after_logits  = base_after_logits,
            outs_added_logits  = outs_added_logits,
        )


# ---------------------------------------------------------------------------
# NumPyro model fragment
# ---------------------------------------------------------------------------

def transition_numpyro(
    shared_context: jnp.ndarray,        # (B, T, 128)
    batted_ball_features: jnp.ndarray,  # (B, T, 4)  — (ls, la, sa, hd)
    base_state: jnp.ndarray,            # (B, T) int {0..7}
    outs: jnp.ndarray,                  # (B, T) int {0,1,2}
    in_play_mask: jnp.ndarray,          # (B, T) bool/int
    # observed values for teacher-forcing
    obs_pa_outcome: Optional[jnp.ndarray]       = None,   # (B, T) int
    obs_runs_scored: Optional[jnp.ndarray]      = None,   # (B, T) int
    obs_base_state_after: Optional[jnp.ndarray] = None,   # (B, T) int
    obs_outs_added: Optional[jnp.ndarray]       = None,   # (B, T) int
    pa_outcome_mask: Optional[jnp.ndarray]      = None,   # (B, T) bool
    runs_mask: Optional[jnp.ndarray]            = None,   # (B, T) bool
    base_state_after_mask: Optional[jnp.ndarray] = None,  # (B, T) bool
    outs_added_mask: Optional[jnp.ndarray]      = None,   # (B, T) bool
    name: str = "transition_net",
) -> TransitionState:
    """
    State transition model: combines the pure-JAX rule engine with learned
    probabilistic components for uncertain events.

    Stochastic sites
    ----------------
    pa_outcome       — Categorical(9), scored only at terminal PAs
    runs_scored      — Categorical(5) = {0,1,2,3,4}
    base_state_after — Categorical(8)
    outs_added       — Categorical(4) = {0,1,2,3}
    The rule engine is applied *after* sampling to ensure inning-end logic
    is deterministic given the sampled counts.
    """
    B, T = base_state.shape

    # --- encode inputs ---
    base_state_oh = encode_base_state_onehot(base_state)             # (B, T, 8)
    outs_oh       = jax.nn.one_hot(jnp.clip(outs, 0, 2), num_classes=3)  # (B, T, 3)

    # --- register Flax module ---
    net = flax_module(
        name,
        TransitionNet(),
        jnp.ones((B, T, D_SHARED_CONTEXT)),
        jnp.ones((B, T, D_BATTED)),
        jnp.ones((B, T, BASE_STATE_DIM)),
        jnp.ones((B, T, 3)),
    )

    raw = net(shared_context, batted_ball_features, base_state_oh, outs_oh)

    def _sample(site, distribution, obs, mask):
        if mask is None:
            return numpyro.sample(site, distribution, obs=obs)
        with handlers.mask(mask=mask.astype(bool)):
            return numpyro.sample(site, distribution, obs=obs)

    # ------------------------------------------------------------------ #
    # 1. PA outcome — the central, observed transition target.           #
    # ------------------------------------------------------------------ #
    pa_outcome = _sample(
        "pa_outcome", dist.Categorical(logits=raw["pa_outcome_logits"]),
        obs_pa_outcome, pa_outcome_mask,
    )
    terminal_mask = pa_outcome_mask if pa_outcome_mask is not None else in_play_mask
    sampled_outcome_type = jnp.where(
        pa_outcome == 0, 0,
        jnp.where((pa_outcome == 1) | (pa_outcome == 2), 1,
                  jnp.where(terminal_mask, 2, 3)),
    ).astype(jnp.int32)
    outcome_type = jnp.where(terminal_mask, sampled_outcome_type, 3)

    # ------------------------------------------------------------------ #
    # 2. runs_scored  — Categorical(5)                                    #
    # ------------------------------------------------------------------ #
    runs_probs = jax.nn.softmax(raw["runs_logits"], axis=-1)   # (B, T, 5)
    runs_scored = _sample("runs_scored", dist.Categorical(probs=runs_probs),
                          obs_runs_scored, runs_mask)  # (B, T) int {0..4}

    # ------------------------------------------------------------------ #
    # 3. base_state_after  — Categorical(8)                               #
    # ------------------------------------------------------------------ #
    base_after_probs = jax.nn.softmax(raw["base_after_logits"], axis=-1)  # (B, T, 8)
    base_state_after = _sample("base_state_after", dist.Categorical(probs=base_after_probs),
                               obs_base_state_after, base_state_after_mask)  # (B, T) int {0..7}

    # ------------------------------------------------------------------ #
    # 4. outs_added  — Categorical(4)  {0,1,2,3}                         #
    # ------------------------------------------------------------------ #
    outs_added_probs = jax.nn.softmax(raw["outs_added_logits"], axis=-1)  # (B, T, 4)
    outs_added = _sample("outs_added", dist.Categorical(probs=outs_added_probs),
                         obs_outs_added,
                         in_play_mask if outs_added_mask is None else outs_added_mask)  # (B, T) int {0..3}

    # ------------------------------------------------------------------ #
    # Rule engine: deterministic post-processing                          #
    # ------------------------------------------------------------------ #
    outs_after, inning_over, _ = rule_engine_step(
        base_state  = base_state,
        outs        = outs,
        outcome_type= outcome_type,
        runs_scored = runs_scored,
        outs_added  = outs_added,
    )

    return TransitionState(
        pa_outcome       = pa_outcome,
        runs_scored      = runs_scored,
        base_state_after = base_state_after,
        outs_added       = outs_added,
        outs_after       = outs_after,
        inning_over      = inning_over,
    )
