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
    error_flag: jnp.ndarray         # (B, T) int {0,1}
    runs_scored: jnp.ndarray        # (B, T) int {0..4}
    base_state_after: jnp.ndarray   # (B, T) int {0..7}
    outs_added: jnp.ndarray         # (B, T) int {0..3}

    # Side events
    wild_pitch: jnp.ndarray         # (B, T) int {0,1}
    passed_ball: jnp.ndarray        # (B, T) int {0,1}
    balk: jnp.ndarray               # (B, T) int {0,1}

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
    # Strikeout: always adds exactly 1 out.
    det_outs_added_so = jnp.where(outcome_type == 0, 1, 0)

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
    error_logit          : (B, T)       Bernoulli
    runs_logits          : (B, T, 5)    Categorical
    base_after_logits    : (B, T, 8)    Categorical
    outs_added_logits    : (B, T, 4)    Categorical  — {0,1,2,3}
    wild_pitch_logit     : (B, T)       Bernoulli
    passed_ball_logit    : (B, T)       Bernoulli
    balk_logit           : (B, T)       Bernoulli
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

        error_logit        = nn.Dense(1, name="head_error")(h)[..., 0]          # (B,T)
        runs_logits        = nn.Dense(MAX_RUNS, name="head_runs")(h)            # (B,T,5)
        base_after_logits  = nn.Dense(BASE_STATE_DIM, name="head_base")(h)     # (B,T,8)
        outs_added_logits  = nn.Dense(MAX_OUTS_ADDED + 1, name="head_outs")(h) # (B,T,4)
        wild_pitch_logit   = nn.Dense(1, name="head_wp")(h)[..., 0]            # (B,T)
        passed_ball_logit  = nn.Dense(1, name="head_pb")(h)[..., 0]            # (B,T)
        balk_logit         = nn.Dense(1, name="head_balk")(h)[..., 0]          # (B,T)

        return dict(
            error_logit        = error_logit,
            runs_logits        = runs_logits,
            base_after_logits  = base_after_logits,
            outs_added_logits  = outs_added_logits,
            wild_pitch_logit   = wild_pitch_logit,
            passed_ball_logit  = passed_ball_logit,
            balk_logit         = balk_logit,
        )


# ---------------------------------------------------------------------------
# NumPyro model fragment
# ---------------------------------------------------------------------------

def transition_numpyro(
    shared_context: jnp.ndarray,        # (B, T, 128)
    batted_ball_features: jnp.ndarray,  # (B, T, 4)  — (ls, la, sa, hd)
    base_state: jnp.ndarray,            # (B, T) int {0..7}
    outs: jnp.ndarray,                  # (B, T) int {0,1,2}
    outcome_type: jnp.ndarray,          # (B, T) int  — see rule_engine_step codes
    in_play_mask: jnp.ndarray,          # (B, T) bool/int
    # observed values for teacher-forcing
    obs_error_flag: Optional[jnp.ndarray]       = None,   # (B, T) int
    obs_runs_scored: Optional[jnp.ndarray]      = None,   # (B, T) int
    obs_base_state_after: Optional[jnp.ndarray] = None,   # (B, T) int
    obs_outs_added: Optional[jnp.ndarray]       = None,   # (B, T) int
    obs_wild_pitch: Optional[jnp.ndarray]       = None,   # (B, T) int
    obs_passed_ball: Optional[jnp.ndarray]      = None,   # (B, T) int
    obs_balk: Optional[jnp.ndarray]             = None,   # (B, T) int
    name: str = "transition_net",
) -> TransitionState:
    """
    State transition model: combines the pure-JAX rule engine with learned
    probabilistic components for uncertain events.

    Stochastic sites
    ----------------
    error_flag       — Bernoulli (in-play positions only; zeroed elsewhere)
    runs_scored      — Categorical(5) = {0,1,2,3,4}
    base_state_after — Categorical(8)
    outs_added       — Categorical(4) = {0,1,2,3}
    wild_pitch       — Bernoulli (always; not gated on in_play)
    passed_ball      — Bernoulli
    balk             — Bernoulli

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

    ipm   = in_play_mask.astype(jnp.float32)    # (B, T)
    no_ip = 1.0 - ipm                            # (B, T)

    # ------------------------------------------------------------------ #
    # 1. error_flag  — Bernoulli (gated to in-play)                       #
    # ------------------------------------------------------------------ #
    # Mask logit to large negative for non-in-play positions.
    err_logit_masked = raw["error_logit"] * ipm - 1e4 * no_ip
    error_flag = numpyro.sample(
        "error_flag",
        dist.Bernoulli(logits=err_logit_masked),
        obs=obs_error_flag,
    )  # (B, T)

    # ------------------------------------------------------------------ #
    # 2. runs_scored  — Categorical(5)                                    #
    # ------------------------------------------------------------------ #
    runs_probs = jax.nn.softmax(raw["runs_logits"], axis=-1)   # (B, T, 5)
    runs_scored = numpyro.sample(
        "runs_scored",
        dist.Categorical(probs=runs_probs),
        obs=obs_runs_scored,
    )  # (B, T) int {0..4}

    # ------------------------------------------------------------------ #
    # 3. base_state_after  — Categorical(8)                               #
    # ------------------------------------------------------------------ #
    base_after_probs = jax.nn.softmax(raw["base_after_logits"], axis=-1)  # (B, T, 8)
    base_state_after = numpyro.sample(
        "base_state_after",
        dist.Categorical(probs=base_after_probs),
        obs=obs_base_state_after,
    )  # (B, T) int {0..7}

    # ------------------------------------------------------------------ #
    # 4. outs_added  — Categorical(4)  {0,1,2,3}                         #
    # ------------------------------------------------------------------ #
    outs_added_probs = jax.nn.softmax(raw["outs_added_logits"], axis=-1)  # (B, T, 4)
    outs_added = numpyro.sample(
        "outs_added",
        dist.Categorical(probs=outs_added_probs),
        obs=obs_outs_added,
    )  # (B, T) int {0..3}

    # ------------------------------------------------------------------ #
    # 5. wild_pitch  — Bernoulli                                          #
    # ------------------------------------------------------------------ #
    wild_pitch = numpyro.sample(
        "wild_pitch",
        dist.Bernoulli(logits=raw["wild_pitch_logit"]),
        obs=obs_wild_pitch,
    )  # (B, T)

    # ------------------------------------------------------------------ #
    # 6. passed_ball  — Bernoulli                                         #
    # ------------------------------------------------------------------ #
    passed_ball = numpyro.sample(
        "passed_ball",
        dist.Bernoulli(logits=raw["passed_ball_logit"]),
        obs=obs_passed_ball,
    )  # (B, T)

    # ------------------------------------------------------------------ #
    # 7. balk  — Bernoulli                                                #
    # ------------------------------------------------------------------ #
    balk = numpyro.sample(
        "balk",
        dist.Bernoulli(logits=raw["balk_logit"]),
        obs=obs_balk,
    )  # (B, T)

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
        error_flag       = error_flag,
        runs_scored      = runs_scored,
        base_state_after = base_state_after,
        outs_added       = outs_added,
        wild_pitch       = wild_pitch,
        passed_ball      = passed_ball,
        balk             = balk,
        outs_after       = outs_after,
        inning_over      = inning_over,
    )
