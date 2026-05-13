from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import flax_module
from typing import Optional, Dict, Any
import dataclasses


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D_MODEL          = 128    # shared_context dimension
D_EXECUTION      = 8      # pitch_execution features (e.g. velocity, break, spin)
D_CONTEXT_TOTAL  = D_MODEL + D_EXECUTION   # 136 — input to all hurdle heads

N_PITCH_TYPES    = 8      # categorical pitch-type vocabulary


# ---------------------------------------------------------------------------
# Reusable head building blocks
# ---------------------------------------------------------------------------

class BinaryHead(nn.Module):
    """Linear head → scalar logit for Bernoulli."""
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (..., D_CONTEXT_TOTAL)
        h = nn.Dense(self.hidden_dim)(x)
        h = nn.relu(h)
        logit = nn.Dense(1)(h)[..., 0]   # (...,)
        return logit


class CategoricalHead(nn.Module):
    """Linear head → K logits for Categorical."""
    n_classes: int
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (..., D_CONTEXT_TOTAL)
        h = nn.Dense(self.hidden_dim)(x)
        h = nn.relu(h)
        logits = nn.Dense(self.n_classes)(h)   # (..., K)
        return logits


class ContinuousHead(nn.Module):
    """Linear head → (mu, log_sigma) for Normal."""
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # x: (..., D_CONTEXT_TOTAL)
        h = nn.Dense(self.hidden_dim)(x)
        h = nn.relu(h)
        params = nn.Dense(2)(h)               # (..., 2)
        mu        = params[..., 0]
        log_sigma = params[..., 1]
        sigma = jax.nn.softplus(log_sigma) + 1e-4
        return mu, sigma


class UmpireNoiseLayer(nn.Module):
    """
    Small learned perturbation applied to called-strike logit to model
    umpire-specific bias.  Conditioned on context only (umpire identity
    could be added later).
    """
    hidden_dim: int = 32

    @nn.compact
    def __call__(self, base_logit: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        # base_logit: (...,)   x: (..., D_CONTEXT_TOTAL)
        noise = nn.Dense(self.hidden_dim)(x)
        noise = nn.tanh(noise)
        noise = nn.Dense(1)(noise)[..., 0]   # (...,)  small correction
        return base_logit + noise


# ---------------------------------------------------------------------------
# Full hurdle network (Flax)
# ---------------------------------------------------------------------------

class HurdleNet(nn.Module):
    """
    All per-pitch head sub-modules as a single Flax module so they are
    registered under one flax_module NumPyro site and share parameter
    initialisation.

    Inputs
    ------
    context  : float32  (B, T, D_MODEL)    — from PitchTransformer
    execution: float32  (B, T, D_EXECUTION) — pitch execution features

    Outputs  (see field names)
    -------
    As a dict of raw logits/params — stochastic sampling happens outside.
    """
    n_pitch_types: int = N_PITCH_TYPES
    hidden_dim: int = 64

    @nn.compact
    def __call__(
        self,
        context: jnp.ndarray,    # (B, T, D_MODEL)
        execution: jnp.ndarray,  # (B, T, D_EXECUTION)
    ) -> Dict[str, Any]:

        x = jnp.concatenate([context, execution], axis=-1)  # (B, T, 136)

        # --- hurdle nodes ---
        swing_logit          = BinaryHead(self.hidden_dim, name="head_swing")(x)
        called_strike_logit  = BinaryHead(self.hidden_dim, name="head_cs_base")(x)
        called_strike_logit  = UmpireNoiseLayer(name="head_cs_umpire")(called_strike_logit, x)
        contact_logit        = BinaryHead(self.hidden_dim, name="head_contact")(x)
        foul_logit           = BinaryHead(self.hidden_dim, name="head_foul")(x)

        # --- pitch type distribution ---
        pitch_type_logits    = CategoricalHead(self.n_pitch_types, self.hidden_dim,
                                               name="head_pitch_type")(x)

        # --- continuous pitch execution outcomes ---
        plate_x_mu,    plate_x_sigma    = ContinuousHead(self.hidden_dim, name="head_px")(x)
        plate_z_mu,    plate_z_sigma    = ContinuousHead(self.hidden_dim, name="head_pz")(x)
        speed_mu,      speed_sigma      = ContinuousHead(self.hidden_dim, name="head_speed")(x)

        return dict(
            swing_logit         = swing_logit,           # (B, T)
            called_strike_logit = called_strike_logit,   # (B, T)
            contact_logit       = contact_logit,         # (B, T)
            foul_logit          = foul_logit,             # (B, T)
            pitch_type_logits   = pitch_type_logits,     # (B, T, K)
            plate_x_mu          = plate_x_mu,            # (B, T)
            plate_x_sigma       = plate_x_sigma,         # (B, T)
            plate_z_mu          = plate_z_mu,            # (B, T)
            plate_z_sigma       = plate_z_sigma,         # (B, T)
            speed_mu            = speed_mu,              # (B, T)
            speed_sigma         = speed_sigma,           # (B, T)
        )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class HurdleOutcomes:
    # Pitch type and location (sampled first — before swing decision)
    pitch_type: jnp.ndarray      # (B, T)  int {0..K-1}
    plate_x: jnp.ndarray         # (B, T)  float
    plate_z: jnp.ndarray         # (B, T)  float
    release_speed: jnp.ndarray   # (B, T)  float

    # Swing/contact tree
    swing: jnp.ndarray           # (B, T)  int {0,1}
    called_strike: jnp.ndarray   # (B, T)  int {0,1}  (valid when swing==0)
    contact: jnp.ndarray         # (B, T)  int {0,1}  (valid when swing==1)
    foul: jnp.ndarray            # (B, T)  int {0,1}  (valid when contact==1)
    in_play: jnp.ndarray         # (B, T)  int {0,1}  deterministic: contact & ~foul


# ---------------------------------------------------------------------------
# NumPyro model fragment
# ---------------------------------------------------------------------------

def hurdle_numpyro(
    shared_context: jnp.ndarray,          # (B, T, 128)
    pitch_execution: jnp.ndarray,         # (B, T, 8)
    # observed values for teacher-forcing (None = free rollout)
    obs_pitch_type: Optional[jnp.ndarray]   = None,   # (B, T) int
    obs_plate_x: Optional[jnp.ndarray]      = None,   # (B, T) float
    obs_plate_z: Optional[jnp.ndarray]      = None,   # (B, T) float
    obs_release_speed: Optional[jnp.ndarray]= None,   # (B, T) float
    obs_swing: Optional[jnp.ndarray]        = None,   # (B, T) int
    obs_called_strike: Optional[jnp.ndarray]= None,   # (B, T) int
    obs_contact: Optional[jnp.ndarray]      = None,   # (B, T) int
    obs_foul: Optional[jnp.ndarray]         = None,   # (B, T) int
    name: str = "hurdle_net",
) -> HurdleOutcomes:
    """
    Full swing/take hurdle tree.  All outputs are stochastically sampled.

    Sampling order
    --------------
    1. pitch_type   — Categorical
    2. plate_x      — Normal
    3. plate_z      — Normal
    4. release_speed — Normal
    5. swing        — Bernoulli
    6. called_strike — Bernoulli  (conditioned on ~swing; logit zeroed for swing=1)
    7. contact      — Bernoulli  (conditioned on swing; logit zeroed for swing=0)
    8. foul         — Bernoulli  (conditioned on contact; logit zeroed for contact=0)

    in_play is derived deterministically as (contact == 1) & (foul == 0).
    """
    B, T, _ = shared_context.shape

    # --- register Flax module ---
    net = flax_module(
        name,
        HurdleNet(),
        input_shape=[
            (B, T, D_MODEL),
            (B, T, D_EXECUTION),
        ],
    )

    raw = net(shared_context, pitch_execution)

    # ------------------------------------------------------------------ #
    # 1. pitch_type  — Categorical(K=8)                                   #
    # ------------------------------------------------------------------ #
    pt_probs = jax.nn.softmax(raw["pitch_type_logits"], axis=-1)  # (B, T, K)
    pitch_type = numpyro.sample(
        "pitch_type",
        dist.Categorical(probs=pt_probs),
        obs=obs_pitch_type,
    )  # (B, T) int

    # ------------------------------------------------------------------ #
    # 2. plate_x  — Normal                                                #
    # ------------------------------------------------------------------ #
    plate_x = numpyro.sample(
        "plate_x",
        dist.Normal(raw["plate_x_mu"], raw["plate_x_sigma"]),
        obs=obs_plate_x,
    )  # (B, T) float

    # ------------------------------------------------------------------ #
    # 3. plate_z  — Normal                                                #
    # ------------------------------------------------------------------ #
    plate_z = numpyro.sample(
        "plate_z",
        dist.Normal(raw["plate_z_mu"], raw["plate_z_sigma"]),
        obs=obs_plate_z,
    )  # (B, T) float

    # ------------------------------------------------------------------ #
    # 4. release_speed  — Normal                                          #
    # ------------------------------------------------------------------ #
    release_speed = numpyro.sample(
        "release_speed",
        dist.Normal(raw["speed_mu"], raw["speed_sigma"]),
        obs=obs_release_speed,
    )  # (B, T) float

    # ------------------------------------------------------------------ #
    # 5. swing  — Bernoulli                                               #
    # ------------------------------------------------------------------ #
    swing = numpyro.sample(
        "swing",
        dist.Bernoulli(logits=raw["swing_logit"]),
        obs=obs_swing,
    )  # (B, T) int {0,1}

    # ------------------------------------------------------------------ #
    # 6. called_strike  — Bernoulli (valid when swing == 0)               #
    #    Mask: zero logit (→ P≈0.5 prior) when swing==1.                 #
    #    The likelihood contribution is still computed, but the semantics #
    #    are gated outside (see transition model).                        #
    # ------------------------------------------------------------------ #
    cs_logit = raw["called_strike_logit"]  # (B, T)
    # When swing==1 the called-strike node is irrelevant; force logit to a
    # large negative value so the model is not penalised on these positions
    # during training.  Use swing sampled value for masking.
    swing_f = swing.astype(jnp.float32)   # (B, T)
    cs_logit_masked = cs_logit * (1.0 - swing_f) - 1e4 * swing_f

    called_strike = numpyro.sample(
        "called_strike",
        dist.Bernoulli(logits=cs_logit_masked),
        obs=obs_called_strike,
    )  # (B, T) int

    # ------------------------------------------------------------------ #
    # 7. contact  — Bernoulli (valid when swing == 1)                     #
    # ------------------------------------------------------------------ #
    contact_logit = raw["contact_logit"]   # (B, T)
    # Zero out for no-swing positions.
    contact_logit_masked = contact_logit * swing_f - 1e4 * (1.0 - swing_f)

    contact = numpyro.sample(
        "contact",
        dist.Bernoulli(logits=contact_logit_masked),
        obs=obs_contact,
    )  # (B, T) int

    # ------------------------------------------------------------------ #
    # 8. foul  — Bernoulli (valid when contact == 1)                      #
    # ------------------------------------------------------------------ #
    contact_f = contact.astype(jnp.float32)
    foul_logit = raw["foul_logit"]   # (B, T)
    foul_logit_masked = foul_logit * contact_f - 1e4 * (1.0 - contact_f)

    foul = numpyro.sample(
        "foul",
        dist.Bernoulli(logits=foul_logit_masked),
        obs=obs_foul,
    )  # (B, T) int

    # ------------------------------------------------------------------ #
    # in_play  — deterministic                                            #
    # ------------------------------------------------------------------ #
    in_play = (contact == 1) & (foul == 0)  # (B, T) bool

    return HurdleOutcomes(
        pitch_type    = pitch_type,
        plate_x       = plate_x,
        plate_z       = plate_z,
        release_speed = release_speed,
        swing         = swing,
        called_strike = called_strike,
        contact       = contact,
        foul          = foul,
        in_play       = in_play.astype(jnp.int32),
    )
