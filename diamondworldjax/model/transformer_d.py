"""Associative two-stage model of launch and batted-ball outcomes.

D1 predicts launch given the pitch and a trunk that includes players, park,
geometry, environment, and history. D2 predicts the outcome given those same
covariates plus realized launch (observed in training, sampled in rollout).

This factorization does not enforce physical or causal separation: D1 can use
park/environment associations, and D2 can bypass launch via its context inputs.
Geometry ablations measure predictive value, not a causal ball-flight effect.
Enforcing that separation requires a new architecture and retraining; existing
checkpoints deliberately retain their original parameterization.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from .pitchformer import Trunk, N_PITCH_TYPES, _loss_valid

N_BATTED = 6
LAUNCH_DIM = 2


class TransformerD(nn.Module):
    n_pitchers: int
    n_batters: int
    n_parks: int
    d_model: int = 192
    n_layers: int = 4
    n_heads: int = 6
    dropout: float = 0.1
    player_mode: str = "id"
    skill_seasons: int = 1
    residual_dim: int = 0
    pitch_history: bool = False
    position_encoding: str = "learned"
    window_size: int = 0
    observation_masks: bool = False
    c_event_mode: str = "legacy"
    c_support: tuple | None = None

    @nn.compact
    def __call__(self, batch, *, train: bool, decode: bool = False,
                 output_heads: bool = True, ss_override=None, hidden_override=None, encode_only=False):
        h = hidden_override
        if h is None:
            h, _ = Trunk(self.n_pitchers, self.n_batters, self.n_parks, self.d_model,
                         self.n_layers, self.n_heads, self.dropout,
                         player_mode=self.player_mode, skill_seasons=self.skill_seasons,
                         residual_dim=self.residual_dim, pitch_history=self.pitch_history,
                         position_encoding=self.position_encoding, window_size=self.window_size,
                         observation_masks=self.observation_masks, c_event_mode=self.c_event_mode, name="trunk")(
                batch, train=train, decode=decode, ss_override=ss_override)
        if encode_only:
            return {"hidden": h}

        # Decode must advance D's independently-trained causal trunk on every
        # pitch.  Its launch/outcome heads are needed only for balls in play,
        # though, so rollout uses this path for the cheap cache update and calls
        # those heads conditionally from its compiled loop.
        if not output_heads:
            return {"hidden": h}

        pitch = jnp.concatenate([
            jax.nn.one_hot(batch["pitch_type"], N_PITCH_TYPES),
            batch["stuff"],
        ], axis=-1)
        pitch = nn.gelu(nn.Dense(self.d_model, name="pitch_proj")(pitch))
        z = nn.gelu(nn.Dense(self.d_model, name="merge")(
            jnp.concatenate([h, pitch], axis=-1)))

        # D1: launch, including park/environment associations in the trunk.
        l_mu = nn.Dense(LAUNCH_DIM, name="launch_mu")(z)
        l_ls = jnp.clip(nn.Dense(LAUNCH_DIM, name="launch_logsigma")(z), -4.0, 2.0)

        # D2: outcome, from context, pitch, the REALISED launch, and the park and
        # air the trunk already carries through the super-state. Launch enters
        # here explicitly; the direct z path can also predict outcomes independently.
        launch_in = nn.gelu(nn.Dense(64, name="launch_proj")(batch["launch"]))
        y = nn.gelu(nn.Dense(self.d_model, name="outcome_merge")(
            jnp.concatenate([z, launch_in, batch["geom"], batch["ctx"]], axis=-1)))
        out_logits = nn.Dense(N_BATTED, name="outcome")(y)

        return {"launch_mu": l_mu, "launch_logsigma": l_ls,
                "outcome_logits": out_logits}


def _masked_mean(x, m):
    m = m.astype(x.dtype)
    return (x * m).sum() / jnp.maximum(m.sum(), 1.0)


def loss_d(out, batch):
    valid = _loss_valid(batch)

    # D1 scored only where a launch was actually measured.
    mu, ls = out["launch_mu"], out["launch_logsigma"]
    zed = (batch["launch"] - mu) / jnp.exp(ls)
    lp_launch = (-0.5 * zed ** 2 - ls - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)
    nll_launch = -_masked_mean(lp_launch, valid * batch["launch_valid"])

    # D2 scored only on balls in play.
    lp = jax.nn.log_softmax(out["outcome_logits"])
    tgt = batch["batted_out"].astype(jnp.int32)[..., None]
    lp_out = jnp.take_along_axis(lp, tgt, axis=-1)[..., 0]
    m_out = valid * batch["batted_valid"]
    nll_out = -_masked_mean(lp_out, m_out)

    # Home-run column on its own: binary log-loss of P(HR) against the HR label.
    p_hr = jnp.exp(lp[..., 4])
    y_hr = (batch["batted_out"] == 4).astype(p_hr.dtype)
    eps = 1e-6
    ll_hr = y_hr * jnp.log(p_hr + eps) + (1 - y_hr) * jnp.log(1 - p_hr + eps)
    nll_hr = -_masked_mean(ll_hr, m_out)

    total = nll_launch + nll_out
    return total, {"nll_launch": nll_launch, "nll_outcome": nll_out,
                   "nll_hr": nll_hr}
