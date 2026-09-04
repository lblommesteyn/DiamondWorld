"""Transformer D: what happens to a ball once it is in play.

WHY THIS HEAD EXISTS

Park geometry was wired into the super-state and produced a null. Weather, wind and
altitude were wired in and produced a null. Both times the explanation given was the
same: A predicts which pitch is thrown and where it crosses the plate, B predicts
whether the batter offers, and neither of those depends on how far away the wall is or
which way the wind blows. Those variables act AFTER the bat meets the ball, and nothing
in the stack modelled that.

D is that model, which makes it two things at once. It completes the stack so a plate
appearance can end in something other than a strikeout or a walk, and it is the test of
the explanation above. If geometry and environment do not help D either, the
explanation was wrong and should be withdrawn.

TWO HEADS, IN PHYSICAL ORDER

  D1  launch: exit velocity and launch angle, as Gaussians, from the super-state and
      the pitch. This is what the bat does to the ball. It should NOT depend on the
      park or the wind, and if it does that is a leak to investigate, not a win.

  D2  outcome: out / single / double / triple / home run, from the super-state, the
      pitch, AND the launch, plus geometry and environment. This is what the park and
      the air do to the ball once it is in flight. The launch is teacher-forced during
      training and sampled from D1 at simulation time.

Splitting them is not decoration. A single head from context to outcome could learn
"this batter hits home runs" without ever representing that a home run is a hard, high
ball that clears a wall. The split forces the park to act through the trajectory,
which is the physics, and it means the two claims "geometry matters" and "launch is
predictable" are scored separately rather than blurred together.

CLASS BALANCE

Outs are 68% of balls in play and triples are 0.6%. The outcome head is scored against
the empirical marginal, so a head that only learned the base rates gets exactly zero
lift, and the home-run column is reported on its own because that is where geometry
and air density have to show up if they show up anywhere.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from .pitchformer import Trunk, N_PITCH_TYPES

N_BATTED = 5
LAUNCH_DIM = 2


class TransformerD(nn.Module):
    n_pitchers: int
    n_batters: int
    n_parks: int
    d_model: int = 192
    n_layers: int = 4
    n_heads: int = 6
    dropout: float = 0.1

    @nn.compact
    def __call__(self, batch, *, train: bool):
        h, _ = Trunk(self.n_pitchers, self.n_batters, self.n_parks, self.d_model,
                     self.n_layers, self.n_heads, self.dropout, name="trunk")(
            batch, train=train)

        pitch = jnp.concatenate([
            jax.nn.one_hot(batch["pitch_type"], N_PITCH_TYPES),
            batch["stuff"],
        ], axis=-1)
        pitch = nn.gelu(nn.Dense(self.d_model, name="pitch_proj")(pitch))
        z = nn.gelu(nn.Dense(self.d_model, name="merge")(
            jnp.concatenate([h, pitch], axis=-1)))

        # D1: launch, from context and pitch only.
        l_mu = nn.Dense(LAUNCH_DIM, name="launch_mu")(z)
        l_ls = jnp.clip(nn.Dense(LAUNCH_DIM, name="launch_logsigma")(z), -4.0, 2.0)

        # D2: outcome, from context, pitch, the REALISED launch, and the park and
        # air the trunk already carries through the super-state. Launch enters
        # here explicitly rather than only through z so the outcome head cannot
        # bypass it.
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
    valid = batch["valid"]

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
