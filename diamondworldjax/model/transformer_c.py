"""Transformer C: the remaining game state, given the plate appearance.

WHAT C IS FOR

A and B produce the pitch and the batter's response. Everything else that moves the
game forward happens around them: a runner steals, a pitch gets away, a fielder
makes an error. transition.py declares heads for exactly these, but they have been
sampling from their priors because the labels were never extracted. data/extract_events.py
now produces them, so C is trainable.

WHAT C DELIBERATELY DOES NOT DO

It does not sample runs or base advancement. v1 through v5 used neural heads for
that and it blew up game-level variance; v6 replaced them with an empirical table
and that is what made the run distribution match reality. The current benchmark says
the run-total distribution is the ONE part of this simulator that works: 1.98x
overdispersion against a real 2.11x. Putting a learned sampler back in that path
risks the only game-level result worth having, to fix nothing that is broken.

So C models the events the empirical table does not cover, and the table keeps
advancement. If C is later shown to beat the table on PIT and coverage, that is an
argument for extending it. It is not an assumption to build in now.

CLASS IMBALANCE

These events are rare: a balk is 0.002% of pitches, a passed ball 0.003%. Trained
with plain BCE, every head would learn to predict zero and score an excellent loss.
Each head is therefore reported as lift over its base rate, and calibration is
checked on the predicted positives, not just the loss.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from .pitchformer import Trunk, _loss_valid

# Order matters: it is the column order of the label array the loader builds.
EVENT_FLAGS = ("wild_pitch", "passed_ball", "balk", "steal", "caught_stealing",
               "pickoff", "error", "defensive_indiff")
N_EVENTS = len(EVENT_FLAGS)


class TransformerC(nn.Module):
    """Per-pitch probabilities for the transition events.

    Shares the super-state and the causal trunk with A and B, and additionally
    sees the realised pitch and the batter's response, because a wild pitch or a
    steal follows from what was actually thrown and whether the batter offered at
    it. That is not leakage: C is asked what happened AROUND this pitch given the
    pitch, in the same way B is asked about the swing given the pitch.
    """
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
                 ss_override=None, hidden_override=None, encode_only=False):
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

        obs = jnp.concatenate([
            jax.nn.one_hot(batch["pitch_type"], 8),
            batch["stuff"],
            batch["swing"][..., None],
            batch["contact"][..., None],
            batch["foul"][..., None],
        ], axis=-1)
        obs = nn.gelu(nn.Dense(self.d_model, name="obs_proj")(obs))
        z = nn.gelu(nn.Dense(self.d_model, name="merge")(
            jnp.concatenate([h, obs], axis=-1)))

        # Bias initialised negative so every head starts near the base rate of a
        # rare event rather than at p=0.5. Starting at 0.5 on an event that occurs
        # twice in 100,000 pitches spends the early steps undoing the init.
        logits = nn.Dense(256 if self.c_event_mode == "bundles" else N_EVENTS, name="events",
                          bias_init=(lambda key, shape, dtype=jnp.float32: jnp.full(shape, -6., dtype).at[0].set(0.))
                          if self.c_event_mode == "bundles" else nn.initializers.constant(-6.0))(z)
        if self.c_event_mode == "bundles" and self.c_support is not None:
            base = sum((batch['ctx'][..., 3+i] > .5).astype(jnp.int32) * (1 << i) for i in range(3))
            outs = jnp.clip(jnp.rint(batch['ctx'][..., 2] * 2).astype(jnp.int32), 0, 2)
            supported = jnp.asarray(self.c_support, dtype=bool).reshape(256, 24).T[base * 3 + outs]
            logits = jnp.where(supported, logits, -1e30)
        return {"event_logits": logits}


def loss_c(out, batch):
    """Masked BCE per event, plus the per-event breakdown for reporting."""
    if out['event_logits'].shape[-1] == 256:
        target = jnp.sum(batch['events'].astype(jnp.int32) * (1 << jnp.arange(8)), -1)
        logp = jnp.take_along_axis(jax.nn.log_softmax(out['event_logits']), target[..., None], -1)[..., 0]
        mask = _loss_valid(batch) * batch.get('c_eligible', 1)
        nll = -jnp.where(mask > 0, logp, 0.).sum() / jnp.maximum(mask.sum(), 1)
        return nll, {'nll_bundle': nll}
    valid = _loss_valid(batch)[..., None]
    y = batch["events"]
    logit = out["event_logits"]
    ll = y * jax.nn.log_sigmoid(logit) + (1 - y) * jax.nn.log_sigmoid(-logit)
    m = jnp.broadcast_to(valid, ll.shape).astype(ll.dtype)
    per_event = -(ll * m).sum(axis=(0, 1)) / jnp.maximum(m.sum(axis=(0, 1)), 1.0)
    total = per_event.sum()
    parts = {f"nll_{f}": per_event[i] for i, f in enumerate(EVENT_FLAGS)}
    return total, parts
