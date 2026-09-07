"""Pitch-level transformers A and B over a shared super-state.

THE DESIGN

Every head conditions on one super-state vector per pitch, assembled once:

    pitcher embedding | batter embedding | park embedding | handedness matchup
    | game state (count, outs, bases, score, inning, TTO) | home/away
    | rules-based arena geometry

Transformer A: causal attention over the pitcher's recent pitches in this game,
predicting the NEXT pitch. Two parts, matching how a pitch is actually chosen and
then thrown:
  A1  pitch TYPE, 8-way categorical over the sequence context
  A2  the STUFF given that type: release speed, both break components, and the
      plate location, as Gaussians with learned per-type mean and spread.
Sampling A1 then A2 is the "sample logits, then stochastically sample outcome
through a stuff model" in the design.

Transformer B: the same super-state and the same causal history, PLUS the pitch A
just produced, predicting the batter's response as the existing hurdle tree does:
swing, then contact given swing, then foul given contact. `in_play` stays the
derived event `contact & ~foul` rather than a fourth head, so B cannot contradict
itself by putting mass on impossible combinations.

WHY THIS IS A SEPARATE MODULE AND NOT AN EDIT TO joint.py

joint.py is a NumPyro model fitted by SVI with hand-written guides. Dropping
attention into it means writing guides for the attention parameters and refitting
the whole joint model to learn anything at all. These are trained instead by
direct maximum likelihood, which is what they are: conditional densities with no
latent variables of their own. That keeps A and B measurable on their own terms
before anything is asked of the joint model, and it leaves the existing PA-level
results untouched while they are evaluated.

ON TRANSFORMER C

C is deliberately absent. It needs error, wild pitch, passed ball, balk and steal
labels, and the processed parquet has no such columns; transition.py's event heads
have been sampling from their priors with obs=None. Those labels have to be
extracted from the raw feed_live JSON before C can be trained, so C is a data
problem first and a modelling problem second.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

N_PITCH_TYPES = 8
STUFF_DIM = 5          # release_speed, pfx_x, pfx_z, plate_x, plate_z


class SuperState(nn.Module):
    """The shared per-pitch conditioning vector.

    Identity is embedded, everything else arrives already normalised from the
    loader. Geometry enters as physical numbers rather than as another learned
    park vector: the park embedding can memorise "runs play up here", but only
    the dimensions can say the wall is 314 feet away, and those are the terms a
    batted-ball head can actually use. The two are kept side by side rather than
    one replacing the other.
    """
    n_pitchers: int
    n_batters: int
    n_parks: int
    d_player: int = 48
    d_park: int = 16
    d_model: int = 192

    @nn.compact
    def __call__(self, pitcher_idx, batter_idx, park_idx, ctx, geom):
        # ctx:  (B, T, D_CTX)  game state + home/away, from the loader
        # geom: (B, T, D_GEO)  arena geometry + has_geometry flag
        p = nn.Embed(self.n_pitchers, self.d_player, name="pitcher_emb")(pitcher_idx)
        b = nn.Embed(self.n_batters, self.d_player, name="batter_emb")(batter_idx)
        k = nn.Embed(self.n_parks, self.d_park, name="park_emb")(park_idx)
        x = jnp.concatenate([p, b, k, ctx, geom], axis=-1)
        x = nn.Dense(self.d_model, name="proj")(x)
        return nn.gelu(x)


class CausalBlock(nn.Module):
    """One pre-norm transformer block with a causal mask."""
    d_model: int
    n_heads: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x, mask, *, train: bool, decode: bool = False):
        h = nn.LayerNorm()(x)
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads, qkv_features=self.d_model,
            dropout_rate=self.dropout, deterministic=not train, decode=decode,
        )(h, h, mask=mask)
        # The FIRST pitch of a sequence has no history, so its attention row is
        # fully masked. That does not raise: softmax over all-masked logits returns
        # a UNIFORM mix over every key, padding included, and position 1 then reads
        # that contaminated vector, so the padding propagates forward through the
        # whole stack. Measured before this line existed: perturbing only padded
        # positions moved the t=0 output by 1.77 and t=1 by 0.81.
        #
        # Zeroing the attention contribution on a fully-masked row leaves the
        # residual stream carrying the super-state alone, which is the intended
        # meaning of "first pitch, no history: predict from who is up, the count,
        # and the park".
        has_key = mask.any(axis=-1)[:, 0, :, None]        # (B, T, 1)
        x = x + h * has_key.astype(h.dtype)
        h = nn.LayerNorm()(x)
        h = nn.Dense(4 * self.d_model)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.d_model)(h)
        h = nn.Dropout(self.dropout, deterministic=not train)(h)
        return x + h


def causal_mask(valid):
    """(B, T) validity -> (B, 1, T, T) mask that is causal AND pad-aware.

    Strictly lower triangular, excluding the diagonal: position t attends to
    pitches BEFORE t only. Including t would let A see the pitch it is being
    asked to predict, which is the whole question.
    """
    B, T = valid.shape
    causal = jnp.tril(jnp.ones((T, T), dtype=bool), k=-1)
    pad = valid[:, None, :]                       # (B, 1, T) key validity
    return (causal[None] & pad)[:, None, :, :]


class Trunk(nn.Module):
    """Shared super-state + causal transformer stack."""
    n_pitchers: int
    n_batters: int
    n_parks: int
    d_model: int = 192
    n_layers: int = 4
    n_heads: int = 6
    dropout: float = 0.1

    @nn.compact
    def __call__(self, batch, *, train: bool, decode: bool = False):
        ss = SuperState(self.n_pitchers, self.n_batters, self.n_parks,
                        d_model=self.d_model, name="super_state")(
            batch["pitcher_idx"], batch["batter_idx"], batch["park_idx"],
            batch["ctx"], batch["geom"])

        T = ss.shape[1]
        if decode and T == 1:
            # During rollout we receive one generated token at a time.  The
            # attention modules retain K/V tensors in Flax's ``cache``
            # collection; ``_cache_valid`` keeps padded or already-ended
            # sequences out of that history.  The strict inequality preserves
            # the training-time contract: a pitch can read *prior* pitches, not
            # itself.
            position = jnp.asarray(batch["_decode_position"], jnp.int32)
            pos = nn.Embed(1024, self.d_model, name="pos_emb")(position[None])
            x = ss + pos[None]
            history_valid = batch["_cache_valid"].astype(bool)
            strict_history = jnp.arange(history_valid.shape[1])[None, :] < position
            mask = (history_valid & strict_history)[:, None, None, :]
        else:
            # The full-sequence path is unchanged.  It is also used once to
            # allocate correctly shaped decode caches before a rollout begins.
            pos = nn.Embed(1024, self.d_model, name="pos_emb")(jnp.arange(T))
            x = ss + pos[None]
            mask = causal_mask(batch["valid"].astype(bool))
        for i in range(self.n_layers):
            x = CausalBlock(self.d_model, self.n_heads, self.dropout,
                            name=f"block_{i}")(x, mask, train=train, decode=decode)
        return nn.LayerNorm(name="out_norm")(x), ss


class TransformerA(nn.Module):
    """What pitch is thrown: type logits, then the stuff given the type."""
    n_pitchers: int
    n_batters: int
    n_parks: int
    d_model: int = 192
    n_layers: int = 4
    n_heads: int = 6
    dropout: float = 0.1

    @nn.compact
    def __call__(self, batch, *, train: bool, decode: bool = False):
        h, _ = Trunk(self.n_pitchers, self.n_batters, self.n_parks, self.d_model,
                     self.n_layers, self.n_heads, self.dropout, name="trunk")(
            batch, train=train, decode=decode)

        type_logits = nn.Dense(N_PITCH_TYPES, name="type_head")(h)

        # Stuff is conditioned on the realised type. Per-type mean and log-sigma
        # rather than one shared Gaussian: a slider and a four-seam differ in
        # both centre and spread, and pooling them would blur exactly the
        # distinction the type head just made.
        mu = nn.Dense(N_PITCH_TYPES * STUFF_DIM, name="stuff_mu")(h)
        ls = nn.Dense(N_PITCH_TYPES * STUFF_DIM, name="stuff_logsigma")(h)
        B, T = type_logits.shape[:2]
        mu = mu.reshape(B, T, N_PITCH_TYPES, STUFF_DIM)
        ls = jnp.clip(ls.reshape(B, T, N_PITCH_TYPES, STUFF_DIM), -4.0, 2.0)
        return {"type_logits": type_logits, "stuff_mu": mu, "stuff_logsigma": ls,
                "hidden": h}


class TransformerB(nn.Module):
    """Batter response to the pitch A produced.

    The realised pitch is injected at the CURRENT position, which is legitimate
    and is the point: B answers "given this pitch was thrown, does he swing", so
    it must see the pitch. The history it attends over remains strictly causal.
    """
    n_pitchers: int
    n_batters: int
    n_parks: int
    d_model: int = 192
    n_layers: int = 4
    n_heads: int = 6
    dropout: float = 0.1

    @nn.compact
    def __call__(self, batch, *, train: bool, decode: bool = False):
        h, _ = Trunk(self.n_pitchers, self.n_batters, self.n_parks, self.d_model,
                     self.n_layers, self.n_heads, self.dropout, name="trunk")(
            batch, train=train, decode=decode)

        pitch = jnp.concatenate([
            jax.nn.one_hot(batch["pitch_type"], N_PITCH_TYPES),
            batch["stuff"],
        ], axis=-1)
        pitch = nn.gelu(nn.Dense(self.d_model, name="pitch_proj")(pitch))
        z = jnp.concatenate([h, pitch], axis=-1)
        z = nn.gelu(nn.Dense(self.d_model, name="merge")(z))

        return {
            "swing_logit":   nn.Dense(1, name="swing")(z)[..., 0],
            "contact_logit": nn.Dense(1, name="contact")(z)[..., 0],
            "foul_logit":    nn.Dense(1, name="foul")(z)[..., 0],
            # HBP is a terminal no-swing pitch.  It cannot belong in D, which
            # is evaluated only after a ball is put in play.
            "hbp_logit":     nn.Dense(1, name="hbp")(z)[..., 0],
        }


# ---------------------------------------------------------------------------
# Losses. Every term is masked; nothing is averaged over padding.
# ---------------------------------------------------------------------------

def _masked_mean(x, m):
    m = m.astype(x.dtype)
    return (x * m).sum() / jnp.maximum(m.sum(), 1.0)


def loss_a(out, batch):
    valid = batch["valid"]
    lp_type = jnp.take_along_axis(
        jax.nn.log_softmax(out["type_logits"]),
        batch["pitch_type"][..., None].astype(jnp.int32), axis=-1)[..., 0]
    nll_type = -_masked_mean(lp_type, valid * batch["type_valid"])

    # Stuff likelihood under the REALISED type only.
    idx = batch["pitch_type"].astype(jnp.int32)[..., None, None]
    mu = jnp.take_along_axis(out["stuff_mu"], jnp.broadcast_to(
        idx, (*idx.shape[:2], 1, STUFF_DIM)), axis=2)[:, :, 0, :]
    ls = jnp.take_along_axis(out["stuff_logsigma"], jnp.broadcast_to(
        idx, (*idx.shape[:2], 1, STUFF_DIM)), axis=2)[:, :, 0, :]
    sig = jnp.exp(ls)
    z = (batch["stuff"] - mu) / sig
    lp_stuff = -0.5 * z ** 2 - ls - 0.5 * jnp.log(2 * jnp.pi)
    # stuff_valid gates rows whose tracking fields are missing, so an absent
    # measurement is not scored as a confident prediction of zero.
    nll_stuff = -_masked_mean(lp_stuff.sum(-1), valid * batch["stuff_valid"])

    return nll_type + nll_stuff, {"nll_type": nll_type, "nll_stuff": nll_stuff}


def _bce(logit, y, m):
    return -_masked_mean(
        y * jax.nn.log_sigmoid(logit) + (1 - y) * jax.nn.log_sigmoid(-logit), m)


def loss_b(out, batch):
    valid = batch["valid"]
    swing = batch["swing"]
    contact = batch["contact"]

    # Each head is scored only where its event is defined: contact only on
    # swings, foul only on contact. Scoring them everywhere would train the
    # heads on rows where the label is structurally absent.
    l_swing = _bce(out["swing_logit"], swing, valid)
    l_contact = _bce(out["contact_logit"], contact, valid * swing)
    l_foul = _bce(out["foul_logit"], batch["foul"], valid * swing * contact)
    l_hbp = _bce(out["hbp_logit"], batch["hbp"], valid * (1 - swing))
    total = l_swing + l_contact + l_foul + l_hbp
    return total, {"nll_swing": l_swing, "nll_contact": l_contact,
                   "nll_foul": l_foul, "nll_hbp": l_hbp}
