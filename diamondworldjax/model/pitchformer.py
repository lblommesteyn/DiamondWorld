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


class HeadResidual(nn.Module):
    n_pitchers: int
    n_batters: int
    d_residual: int = 16
    d_model: int = 192
    player_mode: str = "id"

    @nn.compact
    def __call__(self, pitcher_idx, batter_idx, context=None):
        if self.player_mode != "id":
            # No private ID path can undo the skill ablation.
            x = nn.Dense(self.d_residual, name="context_res")(context)
        else:
            def lookup(name, n, ids):
                value = nn.Embed(n, self.d_residual, name=name)(jnp.clip(ids, 0, n - 1))
                return jnp.where(((ids > 0) & (ids < n))[..., None], value, 0.0)
            x = jnp.concatenate([lookup("pitcher_res", self.n_pitchers, pitcher_idx),
                                 lookup("batter_res", self.n_batters, batter_idx)], -1)
        return nn.Dense(self.d_model, name="res_proj")(x)


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
    player_mode: str = "id"
    skill_seasons: int = 1
    residual_dim: int = 0

    @nn.compact
    def __call__(self, pitcher_idx, batter_idx, park_idx, ctx, geom, season_idx=None,
                 player_vectors=None):
        # ctx:  (B, T, D_CTX)  game state + home/away, from the loader
        # geom: (B, T, D_GEO)  arena geometry + has_geometry flag
        def identity(name, count, ids):
            # Unknowns have a deterministic neutral vector; never read random row 0.
            known = (ids > 0) & (ids < count)
            value = nn.Embed(count, self.d_player, name=name)(jnp.clip(ids, 0, count - 1))
            return jnp.where(known[..., None], value, 0.0)

        if player_vectors is not None:
            p, b = player_vectors
        elif self.player_mode in ("pa", "pa-no-latent"):
            season = jnp.zeros_like(pitcher_idx) if season_idx is None else season_idx
            season = jnp.clip(season, 0, self.skill_seasons - 1)
            def lookup(name, count, ids):
                table = self.variable("player_data", name,
                    lambda: jnp.zeros((count, self.skill_seasons, 64))).value
                value = jax.lax.stop_gradient(table)[jnp.clip(ids, 0, count - 1), season]
                return jnp.where(((ids > 0) & (ids < count))[..., None], value, 0.0)
            p = lookup("pitcher", self.n_pitchers, pitcher_idx)
            b = lookup("batter", self.n_batters, batter_idx)
        elif self.player_mode == "none":
            p = jnp.zeros((*pitcher_idx.shape, self.d_player))
            b = jnp.zeros((*batter_idx.shape, self.d_player))
        elif self.player_mode == "id":
            p = identity("pitcher_emb", self.n_pitchers, pitcher_idx)
            b = identity("batter_emb", self.n_batters, batter_idx)
        else:
            raise ValueError(f"Unknown player mode {self.player_mode!r}")
        k = nn.Embed(self.n_parks, self.d_park, name="park_emb")(park_idx)
        x = jnp.concatenate([p, b, k, ctx, geom], axis=-1)
        x = nn.Dense(self.d_model, name="proj")(x)
        x = nn.gelu(x)
        if self.residual_dim:
            x = x + HeadResidual(self.n_pitchers, self.n_batters, self.residual_dim,
                self.d_model, self.player_mode, name="head_residual")(pitcher_idx, batter_idx, x)
        return x


class CausalBlock(nn.Module):
    """One pre-norm transformer block with a causal mask."""
    d_model: int
    n_heads: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x, mask, *, train: bool, decode: bool = False, history=None):
        h = nn.LayerNorm()(x)
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads, qkv_features=self.d_model,
            dropout_rate=self.dropout, deterministic=not train, decode=decode,
        )(h, h if history is None else h + history, mask=mask)
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
    player_mode: str = "id"
    skill_seasons: int = 1
    residual_dim: int = 0
    pitch_history: bool = False
    position_encoding: str = "learned"
    window_size: int = 0
    observation_masks: bool = False
    c_event_mode: str = "legacy"
    c_support: tuple | None = None

    def _sinusoidal(self, positions, d_model):
        freq = jnp.exp(-jnp.log(10000.0) * jnp.arange(0, d_model, 2) / d_model)
        angle = positions.astype(jnp.float32)[..., None] * freq
        return jnp.concatenate([jnp.sin(angle), jnp.cos(angle)], axis=-1)[..., :d_model]

    @nn.compact
    def __call__(self, batch, *, train: bool, decode: bool = False,
                 ss_override=None):
        if ss_override is not None:
            ss = ss_override
        else:
            ss = SuperState(self.n_pitchers, self.n_batters, self.n_parks,
                            d_model=self.d_model, player_mode=self.player_mode,
                            skill_seasons=self.skill_seasons, residual_dim=self.residual_dim, name="super_state")(
                batch["pitcher_idx"], batch["batter_idx"], batch["park_idx"],
                batch["ctx"], batch["geom"], batch.get("skill_season"))

        history = None
        if self.pitch_history:
            package = jnp.concatenate([
                jax.nn.one_hot(batch.get("history_type", batch["pitch_type"]), N_PITCH_TYPES), batch.get("history_stuff", batch["stuff"]),
                batch["swing"][..., None], batch["contact"][..., None],
                batch["foul"][..., None],
            ], axis=-1)
            if self.observation_masks:
                tm = batch.get('type_valid', jnp.ones_like(batch['valid']))
                sm = batch.get('stuff_observed', jnp.ones_like(batch['stuff'], dtype=bool))
                package = package.at[..., :8].set(package[..., :8] * tm[..., None])
                package = package.at[..., 8:13].set(jnp.where(sm, package[..., 8:13], 0.))
                package = jnp.concatenate([package, tm[..., None], sm.astype(package.dtype)], -1)
            history = nn.Dense(self.d_model, name="history_proj")(package)
        T = ss.shape[1]
        if self.window_size > 0:
            # Strict raw-token window, not a layer-local attention mask. Every
            # query is recomputed from its own last W completed input tokens.
            W = self.window_size
            B = ss.shape[0]
            hist = jnp.zeros_like(ss) if history is None else history
            valid = batch["valid"].astype(bool)
            if decode:
                saved_ss = self.variable("cache", "window_ss", lambda: jnp.zeros((B, W, self.d_model)))
                saved_hist = self.variable("cache", "window_history", lambda: jnp.zeros((B, W, self.d_model)))
                saved_valid = self.variable("cache", "window_valid", lambda: jnp.zeros((B, W), bool))
            if decode and T == 1:
                x = jnp.concatenate([saved_ss.value, ss], 1)
                hist_window = jnp.concatenate([saved_hist.value, hist], 1)
                window_valid = jnp.concatenate([saved_valid.value, valid], 1)
                saved_ss.value = jnp.where(valid[..., None], x[:, 1:], saved_ss.value)
                saved_hist.value = jnp.where(valid[..., None], hist_window[:, 1:], saved_hist.value)
                saved_valid.value = jnp.where(valid, window_valid[:, 1:], saved_valid.value)
            else:
                index = jnp.arange(T)[:, None] + jnp.arange(W + 1)[None, :]
                def windows(value):
                    pads = [(0, 0), (W, 0)] + [(0, 0)] * (value.ndim - 2)
                    value = jnp.pad(value, pads)[:, index]
                    return value.reshape((B * T, W + 1, *value.shape[3:]))
                x, hist_window, window_valid = windows(ss), windows(hist), windows(valid)
            # Fixed window-relative positions, including masked left padding.
            if self.position_encoding == "sinusoidal":
                position = self._sinusoidal(jnp.arange(W + 1), self.d_model)
            else:
                position = nn.Embed(1024, self.d_model, name="pos_emb")(jnp.arange(W + 1))
            x = x + position[None]
            mask = causal_mask(window_valid)
            for i in range(self.n_layers):
                x = CausalBlock(self.d_model, self.n_heads, self.dropout,
                    name=f"block_{i}")(x, mask, train=train, decode=False,
                                      history=hist_window if self.pitch_history else None)
            x = nn.LayerNorm(name="out_norm")(x)[:, -1]
            return x.reshape(B, T, self.d_model), ss
        if decode and T == 1:
            position = jnp.asarray(batch["_decode_position"], jnp.int32)
            if self.position_encoding == "sinusoidal":
                pos = self._sinusoidal(position, self.d_model)[None]
            else:
                pos = nn.Embed(1024, self.d_model, name="pos_emb")(position[None])
            x = ss + pos[None]
            history_valid = batch["_cache_valid"].astype(bool)
            strict_history = jnp.arange(history_valid.shape[1])[None, :] < position
            mask = (history_valid & strict_history)[:, None, None, :]
            if self.window_size > 0:
                too_old = jnp.arange(history_valid.shape[1]) <= (position - self.window_size)
                mask = mask & ~too_old[None, None, None, :]
        else:
            if self.position_encoding == "sinusoidal":
                pos = self._sinusoidal(jnp.arange(T), self.d_model)
            else:
                pos = nn.Embed(1024, self.d_model, name="pos_emb")(jnp.arange(T))
            x = ss + pos[None]
            mask = causal_mask(batch["valid"].astype(bool))
            if self.window_size > 0:
                dist = jnp.arange(T)[:, None] - jnp.arange(T)[None, :]
                window_mask = (dist >= 0) & (dist < self.window_size)
                mask = mask & window_mask[None, None]
        for i in range(self.n_layers):
            x = CausalBlock(self.d_model, self.n_heads, self.dropout,
                            name=f"block_{i}")(x, mask, train=train, decode=decode, history=history)
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


def _loss_valid(batch):
    v = batch["valid"]
    lm = batch.get("loss_mask")
    return v * lm if lm is not None else v


def loss_a(out, batch):
    valid = _loss_valid(batch)
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
    valid = _loss_valid(batch)
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
