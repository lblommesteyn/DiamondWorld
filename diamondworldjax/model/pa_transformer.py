"""PA-level sequence models for the outcome model.

Two architectures share the same (B, T, C) -> (B, T, d_model) contract:

  PATransformer  — causal self-attention, used via ``--pitchformer``
  PAGRU          — gated recurrent unit, used via ``--pitchformer --pa-arch gru``

Both receive the pre-computed context vector from pa_model (game_state, pitcher_z,
batter_z, park_emb already concatenated).  Neither embeds players or parks itself.

PAGRU additionally exposes a ``step`` classmethod for compiled scan rollout: it
advances one timestep given a hidden-state carry. Transformer rollout uses
projected key/value caches.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn
from numpyro.contrib.module import flax_module
import numpyro

from .pitchformer import CausalBlock, causal_mask


class PATransformer(nn.Module):
    """Causal transformer over PA sequences within a game.

    Parameters
    ----------
    d_model : int
        Hidden dimension.  128 is the default, but the value is part of the
        checkpoint architecture and must be supplied again at evaluation time.
    n_layers : int
        Number of CausalBlock layers.  2 by default — PA sequences are shorter
        (~70 PAs per game) than pitch sequences (~300), so depth matters less
        and training / sim speed matters more.
    n_heads : int
        Attention heads.
    dropout : float
        Dropout rate inside CausalBlock.  0.0 by default because SVI already
        regularises through the KL on the player-skill latent; can be revisited
        if the transformer overfits.
    """
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.0
    position_encoding: str = "learned"

    @nn.compact
    def __call__(
        self,
        context_raw: jnp.ndarray,   # (B, T, C)  concatenated PA features
        valid_mask: jnp.ndarray,     # (B, T)     True for real PAs
        *,
        train: bool = False,
    ) -> jnp.ndarray:                # (B, T, d_model)
        """Project, attend, return contextualised PA representations."""
        # Project the raw (variable-width) context to d_model
        x = nn.Dense(self.d_model, name="input_proj")(context_raw)
        x = nn.LayerNorm(name="input_ln")(x)

        # Learned positional embedding (PA index within a game)
        T = x.shape[1]
        if self.position_encoding == "sinusoidal":
            frequency = jnp.exp(-jnp.log(10000.0) * jnp.arange(0, self.d_model, 2) / self.d_model)
            angle = jnp.arange(T)[:, None] * frequency[None, :]
            pos = jnp.stack([jnp.sin(angle), jnp.cos(angle)], -1).reshape(T, -1)[:, :self.d_model]
        elif self.position_encoding == "learned":
            if T > 512:
                raise ValueError("Legacy learned PA positions support at most 512 timesteps")
            pos = nn.Embed(512, self.d_model, name="pos_emb")(jnp.arange(T))
        else:
            raise ValueError("Unknown PA positional encoding")
        x = x + pos[None]

        # Causal self-attention: position t attends to positions < t only.
        # causal_mask returns (B, 1, T, T) with strictly lower-triangular
        # structure ANDed with the key validity mask.
        mask = causal_mask(valid_mask.astype(jnp.bool_))
        for i in range(self.n_layers):
            x = CausalBlock(
                self.d_model, self.n_heads, self.dropout, name=f"block_{i}"
            )(x, mask, train=train)

        return nn.LayerNorm(name="out_norm")(x)  # (B, T, d_model)


# ---------------------------------------------------------------------------
# NumPyro wrapper
# ---------------------------------------------------------------------------

def pa_transformer_numpyro(
    context_raw: jnp.ndarray,    # (B, T, C)
    valid_mask: jnp.ndarray,     # (B, T)
    d_model: int = 128,
    n_layers: int = 2,
    n_heads: int = 4,
    dropout: float = 0.0,
    name: str = "pa_transformer",
    position_encoding: str = "learned",
    train: bool = False,
) -> jnp.ndarray:
    """Register PATransformer as a flax_module site and return (B, T, d_model).

    Same pattern as pitch_transformer_numpyro in pitch_transformer.py: the Flax
    module's parameters become point-estimated NumPyro sites that SVI optimises
    alongside the probabilistic latents.
    """
    if d_model <= 0 or n_layers <= 0 or n_heads <= 0:
        raise ValueError("pitchformer dim, layers, and heads must all be positive")
    if d_model % n_heads:
        raise ValueError("pitchformer dim must be divisible by pitchformer heads")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("pitchformer dropout must be in [0, 1)")

    B, T = valid_mask.shape
    C = context_raw.shape[-1]
    if position_encoding == "auto":
        existing = numpyro.param(name + "$params")
        position_encoding = "sinusoidal" if existing is not None and "pos_emb" not in existing else "learned"

    transformer = flax_module(
        name,
        PATransformer(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            position_encoding=position_encoding,
        ),
        jnp.ones((B, T, C)),           # context_raw dummy
        jnp.ones((B, T), dtype=bool),   # valid_mask dummy
    )

    rngs = {"dropout": numpyro.prng_key()} if train and dropout else {}
    return transformer(context_raw, valid_mask, train=train, rngs=rngs)


# ---------------------------------------------------------------------------
# GRU variant
# ---------------------------------------------------------------------------

class _GRUStack(nn.Module):
    """N stacked GRU layers with LayerNorm between them."""
    d_model: int
    n_layers: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, valid_mask: jnp.ndarray) -> jnp.ndarray:
        B, T, _ = x.shape
        mask = valid_mask.astype(x.dtype)[..., None]  # (B, T, 1)
        for i in range(self.n_layers):
            cell = nn.GRUCell(features=self.d_model, name=f"gru_{i}")
            h = jnp.zeros((B, self.d_model), dtype=x.dtype)
            def advance(module, carry, inputs):
                value, active = inputs
                proposed, _ = module(carry, value)
                carry = jnp.where(active, proposed, carry)
                return carry, carry

            # Lift the recurrence without changing the GRUCell parameter tree.
            _, x = nn.scan(advance, variable_broadcast="params",
                           split_rngs={"params": False}, in_axes=1, out_axes=1)(
                               cell, h, (x, mask))
            if i < self.n_layers - 1:
                x = nn.LayerNorm(name=f"ln_{i}")(x)
        return x


class PAGRU(nn.Module):
    """GRU over PA sequences within a game.

    Same contract as PATransformer: (B, T, C) context + (B, T) valid -> (B, T, d_model).
    No positional embedding — the recurrence carries position implicitly.
    """
    d_model: int = 128
    n_layers: int = 2

    @nn.compact
    def __call__(
        self,
        context_raw: jnp.ndarray,
        valid_mask: jnp.ndarray,
        *,
        train: bool = False,
    ) -> jnp.ndarray:
        x = nn.Dense(self.d_model, name="input_proj")(context_raw)
        x = nn.LayerNorm(name="input_ln")(x)
        x = _GRUStack(self.d_model, self.n_layers, name="gru_stack")(x, valid_mask)
        return nn.LayerNorm(name="out_norm")(x)


def pa_gru_numpyro(
    context_raw: jnp.ndarray,
    valid_mask: jnp.ndarray,
    d_model: int = 128,
    n_layers: int = 2,
    name: str = "pa_gru",
) -> jnp.ndarray:
    """Register PAGRU as a flax_module site and return (B, T, d_model)."""
    if d_model <= 0 or n_layers <= 0:
        raise ValueError("gru dim and layers must be positive")

    B, T = valid_mask.shape
    C = context_raw.shape[-1]

    gru = flax_module(
        name,
        PAGRU(d_model=d_model, n_layers=n_layers),
        jnp.ones((B, T, C)),
        jnp.ones((B, T), dtype=bool),
    )

    return gru(context_raw, valid_mask, train=False)


# ---------------------------------------------------------------------------
# Scan-compatible step functions for fast rollout
# ---------------------------------------------------------------------------

def gru_step_fn(gru_module: PAGRU, params: dict):
    """Return a pure function ``(carry, context_t) -> (carry, output_t)``.

    ``carry`` is a tuple of per-layer hidden states, each (B, d_model).
    ``context_t`` is a single timestep's raw context (B, C).
    ``output_t`` is the GRU output (B, d_model).

    The returned function is suitable for ``jax.lax.scan``.
    """
    p = params

    def step(carry, context_t):
        x = nn.Dense(gru_module.d_model, name="input_proj").apply(
            {"params": p["input_proj"]}, context_t)
        x = nn.LayerNorm(name="input_ln").apply(
            {"params": p["input_ln"]}, x)

        new_carry = []
        for i in range(gru_module.n_layers):
            cell = nn.GRUCell(features=gru_module.d_model, name=f"gru_{i}")
            h_prev = carry[i]
            h_new, _ = cell.apply(
                {"params": p["gru_stack"][f"gru_{i}"]}, h_prev, x)
            new_carry.append(h_new)
            x = h_new
            if i < gru_module.n_layers - 1:
                x = nn.LayerNorm(name=f"ln_{i}").apply(
                    {"params": p["gru_stack"][f"ln_{i}"]}, x)

        out = nn.LayerNorm(name="out_norm").apply(
            {"params": p["out_norm"]}, x)
        return tuple(new_carry), out

    return step


def gru_init_carry(n_layers: int, batch_size: int, d_model: int):
    """Return the initial carry (all-zeros hidden states) for gru_step_fn."""
    return tuple(
        jnp.zeros((batch_size, d_model)) for _ in range(n_layers)
    )


# ---------------------------------------------------------------------------
# Transformer: incremental K/V cache for fast rollout
# ---------------------------------------------------------------------------


def transformer_step_fn(transformer_module: PATransformer, params: dict,
                        max_seq_len: int):
    """Decode one PA using the existing full-sequence checkpoint weights.

    Each layer stores projected keys/values. Per-game positions allow active-game
    bucketing. The simulator grows cache capacity before a position overflows.
    Attention is strictly past-only, matching CausalBlock (including t=0).
    """
    p = params
    heads = transformer_module.n_heads
    width = transformer_module.d_model
    depth = width // heads

    def dense(name, x, subtree=p):
        return nn.Dense(subtree[name]["bias"].shape[0]).apply(
            {"params": subtree[name]}, x)

    def norm(name, x, subtree=p):
        return nn.LayerNorm().apply({"params": subtree[name]}, x)

    def step(carry, context_t):
        layers, valid, positions = carry
        rows = jnp.arange(context_t.shape[0])
        x = norm("input_ln", dense("input_proj", context_t))
        if transformer_module.position_encoding == "learned":
            x = x + p["pos_emb"]["embedding"][positions]
        else:
            frequency = jnp.exp(-jnp.log(10000.0) * jnp.arange(0, width, 2) / width)
            angle = positions[:, None] * frequency[None, :]
            x = x + jnp.stack([jnp.sin(angle), jnp.cos(angle)], -1).reshape(x.shape[0], -1)[:, :width]
        mask = valid & (jnp.arange(valid.shape[1])[None, :] < positions[:, None])
        next_layers = []
        for i, (keys, values) in enumerate(layers):
            block = p[f"block_{i}"]
            h = norm("LayerNorm_0", x, block)
            attention = block["MultiHeadDotProductAttention_0"]
            def project(name):
                return nn.DenseGeneral(features=(heads, depth)).apply(
                    {"params": attention[name]}, h)
            query, key, value = project("query"), project("key"), project("value")
            keys = keys.at[rows, positions].set(key)
            values = values.at[rows, positions].set(value)
            weights = nn.dot_product_attention_weights(
                query[:, None], keys, mask=mask[:, None, None, :], deterministic=True)
            attended = jnp.einsum("bhqk,bkhd->bqhd", weights, values)[:, 0]
            attended = nn.DenseGeneral(features=width, axis=(-2, -1)).apply(
                {"params": attention["out"]}, attended)
            x = x + attended * mask.any(-1)[:, None]
            h = norm("LayerNorm_1", x, block)
            x = x + dense("Dense_1", nn.gelu(dense("Dense_0", h, block)), block)
            next_layers.append((keys, values))
        valid = valid.at[rows, positions].set(True)
        return (tuple(next_layers), valid, positions + 1), norm("out_norm", x)

    return step


def transformer_init_carry(batch_size: int, max_seq_len: int,
                           context_dim: int | None = None, *, n_layers: int = 2,
                           d_model: int = 128, n_heads: int = 4):
    """Allocate projected K/V caches; context_dim is retained for API compatibility."""
    shape = (batch_size, max_seq_len, n_heads, d_model // n_heads)
    return (
        tuple((jnp.zeros(shape), jnp.zeros(shape)) for _ in range(n_layers)),
        jnp.zeros((batch_size, max_seq_len), dtype=jnp.bool_),
        jnp.zeros(batch_size, dtype=jnp.int32),
    )
