from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import flax_module
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D_MODEL      = 128
N_HEADS      = 4
N_LAYERS     = 2
HISTORY_LEN  = 10    # H — number of past pitches attended to

# Token feature dimensions (must sum to D_RAW before projection)
D_PITCH_TYPE_EMB  = 16   # pitch-type embedding
D_LOCATION        = 4    # (plate_x, plate_z, release_x, release_z)
D_OUTCOME_EMB     = 8    # outcome embedding
D_GAME_STATE_SCL  = 8    # game-state scalar features
D_RAW = D_PITCH_TYPE_EMB + D_LOCATION + D_OUTCOME_EMB + D_GAME_STATE_SCL  # = 36

N_PITCH_TYPES = 8    # vocabulary size for pitch_type embeddings
N_OUTCOMES    = 16   # vocabulary size for outcome embeddings

# Fatigue / manager decision dims (concatenated before final projection)
D_FATIGUE  = 16
# Manager decision dim is variable — injected at call time.


# ---------------------------------------------------------------------------
# Utility: multi-head self-attention with masking (pre-norm)
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    d_model: int = D_MODEL
    n_heads: int = N_HEADS
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,                          # (B, S, d_model)
        mask: Optional[jnp.ndarray] = None,      # (B, 1, S, S) or (B, n_heads, S, S)  bool
        deterministic: bool = True,
    ) -> jnp.ndarray:                             # (B, S, d_model)
        B, S, D = x.shape
        head_dim = D // self.n_heads
        scale = head_dim ** -0.5

        # Project to Q, K, V
        q = nn.Dense(D, use_bias=False, name="Wq")(x)  # (B, S, D)
        k = nn.Dense(D, use_bias=False, name="Wk")(x)
        v = nn.Dense(D, use_bias=False, name="Wv")(x)

        # Reshape to (B, n_heads, S, head_dim)
        def split_heads(t):
            return t.reshape(B, S, self.n_heads, head_dim).transpose(0, 2, 1, 3)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Scaled dot-product attention
        attn_weights = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale  # (B, H, S, S)

        if mask is not None:
            # mask == True means "attend", False means "ignore"
            attn_weights = jnp.where(mask, attn_weights, jnp.finfo(jnp.float32).min)

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)         # (B, H, S, S)

        if not deterministic and self.dropout_rate > 0.0:
            attn_weights = nn.Dropout(rate=self.dropout_rate)(
                attn_weights, deterministic=deterministic
            )

        out = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)        # (B, H, S, head_dim)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, D)             # (B, S, D)
        out = nn.Dense(D, use_bias=False, name="Wo")(out)
        return out


# ---------------------------------------------------------------------------
# Transformer encoder layer (pre-norm)
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    d_model: int = D_MODEL
    n_heads: int = N_HEADS
    ffn_dim_mult: int = 4
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,                        # (B, S, d_model)
        mask: Optional[jnp.ndarray] = None,    # (B, 1, S, S) bool
        deterministic: bool = True,
    ) -> jnp.ndarray:                           # (B, S, d_model)

        # Pre-norm self-attention
        h = nn.LayerNorm()(x)
        h = MultiHeadSelfAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate,
        )(h, mask=mask, deterministic=deterministic)
        x = x + h   # residual

        # Pre-norm FFN
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.d_model * self.ffn_dim_mult)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.d_model)(h)
        x = x + h   # residual

        return x


# ---------------------------------------------------------------------------
# Per-pitch token embedding
# ---------------------------------------------------------------------------

class PitchTokenEmbedder(nn.Module):
    """
    Embeds a single pitch-history token from categorical and continuous
    features to d_model.

    Inputs per token (last axis)
    ----------------------------
    pitch_type   : int32   (B, S)       — vocabulary index
    location     : float32 (B, S, 4)   — plate_x, plate_z, rel_x, rel_z
    outcome      : int32   (B, S)       — vocabulary index
    game_state   : float32 (B, S, 8)   — scalar game-state features

    Output
    ------
    token_emb : float32 (B, S, d_model)
    """
    d_model: int = D_MODEL

    @nn.compact
    def __call__(
        self,
        pitch_type: jnp.ndarray,    # (B, S) int32
        location: jnp.ndarray,      # (B, S, 4)
        outcome: jnp.ndarray,       # (B, S) int32
        game_state: jnp.ndarray,    # (B, S, 8)
    ) -> jnp.ndarray:               # (B, S, d_model)

        pt_emb  = nn.Embed(N_PITCH_TYPES, D_PITCH_TYPE_EMB)(pitch_type)  # (B,S,16)
        out_emb = nn.Embed(N_OUTCOMES,    D_OUTCOME_EMB)(outcome)         # (B,S,8)

        raw = jnp.concatenate([pt_emb, location, out_emb, game_state], axis=-1)
        # raw: (B, S, 36)

        # Project to d_model
        token_emb = nn.Dense(self.d_model)(raw)     # (B, S, d_model)
        token_emb = nn.LayerNorm()(token_emb)
        return token_emb


# ---------------------------------------------------------------------------
# Main: PitchTransformer
# ---------------------------------------------------------------------------

class PitchTransformer(nn.Module):
    """
    Transformer over a history of H pitches + a CLS token.

    Sequence layout:  [CLS | pitch_{t-H} | … | pitch_{t-1}]
    Output: representation at the CLS position = shared_context.

    After the transformer, fatigue_state and manager_decision are
    concatenated to the CLS output and passed through a final projection
    back to d_model.

    Inputs
    ------
    hist_pitch_type  : int32   (B, T, H)
    hist_location    : float32 (B, T, H, 4)
    hist_outcome     : int32   (B, T, H)
    hist_game_state  : float32 (B, T, H, 8)
    history_mask     : bool    (B, T, H)    True = valid token
    fatigue_state    : float32 (B, T, 16)
    manager_decision : float32 (B, T, M)   M = number of decision features

    Output
    ------
    shared_context : float32 (B, T, d_model=128)
    """
    d_model: int = D_MODEL
    n_heads: int = N_HEADS
    n_layers: int = N_LAYERS
    history_len: int = HISTORY_LEN

    @nn.compact
    def __call__(
        self,
        hist_pitch_type: jnp.ndarray,   # (B, T, H) int32
        hist_location: jnp.ndarray,     # (B, T, H, 4)
        hist_outcome: jnp.ndarray,      # (B, T, H) int32
        hist_game_state: jnp.ndarray,   # (B, T, H, 8)
        history_mask: jnp.ndarray,      # (B, T, H) bool
        fatigue_state: jnp.ndarray,     # (B, T, 16)
        manager_decision: jnp.ndarray,  # (B, T, M)
        deterministic: bool = True,
    ) -> jnp.ndarray:                   # (B, T, d_model)

        B, T, H = hist_pitch_type.shape
        D = self.d_model

        # ---- flatten (B*T) for sequence processing ----
        def merge_bt(a):
            shape = a.shape
            return a.reshape(B * T, *shape[2:])

        # --- embed each history token ---
        embedder = PitchTokenEmbedder(d_model=D)

        hist_embs = embedder(
            merge_bt(hist_pitch_type),    # (B*T, H)
            merge_bt(hist_location),      # (B*T, H, 4)
            merge_bt(hist_outcome),       # (B*T, H)
            merge_bt(hist_game_state),    # (B*T, H, 8)
        )  # (B*T, H, D)

        # --- CLS token (learnable) ---
        cls_token = self.param(
            "cls_token",
            nn.initializers.normal(0.02),
            (1, 1, D),
        )  # (1, 1, D)
        cls_tokens = jnp.broadcast_to(cls_token, (B * T, 1, D))  # (B*T, 1, D)

        # Sequence: [CLS, h_0, h_1, ..., h_{H-1}]  — length S = H+1
        seq = jnp.concatenate([cls_tokens, hist_embs], axis=1)    # (B*T, H+1, D)
        S = H + 1

        # --- build attention mask ---
        # history_mask: (B, T, H) — True = attend, False = padding
        # CLS always attends; prepend True for CLS column/row.
        cls_valid = jnp.ones((B * T, 1), dtype=bool)
        token_valid = merge_bt(history_mask)    # (B*T, H)
        valid_mask = jnp.concatenate([cls_valid, token_valid], axis=1)  # (B*T, S)

        # Expand to (B*T, 1, S, S): a position can attend to another only
        # if the *key* position is valid.
        attn_mask = valid_mask[:, None, None, :]                          # (B*T, 1, 1, S)
        attn_mask = jnp.broadcast_to(attn_mask, (B * T, 1, S, S))

        # --- transformer layers ---
        x = seq
        for _ in range(self.n_layers):
            x = TransformerEncoderLayer(
                d_model=D,
                n_heads=self.n_heads,
            )(x, mask=attn_mask, deterministic=deterministic)

        # --- extract CLS output ---
        cls_out = x[:, 0, :]   # (B*T, D)

        # --- reshape back to (B, T, D) ---
        cls_out = cls_out.reshape(B, T, D)

        # --- concatenate fatigue and manager decision ---
        M = manager_decision.shape[-1]
        augmented = jnp.concatenate([cls_out, fatigue_state, manager_decision], axis=-1)
        # augmented: (B, T, D + 16 + M)

        # --- final projection back to d_model ---
        shared_context = nn.Dense(D, name="final_proj")(augmented)   # (B, T, D)
        shared_context = nn.LayerNorm()(shared_context)

        return shared_context   # (B, T, 128)


# ---------------------------------------------------------------------------
# NumPyro wrapper
# ---------------------------------------------------------------------------

def pitch_transformer_numpyro(
    hist_pitch_type: jnp.ndarray,    # (B, T, H) int32
    hist_location: jnp.ndarray,      # (B, T, H, 4)
    hist_outcome: jnp.ndarray,       # (B, T, H) int32
    hist_game_state: jnp.ndarray,    # (B, T, H, 8)
    history_mask: jnp.ndarray,       # (B, T, H) bool
    fatigue_state: jnp.ndarray,      # (B, T, 16)
    manager_decision: jnp.ndarray,   # (B, T, M)
    name: str = "pitch_transformer",
    deterministic: bool = True,
) -> jnp.ndarray:
    """
    NumPyro model fragment: registers PitchTransformer as a flax_module site
    and returns shared_context (B, T, 128).
    """
    B, T, H = hist_pitch_type.shape
    M = manager_decision.shape[-1]

    transformer = flax_module(
        name,
        PitchTransformer(),
        input_shape=[
            (B, T, H),          # hist_pitch_type
            (B, T, H, 4),       # hist_location
            (B, T, H),          # hist_outcome
            (B, T, H, 8),       # hist_game_state
            (B, T, H),          # history_mask
            (B, T, 16),         # fatigue_state
            (B, T, M),          # manager_decision
        ],
    )

    shared_context = transformer(
        hist_pitch_type, hist_location, hist_outcome, hist_game_state,
        history_mask, fatigue_state, manager_decision,
        deterministic=deterministic,
    )
    return shared_context  # (B, T, 128)
