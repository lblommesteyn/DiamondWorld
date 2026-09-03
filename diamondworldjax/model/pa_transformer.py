"""PA-level causal transformer for the outcome model.

Adapts the pitchformer's causal-attention architecture (CausalBlock + causal_mask)
to operate over the PA sequence within a game, rather than over individual pitches.

Unlike the pitchformer Trunk, this module does NOT embed players or parks itself.
It receives the pre-computed context vector from pa_model (game_state, pitcher_z,
batter_z, park_emb already concatenated) and runs causal self-attention over the T
PA positions so each PA can attend to all previous PAs in the same game.

The pretrained pitchformer weights do not transfer (different granularity, different
embedding scheme), but the architecture — projection → positional embedding → causal
transformer blocks → layer norm — is reused.

Used via the ``--pitchformer`` flag in train_pa / pa_model.
"""
from __future__ import annotations

import jax.numpy as jnp
import flax.linen as nn
from numpyro.contrib.module import flax_module

from .pitchformer import CausalBlock, causal_mask


class PATransformer(nn.Module):
    """Causal transformer over PA sequences within a game.

    Parameters
    ----------
    d_model : int
        Hidden dimension.  128 matches the PA model's existing CONTEXT_DIM
        convention and keeps the outcome head's first Dense layer the same width
        as without the transformer, so a checkpoint trained without --pitchformer
        can't accidentally be loaded with it (the shape would mismatch).
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
        pos = nn.Embed(512, self.d_model, name="pos_emb")(jnp.arange(T))
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
    name: str = "pa_transformer",
) -> jnp.ndarray:
    """Register PATransformer as a flax_module site and return (B, T, d_model).

    Same pattern as pitch_transformer_numpyro in pitch_transformer.py: the Flax
    module's parameters become point-estimated NumPyro sites that SVI optimises
    alongside the probabilistic latents.
    """
    B, T = valid_mask.shape
    C = context_raw.shape[-1]

    transformer = flax_module(
        name,
        PATransformer(),
        jnp.ones((B, T, C)),           # context_raw dummy
        jnp.ones((B, T), dtype=bool),   # valid_mask dummy
    )

    return transformer(context_raw, valid_mask, train=False)
