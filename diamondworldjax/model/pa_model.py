"""PA-level NumPyro generative model (v2).

Changes from v1:
- Park embedding added to context (captures park factor variance)
- Runs-scored upweighting: non-zero run events get 3x loss weight so the
  model can't collapse to always predicting 0 runs
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpyro
import numpyro.distributions as dist
import numpyro.handlers as _nph
from numpyro.contrib.module import flax_module

from .embeddings import encode_players_numpyro, SKILL_DIM

N_PA_OUTCOMES = 9    # K, BB, HBP, 1B, 2B, 3B, HR, out, E
MAX_RUNS      = 4    # categorical over 0-4 runs per PA
N_BASE_STATES = 8    # 0-7 bitmask
N_PARKS       = 100  # safe upper bound for park embedding table
PARK_DIM      = 8

STATE_DIM   = 8      # inning, half, outs, base_state, score_diff, tto, shift, clock
PLAYER_DIM  = 64
CONTEXT_DIM = STATE_DIM + 2 * PLAYER_DIM + PARK_DIM  # 144

# Weight multiplier for non-zero run events in the ELBO.
# ~85% of PAs score 0 runs; without upweighting the model learns to always
# predict 0 and still gets a good cross-entropy score.
RUNS_UPWEIGHT = 2.0


class ParkEmbedding(nn.Module):
    n_parks: int = N_PARKS
    embed_dim: int = PARK_DIM

    @nn.compact
    def __call__(self, park_ids: jnp.ndarray) -> jnp.ndarray:
        return nn.Embed(num_embeddings=self.n_parks, features=self.embed_dim)(park_ids)


class PAOutcomeHead(nn.Module):
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, context: jnp.ndarray) -> jnp.ndarray:
        h = nn.Dense(self.hidden_dim)(context)
        h = nn.LayerNorm()(h)
        h = nn.relu(h)
        return nn.Dense(N_PA_OUTCOMES)(h)


class PAOutcomeHeadV6(nn.Module):
    """Deeper outcome head for the outcome-only model (v6).

    With the runs_scored / base_state_after heads removed, all model capacity is
    devoted to calibrating the 9-way outcome distribution. A second hidden layer
    helps the head represent rare classes (HR, 2B) the single-layer head
    undersampled.
    """
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, context: jnp.ndarray) -> jnp.ndarray:
        h = nn.Dense(self.hidden_dim)(context)
        h = nn.LayerNorm()(h)
        h = nn.relu(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.LayerNorm()(h)
        h = nn.relu(h)
        return nn.Dense(N_PA_OUTCOMES)(h)


class RunsScoredHead(nn.Module):
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, context: jnp.ndarray, outcome_oh: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([context, outcome_oh], axis=-1)
        h = nn.Dense(self.hidden_dim)(x)
        h = nn.LayerNorm()(h)
        h = nn.relu(h)
        return nn.Dense(MAX_RUNS + 1)(h)


class BaseStateAfterHead(nn.Module):
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, context: jnp.ndarray, outcome_oh: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([context, outcome_oh], axis=-1)
        h = nn.Dense(self.hidden_dim)(x)
        h = nn.LayerNorm()(h)
        h = nn.relu(h)
        return nn.Dense(N_BASE_STATES)(h)


def pa_model(
    batch: dict,
    player_table: dict,
    teacher_force: bool = True,
    outcome_only: bool = False,
    fatigue: bool = False,
    platoon: bool = False,
    kl_scale: float = 1.0,
) -> None:
    B, T = batch["pa_valid"].shape
    P    = player_table["stats"].shape[0]

    # Per-player latent skill vectors. player_skills is a GLOBAL latent over all P
    # players; the likelihood below is over a minibatch of B games. To keep the ELBO
    # balanced (an unbiased estimate of the full-data ELBO), its prior KL must be
    # scaled by kl_scale = batch_games / total_train_games. Without this the global
    # KL is ~total/batch times over-weighted and the posterior collapses to the
    # prior (player_mu ~ 0), switching the latent off. See train_pa for the value.
    with _nph.scale(scale=kl_scale):
        player_skills = numpyro.sample(
            "player_skills",
            dist.Normal(jnp.zeros((P, SKILL_DIM)), jnp.ones((P, SKILL_DIM))).to_event(2),
        )

    # Player embeddings
    pitcher_z, batter_z = encode_players_numpyro(
        player_table["stats"],
        player_table["league"],
        player_table["hand"],
        pitcher_ids   = batch["pitcher_ids"],
        batter_ids    = batch["batter_ids"],
        player_skills = player_skills,
    )  # (B, T, 64) each

    # Park embedding
    park_emb_fn = flax_module("park_embedding", ParkEmbedding(), batch["park_ids"])
    park_emb    = park_emb_fn(batch["park_ids"])  # (B, T, 8)

    # Full context: game state + player embeddings + park
    state_feats = [
        batch["inning"], batch["half"], batch["outs"], batch["base_state"],
        batch["score_diff"], batch["tto"], batch["shift_restricted"], batch["pitch_clock"],
    ]
    if fatigue:
        # Phase-4: pitcher cumulative game pitch count (normalised). The single
        # biggest missing real effect — starters fade as the count climbs.
        state_feats.append(batch["pitch_count_game"])
    if platoon:
        # Platoon: batter side and pitcher throw hand (R=1, L=0), per PA. The
        # (bat_side, pit_hand) pair lets the head learn the platoon interaction
        # directly (incl. its L/R asymmetry) instead of extracting it from two
        # 32-dim player embeddings; bat_side is the real per-PA side, correct
        # for switch hitters.
        state_feats.append(batch["bat_side"])
        state_feats.append(batch["pit_hand"])
    game_state = jnp.stack(state_feats, axis=-1)  # (B, T, 8..11)

    context  = jnp.concatenate([game_state, pitcher_z, batter_z, park_emb], axis=-1)  # (B, T, 144/145)
    dummy_oh = jnp.zeros((B, T, N_PA_OUTCOMES))

    # Outcome-only model (v6): a single deeper head; runs + base_state come from
    # the deterministic/empirical rules engine at rollout time, not the network.
    if outcome_only:
        pa_head   = flax_module("pa_outcome_head_v6", PAOutcomeHeadV6(), context)
        pa_logits = pa_head(context)
        with numpyro.plate("games", B, dim=-2), numpyro.plate("pas", T, dim=-1):
            pa_obs = None
            if teacher_force:
                pa_obs = jnp.where(batch["pa_outcome"] == -1, 0, batch["pa_outcome"])
            numpyro.sample("pa_outcome", dist.Categorical(logits=pa_logits), obs=pa_obs)
        return

    pa_head   = flax_module("pa_outcome_head",       PAOutcomeHead(),      context)
    runs_head = flax_module("runs_scored_head",      RunsScoredHead(),     context, dummy_oh)
    bs_head   = flax_module("base_state_after_head", BaseStateAfterHead(), context, dummy_oh)

    pa_logits = pa_head(context)  # (B, T, 9)

    with numpyro.plate("games", B, dim=-2), numpyro.plate("pas", T, dim=-1):

        # PA outcome
        pa_obs = None
        if teacher_force:
            pa_obs = jnp.where(batch["pa_outcome"] == -1, 0, batch["pa_outcome"])

        pa_outcome = numpyro.sample(
            "pa_outcome", dist.Categorical(logits=pa_logits), obs=pa_obs,
        )

        pa_oh = jax.nn.one_hot(pa_outcome, N_PA_OUTCOMES)  # (B, T, 9)

        # Runs scored with upweighting for non-zero events
        runs_logits = runs_head(context, pa_oh)
        runs_obs    = jnp.clip(batch["runs_scored"], 0, MAX_RUNS) if teacher_force else None
        numpyro.sample("runs_scored", dist.Categorical(logits=runs_logits), obs=runs_obs)

        if teacher_force:
            # Add extra log-prob weight for non-zero run PAs so the model
            # can't minimise loss by always predicting 0.
            extra_lp = dist.Categorical(logits=runs_logits).log_prob(runs_obs)
            numpyro.factor(
                "runs_upweight",
                jnp.where(runs_obs > 0, RUNS_UPWEIGHT * extra_lp, 0.0),
            )

        # Base state after PA
        bs_logits = bs_head(context, pa_oh)
        bs_obs    = batch["base_state_after"] if teacher_force else None
        numpyro.sample("base_state_after", dist.Categorical(logits=bs_logits), obs=bs_obs)
