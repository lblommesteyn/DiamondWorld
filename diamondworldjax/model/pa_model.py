"""PA-level NumPyro generative model (v2).

Changes from v1:
- Park embedding added to context (captures park factor variance)
- Runs-scored upweighting: non-zero run events get 3x loss weight so the
  model can't collapse to always predicting 0 runs
"""
from __future__ import annotations

import contextlib

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


# Outcome-class layout, mirroring rules_engine.PA_OUTCOMES. The order is guarded
# by tests/test_pa_encoding.py, which exists because a silent index bug once made
# the sequence models look far worse than they were.
_IDX_K, _IDX_BB, _IDX_HBP = 0, 1, 2
# Everything the batter put in play (or reached on): 1B, 2B, 3B, HR, out, E.
INPLAY_IDX = (3, 4, 5, 6, 7, 8)
NONCONTACT_IDX = (_IDX_K, _IDX_BB, _IDX_HBP)
N_STAGE1 = 4              # K, BB, HBP, in-play
N_INPLAY = len(INPLAY_IDX)


class BilinearMatchup(nn.Module):
    """Low-rank bilinear interaction between the batter and pitcher embeddings.

    A plate appearance is a MATCHUP, but the baseline context is
    `concat([state, pitcher_z, batter_z, park])`, which forces the MLP to
    discover every interaction from a concatenation. The v11 platoon lever is the
    evidence that this fails: it fixed exactly one interaction (batter side x
    pitcher hand) by feeding it in pre-multiplied, and its docstring says outright
    that this "lets the head learn the platoon interaction directly instead of
    extracting it from two 32-dim player embeddings".

    This is the general form of that fix. For each outcome class k we add a
    bilinear score b^T W_k p, with W_k held at rank `rank` by factorising it as
    U_k^T V_k. That is matrix factorisation in the recommender-systems sense,
    which is the standard structure for a two-sided interaction problem, and it
    costs 2 * rank * n_out parameters per side instead of a full D x D matrix per
    class.

    Initialisation is deliberately small so training starts close to the baseline
    model and the bilinear term has to earn its weight. With unit-variance inputs
    of width D, a kernel of stddev s gives projected elements of std s*sqrt(D);
    their product has std s^2 * D, and summing `rank` of them and dividing by
    sqrt(rank) leaves std ~ s^2 * D. At D=64 and s=0.04 that is ~0.1, roughly a
    tenth of the MLP logit scale. The default Dense init (lecun_normal) instead
    yields ~0.8, which is NOT small relative to the logits and would make this a
    two-lever change rather than a clean single-lever comparison; the guard in
    tests/test_pa_model_variants.py exists because that was the first thing to
    go wrong here.
    """
    rank: int = 8
    n_out: int = N_PA_OUTCOMES
    init_stddev: float = 0.04

    @nn.compact
    def __call__(self, batter_z: jnp.ndarray, pitcher_z: jnp.ndarray) -> jnp.ndarray:
        lead = batter_z.shape[:-1]
        init = nn.initializers.normal(stddev=self.init_stddev)
        u = nn.Dense(self.rank * self.n_out, use_bias=False, kernel_init=init,
                     name="batter_proj")(batter_z)
        v = nn.Dense(self.rank * self.n_out, use_bias=False, kernel_init=init,
                     name="pitcher_proj")(pitcher_z)
        u = u.reshape(*lead, self.n_out, self.rank)
        v = v.reshape(*lead, self.n_out, self.rank)
        return (u * v).sum(-1) / jnp.sqrt(jnp.asarray(self.rank, u.dtype))


class NestedOutcomeHead(nn.Module):
    """Two-stage outcome head matching the generative structure of a PA.

    A plate appearance resolves in two physically distinct stages:

        stage 1 (plate discipline): {K, BB, HBP, in-play}
        stage 2 (contact quality):  in-play -> {1B, 2B, 3B, HR, out, E}

    The flat 9-way softmax forces one representation to serve both, even though
    the features that drive them differ: plate discipline is a K/BB skill, while
    what happens to a batted ball is a contact-quality question. That split is
    exactly what the v16 result showed empirically, where adding xBA-style contact
    features moved HR and hit but left BB flat.

    Returning normalised LOG-PROBABILITIES over all 9 classes (rather than a
    nested distribution object) is deliberate: every downstream consumer, the
    b_heur recalibration vector, the simulator, and the eval scripts that read
    `.logits`, keeps working unchanged, because softmax(log p) == p.
    """
    hidden_dim: int = 256

    # Not decorated: Flax allows exactly one @nn.compact method per module. This
    # helper is called from __call__, so it runs inside that compact context and
    # its explicitly-named submodules are registered normally.
    def _trunk(self, context: jnp.ndarray, name: str) -> jnp.ndarray:
        h = nn.Dense(self.hidden_dim, name=f"{name}_d1")(context)
        h = nn.LayerNorm(name=f"{name}_ln1")(h)
        h = nn.relu(h)
        h = nn.Dense(self.hidden_dim, name=f"{name}_d2")(h)
        h = nn.LayerNorm(name=f"{name}_ln2")(h)
        return nn.relu(h)

    @nn.compact
    def __call__(self, context: jnp.ndarray) -> jnp.ndarray:
        h1 = self._trunk(context, "stage1")
        s1 = nn.Dense(N_STAGE1, name="stage1_out")(h1)          # K, BB, HBP, in-play
        h2 = self._trunk(context, "stage2")
        s2 = nn.Dense(N_INPLAY, name="stage2_out")(h2)          # within in-play

        lp1 = jax.nn.log_softmax(s1, axis=-1)
        lp2 = jax.nn.log_softmax(s2, axis=-1)

        # Chain rule: log p(class) = log p(in-play) + log p(class | in-play).
        inplay_lp = lp1[..., N_STAGE1 - 1:N_STAGE1]
        # Assembled by concatenation rather than scattered .at[].set() writes.
        # Both are mathematically identical because NONCONTACT_IDX and INPLAY_IDX
        # are contiguous and in order (asserted in tests/test_pa_model_variants.py),
        # but the scatter form built nine dynamic-update-slices whose reverse-mode
        # gradient would not compile in reasonable time: a smoke run sat at 4% GPU
        # and never passed step 0. Concatenation is a single fused op.
        return jnp.concatenate([lp1[..., :len(NONCONTACT_IDX)], inplay_lp + lp2], axis=-1)


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


def _player_aggregation_factor(
    pa_logits: jnp.ndarray,
    batch: dict,
    n_players: int,
    weight: float,
    shrink_k: float,
) -> None:
    """Auxiliary loss that reweights the objective toward the PLAYER axis.

    THE DIAGNOSIS. Training maximises the per-PA likelihood, but the metric the
    project is judged on is cross-player rate correlation. Those are not the same
    objective, and the mismatch is extreme. Marginal outcome entropy is 1.495
    nats and the best model reaches about 1.49, so the entire player-attributable
    share of the training signal is roughly 0.005 nats: about 0.3% of the loss.
    Put plainly, ~99.7% of every gradient step goes into getting the league-average
    plate appearance right, while 100% of the evaluation is about telling players
    apart. That is why architecture levers keep landing on zero (they add capacity
    to fit the saturated 99.7%) while feature levers move the number.

    THE MECHANISM. This is NOT a new source of information, and it is important
    not to oversell it as one. The per-PA likelihood already drives predicted
    probabilities toward observed outcomes, and this term has the same optimum.
    What it changes is the WEIGHTING. Cross-entropy weights every plate
    appearance equally, so a batter with 700 PA dominates one with 150. Squared
    error on per-batter AGGREGATES weights every BATTER equally, which is exactly
    the axis the metric measures. The gradient is unbiased for the same target:
    d/dp E[(p - obs)^2] = 2(p - true_rate).

    Aggregation is over the minibatch, so each batter contributes only a handful
    of PAs and the observed rate is very noisy. That is tolerable because the
    noise is unbiased, but batters with one or two PAs in a batch carry almost no
    information, so contributions are shrunk by n/(n + shrink_k).

    THE RISK, stated up front. Pushed hard enough this degenerates toward
    reproducing each batter's own historical rate, and a rate-features-only model
    scores 0.438 AVG, well below v16's 0.611. So the weight matters and larger is
    not better: if there is a win here it is at an interior optimum, and a null or
    a regression is a real possible outcome.
    """
    p = jax.nn.softmax(pa_logits, axis=-1)
    valid = batch["pa_valid"].astype(p.dtype)
    y = batch["pa_outcome"]

    # The four stats the evaluation metric scores, in its own definitions:
    # K, BB (incl. HBP), Hit (1B+2B+3B+HR), HR.
    pred = [
        p[..., _IDX_K],
        p[..., _IDX_BB] + p[..., _IDX_HBP],
        p[..., 3] + p[..., 4] + p[..., 5] + p[..., 6],
        p[..., 6],
    ]
    obs = [
        (y == _IDX_K),
        (y == _IDX_BB) | (y == _IDX_HBP),
        (y >= 3) & (y <= 6),
        (y == 6),
    ]

    idx = batch["batter_ids"].reshape(-1)
    w = valid.reshape(-1)

    def _seg(v):
        return jax.ops.segment_sum(v.reshape(-1) * w, idx, num_segments=n_players)

    n = jax.ops.segment_sum(w, idx, num_segments=n_players)
    safe_n = jnp.maximum(n, 1.0)

    se = jnp.zeros((n_players,), p.dtype)
    for pr, ob in zip(pred, obs):
        se = se + (_seg(pr) / safe_n - _seg(ob.astype(p.dtype)) / safe_n) ** 2

    shrink = n / (n + shrink_k)
    loss = (shrink * se).sum() / jnp.maximum(shrink.sum(), 1e-6)

    # Scale by the number of valid PAs so the term is commensurate with the
    # likelihood (which is ~1.5 nats per PA); `weight` is then an interpretable
    # fraction of the total objective rather than an arbitrary constant.
    numpyro.factor("player_agg", -weight * w.sum() * loss)


def pa_model(
    batch: dict,
    player_table: dict,
    teacher_force: bool = True,
    outcome_only: bool = False,
    fatigue: bool = False,
    platoon: bool = False,
    kl_scale: float = 1.0,
    bilinear_rank: int = 0,
    nested: bool = False,
    skill_prior: str = "iso",
    player_agg_weight: float = 0.0,
    player_agg_shrink: float = 20.0,
    season_base: int = 2015,
    n_seasons: int = 9,
    pitchformer: bool = False,
    pitchformer_dim: int = 128,
    pitchformer_layers: int = 2,
    pitchformer_heads: int = 4,
    pitchformer_dropout: float = 0.0,
    pa_arch: str = "transformer",
    pitchformer_position: str = "auto",
    runs_upweight: float = 2.0,
    player_skills_override: jnp.ndarray | None = None,
) -> None:
    B, T = batch["pa_valid"].shape
    P    = player_table["stats"].shape[0]

    # Per-player latent skill vectors. player_skills is a GLOBAL latent over all P
    # players; the likelihood below is over a minibatch of B games. To keep the ELBO
    # balanced (an unbiased estimate of the full-data ELBO), its prior KL must be
    # scaled by kl_scale = batch_games / total_train_games. Without this the global
    # KL is ~total/batch times over-weighted and the posterior collapses to the
    # prior (player_mu ~ 0), switching the latent off. See train_pa for the value.
    #
    # skill_prior controls the shape of that prior:
    #   "iso"     N(0, I), the v6..v16 behavior.
    #   "learned" N(0, diag(tau)) with tau a learned per-dimension scale, so the
    #             model sets its own shrinkage strength instead of having it
    #             pinned at 1.0. This is the lever that matters for a metric that
    #             is fundamentally about shrinkage quality.
    #   "lkj"     N(0, LL^T) with a full learned correlation via an LKJ prior.
    #             NOTE: this is partly redundant. player_skills is already pushed
    #             through SkillFusionLayer, a Dense map, and a linear map of an
    #             isotropic Gaussian is already a correlated Gaussian, so the
    #             model can represent correlated skills today. What "lkj" adds is
    #             correlation in the PRIOR (hence in the KL), not new expressive
    #             power. Included so the redundancy can be measured rather than
    #             assumed.
    if player_skills_override is not None:
        if player_skills_override.shape[0] != P:
            raise ValueError("player_skills_override must have one row per player")
        player_skills = player_skills_override
    else:
        with _nph.scale(scale=kl_scale):
            if skill_prior == "iso":
                skill_dist = dist.Normal(
                    jnp.zeros((P, SKILL_DIM)), jnp.ones((P, SKILL_DIM))
                ).to_event(2)
            elif skill_prior == "learned":
                tau = numpyro.sample(
                    "skill_tau", dist.HalfNormal(jnp.ones(SKILL_DIM)).to_event(1)
                )
                skill_dist = dist.Normal(
                    jnp.zeros((P, SKILL_DIM)), jnp.broadcast_to(tau, (P, SKILL_DIM))
                ).to_event(2)
            elif skill_prior == "lkj":
                tau = numpyro.sample(
                    "skill_tau", dist.HalfNormal(jnp.ones(SKILL_DIM)).to_event(1)
                )
                L_omega = numpyro.sample(
                    "skill_L", dist.LKJCholesky(SKILL_DIM, concentration=2.0)
                )
                scale_tril = tau[:, None] * L_omega
                skill_dist = dist.MultivariateNormal(
                    jnp.zeros((P, SKILL_DIM)), scale_tril=scale_tril
                ).to_event(1)
            elif skill_prior == "walk":
                sigma_walk = numpyro.sample("skill_walk_sigma", dist.HalfNormal(0.3))
                eps = numpyro.sample(
                    "player_skill_eps",
                    dist.Normal(jnp.zeros((P, n_seasons, SKILL_DIM)),
                                jnp.ones((P, n_seasons, SKILL_DIM))).to_event(3),
                )
                steps = jnp.concatenate(
                    [eps[:, :1, :], eps[:, 1:, :] * sigma_walk], axis=1
                )
                player_skills = numpyro.deterministic(
                    "player_skills_walk", jnp.cumsum(steps, axis=1)
                )
                skill_dist = None
            else:
                raise ValueError(f"unknown skill_prior {skill_prior!r}")
            if skill_dist is not None:
                player_skills = numpyro.sample("player_skills", skill_dist)

    # Player embeddings
    season_idx = None
    if skill_prior == "walk":
        # Clamp into [0, n_seasons-1]. The clamp is what makes the test season work:
        # 2024 is unseen in training, so it maps to the last TRAINED season's skill,
        # which is exactly the "most recent form" quantity the recency lever
        # approximates by hand.
        season_idx = jnp.clip(
            batch["season"].astype(jnp.int32) - season_base, 0, n_seasons - 1
        )
    pitcher_z, batter_z = encode_players_numpyro(
        player_table["stats"],
        player_table["league"],
        player_table["hand"],
        pitcher_ids   = batch["pitcher_ids"],
        batter_ids    = batch["batter_ids"],
        player_skills = player_skills,
        season_idx    = season_idx,
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

    context_raw = jnp.concatenate([game_state, pitcher_z, batter_z, park_emb], axis=-1)  # (B, T, 144/145)

    if pitchformer:
        if pa_arch in ("gru", "gru_skip"):
            from .pa_transformer import pa_gru_numpyro
            context = pa_gru_numpyro(
                context_raw,
                batch["pa_valid"],
                d_model=pitchformer_dim,
                n_layers=pitchformer_layers,
                # Keep the checkpoint namespace distinct: this is an
                # experiment, not a compatible re-interpretation of a GRU.
                name="pa_gru_skip" if pa_arch == "gru_skip" else "pa_gru",
            )
            # The recurrent state is useful for recent PA history, but it
            # should not be a bottleneck for matchup/player features.  This
            # residual path lets the outcome head use both representations.
            if pa_arch == "gru_skip":
                context = jnp.concatenate([context_raw, context], axis=-1)
        else:
            from .pa_transformer import pa_transformer_numpyro
            context = pa_transformer_numpyro(
                context_raw,
                batch["pa_valid"],
                d_model=pitchformer_dim,
                n_layers=pitchformer_layers,
                n_heads=pitchformer_heads,
                dropout=pitchformer_dropout,
                train=teacher_force, position_encoding=pitchformer_position,
            )
    else:
        context = context_raw

    dummy_oh = jnp.zeros((B, T, N_PA_OUTCOMES))

    # Outcome-only model (v6): a single deeper head; runs + base_state come from
    # the deterministic/empirical rules engine at rollout time, not the network.
    if outcome_only:
        if nested:
            pa_head = flax_module("pa_outcome_head_nested", NestedOutcomeHead(), context)
        else:
            pa_head = flax_module("pa_outcome_head_v6", PAOutcomeHeadV6(), context)
        pa_logits = pa_head(context)

        if bilinear_rank > 0:
            # Additive in logit space. For the nested head pa_logits are already
            # normalised log-probs, so adding here makes them unnormalised again;
            # that is fine and intended, since every consumer applies a softmax
            # (dist.Categorical(logits=...) normalises internally). The bilinear
            # term therefore adjusts the matchup on the same scale in both heads.
            bl = flax_module(
                "bilinear_matchup",
                BilinearMatchup(rank=bilinear_rank),
                batter_z, pitcher_z,
            )
            pa_logits = pa_logits + bl(batter_z, pitcher_z)

        with numpyro.plate("games", B, dim=-2), numpyro.plate("pas", T, dim=-1):
            if teacher_force:
                # Mask padded PAs out of the likelihood. Padded positions carry
                # pa_outcome=-1, remapped to 0 (=K); without this mask they were
                # counted as observed strikeouts (~16% of positions), inflating the
                # model's K rate. Only applied when observing (teacher_force); at
                # eval the site stays a plain Categorical so .logits is accessible.
                pa_obs = jnp.where(batch["pa_outcome"] == -1, 0, batch["pa_outcome"])
                with _nph.mask(mask=batch["pa_valid"]):
                    numpyro.sample("pa_outcome", dist.Categorical(logits=pa_logits), obs=pa_obs)
            else:
                numpyro.sample("pa_outcome", dist.Categorical(logits=pa_logits), obs=None)

        if teacher_force and player_agg_weight > 0.0:
            _player_aggregation_factor(
                pa_logits, batch, P, player_agg_weight, player_agg_shrink
            )
        return

    pa_head   = flax_module("pa_outcome_head",       PAOutcomeHead(),      context)
    runs_head = flax_module("runs_scored_head",      RunsScoredHead(),     context, dummy_oh)
    bs_head   = flax_module("base_state_after_head", BaseStateAfterHead(), context, dummy_oh)

    pa_logits = pa_head(context)  # (B, T, 9)

    # Mask padded PAs out of the observed likelihood, but only when teacher-forcing
    # (observing); at eval the sites stay plain Categoricals so .logits is readable.
    _pctx = _nph.mask(mask=batch["pa_valid"]) if teacher_force else contextlib.nullcontext()
    with numpyro.plate("games", B, dim=-2), numpyro.plate("pas", T, dim=-1), _pctx:

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
                jnp.where(runs_obs > 0, runs_upweight * extra_lp, 0.0),
            )

        # Base state after PA
        bs_logits = bs_head(context, pa_oh)
        bs_obs    = batch["base_state_after"] if teacher_force else None
        numpyro.sample("base_state_after", dist.Categorical(logits=bs_logits), obs=bs_obs)
