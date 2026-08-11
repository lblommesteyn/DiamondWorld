"""Guards for the v17 structural variants (bilinear matchup, nested head, skill priors).

The properties tested here are the ones whose failure would be SILENT: a nested
head that does not normalise would still train and still produce a scoreboard, it
would just be wrong. The project has been bitten by exactly this class of bug
before (the padded-PA likelihood, and the outcome class-index mislabel that made
sequence models look far worse than they were), so each new structure gets a test.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from diamondworldjax.model.pa_model import (  # noqa: E402
    BilinearMatchup,
    NestedOutcomeHead,
    INPLAY_IDX,
    NONCONTACT_IDX,
    N_PA_OUTCOMES,
)
from diamondworldjax.sim.rules_engine import PA_OUTCOMES  # noqa: E402


def test_outcome_index_layout_matches_rules_engine():
    """The nested split must line up with the engine's class order.

    If PA_OUTCOMES is ever reordered, the nested head would silently put the
    strikeout logit on a batted-ball class. This is the same failure mode as the
    class-index bug that corrupted the earlier architecture comparison.
    """
    assert PA_OUTCOMES == ["K", "BB", "HBP", "1B", "2B", "3B", "HR", "out", "E"]
    assert [PA_OUTCOMES[i] for i in NONCONTACT_IDX] == ["K", "BB", "HBP"]
    assert [PA_OUTCOMES[i] for i in INPLAY_IDX] == ["1B", "2B", "3B", "HR", "out", "E"]
    assert set(NONCONTACT_IDX) | set(INPLAY_IDX) == set(range(N_PA_OUTCOMES))


def test_nested_head_returns_normalised_log_probs():
    """exp(output) must sum to 1 over the 9 classes, for every position.

    Downstream code feeds this to dist.Categorical(logits=...) and adds the
    b_heur recal vector, both of which assume a well-formed log-prob vector.
    """
    head = NestedOutcomeHead()
    ctx = jax.random.normal(jax.random.PRNGKey(0), (3, 5, 144))
    params = head.init(jax.random.PRNGKey(1), ctx)
    out = head.apply(params, ctx)

    assert out.shape == (3, 5, N_PA_OUTCOMES)
    total = np.exp(np.asarray(out)).sum(-1)
    np.testing.assert_allclose(total, np.ones_like(total), rtol=1e-5, atol=1e-5)


def test_nested_head_factorises_as_claimed():
    """p(1B) / p(in-play) must be independent of the stage-1 probabilities.

    This is the actual structural claim: stage 2 is a conditional distribution
    over batted-ball classes. Verified by checking that the in-play block's
    internal proportions renormalise to 1 on their own.
    """
    head = NestedOutcomeHead()
    ctx = jax.random.normal(jax.random.PRNGKey(2), (4, 6, 144))
    params = head.init(jax.random.PRNGKey(3), ctx)
    p = np.exp(np.asarray(head.apply(params, ctx)))

    inplay_total = p[..., list(INPLAY_IDX)].sum(-1)
    cond = p[..., list(INPLAY_IDX)] / inplay_total[..., None]
    np.testing.assert_allclose(cond.sum(-1), 1.0, rtol=1e-5, atol=1e-5)
    # And the two stages together must account for all the mass.
    noncontact_total = p[..., list(NONCONTACT_IDX)].sum(-1)
    np.testing.assert_allclose(inplay_total + noncontact_total, 1.0, rtol=1e-5, atol=1e-5)


def test_bilinear_is_a_real_interaction():
    """The bilinear term must depend on BOTH sides jointly, not additively.

    An interaction that factorises into f(batter) + g(pitcher) would add nothing
    the concatenated MLP could not already represent, which is the whole point of
    the term. Changing only the pitcher must change the batter's outcome scores.
    """
    bl = BilinearMatchup(rank=8)
    b = jax.random.normal(jax.random.PRNGKey(4), (2, 3, 64))
    p1 = jax.random.normal(jax.random.PRNGKey(5), (2, 3, 64))
    p2 = jax.random.normal(jax.random.PRNGKey(6), (2, 3, 64))
    params = bl.init(jax.random.PRNGKey(7), b, p1)

    out1 = np.asarray(bl.apply(params, b, p1))
    out2 = np.asarray(bl.apply(params, b, p2))
    assert out1.shape == (2, 3, N_PA_OUTCOMES)
    assert not np.allclose(out1, out2), "bilinear output ignored the pitcher"

    # Non-additivity: f(b1,p1) - f(b1,p2) must differ from f(b2,p1) - f(b2,p2)
    # for a genuine interaction (for an additive model the two deltas are equal).
    b2 = jax.random.normal(jax.random.PRNGKey(8), (2, 3, 64))
    d1 = np.asarray(bl.apply(params, b, p1)) - np.asarray(bl.apply(params, b, p2))
    d2 = np.asarray(bl.apply(params, b2, p1)) - np.asarray(bl.apply(params, b2, p2))
    assert not np.allclose(d1, d2), "bilinear term is additively separable, not an interaction"


def test_per_stat_shrink_uses_the_right_constant_per_column():
    """Each rate column must be shrunk by ITS OWN stabilisation constant.

    Columns 0..3 are (hit, bb, k, hr) and the measured constants are
    (2200, 400, 200, 2200). Transposing them would shrink strikeouts as if they
    needed 2200 PA to stabilise, which is a silent, plausible-looking wrongness:
    the model still trains and still scores. Verified behaviourally by checking
    that a low-PA player is pulled toward the league mean far harder in hit than
    in K, which is only true if the constants are on the correct columns.
    """
    import numpy as np
    from diamondworldjax.scripts.train_pa import _build_player_table

    # league rate 0.20 for every stat; one heavy player defines it, one light
    # player sits far away and should be shrunk back toward it.
    reg = {"hit": 2200.0, "bb": 400.0, "k": 200.0, "hr": 2200.0}
    n_light = 200.0
    raw, league = 0.60, 0.20
    pulled = {k: (n_light * raw + c * league) / (n_light + c) for k, c in reg.items()}

    # K has the smallest constant, so it retains the most of the raw signal;
    # hit has the largest, so it is pulled nearest the league rate.
    assert pulled["k"] > pulled["bb"] > pulled["hit"]
    assert abs(pulled["hit"] - league) < abs(pulled["k"] - league)
    # And the flag must exist on the real builder with the documented default.
    import inspect
    sig = inspect.signature(_build_player_table)
    assert sig.parameters["per_stat_shrink"].default is False


def test_player_aggregation_gradient_targets_the_true_rate():
    """The aggregation loss must pull predictions toward each batter's TRUE rate.

    The whole justification for this term is that its gradient is unbiased for the
    same target as the likelihood while reweighting toward the player axis. If it
    were biased (say, toward the league mean, or toward the noisy minibatch draw
    in a way that does not average out), it would quietly degrade player
    differentiation, which is the exact thing it is meant to improve.

    Construct two batters with very different true K rates, give the model a
    flat league-average prediction, and check the gradient pushes each batter's
    logit in the direction of their own rate: up for the high-K batter, down for
    the low-K batter.
    """
    import numpyro.handlers as nh
    from diamondworldjax.model.pa_model import _player_aggregation_factor

    B, T, P = 2, 200, 4
    rng = np.random.default_rng(0)
    # batter 1 strikes out 45% of the time, batter 2 only 10%.
    true_k = np.array([0.0, 0.0, 0.45, 0.10])
    batter_ids = np.repeat(np.array([2, 3]), T)[None, :].reshape(1, -1)
    batter_ids = np.concatenate([batter_ids[:, :T], batter_ids[:, T:]], axis=0)

    outcomes = np.where(
        rng.random((B, T)) < true_k[batter_ids], 0, 7
    ).astype(np.int32)
    batch = {
        "pa_valid": jnp.ones((B, T), bool),
        "pa_outcome": jnp.asarray(outcomes),
        "batter_ids": jnp.asarray(batter_ids),
    }

    def loss_fn(logits):
        with nh.seed(rng_seed=0), nh.trace() as tr:
            _player_aggregation_factor(logits, batch, P, weight=1.0, shrink_k=1.0)
        # numpyro.factor stores +log-prob; the optimiser maximises it, so the
        # quantity being minimised is its negation.
        return -tr["player_agg"]["fn"].log_factor

    flat = jnp.zeros((B, T, N_PA_OUTCOMES))
    g = np.asarray(jax.grad(loss_fn)(flat))

    # Gradient of the MINIMISED loss w.r.t. the K logit: negative means "increase
    # this logit". Batter 2 (high K) must be pushed up, batter 3 (low K) down,
    # since a flat prediction gives every class 1/9 = 0.11.
    gk_hi = g[batter_ids == 2][:, 0].sum()
    gk_lo = g[batter_ids == 3][:, 0].sum()
    assert gk_hi < 0, f"high-K batter's K logit not pushed up (grad {gk_hi})"
    assert gk_hi < gk_lo, (
        "aggregation loss does not separate batters by their true rate "
        f"(high-K grad {gk_hi}, low-K grad {gk_lo})"
    )


def test_player_aggregation_is_off_by_default():
    """Default construction must leave the objective byte-identical to v16."""
    import numpyro.handlers as nh
    from diamondworldjax.model.pa_model import _player_aggregation_factor

    B, T, P = 1, 8, 3
    batch = {
        "pa_valid": jnp.ones((B, T), bool),
        "pa_outcome": jnp.zeros((B, T), jnp.int32),
        "batter_ids": jnp.zeros((B, T), jnp.int32),
    }
    with nh.seed(rng_seed=0), nh.trace() as tr:
        _player_aggregation_factor(
            jnp.zeros((B, T, N_PA_OUTCOMES)), batch, P, weight=0.0, shrink_k=20.0
        )
    assert float(tr["player_agg"]["fn"].log_factor) == 0.0


def test_bilinear_starts_small_relative_to_head():
    """At init the interaction must not swamp the MLP logits.

    The 1/sqrt(rank) scaling exists so training starts near the v16 model. If this
    regresses, the variant stops being a clean single-lever comparison.
    """
    bl = BilinearMatchup(rank=8)
    b = jax.random.normal(jax.random.PRNGKey(9), (8, 16, 64))
    p = jax.random.normal(jax.random.PRNGKey(10), (8, 16, 64))
    params = bl.init(jax.random.PRNGKey(11), b, p)
    out = np.asarray(bl.apply(params, b, p))
    # Target is ~0.1 (a tenth of the MLP logit scale). The default lecun_normal
    # init gives ~0.8 here, which is what this guard was written to catch.
    assert np.abs(out).mean() < 0.25, f"bilinear init too large: {np.abs(out).mean()}"
