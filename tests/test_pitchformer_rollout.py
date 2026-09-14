import jax.numpy as jnp
import jax
import numpy as np
from flax.core import unfreeze

from diamondworldjax.domain import PAOutcome
from diamondworldjax.model.pitchformer import TransformerA, TransformerB
from diamondworldjax.sim.rules_engine import EmpiricalEngine
from diamondworldjax.simulate.pitchformer_rollout import (
    PitchformerHeads, _cache_template, _cached_apply, rollout_batch,
)


class _A:
    def apply(self, _params, batch, *, train):
        b, t = batch["valid"].shape
        logits = jnp.zeros((b, t, 8)).at[..., 0].set(100.0)
        return {
            "type_logits": logits,
            "stuff_mu": jnp.zeros((b, t, 8, 5)),
            "stuff_logsigma": jnp.full((b, t, 8, 5), -30.0),
        }


class _B:
    def apply(self, _params, batch, *, train):
        b, t = batch["valid"].shape
        return {
            "swing_logit": jnp.full((b, t), 100.0),
            "contact_logit": jnp.full((b, t), 100.0),
            "foul_logit": jnp.full((b, t), -100.0),
            "hbp_logit": jnp.full((b, t), -100.0),
        }


class _C:
    def apply(self, _params, batch, *, train):
        b, t = batch["valid"].shape
        return {"event_logits": jnp.full((b, t, 8), 100.0)}


class _D:
    def apply(self, _params, batch, *, train):
        b, t = batch["valid"].shape
        return {
            "launch_mu": jnp.zeros((b, t, 2)),
            "launch_logsigma": jnp.full((b, t, 2), -30.0),
            "outcome_logits": jnp.zeros((b, t, 6)).at[..., 4].set(100.0),
        }


class _NoSwingHbpB:
    def apply(self, _params, batch, *, train):
        b, t = batch["valid"].shape
        return {
            "swing_logit": jnp.full((b, t), -100.0),
            "contact_logit": jnp.full((b, t), -100.0),
            "foul_logit": jnp.full((b, t), -100.0),
            "hbp_logit": jnp.full((b, t), 100.0),
        }


class _NoEventC:
    def apply(self, _params, batch, *, train):
        b, t = batch["valid"].shape
        return {"event_logits": jnp.full((b, t, 8), -100.0)}


def _batch(b=2, t=3):
    return {
        "valid": np.ones((b, t), np.float32),
        "ctx": np.zeros((b, t, 24), np.float32),
        "geom": np.zeros((b, t, 10), np.float32),
        "pitcher_idx": np.ones((b, t), np.int32),
        "batter_idx": np.ones((b, t), np.int32),
        "park_idx": np.ones((b, t), np.int32),
        "pitch_type": np.zeros((b, t), np.int32),
        "stuff": np.zeros((b, t, 5), np.float32),
        "swing": np.zeros((b, t), np.float32),
        "contact": np.zeros((b, t), np.float32),
        "foul": np.zeros((b, t), np.float32),
        "launch": np.zeros((b, t, 2), np.float32),
    }


def test_rollout_composes_abcd_and_uses_generated_terminal_outcome():
    heads = PitchformerHeads(_A(), _B(), _C(), _D(), None, None, {}, {})
    got = rollout_batch(heads, _batch(), seed=0, engine=EmpiricalEngine())

    # Every generated pitch is put in play and D always selects HR.  This proves
    # that the PA result is composed from generated A/B/D output, rather than
    # copied from the recorded batch.
    assert got["active"].all()
    assert got["pa_terminal"].all()
    assert np.all(got["pa_outcome"] == int(PAOutcome.HOME_RUN))
    # C produces one mutually-exclusive event type, never incompatible flags.
    assert np.all(got["event"].sum(axis=-1) == 1)


def test_rollout_hybrid_replaces_only_generated_in_play_outcomes():
    calls = []

    def pa_in_play(state, source, in_play, original, _key):
        calls.append((state["outs"].copy(), source.copy(), in_play.copy(), original["ctx"].shape))
        return np.full(len(source), int(PAOutcome.SINGLE), np.int32)

    # A/B make every pitch a ball in play and D selects HR.  The hybrid callback
    # must replace just that D-derived PA class, while retaining normal rollout
    # scheduling and state advancement.
    heads = PitchformerHeads(_A(), _B(), _C(), _D(), None, None, {}, {})
    got = rollout_batch(
        heads, _batch(b=1, t=2), seed=0, engine=EmpiricalEngine(),
        terminal_outcome_sampler=pa_in_play,
    )

    assert len(calls) == 2
    assert calls[0][1].tolist() == [0]
    assert calls[0][2].tolist() == [True]
    assert np.all(got["pa_outcome"] == int(PAOutcome.SINGLE))


def test_rollout_advances_observed_pa_schedule_after_generated_terminal():
    batch = _batch(b=1, t=3)
    batch["pa_start"] = np.array([[True, False, True]])
    batch["batter_idx"][:] = np.array([[2, 99, 3]])
    heads = PitchformerHeads(_A(), _B(), _C(), _D(), None, None, {}, {})
    got = rollout_batch(heads, batch, seed=0, engine=EmpiricalEngine())

    # The first generated in-play ends the PA immediately, so the next pitch
    # uses the second scheduled PA's batter (3), not the recorded mid-PA actor.
    assert got["active"].tolist() == [[True, True, False]]
    assert got["batter_idx"][0, :2].tolist() == [2, 3]


def test_rollout_samples_hbp_as_a_terminal_no_swing_outcome():
    heads = PitchformerHeads(_A(), _NoSwingHbpB(), None, None, None, None)
    got = rollout_batch(heads, _batch(b=1, t=2), seed=0, engine=EmpiricalEngine())

    assert got["hbp"].all()
    assert got["pa_terminal"].all()
    assert np.all(got["pa_outcome"] == int(PAOutcome.HIT_BY_PITCH))


def test_rollout_keeps_c_rare_events_rare():
    heads = PitchformerHeads(_A(), _B(), _NoEventC(), None, None, None, {}, None)
    got = rollout_batch(heads, _batch(b=1, t=2), seed=0, engine=EmpiricalEngine())

    assert not got["event"].any()


def test_kv_decode_matches_full_causal_forward():
    """The rollout cache must retain the original strict-causal semantics."""
    batch = _batch(b=2, t=4)
    batch["valid"][1, 2:] = 0
    batch["ctx"] = np.random.default_rng(4).normal(size=(2, 4, 24)).astype(np.float32)
    model = TransformerA(n_pitchers=4, n_batters=4, n_parks=4,
                         d_model=12, n_layers=2, n_heads=3, dropout=0.0)
    full_batch = {name: jnp.asarray(value) for name, value in batch.items()}
    params = model.init(jax.random.PRNGKey(0), full_batch, train=False)
    full = model.apply(params, full_batch, train=False)["type_logits"]

    cache = _cache_template(model, params, batch)
    history_valid = jnp.zeros((2, 4), bool)
    decoded = []
    for t in range(4):
        token = {name: jnp.asarray(value)[:, t:t + 1] for name, value in batch.items()}
        token["_decode_position"] = jnp.array(t, jnp.int32)
        token["_cache_valid"] = history_valid
        out, cache = _cached_apply(model, params, cache, token)
        decoded.append(out["type_logits"][:, 0])
        history_valid = history_valid.at[:, t].set(jnp.asarray(batch["valid"][:, t], bool))

    got = jnp.stack(decoded, axis=1)
    np.testing.assert_allclose(got, full, rtol=2e-5, atol=2e-5)


def test_cached_rollout_restores_trailing_padding():
    """The compiled scan trims only internal padding, not its public layout."""
    batch = _batch(b=1, t=4)
    batch["valid"][0, 2:] = 0
    kw = dict(n_pitchers=4, n_batters=4, n_parks=4,
              d_model=12, n_layers=2, n_heads=3, dropout=0.0)
    a, b = TransformerA(**kw), TransformerB(**kw)
    jbatch = {name: jnp.asarray(value) for name, value in batch.items()}
    a_params = a.init(jax.random.PRNGKey(10), jbatch, train=False)
    b_params = b.init(jax.random.PRNGKey(11), jbatch, train=False)

    got = rollout_batch(PitchformerHeads(a, b, None, None, a_params, b_params),
                        batch, seed=3, engine=EmpiricalEngine(), decode_len=4)
    assert got["active"].shape == (1, 4)
    assert not got["active"][0, 2:].any()
    assert np.all(got["pa_outcome"][0, 2:] == -1)


def test_cached_rollout_stops_when_generated_third_out_ends_half():
    """A per-half cached rollout must not consume rows from the next half."""
    batch = _batch(b=1, t=12)
    kw = dict(n_pitchers=4, n_batters=4, n_parks=4,
              d_model=12, n_layers=1, n_heads=3, dropout=0.0,
              window_size=1)
    a, b = TransformerA(**kw), TransformerB(**kw)
    jbatch = {name: jnp.asarray(value) for name, value in batch.items()}
    a_params = unfreeze(a.init(jax.random.PRNGKey(10), jbatch, train=False))
    b_params = unfreeze(b.init(jax.random.PRNGKey(11), jbatch, train=False))

    # Make every pitch a strike in the zone: three strikeouts produce the
    # third out on pitch nine.  These are real Transformer heads so the test
    # exercises the cached compiled scan rather than the reference fallback.
    a_params["params"]["stuff_mu"]["kernel"] = jnp.zeros_like(
        a_params["params"]["stuff_mu"]["kernel"])
    a_params["params"]["stuff_mu"]["bias"] = jnp.zeros_like(
        a_params["params"]["stuff_mu"]["bias"])
    a_params["params"]["stuff_logsigma"]["kernel"] = jnp.zeros_like(
        a_params["params"]["stuff_logsigma"]["kernel"])
    a_params["params"]["stuff_logsigma"]["bias"] = jnp.full_like(
        a_params["params"]["stuff_logsigma"]["bias"], -30.0)
    for name in ("swing", "hbp"):
        b_params["params"][name]["kernel"] = jnp.zeros_like(
            b_params["params"][name]["kernel"])
        b_params["params"][name]["bias"] = jnp.full_like(
            b_params["params"][name]["bias"], -100.0)

    got = rollout_batch(
        PitchformerHeads(a, b, None, None, a_params, b_params),
        batch,
        seed=3,
        engine=EmpiricalEngine(),
        decode_len=12,
    )

    np.testing.assert_array_equal(got["active"][0, :9], np.ones(9, bool))
    assert not got["active"][0, 9:].any()
    # _initial_state starts synthetic rollouts in the fifth inning.
    assert got["final_state"]["inning"][0] == 5
    assert got["final_state"]["half"][0] == 1
    assert not got["final_state"]["ended"][0]
