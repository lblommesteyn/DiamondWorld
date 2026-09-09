"""Autoregressive rollout for the separately trained pitchformer A--D heads.

This is deliberately a *generated-state* evaluator, not a replacement for the
PA simulator.  It generates pitch type/stuff (A), batter response (B), and a
batted-ball outcome (D) one pitch at a time.  C is executed and its sampled
event rates are recorded, but runner-event placement is not yet modelled: C's
current labels do not identify the runner involved.  Base advancement therefore
continues to come from the validated empirical PA transition engine.

The observed pitcher/batter/park schedule is exogenous.  That makes this a
useful first free-roll gate for the pitch process while keeping roster/manager
generation out of scope.
"""
from __future__ import annotations

from functools import partial
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from diamondworldjax.data.pitch_seq import STUFF_CENTRE, STUFF_SCALE
from diamondworldjax.domain import PAOutcome
from diamondworldjax.model.pitchformer import TransformerA, TransformerB
from diamondworldjax.model.transformer_c import EVENT_FLAGS, TransformerC
from diamondworldjax.model.transformer_d import TransformerD
from diamondworldjax.sim.rules_engine import BS_AFTER, N_KEYS, OUT_INC, RUNS, EmpiricalEngine
from diamondworldjax.sim.c_transition_engine import CTransitionEngine

ZONE_HALF_WIDTH = 0.83
ZONE_BOTTOM = 1.52
ZONE_TOP = 3.42
_D_TO_PA = np.array([
    int(PAOutcome.OUT), int(PAOutcome.SINGLE), int(PAOutcome.DOUBLE),
    int(PAOutcome.TRIPLE), int(PAOutcome.HOME_RUN), int(PAOutcome.ERROR),
], dtype=np.int32)


@dataclass(frozen=True)
class PitchformerHeads:
    """Independent A--D modules and their parameter pytrees."""

    a: TransformerA
    b: TransformerB
    c: TransformerC | None
    d: TransformerD | None
    a_params: Any
    b_params: Any
    c_params: Any | None = None
    d_params: Any | None = None


def _sample_c_event(logits: jax.Array, key: jax.Array) -> jax.Array:
    """Sample one legal C event from independently-trained Bernoulli heads.

    C's loss estimates marginal probabilities, not categorical logits.  The
    previous code passed those logits straight to ``categorical``, which turns
    a rare-event logit into an arbitrary relative category weight.  Preserve
    P(any event) under independent Bernoulli heads, then distribute an event
    across its possible types in proportion to their marginal probabilities.
    """
    if logits.shape[-1] == 256:
        return jax.random.categorical(key, logits, axis=-1) - 1
    p = jax.nn.sigmoid(logits)
    p_none = jnp.prod(1.0 - p, axis=-1, keepdims=True)
    p_event = (1.0 - p_none) * p / jnp.maximum(p.sum(axis=-1, keepdims=True), 1e-8)
    probs = jnp.concatenate([p_none, p_event], axis=-1)
    return jax.random.categorical(key, jnp.log(jnp.clip(probs, 1e-8, 1.0)), axis=-1) - 1


def _event_flags(event, mode):
    if mode == 'bundles':
        return (((event + 1)[..., None] & (1 << jnp.arange(8))) != 0)
    return jax.nn.one_hot(jnp.maximum(event, 0), 8, dtype=bool) & (event >= 0)[..., None]


def _initial_state(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Decode the first recorded pre-pitch state for every half-inning.

    We retain the observed score differential at the beginning of a half.  From
    that point onward every count, base, out, and batting-team score change is
    generated.  A half is the pitchformer sequence unit, so crossing into the
    next half is intentionally outside this evaluator's contract.
    """
    ctx = np.asarray(batch["ctx"])
    base = ((ctx[:, 0, 3] > .5).astype(np.int32)
            + 2 * (ctx[:, 0, 4] > .5).astype(np.int32)
            + 4 * (ctx[:, 0, 5] > .5).astype(np.int32))
    half = np.rint(ctx[:, 0, 9]).astype(np.int32)
    diff = ctx[:, 0, 7].astype(np.float32) * 10.0
    return {
        "balls": np.rint(ctx[:, 0, 0] * 3).astype(np.int32),
        "strikes": np.rint(ctx[:, 0, 1] * 2).astype(np.int32),
        "outs": np.rint(ctx[:, 0, 2] * 2).astype(np.int32),
        "base": base,
        "home_score": np.where(half == 1, np.maximum(diff, 0), np.maximum(-diff, 0)).astype(np.int32),
        "away_score": np.where(half == 0, np.maximum(diff, 0), np.maximum(-diff, 0)).astype(np.int32),
        "inning": np.rint(ctx[:, 0, 8] * 4.0 + 5.0).astype(np.int32),
        "half": half,
        "tto": np.rint(ctx[:, 0, 12] * 4.0).astype(np.int32),
        "pitch_count": np.zeros(ctx.shape[0], dtype=np.int32),
        "pa_slot": np.zeros(ctx.shape[0], dtype=np.int32),
        "ended": np.zeros(ctx.shape[0], dtype=bool),
    }


def _batting_diff(state: dict[str, np.ndarray]) -> np.ndarray:
    return np.where(state["half"] == 0,
                    state["away_score"] - state["home_score"],
                    state["home_score"] - state["away_score"])


def _write_generated_context(
    target: np.ndarray, template: np.ndarray, state: dict[str, np.ndarray], t: int,
) -> None:
    """Write generated dynamic features while retaining exogenous inputs."""
    b = state["base"]
    target[:, t] = template[:, t]
    target[:, t, 0] = state["balls"] / 3.0
    target[:, t, 1] = state["strikes"] / 2.0
    target[:, t, 2] = state["outs"] / 2.0
    target[:, t, 3] = (b & 1 > 0)
    target[:, t, 4] = (b & 2 > 0)
    target[:, t, 5] = (b & 4 > 0)
    target[:, t, 6] = ((b & 2 > 0) | (b & 4 > 0))
    diff = _batting_diff(state)
    target[:, t, 7] = np.clip(diff, -10, 10) / 10.0
    target[:, t, 8] = (state["inning"] - 5.0) / 4.0
    target[:, t, 9] = state["half"]
    target[:, t, 10] = 1.0 - state["half"]
    target[:, t, 11] = ((state["inning"] >= 7)
                         & (np.abs(diff) <= 1)).astype(np.float32)
    target[:, t, 12] = np.clip(state["tto"], 0, 4) / 4.0


def _sample_a(out: dict[str, jnp.ndarray], t: int, key: jax.Array) -> tuple[np.ndarray, np.ndarray]:
    kt, ks = jax.random.split(key)
    typ = jax.random.categorical(kt, out["type_logits"][:, t]).astype(jnp.int32)
    idx = typ[:, None, None]
    mu = jnp.take_along_axis(
        out["stuff_mu"][:, t], jnp.broadcast_to(idx, (typ.shape[0], 1, 5)), axis=1,
    )[:, 0]
    ls = jnp.take_along_axis(
        out["stuff_logsigma"][:, t], jnp.broadcast_to(idx, (typ.shape[0], 1, 5)), axis=1,
    )[:, 0]
    stuff = mu + jnp.exp(ls) * jax.random.normal(ks, mu.shape)
    return np.asarray(typ), np.asarray(stuff)


# ---------------------------------------------------------------------------
# Cached, compiled decode path
# ---------------------------------------------------------------------------

# Flax's decode cache is shape-dependent but parameter-independent.  Reusing a
# zero template avoids a complete dummy forward for every evaluation batch.
_CACHE_TEMPLATES: dict[tuple[object, ...], Any] = {}


def _supports_cached_decode(heads: PitchformerHeads) -> bool:
    """Whether the supplied heads are real Flax checkpoint modules.

    Lightweight fake heads in unit tests intentionally retain the readable
    NumPy reference path below.  Production checkpoints all satisfy this.
    """
    pairs = ((heads.a, heads.a_params), (heads.b, heads.b_params),
             (heads.c, heads.c_params), (heads.d, heads.d_params))
    return all(model is None or (hasattr(model, "init") and isinstance(params, dict)
                                 and "params" in params)
               for model, params in pairs)


def _cache_template(head: Any, params: dict[str, Any], batch: dict[str, np.ndarray]) -> Any:
    """Create an empty Flax K/V cache for one head and one batch shape."""
    B, T = batch["valid"].shape
    key = (head, B, T)
    cached = _CACHE_TEMPLATES.get(key)
    if cached is not None:
        return cached
    init_batch = {name: jnp.asarray(value) for name, value in batch.items()}
    init_batch["valid"] = jnp.zeros_like(init_batch["valid"])
    init_batch["_decode_position"] = jnp.array(0, jnp.int32)
    init_batch["_cache_valid"] = init_batch["valid"].astype(bool)
    # The full token shape only allocates cache tensors here; the subsequent
    # decode calls use T=1 and start from cache_index=0.
    variables = head.init(jax.random.PRNGKey(0), init_batch, train=False, decode=True)
    cached = variables["cache"]
    _CACHE_TEMPLATES[key] = cached
    return cached


def _cached_apply(model: Any, params: dict[str, Any], cache: Any,
                  token: dict[str, jax.Array], *, output_heads: bool = True) -> tuple[dict[str, jax.Array], Any]:
    """Decode one token and return the updated K/V cache."""
    variables = {**params, "cache": cache}
    if output_heads:
        out, updated = model.apply(variables, token, train=False, decode=True,
                                   mutable=("cache",))
    else:
        out, updated = model.apply(variables, token, train=False, decode=True,
                                   output_heads=False, mutable=("cache",))
    return out, updated["cache"]


def _empirical_tables(engine: EmpiricalEngine) -> dict[str, jax.Array]:
    """Pack ragged empirical transitions into JAX-indexable tables."""
    max_choices = max([1, *(len(v) for v in engine._bsa.values())])
    base = np.zeros((N_KEYS, max_choices), np.int32)
    runs = np.zeros((N_KEYS, max_choices), np.int32)
    outs = np.zeros((N_KEYS, max_choices), np.int32)
    count = np.ones(N_KEYS, np.int32)
    for key in range(N_KEYS):
        state = key // (3 * 9)
        remainder = key % (3 * 9)
        outcome = remainder % 9
        base[key, 0] = BS_AFTER[state, outcome]
        runs[key, 0] = RUNS[state, outcome]
        outs[key, 0] = OUT_INC[outcome]
    for key, values in engine._bsa.items():
        size = len(values)
        if not size:
            continue
        base[key, :size] = values
        runs[key, :size] = engine._runs[key]
        outs[key, :size] = engine._outs_added[key]
        count[key] = size
    return {"base": jnp.asarray(base), "runs": jnp.asarray(runs),
            "outs": jnp.asarray(outs), "count": jnp.asarray(count)}


def _c_tables(c_engine: CTransitionEngine | None) -> dict[str, jax.Array]:
    """Pack C's sparse empirical adapter into a static JAX lookup table."""
    event_count = 255 if getattr(c_engine, "event_mode", "legacy") == "bundles" else len(EVENT_FLAGS)
    choices = [1]
    if c_engine is not None:
        choices.extend(len(v) for v in c_engine._base.values())
    max_choices = max(choices)
    base = np.zeros((event_count, 8, 3, max_choices), np.int32)
    outs = np.zeros_like(base)
    runs = np.zeros_like(base)
    count = np.ones((event_count, 8, 3), np.int32)
    for event in range(event_count):
        for bs in range(8):
            for out in range(3):
                base[event, bs, out, 0] = bs
                outs[event, bs, out, 0] = out
    if c_engine is not None:
        for (event, bs, out), values in c_engine._base.items():
            size = len(values)
            if not size:
                continue
            base[event, bs, out, :size] = values
            outs[event, bs, out, :size] = c_engine._outs[(event, bs, out)]
            runs[event, bs, out, :size] = c_engine._runs[(event, bs, out)]
            count[event, bs, out] = size
    return {"base": jnp.asarray(base), "outs": jnp.asarray(outs),
            "runs": jnp.asarray(runs), "count": jnp.asarray(count)}


def _sample_empirical(tables: dict[str, jax.Array], base: jax.Array, outs: jax.Array,
                      outcome: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    idx = (jnp.clip(base, 0, 7) * 3 + jnp.clip(outs, 0, 2)) * 9 + jnp.clip(outcome, 0, 8)
    count = tables["count"][idx]
    choice = jnp.floor(jax.random.uniform(key, idx.shape) * count).astype(jnp.int32)
    return tables["base"][idx, choice], tables["runs"][idx, choice], tables["outs"][idx, choice]


def _sample_c_transition(tables: dict[str, jax.Array], event: jax.Array,
                         base: jax.Array, outs: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    event = jnp.clip(event, 0, tables["count"].shape[0] - 1)
    base = jnp.clip(base, 0, 7)
    outs = jnp.clip(outs, 0, 2)
    count = tables["count"][event, base, outs]
    choice = jnp.floor(jax.random.uniform(key, event.shape) * count).astype(jnp.int32)
    return (tables["base"][event, base, outs, choice],
            tables["outs"][event, base, outs, choice],
            tables["runs"][event, base, outs, choice])


def _dense(params: dict[str, Any], name: str, x: jax.Array) -> jax.Array:
    """Apply a checkpoint Dense layer without entering a second model decode."""
    layer = params["params"][name]
    return jnp.einsum("...i,ij->...j", x, layer["kernel"]) + layer["bias"]


def _d_heads(params: dict[str, Any], hidden: jax.Array, pitch_type: jax.Array,
             stuff: jax.Array, launch_key: jax.Array, outcome_key: jax.Array,
             ctx: jax.Array, geom: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Run D's post-trunk heads once a ball in play has been established."""
    pitch = jnp.concatenate([jax.nn.one_hot(pitch_type, 8), stuff], axis=-1)
    pitch = jax.nn.gelu(_dense(params, "pitch_proj", pitch))
    z = jax.nn.gelu(_dense(params, "merge", jnp.concatenate([hidden, pitch], axis=-1)))
    mu = _dense(params, "launch_mu", z)
    ls = jnp.clip(_dense(params, "launch_logsigma", z), -4.0, 2.0)
    launch = mu + jnp.exp(ls) * jax.random.normal(launch_key, mu.shape)
    launch_in = jax.nn.gelu(_dense(params, "launch_proj", launch))
    y = jax.nn.gelu(_dense(params, "outcome_merge",
                            jnp.concatenate([z, launch_in, geom, ctx], axis=-1)))
    outcome = jax.random.categorical(outcome_key, _dense(params, "outcome", y)).astype(jnp.int32)
    return launch, outcome


def _pa_sources(batch: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Map generated PA slots back to the observed exogenous schedule."""
    B, T = batch["valid"].shape
    source = np.zeros((B, T), np.int32)
    counts = np.zeros(B, np.int32)
    if "pa_start" not in batch:
        source[:] = np.arange(T, dtype=np.int32)
        counts[:] = np.asarray(batch["valid"], bool).sum(axis=1)
        return source, counts
    for i in range(B):
        starts = np.flatnonzero(np.asarray(batch["pa_start"][i], bool))
        source[i, :len(starts)] = starts
        counts[i] = len(starts)
    return source, counts


def _trim_trailing_padding(batch: dict[str, np.ndarray], decode_len: int | None = None) -> tuple[dict[str, np.ndarray], int]:
    """Remove shared trailing padding before compiling a batch-sized scan.

    ``decode_len`` may retain a small amount of padding to place calls into a
    fixed set of JIT shapes.  It is useful for full-game evaluation, where
    exact half-inning lengths would otherwise produce many compilations.
    """
    full_t = batch["valid"].shape[1]
    nonempty = np.flatnonzero(np.asarray(batch["valid"], bool).any(axis=0))
    used_t = int(nonempty[-1] + 1) if len(nonempty) else 1
    if decode_len is not None:
        if decode_len < 1:
            raise ValueError(f"decode_len must be positive, got {decode_len}")
        # The input is only an observed schedule.  A final decode bucket can
        # legitimately exceed it once rollout gets its padded extension.
        used_t = max(used_t, min(decode_len, full_t))
    if used_t == full_t:
        return batch, full_t
    B = batch["valid"].shape[0]
    trimmed = {
        name: (value[:, :used_t].copy()
               if value.ndim >= 2 and value.shape[0] == B and value.shape[1] == full_t
               else value.copy())
        for name, value in batch.items()
    }
    return trimmed, full_t


def _restore_trailing_padding(records: dict[str, np.ndarray], full_t: int) -> dict[str, np.ndarray]:
    """Restore the public ``(B, T)`` layout after a trimmed compiled scan."""
    used_t = records["active"].shape[1]
    if used_t == full_t:
        return records
    restored: dict[str, np.ndarray] = {}
    for name, value in records.items():
        if name == "final_state":
            restored[name] = value
            continue
        fill = -1 if name == "pa_outcome" else 0
        padded = np.full((value.shape[0], full_t, *value.shape[2:]), fill, value.dtype)
        padded[:, :used_t] = value
        restored[name] = padded
    return restored


@partial(jax.jit, static_argnames=("a", "b", "c", "d", "has_c", "has_d", "has_schedule", "stop_when_decided"))
def _compiled_rollout(
    *,
    a: Any, b: Any, c: Any, d: Any, has_c: bool, has_d: bool, has_schedule: bool, stop_when_decided: bool,
    a_params: dict[str, Any], b_params: dict[str, Any], c_params: dict[str, Any] | None,
    d_params: dict[str, Any] | None, original: dict[str, jax.Array], state: dict[str, jax.Array],
    sources: jax.Array, source_counts: jax.Array, cache_a: Any, cache_b: Any,
    cache_c: Any, cache_d: Any, empirical: dict[str, jax.Array], c_transitions: dict[str, jax.Array],
    key: jax.Array,
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    """One whole half-inning batch as a single XLA program.

    The Python transition adapters are converted to tables before entry, so
    sampling and the generated state evolution stay on device inside ``scan``.
    """

    B, T = original["valid"].shape
    empty_launch = jnp.zeros((B, 2), jnp.float32)
    empty_batted = jnp.zeros(B, jnp.int32)

    '''def body(carry, t):
        state, history_valid, cache_a, cache_b, cache_c, cache_d, key = carry
        active = original["valid"][:, t].astype(bool) & ~state["ended"]
        if has_schedule:
            source_ok = state["pa_slot"] < source_counts
            active = active & source_ok
            source = jnp.take_along_axis(
                sources, jnp.clip(state["pa_slot"], 0, T - 1)[:, None], axis=1)[:, 0]
        else:
            source = jnp.full(B, t, jnp.int32)

        ctx = original["ctx"][:, t]
        source_ctx = jnp.take_along_axis(original["ctx"], source[:, None, None], axis=1)[:, 0]
        base = state["base"]
        diff = jnp.where(state["half"] == 0,
                         state["away_score"] - state["home_score"],
                         state["home_score"] - state["away_score"])
        ctx = ctx.at[:, 0].set(state["balls"] / 3.0)
        ctx = ctx.at[:, 1].set(state["strikes"] / 2.0)
        ctx = ctx.at[:, 2].set(state["outs"] / 2.0)
        ctx = ctx.at[:, 3].set((base & 1 > 0).astype(jnp.float32))
        ctx = ctx.at[:, 4].set((base & 2 > 0).astype(jnp.float32))
        ctx = ctx.at[:, 5].set((base & 4 > 0).astype(jnp.float32))
        ctx = ctx.at[:, 6].set(((base & 2 > 0) | (base & 4 > 0)).astype(jnp.float32))
        ctx = ctx.at[:, 7].set(jnp.clip(diff, -10, 10) / 10.0)
        ctx = ctx.at[:, 8].set((state["inning"] - 5.0) / 4.0)
        ctx = ctx.at[:, 9].set(state["half"])
        ctx = ctx.at[:, 10].set(1.0 - state["half"])
        ctx = ctx.at[:, 11].set(((state["inning"] >= 7) & (jnp.abs(diff) <= 1)).astype(jnp.float32))
        ctx = ctx.at[:, 12].set(jnp.clip(state["tto"], 0, 4) / 4.0)
        # Batter handedness and matchup are exogenous PA-schedule inputs.
        ctx = ctx.at[:, 13:16].set(source_ctx[:, 13:16])
        pitcher = jnp.take_along_axis(original["pitcher_idx"], source[:, None], axis=1)[:, 0]
        batter = jnp.take_along_axis(original["batter_idx"], source[:, None], axis=1)[:, 0]
        park = jnp.take_along_axis(original["park_idx"], source[:, None], axis=1)[:, 0]
        geom = original["geom"][:, t]

        token = {
            "pitcher_idx": pitcher[:, None], "batter_idx": batter[:, None], "park_idx": park[:, None],
            "ctx": ctx[:, None], "geom": geom[:, None], "valid": active[:, None],
            "pitch_type": jnp.zeros(B, jnp.int32)[:, None], "stuff": jnp.zeros((B, 1, 5), jnp.float32),
            "swing": jnp.zeros((B, 1), jnp.float32), "contact": jnp.zeros((B, 1), jnp.float32),
            "foul": jnp.zeros((B, 1), jnp.float32), "launch": jnp.zeros((B, 1, 2), jnp.float32),
            "_decode_position": t, "_cache_valid": history_valid,
        }
        key, ktype, kstuff, kb, kcontact, kfoul, kc, kd1, kd2, kpa, kct = jax.random.split(key, 11)
        out_a, cache_a = _cached_apply(a, a_params, cache_a, token)
        typ = jax.random.categorical(ktype, out_a["type_logits"][:, 0]).astype(jnp.int32)
        a_idx = typ[:, None, None]
        mu = jnp.take_along_axis(out_a["stuff_mu"][:, 0], jnp.broadcast_to(a_idx, (B, 1, 5)), axis=1)[:, 0]
        ls = jnp.take_along_axis(out_a["stuff_logsigma"][:, 0], jnp.broadcast_to(a_idx, (B, 1, 5)), axis=1)[:, 0]
        stuff = mu + jnp.exp(ls) * jax.random.normal(kstuff, mu.shape)
        token["pitch_type"] = typ[:, None]
        token["stuff"] = stuff[:, None]

        out_b, cache_b = _cached_apply(b, b_params, cache_b, token)
        swing = jax.random.bernoulli(kb, jax.nn.sigmoid(out_b["swing_logit"][:, 0])) & active
        contact = jax.random.bernoulli(kcontact, jax.nn.sigmoid(out_b["contact_logit"][:, 0])) & swing
        foul = jax.random.bernoulli(kfoul, jax.nn.sigmoid(out_b["foul_logit"][:, 0])) & contact
        token["swing"] = swing[:, None].astype(jnp.float32)
        token["contact"] = contact[:, None].astype(jnp.float32)
        token["foul"] = foul[:, None].astype(jnp.float32)

        if has_c:
            out_c, cache_c = _cached_apply(c, c_params, cache_c, token)
            event = _sample_c_event(out_c["event_logits"][:, 0], kc).astype(jnp.int32)
            event = jnp.where(active, event, -1)
        else:
            event = jnp.full(B, -1, jnp.int32)

        # D's independently-trained trunk must receive every pitch so its K/V
        # history stays semantically identical to a full causal forward.  Its
        # launch/outcome MLPs run only when this step contains a ball in play.
        if has_d:
            d_out, cache_d = _cached_apply(d, d_params, cache_d, token, output_heads=False)
            d_hidden = d_out["hidden"][:, 0]
        else:
            d_hidden = jnp.zeros((B, 1), jnp.float32)

        px = stuff[:, 3] * STUFF_SCALE[3] + STUFF_CENTRE[3]
        pz = stuff[:, 4] * STUFF_SCALE[4] + STUFF_CENTRE[4]
        called_strike = (~swing) & (jnp.abs(px) <= ZONE_HALF_WIDTH) & (pz >= ZONE_BOTTOM) & (pz <= ZONE_TOP)
        in_play = swing & contact & ~foul

        def run_d(_: None) -> tuple[jax.Array, jax.Array]:
            return _d_heads(d_params, d_hidden, typ, stuff, kd1, kd2, ctx, geom)

        if has_d:
            launch, batted = jax.lax.cond(
                jnp.any(in_play), run_d, lambda _: (empty_launch, empty_batted), None)
        else:
            launch, batted = empty_launch, empty_batted
        outcome = jnp.full(B, int(PAOutcome.OUT), jnp.int32)
        strikeout = state["strikes"] + ((swing & ~contact) | called_strike) >= 3
        walk = state["balls"] + ((~swing) & ~called_strike) >= 4
        outcome = jnp.where(strikeout, int(PAOutcome.STRIKEOUT), outcome)
        outcome = jnp.where(walk, int(PAOutcome.WALK), outcome)
        outcome = jnp.where(in_play & has_d, jnp.asarray(_D_TO_PA)[batted], outcome)
        terminal = active & (strikeout | walk | in_play)

        pa_base, pa_runs, pa_outs = _sample_empirical(empirical, state["base"], state["outs"], outcome, kpa)
        next_outs = state["outs"] + pa_outs
        inning_over = terminal & (next_outs >= 3)
        next_state = dict(state)
        next_state["base"] = jnp.where(terminal, pa_base, state["base"])
        next_state["base"] = jnp.where(inning_over, 0, next_state["base"])
        next_state["outs"] = jnp.where(inning_over, 0, next_outs)
        runs = jnp.where(terminal, pa_runs, 0)
        next_state["home_score"] = state["home_score"] + runs * (state["half"] == 1)
        next_state["away_score"] = state["away_score"] + runs * (state["half"] == 0)
        next_state["balls"] = jnp.where(terminal, 0, state["balls"] + ((~swing) & ~called_strike))
        next_state["strikes"] = jnp.where(
            terminal, 0, jnp.minimum(2, state["strikes"] + ((swing & ~contact) | called_strike | foul)))

        c_apply = active & ~terminal & (event >= 0) & (c_transitions["enabled"] == 1)
        c_base, c_outs, c_runs = _sample_c_transition(c_transitions, event, next_state["base"], next_state["outs"], kct)
        next_state["base"] = jnp.where(c_apply, c_base, next_state["base"])
        next_state["outs"] = jnp.where(c_apply, c_outs, next_state["outs"])
        next_state["home_score"] = next_state["home_score"] + jnp.where(c_apply, c_runs, 0) * (state["half"] == 1)
        next_state["away_score"] = next_state["away_score"] + jnp.where(c_apply, c_runs, 0) * (state["half"] == 0)
        c_third_out = c_apply & (next_state["outs"] >= 3)
        next_state["base"] = jnp.where(c_third_out, 0, next_state["base"])
        next_state["outs"] = jnp.where(c_third_out, 0, next_state["outs"])
        ended = state["ended"] | inning_over | c_third_out
        if stop_when_decided:
            ended = ended | ((state["half"] == 1) & (state["inning"] >= 9)
                             & (next_state["home_score"] > next_state["away_score"]))
        next_state["ended"] = ended
        next_state["pitch_count"] = state["pitch_count"] + active.astype(jnp.int32)
        next_state["pa_slot"] = state["pa_slot"] + terminal.astype(jnp.int32)
        history_valid = history_valid.at[:, t].set(active)
        record = {
            "ctx": ctx, "active": active, "pa_terminal": terminal,
            "pa_outcome": jnp.where(terminal, outcome, -1), "pitch_type": typ, "stuff": stuff,
            "launch": launch, "swing": swing, "contact": contact, "foul": foul,
            "event": jax.nn.one_hot(jnp.maximum(event, 0), len(EVENT_FLAGS), dtype=bool)
                     & (event >= 0)[:, None],
            "batter_idx": batter,
        }
        return (next_state, history_valid, cache_a, cache_b, cache_c, cache_d, key), record'''

    def body(carry, t):
        (state, half_complete, history_valid,
         cache_a, cache_b, cache_c, cache_d, key) = carry

        # ------------------------------------------------------------------
        # 1. Determine whether this rollout row is active and which scheduled
        #    PA supplies exogenous batter/pitcher/park/matchup information.
        # ------------------------------------------------------------------
        # ``rollout_batch`` receives one observed half-inning at a time.  A
        # generated third out must therefore stop this *call*, while leaving
        # ``state["ended"]`` reserved for an actually completed game.  Otherwise
        # the remaining source rows from (say) the top half would be consumed as
        # the bottom half before the outer game evaluator can supply its proper
        # schedule.
        active = (original["valid"][:, t].astype(bool)
                  & ~state["ended"] & ~half_complete)

        if has_schedule:
            source_ok = state["pa_slot"] < source_counts
            active = active & source_ok

            source = jnp.take_along_axis(
                sources,
                jnp.clip(state["pa_slot"], 0, T - 1)[:, None],
                axis=1,
            )[:, 0]
        else:
            source = jnp.full(B, t, jnp.int32)

        # ------------------------------------------------------------------
        # 2. Build generated game context.
        # ------------------------------------------------------------------
        source_ctx = jnp.take_along_axis(
            original["ctx"],
            source[:, None, None],
            axis=1,
        )[:, 0]
        # In the padded extension there is no observed pitch row at ``t``.
        # Start from the scheduled PA's exogenous context so park/weather and
        # matchup features are retained, then overwrite generated game state.
        ctx = source_ctx

        base = state["base"]

        diff = jnp.where(
            state["half"] == 0,
            state["away_score"] - state["home_score"],
            state["home_score"] - state["away_score"],
        )

        ctx = ctx.at[:, 0].set(state["balls"] / 3.0)
        ctx = ctx.at[:, 1].set(state["strikes"] / 2.0)
        ctx = ctx.at[:, 2].set(state["outs"] / 2.0)

        ctx = ctx.at[:, 3].set(
            ((base & 1) > 0).astype(jnp.float32)
        )
        ctx = ctx.at[:, 4].set(
            ((base & 2) > 0).astype(jnp.float32)
        )
        ctx = ctx.at[:, 5].set(
            ((base & 4) > 0).astype(jnp.float32)
        )
        ctx = ctx.at[:, 6].set(
            (((base & 2) > 0) | ((base & 4) > 0)).astype(jnp.float32)
        )

        ctx = ctx.at[:, 7].set(
            jnp.clip(diff, -10, 10) / 10.0
        )
        ctx = ctx.at[:, 8].set(
            (state["inning"] - 5.0) / 4.0
        )
        ctx = ctx.at[:, 9].set(
            state["half"]
        )
        ctx = ctx.at[:, 10].set(
            1.0 - state["half"]
        )
        ctx = ctx.at[:, 11].set(
            (
                (state["inning"] >= 7)
                & (jnp.abs(diff) <= 1)
            ).astype(jnp.float32)
        )
        ctx = ctx.at[:, 12].set(
            jnp.clip(state["tto"], 0, 4) / 4.0
        )

        # Exogenous PA-level matchup information.
        ctx = ctx.at[:, 13:16].set(source_ctx[:, 13:16])

        pitcher = jnp.take_along_axis(
            original["pitcher_idx"],
            source[:, None],
            axis=1,
        )[:, 0]

        batter = jnp.take_along_axis(
            original["batter_idx"],
            source[:, None],
            axis=1,
        )[:, 0]

        park = jnp.take_along_axis(
            original["park_idx"],
            source[:, None],
            axis=1,
        )[:, 0]

        geom = jnp.take_along_axis(
            original["geom"], source[:, None, None], axis=1
        )[:, 0]

        # ------------------------------------------------------------------
        # 3. Start current token blank.
        #
        # IMPORTANT:
        # cache_* contains only COMPLETED pitches < t.
        # None of the provisional caches returned below are retained.
        # ------------------------------------------------------------------
        token = {
            "pitcher_idx": pitcher[:, None],
            "batter_idx": batter[:, None],
            "park_idx": park[:, None],
            "ctx": ctx[:, None],
            "geom": geom[:, None],
            "valid": active[:, None],

            "pitch_type": jnp.zeros((B, 1), jnp.int32),
            "stuff": jnp.zeros((B, 1, 5), jnp.float32),

            "swing": jnp.zeros((B, 1), jnp.float32),
            "contact": jnp.zeros((B, 1), jnp.float32),
            "foul": jnp.zeros((B, 1), jnp.float32),

            "launch": jnp.zeros((B, 1, 2), jnp.float32),

            "_decode_position": t,
            "_cache_valid": history_valid,
        }

        if "skill_season" in original:
            token["skill_season"] = jnp.take_along_axis(original["skill_season"], source[:, None], axis=1)

        key, ktype, kstuff, kb, kcontact, kfoul, khbp, kc, kd1, kd2, kpa, kct = (
            jax.random.split(key, 12)
        )

        # ================================================================
        # HEAD A
        #
        # Predict from:
        #   completed history < t
        #   + blank current pitch t
        #
        # DO NOT keep the returned provisional cache.
        # ================================================================
        out_a, _ = _cached_apply(
            a,
            a_params,
            cache_a,
            token,
        )

        typ = jax.random.categorical(
            ktype,
            out_a["type_logits"][:, 0],
        ).astype(jnp.int32)

        a_idx = typ[:, None, None]

        mu = jnp.take_along_axis(
            out_a["stuff_mu"][:, 0],
            jnp.broadcast_to(a_idx, (B, 1, 5)),
            axis=1,
        )[:, 0]

        ls = jnp.take_along_axis(
            out_a["stuff_logsigma"][:, 0],
            jnp.broadcast_to(a_idx, (B, 1, 5)),
            axis=1,
        )[:, 0]

        stuff = (
            mu
            + jnp.exp(ls)
            * jax.random.normal(kstuff, mu.shape)
        )

        token["pitch_type"] = typ[:, None]
        token["stuff"] = stuff[:, None]

        # ================================================================
        # HEAD B
        #
        # B sees pitch type/stuff for t, but not its own outcomes.
        # Again: discard provisional cache.
        # ================================================================
        out_b, _ = _cached_apply(
            b,
            b_params,
            cache_b,
            token,
        )

        swing = (
            jax.random.bernoulli(
                kb,
                jax.nn.sigmoid(out_b["swing_logit"][:, 0]),
            )
            & active
        )

        contact = (
            jax.random.bernoulli(
                kcontact,
                jax.nn.sigmoid(out_b["contact_logit"][:, 0]),
            )
            & swing
        )

        foul = (
            jax.random.bernoulli(
                kfoul,
                jax.nn.sigmoid(out_b["foul_logit"][:, 0]),
            )
            & contact
        )

        hit_by_pitch = (
            jax.random.bernoulli(
                khbp,
                jax.nn.sigmoid(out_b["hbp_logit"][:, 0]),
            )
            & active
            & ~swing
        )

        token["swing"] = swing[:, None].astype(jnp.float32)
        token["contact"] = contact[:, None].astype(jnp.float32)
        token["foul"] = foul[:, None].astype(jnp.float32)

        # ================================================================
        # HEAD C
        #
        # Sees pitch + swing/contact/foul.
        # ================================================================
        if has_c:
            out_c, _ = _cached_apply(
                c,
                c_params,
                cache_c,
                token,
            )

            event = _sample_c_event(out_c["event_logits"][:, 0], kc)

            event = jnp.where(active, event, -1)

        else:
            event = jnp.full(B, -1, jnp.int32)

        # ------------------------------------------------------------------
        # 4. Resolve pitch location / count state.
        # ------------------------------------------------------------------
        px = (
            stuff[:, 3] * STUFF_SCALE[3]
            + STUFF_CENTRE[3]
        )
        pz = (
            stuff[:, 4] * STUFF_SCALE[4]
            + STUFF_CENTRE[4]
        )

        zone = (
            (jnp.abs(px) <= ZONE_HALF_WIDTH)
            & (pz >= ZONE_BOTTOM)
            & (pz <= ZONE_TOP)
        )

        called_strike = (
            active
            & ~swing
            & ~hit_by_pitch
            & zone
        )

        in_play = (
            active
            & swing
            & contact
            & ~foul
        )

        # These are the count increments for a NON-terminal pitch.
        ball_inc = (
            active
            & ~swing
            & ~hit_by_pitch
            & ~called_strike
        )

        # Foul is included here because jnp.minimum(2, ...) prevents
        # an ordinary two-strike foul from becoming strike three.
        strike_inc = (
            active
            & (
                (swing & ~contact)
                | called_strike
                | foul
            )
        )

        # Strikeout does NOT include an ordinary foul at two strikes.
        strikeout = (
            active
            & (
                state["strikes"]
                + (
                    (swing & ~contact)
                    | called_strike
                )
                >= 3
            )
        )

        walk = (
            active
            & (
                state["balls"] + ball_inc
                >= 4
            )
        )

        # ================================================================
        # HEAD D -- first pass
        #
        # This mirrors the original:
        #
        #   out_d = D(...)
        #   sample launch
        #
        # using the PRE-LAUNCH current token.
        # ================================================================
        if has_d:
            out_d_pre, _ = _cached_apply(
                d,
                d_params,
                cache_d,
                token,
            )

            launch_mu = out_d_pre["launch_mu"][:, 0]
            launch_ls = out_d_pre["launch_logsigma"][:, 0]

            sampled_launch = (
                launch_mu
                + jnp.exp(launch_ls)
                * jax.random.normal(
                    kd1,
                    launch_mu.shape,
                )
            )

            # Preserve the original loop semantics: launch is only meaningful
            # for a ball in play.
            launch = jnp.where(
                in_play[:, None],
                sampled_launch,
                jnp.zeros_like(sampled_launch),
            )

        else:
            launch = empty_launch

        token["launch"] = launch[:, None]

        # ================================================================
        # HEAD D -- second pass
        #
        # Now D sees the sampled launch, exactly like rebuilding rolling
        # and applying D again in the Python loop.
        # ================================================================
        if has_d:
            out_d_post, _ = _cached_apply(
                d,
                d_params,
                cache_d,
                token,
            )

            batted = jax.random.categorical(
                kd2,
                out_d_post["outcome_logits"][:, 0],
            ).astype(jnp.int32)

        else:
            batted = empty_batted

        # ------------------------------------------------------------------
        # 5. Resolve PA outcome.
        # ------------------------------------------------------------------
        outcome = jnp.full(
            B,
            int(PAOutcome.OUT),
            jnp.int32,
        )

        outcome = jnp.where(
            strikeout,
            int(PAOutcome.STRIKEOUT),
            outcome,
        )

        outcome = jnp.where(
            walk,
            int(PAOutcome.WALK),
            outcome,
        )

        outcome = jnp.where(
            hit_by_pitch,
            int(PAOutcome.HIT_BY_PITCH),
            outcome,
        )

        if has_d:
            outcome = jnp.where(
                in_play,
                jnp.asarray(_D_TO_PA)[batted],
                outcome,
            )

        terminal = (
            active
            & (
                strikeout
                | walk
                | hit_by_pitch
                | in_play
            )
        )

        # ------------------------------------------------------------------
        # 6. PA-level empirical transition.
        #
        # _sample_empirical can still run for all B rows for static-shape JAX,
        # but its result only affects terminal rows.
        # ------------------------------------------------------------------
        pa_base, pa_runs, pa_outs = _sample_empirical(
            empirical,
            state["base"],
            state["outs"],
            outcome,
            kpa,
        )

        # Important: do NOT allow sampled PA outs to affect non-terminal
        # pitches.
        outs_added = jnp.where(
            terminal,
            pa_outs,
            0,
        )

        next_outs = (
            state["outs"]
            + outs_added
        )

        inning_over = (
            terminal
            & (next_outs >= 3)
        )

        next_state = dict(state)

        next_state["base"] = jnp.where(
            terminal,
            pa_base,
            state["base"],
        )

        next_state["base"] = jnp.where(
            terminal,
            pa_base,
            state["base"],
        )

        next_state["outs"] = next_outs

        runs = jnp.where(
            terminal,
            pa_runs,
            0,
        )

        next_state["home_score"] = (
            state["home_score"]
            + runs * (state["half"] == 1)
        )

        next_state["away_score"] = (
            state["away_score"]
            + runs * (state["half"] == 0)
        )

        next_state["balls"] = jnp.where(
            terminal,
            0,
            state["balls"] + ball_inc,
        )

        next_state["strikes"] = jnp.where(
            terminal,
            0,
            jnp.minimum(
                2,
                state["strikes"] + strike_inc,
            ),
        )

        # ------------------------------------------------------------------
        # 7. Head-C runner event.
        #
        # Same convention as original Python rollout:
        # apply only on non-terminal pitches so empirical PA advancement and
        # C advancement cannot double-count the same runner movement.
        # ------------------------------------------------------------------
        c_apply = (
            active
            & ~terminal
            & (event >= 0)
            & (c_transitions["enabled"] == 1)
        )

        # Never pass -1 as an array index into the transition sampler.
        safe_event = jnp.maximum(event, 0)

        c_base, c_outs, c_runs = _sample_c_transition(
            c_transitions,
            safe_event,
            next_state["base"],
            next_state["outs"],
            kct,
        )

        next_state["base"] = jnp.where(
            c_apply,
            c_base,
            next_state["base"],
        )

        next_state["outs"] = jnp.where(
            c_apply,
            c_outs,
            next_state["outs"],
        )

        c_runs_applied = jnp.where(
            c_apply,
            c_runs,
            0,
        )

        next_state["home_score"] = (
            next_state["home_score"]
            + c_runs_applied * (state["half"] == 1)
        )

        next_state["away_score"] = (
            next_state["away_score"]
            + c_runs_applied * (state["half"] == 0)
        )

        c_third_out = c_apply & (next_state["outs"] >= 3)

        # A third out ends the HALF-INNING, not the game.  ``half_complete``
        # stops the remainder of this per-half rollout without poisoning the
        # persistent game-level ``ended`` flag.
        half_over = inning_over | c_third_out
        next_half_complete = half_complete | half_over

        # Clear inning-specific state.
        next_state["base"] = jnp.where(
            half_over,
            0,
            next_state["base"],
        )

        next_state["outs"] = jnp.where(
            half_over,
            0,
            next_state["outs"],
        )

        next_state["balls"] = jnp.where(
            half_over,
            0,
            next_state["balls"],
        )

        next_state["strikes"] = jnp.where(
            half_over,
            0,
            next_state["strikes"],
        )

        # Preserve the half in which the pitch actually occurred.
        old_half = state["half"]
        old_inning = state["inning"]

        # Bottom half ending advances the inning.
        next_state["inning"] = jnp.where(
            half_over & (old_half == 1),
            old_inning + 1,
            old_inning,
        )

        # Top -> bottom, bottom -> top.
        next_state["half"] = jnp.where(
            half_over,
            1 - old_half,
            old_half,
        )

        # ------------------------------------------------------------
        # PA schedule
        # ------------------------------------------------------------
        next_pa_slot = (
            state["pa_slot"]
            + terminal.astype(jnp.int32)
        )

        next_state["pa_slot"] = next_pa_slot

        # ------------------------------------------------------------
        # Actual rollout/game-ending conditions
        # ------------------------------------------------------------
        ended = state["ended"]

        if stop_when_decided:
            # Walkoff: home takes the lead during bottom of 9th or later.
            walkoff = (
                (old_half == 1)
                & (old_inning >= 9)
                & (next_state["home_score"] > next_state["away_score"])
            )

            # If the top of the 9th+ finishes with home already ahead,
            # the bottom half is not played.
            home_wins_after_top = (
                half_over
                & (old_half == 0)
                & (old_inning >= 9)
                & (next_state["home_score"] > next_state["away_score"])
            )

            # Bottom of the 9th+ finishes with a non-tied score.
            game_over_after_bottom = (
                half_over
                & (old_half == 1)
                & (old_inning >= 9)
                & (next_state["home_score"] != next_state["away_score"])
            )

            ended = (
                ended
                | walkoff
                | home_wins_after_top
                | game_over_after_bottom
            )

        next_state["ended"] = ended

        next_state["pitch_count"] = (
            state["pitch_count"]
            + active.astype(jnp.int32)
        )

        next_state["pa_slot"] = (
            state["pa_slot"]
            + terminal.astype(jnp.int32)
        )

        # ================================================================
        # 9. COMMIT COMPLETED TOKEN TO ALL FOUR CACHES
        #
        # This is the crucial difference from the previous implementation.
        #
        # Every head's historical cache now receives the SAME COMPLETED
        # representation of pitch t.
        #
        # We feed the OLD cache into each commit -- not one of the
        # provisional caches above.
        # ================================================================
        # Commit the fully completed pitch to each head's historical cache.
        #
        # Important: use the OLD cache for each head, not the provisional cache
        # returned during prediction earlier in this step.

        _, next_cache_a = _cached_apply(
            a,
            a_params,
            cache_a,
            token,
        )

        _, next_cache_b = _cached_apply(
            b,
            b_params,
            cache_b,
            token,
        )

        if has_c:
            _, next_cache_c = _cached_apply(
                c,
                c_params,
                cache_c,
                token,
            )
        else:
            next_cache_c = cache_c

        if has_d:
            _, next_cache_d = _cached_apply(
                d,
                d_params,
                cache_d,
                token,
            )
        else:
            next_cache_d = cache_d
        # Mark t as historical for the NEXT iteration.
        history_valid = history_valid.at[:, t].set(active)

        # ------------------------------------------------------------------
        # 10. Record generated pitch.
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # Probability fields for PIT calibration.
        # Record the B-head logit-probabilities and D-head outcome logits
        # at every pitch so downstream code can compose an implied PA
        # outcome distribution at terminal positions.
        # ------------------------------------------------------------------
        swing_prob = jax.nn.sigmoid(out_b["swing_logit"][:, 0])
        contact_prob = jax.nn.sigmoid(out_b["contact_logit"][:, 0])
        foul_prob = jax.nn.sigmoid(out_b["foul_logit"][:, 0])
        hbp_prob = jax.nn.sigmoid(out_b["hbp_logit"][:, 0])

        if has_d:
            d_outcome_logits = out_d_post["outcome_logits"][:, 0]   # (B, 6)
            d_outcome_probs = jax.nn.softmax(d_outcome_logits, axis=-1)
        else:
            d_outcome_probs = jnp.zeros((B, 6), jnp.float32)

        record = {
            "ctx": ctx,
            "active": active,
            "pa_terminal": terminal,
            "pa_outcome": jnp.where(
                terminal,
                outcome,
                -1,
            ),
            "pitch_type": typ,
            "stuff": stuff,
            "launch": launch,
            "swing": swing,
            "contact": contact,
            "foul": foul,
            "hbp": hit_by_pitch,
            "event": _event_flags(jnp.where(c_apply, event, -1), getattr(c, 'c_event_mode', 'legacy')),
            "batter_idx": batter,
            # PIT calibration fields
            "swing_prob": swing_prob,
            "contact_prob": contact_prob,
            "foul_prob": foul_prob,
            "hbp_prob": hbp_prob,
            "d_outcome_probs": d_outcome_probs,
            "zone": zone,
            "balls": state["balls"],
            "strikes": state["strikes"],
        }

        return (
            (
                next_state,
                next_half_complete,
                history_valid,
                next_cache_a,
                next_cache_b,
                next_cache_c,
                next_cache_d,
                key,
            ),
            record,
        )

    carry = (state, jnp.zeros(B, bool), jnp.zeros((B, T), bool),
             cache_a, cache_b, cache_c, cache_d, key)
    final, records = jax.lax.scan(body, carry, jnp.arange(T, dtype=jnp.int32))
    final_state = final[0]
    return records, final_state, final[3:7]


def _rollout_batch_cached(
    heads: PitchformerHeads, batch: dict[str, np.ndarray], *, seed: int, engine: EmpiricalEngine,
    c_engine: CTransitionEngine | None, initial_state: dict[str, np.ndarray] | None,
    stop_when_decided: bool, decode_len: int | None, initial_cache=None,
) -> dict[str, np.ndarray]:
    """Run the checkpoint path with K/V cache + one compiled scan."""

    original_full = {name: np.asarray(value) for name, value in batch.items()}
    original, full_t = _trim_trailing_padding(original_full, decode_len)
    old_t = original["valid"].shape[1]
    nonempty = np.flatnonzero(original["valid"].any(axis=0))
    observed_t = int(nonempty[-1] + 1) if len(nonempty) else 1
    # ``decode_len`` is the final compiled bucket used by the game evaluator.
    # Outside that path retain the deliberate 1.5x extension so a generated PA
    # is not forced to inherit the observed pitch count.
    rollout_t = max(old_t, int(np.ceil(observed_t * 1.5)), decode_len or 0)

    def pad_time(x, new_t):
        pad = new_t - x.shape[1]
        if pad <= 0:
            return x

        pads = [(0, 0)] * x.ndim
        pads[1] = (0, pad)

        return np.pad(x, pads, mode="constant")

    original = {
        name: pad_time(value, rollout_t)
        for name, value in original.items()
    }
    if "pa_start" in original and rollout_t > old_t:
        # Extra time positions are real decode opportunities, not inert zero
        # padding.  ``source_counts`` below limits them to observed scheduled
        # plate appearances, while the active body copies each PA's static
        # matchup/park/environment values from its scheduled source row.
        original["valid"][:, old_t:rollout_t] = 1.0
    state = _initial_state(original) if initial_state is None else {
        name: np.asarray(value).copy() for name, value in initial_state.items()
    }
    sources, counts = _pa_sources(original)
    cache_a = _cache_template(heads.a, heads.a_params, original)
    cache_b = _cache_template(heads.b, heads.b_params, original)
    cache_c = _cache_template(heads.c, heads.c_params, original) if heads.c is not None else None
    cache_d = _cache_template(heads.d, heads.d_params, original) if heads.d is not None else None
    c_tables = _c_tables(c_engine)
    c_tables["enabled"] = jnp.array(int(c_engine is not None and heads.c is not None), jnp.int32)
    if initial_cache is not None:
        cache_a, cache_b, cache_c, cache_d = initial_cache
    records, final_state, final_cache = _compiled_rollout(
        a=heads.a, b=heads.b, c=heads.c, d=heads.d,
        has_c=heads.c is not None and heads.c_params is not None,
        has_d=heads.d is not None and heads.d_params is not None,
        has_schedule="pa_start" in original,
        stop_when_decided=stop_when_decided,
        a_params=heads.a_params, b_params=heads.b_params, c_params=heads.c_params, d_params=heads.d_params,
        original={name: jnp.asarray(value) for name, value in original.items()},
        state={name: jnp.asarray(value) for name, value in state.items()},
        sources=jnp.asarray(sources), source_counts=jnp.asarray(counts),
        cache_a=cache_a, cache_b=cache_b, cache_c=cache_c, cache_d=cache_d,
        empirical=_empirical_tables(engine), c_transitions=c_tables, key=jax.random.PRNGKey(seed),
    )
    out = {name: np.asarray(jnp.swapaxes(value, 0, 1)) for name, value in records.items()}
    out["final_state"] = {name: np.asarray(value) for name, value in final_state.items()}
    out = _restore_trailing_padding(out, rollout_t)
    out["final_cache"] = jax.tree.map(np.asarray, final_cache)
    return out


def _rollout_batch_reference(
    heads: PitchformerHeads,
    batch: dict[str, np.ndarray],
    *,
    seed: int,
    engine: EmpiricalEngine,
    c_engine: CTransitionEngine | None = None,
    initial_state: dict[str, np.ndarray] | None = None,
    stop_when_decided: bool = False,
) -> dict[str, np.ndarray]:
    """Free-roll one padded batch of half-inning pitch sequences.

    Returned arrays use the input's ``(B, T)`` layout.  Padded and post-third-out
    positions are marked inactive.  This bounded layout lets the existing causal
    transformer run one compiled shape while every prefix contains generated—not
    observed—pitch and game-state history.
    """
    original = {name: np.asarray(value) for name, value in batch.items()}
    B, T = original["valid"].shape
    rolling = {name: value.copy() for name, value in original.items()}
    # These are generated causal-history fields; no real values survive.
    for name in ("pitch_type", "stuff", "swing", "contact", "foul", "launch"):
        if name in rolling:
            rolling[name].fill(0)
    rolling["valid"].fill(0)
    # Generated measurements are observed; source-feed missingness is irrelevant.
    for name in ("type_valid", "stuff_observed", "launch_observed"):
        if name in rolling:
            rolling[name].fill(1)
    state = _initial_state(original) if initial_state is None else {
        name: np.asarray(value).copy() for name, value in initial_state.items()
    }
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    records = {
        "ctx": np.zeros_like(original["ctx"]),
        "active": np.zeros((B, T), bool),
        "pa_terminal": np.zeros((B, T), bool),
        "pa_outcome": np.full((B, T), -1, np.int32),
        "pitch_type": np.zeros((B, T), np.int32),
        "stuff": np.zeros((B, T, 5), np.float32),
        "launch": np.zeros((B, T, 2), np.float32),
        "swing": np.zeros((B, T), bool),
        "contact": np.zeros((B, T), bool),
        "foul": np.zeros((B, T), bool),
        "hbp": np.zeros((B, T), bool),
        "event": np.zeros((B, T, len(EVENT_FLAGS)), bool),
        "batter_idx": np.zeros((B, T), np.int32),
        # PIT calibration fields
        "swing_prob": np.zeros((B, T), np.float32),
        "contact_prob": np.zeros((B, T), np.float32),
        "foul_prob": np.zeros((B, T), np.float32),
        "hbp_prob": np.zeros((B, T), np.float32),
        "d_outcome_probs": np.zeros((B, T, 6), np.float32),
        "zone": np.zeros((B, T), bool),
        "balls": np.zeros((B, T), np.int32),
        "strikes": np.zeros((B, T), np.int32),
    }

    for t in range(T):
        active = original["valid"][:, t].astype(bool) & ~state["ended"]
        if not active.any():
            continue
        if "pa_start" in original:
            for bi in np.flatnonzero(active):
                starts = np.flatnonzero(original["pa_start"][bi])
                slot = int(state["pa_slot"][bi])
                if slot >= len(starts):
                    state["ended"][bi] = True
                    active[bi] = False
                    continue
                source = starts[slot]
                for name in ("pitcher_idx", "batter_idx", "park_idx"):
                    rolling[name][bi, t] = original[name][bi, source]
        _write_generated_context(rolling["ctx"], original["ctx"], state, t)
        if "pa_start" in original:
            for bi in np.flatnonzero(active):
                source = np.flatnonzero(original["pa_start"][bi])[int(state["pa_slot"][bi])]
                rolling["ctx"][bi, t, 13:16] = original["ctx"][bi, source, 13:16]
        rolling["valid"][:, t] = active
        model_batch = {name: jnp.asarray(value) for name, value in rolling.items()}

        key, ka, kb, khbp, kc, kd1, kd2 = jax.random.split(key, 7)
        out_a = heads.a.apply(heads.a_params, model_batch, train=False)
        typ, stuff = _sample_a(out_a, t, ka)
        rolling["pitch_type"][:, t] = typ
        rolling["stuff"][:, t] = stuff
        model_batch = {name: jnp.asarray(value) for name, value in rolling.items()}
        out_b = heads.b.apply(heads.b_params, model_batch, train=False)
        swing = np.asarray(jax.random.bernoulli(kb, jax.nn.sigmoid(out_b["swing_logit"][:, t]))).copy()
        key, kcontact, kfoul = jax.random.split(key, 3)
        contact = np.asarray(jax.random.bernoulli(kcontact, jax.nn.sigmoid(out_b["contact_logit"][:, t]))).copy()
        foul = np.asarray(jax.random.bernoulli(kfoul, jax.nn.sigmoid(out_b["foul_logit"][:, t]))).copy()
        swing &= active
        contact &= swing
        foul &= contact
        hit_by_pitch = np.asarray(jax.random.bernoulli(
            khbp, jax.nn.sigmoid(out_b["hbp_logit"][:, t]))).copy()
        hit_by_pitch &= active & ~swing
        rolling["swing"][:, t] = swing
        rolling["contact"][:, t] = contact
        rolling["foul"][:, t] = foul

        event = np.full(B, -1, np.int32)
        if heads.c is not None and heads.c_params is not None:
            out_c = heads.c.apply(heads.c_params, {name: jnp.asarray(value) for name, value in rolling.items()}, train=False)
            event = np.asarray(_sample_c_event(out_c["event_logits"][:, t], kc)).astype(np.int32)
            event[~active] = -1

        px = stuff[:, 3] * STUFF_SCALE[3] + STUFF_CENTRE[3]
        pz = stuff[:, 4] * STUFF_SCALE[4] + STUFF_CENTRE[4]
        called_strike = (~swing) & ~hit_by_pitch & (np.abs(px) <= ZONE_HALF_WIDTH) & (pz >= ZONE_BOTTOM) & (pz <= ZONE_TOP)
        in_play = swing & contact & ~foul
        outcome = np.full(B, int(PAOutcome.OUT), np.int32)
        outcome[(state["strikes"] + ((swing & ~contact) | called_strike) >= 3)] = int(PAOutcome.STRIKEOUT)
        outcome[(state["balls"] + ((~swing) & ~hit_by_pitch & ~called_strike) >= 4)] = int(PAOutcome.WALK)
        outcome[hit_by_pitch] = int(PAOutcome.HIT_BY_PITCH)

        if in_play.any() and heads.d is not None and heads.d_params is not None:
            d_batch = {name: jnp.asarray(value) for name, value in rolling.items()}
            out_d = heads.d.apply(heads.d_params, d_batch, train=False)
            mu, ls = out_d["launch_mu"][:, t], out_d["launch_logsigma"][:, t]
            launch = np.asarray(mu + jnp.exp(ls) * jax.random.normal(kd1, mu.shape))
            rolling["launch"][:, t] = launch
            d_batch = {name: jnp.asarray(value) for name, value in rolling.items()}
            out_d = heads.d.apply(heads.d_params, d_batch, train=False)
            d_out = np.asarray(jax.random.categorical(kd2, out_d["outcome_logits"][:, t]))
            outcome[in_play] = _D_TO_PA[d_out[in_play]]

        # The empirical engine is the established source of base advancement and
        # run totals.  It also gives the same outcome a context-specific chance of
        # a double play or sacrifice out.
        terminal = active & ((outcome == int(PAOutcome.STRIKEOUT))
                             | (outcome == int(PAOutcome.WALK))
                             | (outcome == int(PAOutcome.HIT_BY_PITCH)) | in_play)
        trans = engine.sample(state["base"], state["outs"], outcome, rng)
        runs = np.where(terminal, trans["runs"], 0)
        outs_added = np.where(terminal, trans["out_inc"], 0)
        next_outs = state["outs"] + outs_added
        inning_over = terminal & (next_outs >= 3)
        state["base"] = np.where(terminal, trans["bs_after"], state["base"])
        state["base"] = np.where(inning_over, 0, state["base"])
        state["outs"] = np.where(inning_over, 0, next_outs)
        state["home_score"] += runs * (state["half"] == 1)
        state["away_score"] += runs * (state["half"] == 0)
        state["balls"] = np.where(terminal, 0, state["balls"] + ((~swing) & ~hit_by_pitch & ~called_strike))
        state["strikes"] = np.where(terminal, 0, np.minimum(2, state["strikes"] + ((swing & ~contact) | called_strike | foul)))
        # C events occur around a pitch.  Only non-terminal rows are applied:
        # terminal PA rows already use the empirical PA transition, where a
        # second runner transition would double count the same movement.
        c_apply = active & ~terminal & (event >= 0) & (c_engine is not None)
        records["event"][:, t] = np.asarray(_event_flags(jnp.asarray(np.where(c_apply, event, -1)), getattr(heads.c, "c_event_mode", "legacy")))
        if c_apply.any() and c_engine is not None:
            ct = c_engine.sample(event, state["base"], state["outs"], rng)
            state["base"] = np.where(c_apply, ct["base"], state["base"])
            state["outs"] = np.where(c_apply, ct["outs"], state["outs"])
            c_runs = np.where(c_apply, ct["runs"], 0)
            state["home_score"] += c_runs * (state["half"] == 1)
            state["away_score"] += c_runs * (state["half"] == 0)
            c_third_out = c_apply & (state["outs"] >= 3)
            state["base"] = np.where(c_third_out, 0, state["base"])
            state["outs"] = np.where(c_third_out, 0, state["outs"])
            state["ended"] |= c_third_out
        if stop_when_decided:
            walkoff = ((state["half"] == 1) & (state["inning"] >= 9)
                       & (state["home_score"] > state["away_score"]))
            state["ended"] |= walkoff
        state["pitch_count"] += active
        state["ended"] |= inning_over
        state["pa_slot"] += terminal.astype(np.int32)
        records["active"][:, t] = active
        records["ctx"][:, t] = rolling["ctx"][:, t]
        records["pa_terminal"][:, t] = terminal
        records["pa_outcome"][:, t] = np.where(terminal, outcome, -1)
        records["pitch_type"][:, t] = typ
        records["stuff"][:, t] = stuff
        records["launch"][:, t] = rolling["launch"][:, t]
        records["swing"][:, t] = swing
        records["contact"][:, t] = contact
        records["foul"][:, t] = foul
        records["hbp"][:, t] = hit_by_pitch
        records["batter_idx"][:, t] = rolling["batter_idx"][:, t]
        # PIT calibration fields
        records["swing_prob"][:, t] = np.asarray(jax.nn.sigmoid(out_b["swing_logit"][:, t]))
        records["contact_prob"][:, t] = np.asarray(jax.nn.sigmoid(out_b["contact_logit"][:, t]))
        records["foul_prob"][:, t] = np.asarray(jax.nn.sigmoid(out_b["foul_logit"][:, t]))
        records["hbp_prob"][:, t] = np.asarray(jax.nn.sigmoid(out_b["hbp_logit"][:, t]))
        zone_mask = (np.abs(px) <= ZONE_HALF_WIDTH) & (pz >= ZONE_BOTTOM) & (pz <= ZONE_TOP)
        records["zone"][:, t] = zone_mask
        records["balls"][:, t] = state["balls"]
        records["strikes"][:, t] = state["strikes"]
        if in_play.any() and heads.d is not None and heads.d_params is not None:
            records["d_outcome_probs"][:, t] = np.asarray(jax.nn.softmax(out_d["outcome_logits"][:, t], axis=-1))
    records["final_state"] = state
    return records


class GameHistory:
    """Per-game strict-window caches, independent of batching and decode padding."""
    def __init__(self, policy='game'):
        self.policy = policy
        self.rows = {}

    def key(self, game, inning, half):
        if self.policy == 'half_inning':
            return (int(game), int(inning), int(half))
        if self.policy == 'batting_side':
            return (int(game), int(half))
        return int(game)

    def get(self, heads, batch, games, inning, half):
        if self.policy == 'legacy':
            return None
        if heads.a.window_size <= 0:
            raise ValueError('History continuation requires a strict window checkpoint')
        values = []
        for i, game in enumerate(games):
            key = self.key(game, inning, half)
            if key in self.rows:
                values.append(self.rows[key])
            else:
                one = {k: v[i:i+1] for k, v in batch.items()}
                values.append(tuple(_cache_template(model, params, one) if model is not None else None
                    for model, params in [(heads.a,heads.a_params),(heads.b,heads.b_params),
                                          (heads.c,heads.c_params),(heads.d,heads.d_params)]))
        return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *values)

    def put(self, games, inning, half, cache):
        if self.policy != 'legacy':
            for i, game in enumerate(games):
                self.rows[self.key(game, inning, half)] = jax.tree.map(lambda x: x[i:i+1], cache)


def rollout_batch(
    heads: PitchformerHeads,
    batch: dict[str, np.ndarray],
    *,
    seed: int,
    engine: EmpiricalEngine,
    c_engine: CTransitionEngine | None = None,
    initial_state: dict[str, np.ndarray] | None = None,
    stop_when_decided: bool = False,
    decode_len: int | None = None,
    initial_cache=None,
) -> dict[str, np.ndarray]:
    """Free-roll a padded batch of half-inning pitch sequences.

    Checkpoint-backed A--D heads use the cached, whole-loop compiled path.  A
    small reference implementation is retained for test doubles and is useful
    when inspecting a single hand-written transition fixture.
    """
    if _supports_cached_decode(heads):
        return _rollout_batch_cached(
            heads, batch, seed=seed, engine=engine, c_engine=c_engine,
            initial_state=initial_state, stop_when_decided=stop_when_decided, decode_len=decode_len, initial_cache=initial_cache,
        )
    if initial_cache is not None:
        raise ValueError("History continuation requires checkpoint-backed cached heads")
    return _rollout_batch_reference(
        heads, batch, seed=seed, engine=engine, c_engine=c_engine,
        initial_state=initial_state, stop_when_decided=stop_when_decided,
    )
