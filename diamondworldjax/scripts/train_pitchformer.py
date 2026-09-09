"""Train transformers A and B by maximum likelihood, and score them honestly.

WHAT IS BEING MEASURED

Per-pitch held-out NLL for each head, against a baseline that already knows the
easy part. The lesson from the PA-level work is that a likelihood number on its
own is nearly uninterpretable here: per-PA NLL is saturated, every model scores
1.49 to 1.55 nats against a marginal entropy of 1.495, and JEPA got the best NLL
of any variant while differentiating players not at all. So every head reports
its improvement over an explicit baseline rather than a bare NLL:

  pitch type   vs the empirical marginal over the 8 classes
  swing        vs the count-conditional swing rate (balls x strikes), which is
               most of what a naive model would get right
  contact      vs the marginal contact-given-swing rate
  foul         vs the marginal foul-given-contact rate

A head that cannot beat its baseline has learned nothing worth having, whatever
its absolute NLL looks like.

TRAIN/TEST SPLIT

By SEASON, never by random row. Pitches inside one game are heavily dependent, so
a random split would put the same half-inning on both sides and report memorised
context as generalisation. 2015-2023 train, 2024 test, which is the same split the
PA-level results use.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..data.pitch_seq import (build_id_maps, load_seasons, make_sequences,
                              EVENT_FLAGS)
import flax
import flax.linen as nn
from functools import partial

from ..model.pitchformer import (TransformerA, TransformerB, SuperState,
                                 loss_a, loss_b, N_PITCH_TYPES)
from ..model.transformer_c import TransformerC, loss_c
from ..model.transformer_d import N_BATTED, TransformerD, loss_d


def batches(arrs, bs, rng=None, shuffle=True):
    n = len(arrs["valid"])
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, n if not shuffle else n - bs + 1, bs):
        j = idx[i:i + bs]
        yield {k: jnp.asarray(v[j]) for k, v in arrs.items()}


def aggregate_evaluation(rows):
    """Weight every reported loss by the head's observed target count, including tails."""
    sums, counts = {}, {}
    for parts, batch in rows:
        valid = np.asarray(batch["valid"]) * np.asarray(batch.get("loss_mask", 1))
        masks = {"nll_bundle": valid * np.asarray(batch.get("c_eligible", 1)), "nll_type": valid * np.asarray(batch["type_valid"]),
                 "nll_stuff": valid * np.asarray(batch["stuff_valid"]),
                 "nll_swing": valid, "nll_contact": valid * np.asarray(batch["swing"]),
                 "nll_foul": valid * np.asarray(batch["swing"]) * np.asarray(batch["contact"]),
                 "nll_hbp": valid * (1 - np.asarray(batch["swing"])),
                 "nll_launch": valid * np.asarray(batch["launch_valid"]),
                 "nll_outcome": valid * np.asarray(batch["batted_valid"]),
                 "nll_hr": valid * np.asarray(batch["batted_valid"])}
        for name, value in parts.items():
            n = float(masks.get(name, valid).sum())
            sums[name] = sums.get(name, 0.0) + float(value) * n
            counts[name] = counts.get(name, 0.0) + n
    if not sums:
        raise ValueError("No held-out sequences to evaluate")
    result = {name: sums[name] / counts[name] if counts[name] else 0.0 for name in sums}
    result["loss"] = sum(v for k, v in result.items() if k != "nll_hr")
    result["target_counts"] = counts
    return result


def baselines(train, test):
    """Marginal / count-conditional baselines the heads must beat."""
    def flat(a, m):
        return a[m > 0]

    v = train["valid"] * train["type_valid"]
    tt = flat(train["pitch_type"], v)
    p_type = np.bincount(tt, minlength=N_PITCH_TYPES).astype(np.float64)
    p_type /= p_type.sum()

    # Swing rate by (balls, strikes). ctx columns 0 and 1 are balls/3 and
    # strikes/2, so they invert exactly.
    b = np.rint(train["ctx"][..., 0] * 3).astype(int)
    s = np.rint(train["ctx"][..., 1] * 2).astype(int)
    vt = train["valid"] > 0
    sw_tab = np.zeros((4, 3))
    for bb in range(4):
        for ss in range(3):
            m = vt & (b == bb) & (s == ss)
            sw_tab[bb, ss] = train["swing"][m].mean() if m.sum() else 0.5

    sw = train["swing"][vt]
    ct = train["contact"][vt * (train["swing"] > 0)]
    fl = train["foul"][vt * (train["swing"] > 0) * (train["contact"] > 0)]

    # Score the baselines on TEST.
    vte = test["valid"] > 0
    vty = vte & (test["type_valid"] > 0)
    nll_type = -np.log(np.clip(p_type[test["pitch_type"][vty]], 1e-12, None)).mean()

    bt = np.rint(test["ctx"][..., 0] * 3).astype(int)
    st = np.rint(test["ctx"][..., 1] * 2).astype(int)
    p_sw = np.clip(sw_tab[np.clip(bt, 0, 3), np.clip(st, 0, 2)], 1e-6, 1 - 1e-6)
    y = test["swing"]
    nll_sw = -(y * np.log(p_sw) + (1 - y) * np.log(1 - p_sw))[vte].mean()

    def bern(p, y, m):
        p = float(np.clip(p, 1e-6, 1 - 1e-6))
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))[m].mean()

    m_ct = vte & (test["swing"] > 0)
    m_fl = m_ct & (test["contact"] > 0)
    m_hbp = vte & (test["swing"] == 0)
    return {
        "type": float(nll_type),
        "swing": float(nll_sw),
        "contact": float(bern(ct.mean(), test["contact"], m_ct)),
        "foul": float(bern(fl.mean(), test["foul"], m_fl)),
        "hbp": float(bern(train["hbp"][vt & (train["swing"] == 0)].mean(),
                           test["hbp"], m_hbp)),
        "swing_marginal": float(sw.mean()),
    }


def run(model, loss_fn, train, test, *, steps, bs, lr, seed, name, out_dir, player_tables=None):
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)

    init_batch = {k: jnp.asarray(v[:2]) for k, v in train.items()}
    key, k0 = jax.random.split(key)
    params = install_player_data(model.init(k0, init_batch, train=False), player_tables)

    n_par = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"[{name}] {n_par:,} parameters", flush=True)

    sched = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.1, peak_value=lr, warmup_steps=max(1, steps // 20),
        decay_steps=steps, end_value=lr * 0.05)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4))
    opt = trainable_optimizer(opt)
    state = opt.init(params)

    @jax.jit
    def step(params, state, batch, key):
        def f(p):
            out = model.apply(p, batch, train=True, rngs={"dropout": key})
            l, parts = loss_fn(out, batch)
            return l, parts
        (l, parts), g = jax.value_and_grad(f, has_aux=True)(params)
        upd, state = opt.update(g, state, params)
        return optax.apply_updates(params, upd), state, l, parts

    @jax.jit
    def evaluate(params, batch):
        out = model.apply(params, batch, train=False)
        return loss_fn(out, batch)

    t0 = time.time()
    it = 0
    while it < steps:
        for batch in batches(train, bs, rng):
            key, sk = jax.random.split(key)
            params, state, l, parts = step(params, state, batch, sk)
            it += 1
            if it % 200 == 0:
                print(f"[{name}] step {it}/{steps} loss {float(l):.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if it >= steps:
                break

    res = aggregate_evaluation(
        (evaluate(params, batch)[1], batch) for batch in batches(test, bs, shuffle=False))
    print(f"[{name}] held-out: {res}", flush=True)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{out_dir}/{name}_params.pkl", "wb") as f:
        pickle.dump(jax.device_get(params), f)
    return res


# ---------------------------------------------------------------------------
# Shared-embedding training (--shared-emb)
# ---------------------------------------------------------------------------

from ..model.pitchformer import HeadResidual
from ..model.pitchformer_checkpoint import (install_player_data, trainable_optimizer,
    transfer_pa_skills, save_metadata, export_shared_head)


class SharedPitchformer(nn.Module):
    """All heads sharing one set of player/park embeddings.

    One SuperState produces the global identity vector.  Each head adds a small
    learned residual (HeadResidual) and passes the result as ``ss_override`` to
    its own independently-parameterised Trunk + output layers. Each training
    step scores every head on the same batch; shared parameters accumulate all
    head gradients while every private head receives one update.
    """
    n_pitchers: int
    n_batters: int
    n_parks: int
    d_model: int = 192
    n_layers: int = 4
    n_heads: int = 6
    d_residual: int = 16
    dropout: float = 0.1
    heads: str = "abcd"
    player_mode: str = "id"
    skill_seasons: int = 1
    pitch_history: bool = False
    position_encoding: str = "learned"
    window_size: int = 0
    observation_masks: bool = False
    c_event_mode: str = "legacy"
    c_support: tuple | None = None

    def setup(self):
        self.shared_ss = SuperState(
            self.n_pitchers, self.n_batters, self.n_parks,
            d_model=self.d_model, player_mode=self.player_mode, skill_seasons=self.skill_seasons)

        kw = dict(n_pitchers=self.n_pitchers, n_batters=self.n_batters,
                  n_parks=self.n_parks, d_model=self.d_model,
                  n_layers=self.n_layers, n_heads=self.n_heads,
                  dropout=self.dropout, player_mode=self.player_mode,
                  skill_seasons=self.skill_seasons, pitch_history=self.pitch_history,
                  position_encoding=self.position_encoding, window_size=self.window_size,
                  observation_masks=self.observation_masks, c_event_mode=self.c_event_mode, c_support=self.c_support)

        if "a" in self.heads:
            self.res_a = HeadResidual(self.n_pitchers, self.n_batters,
                                      self.d_residual, self.d_model, self.player_mode)
            self.head_a = TransformerA(**kw)
        if "b" in self.heads:
            self.res_b = HeadResidual(self.n_pitchers, self.n_batters,
                                      self.d_residual, self.d_model, self.player_mode)
            self.head_b = TransformerB(**kw)
        if "c" in self.heads:
            self.res_c = HeadResidual(self.n_pitchers, self.n_batters,
                                      self.d_residual, self.d_model, self.player_mode)
            self.head_c = TransformerC(**kw)
        if "d" in self.heads:
            self.res_d = HeadResidual(self.n_pitchers, self.n_batters,
                                      self.d_residual, self.d_model, self.player_mode)
            self.head_d = TransformerD(**kw)

    def __call__(self, batch, head: str, *, train: bool, decode: bool = False):
        ss = self.shared_ss(
            batch["pitcher_idx"], batch["batter_idx"], batch["park_idx"],
            batch["ctx"], batch["geom"], batch.get("skill_season"))

        res_map = {"a": getattr(self, "res_a", None),
                   "b": getattr(self, "res_b", None),
                   "c": getattr(self, "res_c", None),
                   "d": getattr(self, "res_d", None)}
        head_map = {"a": getattr(self, "head_a", None),
                    "b": getattr(self, "head_b", None),
                    "c": getattr(self, "head_c", None),
                    "d": getattr(self, "head_d", None)}

        ss_h = ss + res_map[head](batch["pitcher_idx"], batch["batter_idx"], ss)
        return head_map[head](batch, train=train, decode=decode, ss_override=ss_h)


def _deep_merge(base, override):
    """Recursively merge two nested dicts; base values win on conflict."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        elif k not in result:
            result[k] = v
    return result


def _init_shared(model, key, init_batch, heads):
    """Initialise SharedPitchformer by calling each head and merging params.

    Each head's nn.compact sub-modules materialise only when called, so we
    init once per head and deep-merge the resulting param trees.  The shared
    SuperState appears in every init; we keep the first copy.
    """
    merged = None
    for h in heads:
        key, k = jax.random.split(key)
        p = flax.core.unfreeze(model.init(k, init_batch, head=h, train=False))
        if merged is None:
            merged = p
        else:
            merged = _deep_merge(merged, p)
    return flax.core.freeze(merged)


def run_shared(model, train, test, *, steps, bs, lr, seed, out_dir, tag,
               heads, loss_fns, base, player_tables=None, missing_samples=0):
    """Joint updates: every head receives `steps` supervised batches and updates.

    Summing head losses avoids dormant heads receiving extra Adam momentum or
    weight-decay updates between turns. The per-head budget matches `run`.
    """
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)

    init_batch = {k: jnp.asarray(v[:2]) for k, v in train.items()}
    params = install_player_data(_init_shared(model, key, init_batch, heads), player_tables)
    key, _ = jax.random.split(key)  # consume one to stay aligned

    n_par = sum(x.size for x in jax.tree_util.tree_leaves(params))
    n_shared = sum(x.size for x in jax.tree_util.tree_leaves(
        flax.core.unfreeze(params)["params"]["shared_ss"]))
    print(f"[shared] {n_par:,} total parameters "
          f"({n_shared:,} shared embeddings)", flush=True)

    sched = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.1, peak_value=lr, warmup_steps=max(1, steps // 20),
        decay_steps=steps, end_value=lr * 0.05)
    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.adamw(sched, weight_decay=1e-4))
    opt = trainable_optimizer(opt)
    state = opt.init(params)

    @jax.jit
    def step(params, state, batch, key):
        def f(p):
            if missing_samples:
                from diamondworldjax.model.marginal_pitch_likelihood import marginal_log_likelihood
                def apply(data):
                    return {h: model.apply(p, data, head=h, train=True,
                        rngs={'dropout': jax.random.fold_in(key, ord(h))}) for h in heads}
                # Sum likelihood rather than independent mean losses; one
                # complete probability model for all representation variants.
                return -marginal_log_likelihood(apply, batch, key, missing_samples) / batch['valid'].shape[0]
            keys = jax.random.split(key, len(heads))
            losses = []
            for head, head_key in zip(heads, keys):
                out = model.apply(p, batch, head=head, train=True,
                                  rngs={"dropout": head_key})
                losses.append(loss_fns[head](out, batch)[0])
            return jnp.stack(losses).sum()
        loss, gradients = jax.value_and_grad(f)(params)
        updates, new_state = opt.update(gradients, state, params)
        return optax.apply_updates(params, updates), new_state, loss

    t0 = time.time()
    it = 0
    while it < steps:
        for batch in batches(train, bs, rng):
            key, sk = jax.random.split(key)
            params, state, loss = step(params, state, batch, sk)
            it += 1
            if it % 200 == 0:
                print(f"[shared] step {it}/{steps} all heads loss {float(loss):.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if it >= steps:
                break

    # Per-head legacy diagnostics are explicitly separate from the joint score.
    results = {}
    if missing_samples:
        from diamondworldjax.model.marginal_pitch_likelihood import marginal_log_likelihood
        @jax.jit
        def score_marginal(batch):
            apply = lambda data: {h: model.apply(params, data, head=h, train=False) for h in heads}
            return marginal_log_likelihood(apply, batch, key, missing_samples)
        results['joint_marginal'] = {'log_likelihood': sum(float(score_marginal(b))
            for b in batches(test, bs, shuffle=False)), 'samples': missing_samples}

    for head in heads:
        @partial(jax.jit, static_argnums=(2,))
        def evaluate(params, batch, head):
            out = model.apply(params, batch, head=head, train=False)
            return loss_fns[head](out, batch)

        results[head] = aggregate_evaluation(
            (evaluate(params, batch, head)[1], batch) for batch in batches(test, bs, shuffle=False))
        print(f"[shared/{head}] held-out: {results[head]}", flush=True)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{out_dir}/shared_{tag}_params.pkl", "wb") as f:
        pickle.dump(jax.device_get(params), f)

    for head in heads:
        with open(f"{out_dir}/{head.upper()}_{tag}_params.pkl", "wb") as f:
            pickle.dump(jax.device_get(export_shared_head(params, head)), f)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-seasons", default="2015,2016,2017,2018,2019,2021,2022,2023")
    ap.add_argument("--test-season", type=int, default=2024)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-train-rows", type=int, default=None,
                    help="smoke-test escape hatch; None uses everything")
    ap.add_argument("--out", default="checkpoints/pitchformer")
    ap.add_argument("--tag", default="ab")
    ap.add_argument("--stack", default="ab",
                    help="which heads to train: any of a, b, c, d")
    ap.add_argument("--events", default="data/processed/events.parquet")
    ap.add_argument("--game-context", default="data/processed/game_context.parquet")
    ap.add_argument("--no-env", action="store_true",
                    help="zero the weather/altitude block, for the A/B comparison")
    ap.add_argument("--no-geom", action="store_true",
                    help="zero the park geometry block, for the A/B comparison")
    ap.add_argument("--shared-emb", action="store_true",
                    help="Train all heads with shared player embeddings and "
                         "per-head residuals via joint updates. Saves a single "
                         "checkpoint with all heads.")
    ap.add_argument("--d-residual", type=int, default=16,
                    help="Per-head residual embedding dimension (shared-emb only).")
    ap.add_argument("--player-skills", choices=["id", "pa", "pa-no-latent", "none", "bayesian"], default="id",
                    help="bayesian learns shared skills plus a separate Bayesian residual per head; pa transfers the PA encoder and posterior mean; pa-no-latent zeros the same latent before fusion; none removes player IDs.")
    ap.add_argument("--pa-skills-ckpt", default=None)
    ap.add_argument("--skill-prior", choices=["iso", "walk"], default="walk")
    ap.add_argument("--skill-residual-scale", type=float, default=0.35)
    ap.add_argument("--skill-walk-scale", type=float, default=0.3)
    ap.add_argument("--skill-feature-mode", choices=["pa", "neutral"], default="pa",
                    help="Native Bayesian covariates: pooled training-only PA statistics or zeros")
    ap.add_argument("--recency-halflife", type=float, default=None)
    ap.add_argument("--contact-quality", action="store_true")
    ap.add_argument("--per-stat-shrink", action="store_true")
    ap.add_argument("--skill-features", help="Optional NPZ: player_ids, stats, league, hand, through_year; no held-out-year covariates")
    ap.add_argument("--pitch-history", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--c-event-mode", choices=["bundles", "legacy"], default="bundles")
    ap.add_argument("--history-reset", choices=["game", "half_inning", "batting_side"], default="game")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--missing-samples", type=int, default=2)
    ap.add_argument("--position-encoding", choices=["learned", "sinusoidal"], default="sinusoidal")
    ap.add_argument("--window-size", type=int, default=32, help="Number of prior completed tokens in the strict window (0=legacy unlimited)")
    ap.add_argument("--context-len", type=int, default=None, help="Overlapping context tokens for sliding window training")
    args = ap.parse_args()
    if args.missing_samples < 0:
        ap.error('missing-samples must be nonnegative')
    if args.missing_samples:
        if 'a' not in args.stack:
            ap.error('Missing-pitch marginalization requires A in the trained stack')
        args.shared_emb = True
    if args.context_len is None:
        args.context_len = args.window_size
    if not 0 <= args.dropout < 1:
        ap.error('dropout must be in [0, 1)')
    if args.window_size == 0:
        ap.error('Configurable history continuation requires window-size > 0')
    if args.window_size < 0 or not 0 <= args.context_len < args.max_len:
        ap.error('Require window-size >= 0 and 0 <= context-len < max-len')
    if args.context_len < args.window_size:
        ap.error('context-len must cover window-size for identical target histories')
    if args.player_skills.startswith("pa") and not args.pa_skills_ckpt:
        ap.error("PA-compatible skills require --pa-skills-ckpt")
    if args.bs < 1 or args.steps < 2 or not args.stack or set(args.stack) - set("abcd") or len(set(args.stack)) != len(args.stack):
        ap.error("Use positive bs, steps >= 2, and distinct heads from abcd")

    seasons = [int(s) for s in args.train_seasons.split(",")]
    if args.test_season in seasons:
        ap.error("Test season must not occur in training seasons")
    print(f"loading train seasons {seasons}, test {args.test_season}", flush=True)
    tr_dfs = load_seasons(seasons)
    te_dfs = load_seasons([args.test_season])

    # Keep checkpoint embedding shapes compatible with evaluation even for a
    # row-limited smoke run.  The limit is a data-volume control, not a change
    # to the player/park vocabulary contract.
    maps = build_id_maps(tr_dfs)

    if args.limit_train_rows:
        tr_dfs = [d.head(args.limit_train_rows) for d in tr_dfs]
        te_dfs = [d.head(args.limit_train_rows) for d in te_dfs]

    player_tables, skill_season_base = None, min(seasons)
    if args.player_skills.startswith("pa"):
        player_tables, skill_season_base = transfer_pa_skills(
            args.pa_skills_ckpt, maps, args.player_skills, seasons)
    skill_seasons = player_tables["pitcher"].shape[1] if player_tables else 1
    print(f"pitchers={maps['n_pitcher']} batters={maps['n_batter']} "
          f"parks={maps['n_park']} (index 0 reserved for unknown)", flush=True)

    import polars as pl
    ev = None
    if "c" in args.stack:
        ev = pl.read_parquet(args.events)
        print(f"events table: {ev.height:,} rows", flush=True)
    gctx = None
    if not args.no_env:
        import os
        if os.path.exists(args.game_context):
            gctx = pl.read_parquet(args.game_context)
            print(f"game context: {gctx.height:,} games", flush=True)
        else:
            print("game context file absent, running without environment", flush=True)
            args.no_env = True
    else:
        print("environment block DISABLED (--no-env)", flush=True)

    train = make_sequences(pl.concat(tr_dfs), maps, args.max_len, events=ev,
                           game_ctx=gctx, context_len=args.context_len, history_reset=args.history_reset)
    test = make_sequences(pl.concat(te_dfs), maps, args.max_len, events=ev,
                          game_ctx=gctx, context_len=args.context_len, history_reset=args.history_reset)
    print(f"train seqs {train['valid'].shape}, pitches {int(train['valid'].sum()):,}",
          flush=True)
    print(f"test  seqs {test['valid'].shape}, pitches {int(test['valid'].sum()):,}",
          flush=True)

    for arrays in (train, test):
        arrays["skill_season"] = np.clip(arrays["season"] - skill_season_base, 0, skill_seasons - 1)
    if args.bs > len(train["valid"]):
        raise ValueError("Batch size exceeds available training sequences")
    if args.no_geom:
        train["geom"][:] = 0.0
        test["geom"][:] = 0.0
        print("geometry block DISABLED (--no-geom)", flush=True)

    args.c_support = None
    if 'c' in args.stack and args.c_event_mode == 'bundles':
        from diamondworldjax.sim.c_transition_engine import CTransitionEngine
        args.c_support = CTransitionEngine(event_mode='bundles').fit(pl.concat(tr_dfs), ev).support()
        args.c_support_coverage = {}
        support = np.asarray(args.c_support).reshape(256, 24)
        for split, arrays in [('train', train), ('test', test)]:
            target = (arrays['events'].astype(np.int32) * (1 << np.arange(8))).sum(-1)
            base_state = sum((arrays['ctx'][..., 3+i] > .5).astype(np.int32) * (1 << i) for i in range(3))
            outs = np.clip(np.rint(arrays['ctx'][..., 2] * 2).astype(np.int32), 0, 2)
            supported = support[target, base_state * 3 + outs]
            eligible = arrays['c_eligible'].astype(bool)
            valid = arrays['valid'] * arrays.get('loss_mask', 1) > 0
            args.c_support_coverage[split] = dict(eligible=int((valid & eligible).sum()),
                unsupported=int((valid & eligible & ~supported).sum()))
            arrays['c_eligible'] = eligible & supported
        print(f'C transition support coverage: {args.c_support_coverage}', flush=True)
    if args.player_skills == "bayesian":
        from diamondworldjax.model.bayesian_pitchformer import run_bayesian
        run_bayesian(args, train, test, maps, seasons, feature_pitches=pl.concat(tr_dfs))
        return

    base = baselines(train, test)
    print(f"baselines (test NLL): {base}", flush=True)

    kw = dict(n_pitchers=maps["n_pitcher"], n_batters=maps["n_batter"],
              n_parks=maps["n_park"], d_model=args.d_model, dropout=args.dropout,
              n_layers=args.layers, n_heads=args.heads, player_mode=args.player_skills,
              skill_seasons=skill_seasons, pitch_history=args.pitch_history,
              position_encoding=args.position_encoding, window_size=args.window_size,
              observation_masks=True, c_event_mode=args.c_event_mode, c_support=args.c_support)

    metadata = {"version": 1, "config": vars(args), "maps": maps, "train_years": seasons,
                "skill_season_base": skill_season_base,
                "model_options": {"player_mode": args.player_skills, "skill_seasons": skill_seasons,
                                  "pitch_history": args.pitch_history, "dropout": args.dropout,
                                  "observation_masks": True, "c_event_mode": args.c_event_mode, "c_support": args.c_support,
                                  "position_encoding": args.position_encoding,
                                  "window_size": args.window_size,
                                  "residual_dim": args.d_residual if args.shared_emb else 0}}
    save_metadata(args.out, args.tag, metadata)
    report = {"likelihood_note": "joint_marginal is the primary score when enabled; per-head scores are complete-case/plug-in diagnostics", "baselines": base, "config": vars(args), "improvement_nats": {}}

    # Pre-compute C and D baselines when those heads are in the stack (needed
    # by both the shared-emb and independent training paths).
    if "c" in args.stack:
        cb = {}
        for i, f in enumerate(EVENT_FLAGS):
            vtr = train["valid"] > 0
            rate = float(np.clip(train["events"][..., i][vtr].mean(), 1e-7, 1 - 1e-7))
            vte = test["valid"] > 0
            y = test["events"][..., i][vte]
            cb[f] = float(-(y * np.log(rate) + (1 - y) * np.log(1 - rate)).mean())
            cb[f"{f}_rate"] = rate
        report["baselines_c"] = cb

    if "d" in args.stack:
        vtr = (train["valid"] * train["batted_valid"]) > 0
        vte = (test["valid"] * test["batted_valid"]) > 0
        p_out = np.bincount(train["batted_out"][vtr], minlength=N_BATTED).astype(np.float64)
        p_out /= p_out.sum()
        nll_out_b = float(-np.log(p_out[test["batted_out"][vte]]).mean())
        p_hr = float(np.clip(p_out[4], 1e-6, 1 - 1e-6))
        y = (test["batted_out"][vte] == 4).astype(np.float64)
        nll_hr_b = float(-(y * np.log(p_hr) + (1 - y) * np.log(1 - p_hr)).mean())
        ltr = (train["valid"] * train["launch_valid"]) > 0
        lte = (test["valid"] * test["launch_valid"]) > 0
        mu_l = train["launch"][ltr].mean(0); sd_l = train["launch"][ltr].std(0) + 1e-6
        zz = (test["launch"][lte] - mu_l) / sd_l
        nll_launch_b = float((0.5 * zz ** 2 + np.log(sd_l) + 0.5 * np.log(2 * np.pi)).sum(-1).mean())
        report["baselines_d"] = {"outcome": nll_out_b, "hr": nll_hr_b,
                                 "launch": nll_launch_b, "p_outcome": p_out.tolist()}
        print(f"D baselines: outcome {nll_out_b:.4f}  hr {nll_hr_b:.4f}  launch {nll_launch_b:.4f}",
              flush=True)

    # ----- Shared-embedding joint path -----
    if args.shared_emb:
        loss_fns = {"a": loss_a, "b": loss_b, "c": loss_c, "d": loss_d}
        model = SharedPitchformer(
            heads=args.stack, d_residual=args.d_residual, **kw)
        results = run_shared(
            model, train, test, steps=args.steps, bs=args.bs, lr=args.lr,
            seed=args.seed, out_dir=args.out, tag=args.tag,
            heads=args.stack, loss_fns=loss_fns, base=base, player_tables=player_tables, missing_samples=args.missing_samples)

        # Map per-head results into the report.
        _head_metric_map = {
            "a": [("type", "nll_type")],
            "b": [("swing", "nll_swing"), ("contact", "nll_contact"),
                   ("foul", "nll_foul"), ("hbp", "nll_hbp")],
        }
        for h, res in results.items():
            if h == 'joint_marginal':
                report[h] = res
                continue
            report[h.upper()] = res
            if h in _head_metric_map:
                for bk, rk in _head_metric_map[h]:
                    report["improvement_nats"][bk] = base[bk] - res[rk]
            if h == "c":
                for f in EVENT_FLAGS:
                    if f in report.get("baselines_c", {}) and f"nll_{f}" in res:
                        report["improvement_nats"][f] = (
                            report["baselines_c"][f] - res.get(f"nll_{f}", float("nan")))
            if h == "d":
                bd = report.get("baselines_d", {})
                if bd:
                    report["improvement_nats"]["outcome"] = (
                        bd["outcome"] - res["nll_outcome"])
                    report["improvement_nats"]["hr"] = (
                        bd["hr"] - res["nll_hr"])
                    report["improvement_nats"]["launch"] = (
                        bd["launch"] - res["nll_launch"])

        Path("data/eval2").mkdir(parents=True, exist_ok=True)
        out = f"data/eval2/pitchformer_{args.tag}.json"
        with open(out, "w") as f:
            json.dump(report, f, indent=2)

        print("\n=== HELD-OUT IMPROVEMENT OVER BASELINE (nats/pitch, higher better) ===")
        for k, v in report["improvement_nats"].items():
            print(f"  {k:9} {v:+.4f}")
        print(f"saved -> {out}")
        return

    # ----- Independent per-head training (default) -----
    if "a" in args.stack:
        res_a = run(TransformerA(**kw), loss_a, train, test, steps=args.steps,
                    bs=args.bs, lr=args.lr, seed=args.seed, name=f"A_{args.tag}",
                    out_dir=args.out, player_tables=player_tables)
        report["A"] = res_a
        report["improvement_nats"]["type"] = base["type"] - res_a["nll_type"]

    if "b" in args.stack:
        res_b = run(TransformerB(**kw), loss_b, train, test, steps=args.steps,
                    bs=args.bs, lr=args.lr, seed=args.seed, name=f"B_{args.tag}",
                    out_dir=args.out, player_tables=player_tables)
        report["B"] = res_b
        for k in ("swing", "contact", "foul", "hbp"):
            report["improvement_nats"][k] = base[k] - res_b[f"nll_{k}"]

    if "c" in args.stack:
        cb = report["baselines_c"]
        res_c = run(TransformerC(**kw), loss_c, train, test, steps=args.steps,
                    bs=args.bs, lr=args.lr, seed=args.seed, name=f"C_{args.tag}",
                    out_dir=args.out, player_tables=player_tables)
        report["C"] = res_c
        for f in EVENT_FLAGS:
            report["improvement_nats"][f] = cb[f] - res_c[f"nll_{f}"]

    if "d" in args.stack:
        bd = report["baselines_d"]
        res_d = run(TransformerD(**kw), loss_d, train, test, steps=args.steps,
                    bs=args.bs, lr=args.lr, seed=args.seed, name=f"D_{args.tag}",
                    out_dir=args.out, player_tables=player_tables)
        report["D"] = res_d
        report["improvement_nats"]["outcome"] = bd["outcome"] - res_d["nll_outcome"]
        report["improvement_nats"]["hr"] = bd["hr"] - res_d["nll_hr"]
        report["improvement_nats"]["launch"] = bd["launch"] - res_d["nll_launch"]
    Path("data/eval2").mkdir(parents=True, exist_ok=True)
    out = f"data/eval2/pitchformer_{args.tag}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== HELD-OUT IMPROVEMENT OVER BASELINE (nats/pitch, higher better) ===")
    for k, v in report["improvement_nats"].items():
        print(f"  {k:9} {v:+.4f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
