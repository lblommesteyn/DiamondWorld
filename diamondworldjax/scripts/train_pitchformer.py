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
from ..model.pitchformer import (TransformerA, TransformerB, loss_a, loss_b,
                                 N_PITCH_TYPES)
from ..model.transformer_c import TransformerC, loss_c


def batches(arrs, bs, rng=None, shuffle=True):
    n = len(arrs["valid"])
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, n - bs + 1, bs):
        j = idx[i:i + bs]
        yield {k: jnp.asarray(v[j]) for k, v in arrs.items()}


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
    return {
        "type": float(nll_type),
        "swing": float(nll_sw),
        "contact": float(bern(ct.mean(), test["contact"], m_ct)),
        "foul": float(bern(fl.mean(), test["foul"], m_fl)),
        "swing_marginal": float(sw.mean()),
    }


def run(model, loss_fn, train, test, *, steps, bs, lr, seed, name, out_dir):
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)

    init_batch = {k: jnp.asarray(v[:2]) for k, v in train.items()}
    key, k0 = jax.random.split(key)
    params = model.init(k0, init_batch, train=False)

    n_par = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"[{name}] {n_par:,} parameters", flush=True)

    sched = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.1, peak_value=lr, warmup_steps=max(1, steps // 20),
        decay_steps=steps, end_value=lr * 0.05)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4))
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

    # Held-out score, accumulated over full batches only so every batch has the
    # same shape and the mean is not weighted by a ragged tail.
    tot, parts_sum, nb = 0.0, {}, 0
    for batch in batches(test, bs, shuffle=False):
        l, parts = evaluate(params, batch)
        tot += float(l)
        for k, v in parts.items():
            parts_sum[k] = parts_sum.get(k, 0.0) + float(v)
        nb += 1
    res = {"loss": tot / nb, **{k: v / nb for k, v in parts_sum.items()}}
    print(f"[{name}] held-out: {res}", flush=True)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{out_dir}/{name}_params.pkl", "wb") as f:
        pickle.dump(jax.device_get(params), f)
    return res


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
                    help="which heads to train: any of a, b, c")
    ap.add_argument("--events", default="data/processed/events.parquet")
    ap.add_argument("--game-context", default="data/processed/game_context.parquet")
    ap.add_argument("--no-env", action="store_true",
                    help="zero the weather/altitude block, for the A/B comparison")
    args = ap.parse_args()

    seasons = [int(s) for s in args.train_seasons.split(",")]
    print(f"loading train seasons {seasons}, test {args.test_season}", flush=True)
    tr_dfs = load_seasons(seasons)
    te_dfs = load_seasons([args.test_season])

    if args.limit_train_rows:
        tr_dfs = [d.head(args.limit_train_rows) for d in tr_dfs]
        te_dfs = [d.head(args.limit_train_rows) for d in te_dfs]

    maps = build_id_maps(tr_dfs)
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
    else:
        print("environment block DISABLED (--no-env)", flush=True)

    train = make_sequences(pl.concat(tr_dfs), maps, args.max_len, events=ev,
                           game_ctx=gctx)
    test = make_sequences(pl.concat(te_dfs), maps, args.max_len, events=ev,
                          game_ctx=gctx)
    print(f"train seqs {train['valid'].shape}, pitches {int(train['valid'].sum()):,}",
          flush=True)
    print(f"test  seqs {test['valid'].shape}, pitches {int(test['valid'].sum()):,}",
          flush=True)

    base = baselines(train, test)
    print(f"baselines (test NLL): {base}", flush=True)

    kw = dict(n_pitchers=maps["n_pitcher"], n_batters=maps["n_batter"],
              n_parks=maps["n_park"], d_model=args.d_model,
              n_layers=args.layers, n_heads=args.heads)

    report = {"baselines": base, "config": vars(args), "improvement_nats": {}}

    if "a" in args.stack:
        res_a = run(TransformerA(**kw), loss_a, train, test, steps=args.steps,
                    bs=args.bs, lr=args.lr, seed=args.seed, name=f"A_{args.tag}",
                    out_dir=args.out)
        report["A"] = res_a
        report["improvement_nats"]["type"] = base["type"] - res_a["nll_type"]

    if "b" in args.stack:
        res_b = run(TransformerB(**kw), loss_b, train, test, steps=args.steps,
                    bs=args.bs, lr=args.lr, seed=args.seed, name=f"B_{args.tag}",
                    out_dir=args.out)
        report["B"] = res_b
        for k in ("swing", "contact", "foul"):
            report["improvement_nats"][k] = base[k] - res_b[f"nll_{k}"]

    if "c" in args.stack:
        # Each event head is scored against its own BASE RATE, fitted on train
        # and evaluated on test. These events are rare enough (a balk is 0.016%
        # of pitches) that a head predicting zero everywhere scores a superb
        # loss, so the absolute NLL says nothing and only the lift does.
        cb = {}
        for i, f in enumerate(EVENT_FLAGS):
            vtr = train["valid"] > 0
            rate = float(np.clip(train["events"][..., i][vtr].mean(), 1e-7, 1 - 1e-7))
            vte = test["valid"] > 0
            y = test["events"][..., i][vte]
            cb[f] = float(-(y * np.log(rate) + (1 - y) * np.log(1 - rate)).mean())
            cb[f"{f}_rate"] = rate
        report["baselines_c"] = cb

        res_c = run(TransformerC(**kw), loss_c, train, test, steps=args.steps,
                    bs=args.bs, lr=args.lr, seed=args.seed, name=f"C_{args.tag}",
                    out_dir=args.out)
        report["C"] = res_c
        for f in EVENT_FLAGS:
            report["improvement_nats"][f] = cb[f] - res_c[f"nll_{f}"]
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
