"""Simulate plate appearances pitch by pitch with A and B, and score the result.

THE QUESTION THIS ANSWERS

A and B beat their baselines on held-out likelihood by a wide margin. That does not
establish that simulating every pitch produces a better SIMULATOR. v1-v5 improved
components and made the simulator worse, and JEPA had the best per-PA NLL of any
variant while differentiating players not at all. So the pitch-level stack is scored
on the project's actual gate: cross-player correlation of K and BB rates over batters
with at least 150 PA, which v16 gets 0.792 and 0.651 on.

HOW A PLATE APPEARANCE IS PLAYED OUT

Starting from the real game context, the PA is rolled forward one pitch at a time:

  A     samples the pitch type, then the stuff (location and velocity) for it
  B     samples swing given that pitch, then contact given swing, then foul
  call  B samples a called strike on a taken non-HBP pitch (or, for older
        checkpoints, a rules-based-zone fallback is used)

  swing, no contact           -> strike
  swing, contact, foul        -> strike, except with two strikes, where the count
                                 holds, which is why a PA can run long
  swing, contact, not foul    -> ball in play, PA ends
  take, called strike         -> strike
  take, no called strike      -> ball

  three strikes -> strikeout      four balls -> walk

The processed pitch export has no raw call-description field, but it does have the
pre-pitch count.  The training loader infers calls from the next pitch's count (and
terminal taken strikeouts), so new B checkpoints learn this conditional event.
Older checkpoints fall back to the geometric zone, retaining compatibility.

CONTEXT IS REAL, THE PITCHES ARE SIMULATED

Each PA is rolled out conditioned on the REAL pitches that preceded it in that game
half. The alternative, simulating a whole half-inning, would drift away from the
distribution the trunk was trained on and confound "are the conditionals good" with
"does error compound over an inning". Those are separate questions and this script
answers the first. Compounding is the harder one and is not claimed here.
"""
from __future__ import annotations

import argparse
import json
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

from ..model.pitchformer_checkpoint import restore_metadata, head_kwargs, add_skill_season
from ..data.pitch_seq import (build_id_maps, load_seasons, make_sequences,
                              STUFF_CENTRE, STUFF_SCALE)
from ..model.pitchformer import TransformerA, TransformerB

# Rules-based strike zone in feet. Half-width is the plate (17in) plus a ball on
# each side, which is the zone as called rather than as written.
ZONE_HALF_WIDTH = 0.83
ZONE_BOTTOM = 1.52
ZONE_TOP = 3.42

MAX_PITCHES_PER_PA = 20      # a PA longer than this is vanishingly rare


def load_params(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--params-dir", default="checkpoints/pitchformer")
    ap.add_argument("--tag", default="ab_v1")
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-pa", type=int, default=150)
    ap.add_argument("--limit-seqs", type=int, default=None)
    ap.add_argument("--out", default="data/eval2/pitchlevel_playercorr.json")
    args = ap.parse_args()
    metadata = restore_metadata(args)

    train_seasons = metadata["train_years"] if metadata else [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]
    tr_dfs = load_seasons(train_seasons)
    maps = metadata["maps"] if metadata else build_id_maps(tr_dfs)
    del tr_dfs

    te = load_seasons([args.season])[0]
    from pathlib import Path
    context_path = Path(metadata["config"].get("game_context", "data/processed/game_context.parquet")) if metadata else None
    gctx = pl.read_parquet(context_path) if context_path and context_path.exists() and not args.no_env else None
    seqs = add_skill_season(make_sequences(te, maps, args.max_len, game_ctx=gctx), metadata)
    if metadata and args.no_geom:
        seqs["geom"][:] = 0
    if args.limit_seqs:
        seqs = {k: v[:args.limit_seqs] for k, v in seqs.items()}
    n_seq, T = seqs["valid"].shape
    print(f"{n_seq:,} sequences x {T}", flush=True)

    kw = head_kwargs(args, maps, metadata)
    A, B = TransformerA(**kw), TransformerB(**kw)
    pa = load_params(f"{args.params_dir}/A_{args.tag}_params.pkl")
    pb = load_params(f"{args.params_dir}/B_{args.tag}_params.pkl")

    @jax.jit
    def fwd_a(batch):
        return A.apply(pa, batch, train=False)

    @jax.jit
    def fwd_b(batch):
        return B.apply(pb, batch, train=False)

    key = jax.random.PRNGKey(args.seed)

    # Real per-batter K and BB, from the same rows the simulation covers, so the
    # comparison is like for like rather than against a season total that includes
    # plate appearances the simulator never saw.
    real_k, real_bb, real_pa = {}, {}, {}
    sim_k, sim_bb, sim_pa = {}, {}, {}
    zone_hits = zone_n = 0        # calibration of A's location model

    bs_eval = 64
    for start in range(0, n_seq, bs_eval):
        sl = slice(start, min(start + bs_eval, n_seq))
        batch = {k: jnp.asarray(v[sl]) for k, v in seqs.items()}
        outa = fwd_a(batch)
        outb = fwd_b(batch)

        key, k1, k2, k3, k4, k5, k6 = jax.random.split(key, 7)
        # A: pitch type, then the stuff for that type.
        ptype = jax.random.categorical(k1, outa["type_logits"])
        idx = ptype[..., None, None]
        mu = jnp.take_along_axis(outa["stuff_mu"],
                                 jnp.broadcast_to(idx, (*idx.shape[:2], 1, 5)),
                                 axis=2)[:, :, 0, :]
        ls = jnp.take_along_axis(outa["stuff_logsigma"],
                                 jnp.broadcast_to(idx, (*idx.shape[:2], 1, 5)),
                                 axis=2)[:, :, 0, :]
        stuff = mu + jnp.exp(ls) * jax.random.normal(k2, mu.shape)

        # B is asked about the pitch A just produced, not the real one.
        bb_in = dict(batch)
        bb_in["pitch_type"] = ptype
        bb_in["stuff"] = stuff
        outb = fwd_b(bb_in)

        swing = jax.random.bernoulli(k3, jax.nn.sigmoid(outb["swing_logit"]))
        contact = jax.random.bernoulli(k4, jax.nn.sigmoid(outb["contact_logit"]))
        foul = jax.random.bernoulli(k5, jax.nn.sigmoid(outb["foul_logit"]))

        # De-standardise location to feet for the zone test.
        px = np.asarray(stuff[..., 3]) * STUFF_SCALE[3] + STUFF_CENTRE[3]
        pz = np.asarray(stuff[..., 4]) * STUFF_SCALE[4] + STUFF_CENTRE[4]
        in_zone = (np.abs(px) <= ZONE_HALF_WIDTH) & (pz >= ZONE_BOTTOM) & (pz <= ZONE_TOP)
        called_strike = (jax.random.bernoulli(
            k6, jax.nn.sigmoid(outb["called_strike_logit"])
        ) if "called_strike_logit" in outb else jnp.asarray(in_zone))

        # How often do SAMPLED locations land in the zone, against 47.7% for real
        # 2024 pitches? If the model's Gaussian is over-dispersed, every taken
        # pitch is too often a ball and the simulated walk rate inflates, which is
        # the first thing to check when BB comes out wrong.
        _v = np.asarray(seqs["valid"][sl]) > 0
        zone_hits += int(in_zone[_v].sum())
        zone_n += int(_v.sum())

        sw = np.asarray(swing)
        ct = np.asarray(contact)
        fl = np.asarray(foul)
        cs = np.asarray(called_strike)
        valid = np.asarray(seqs["valid"][sl]) > 0
        bidx = np.asarray(seqs["batter_idx"][sl])

        # Walk the sequence, running the count machine per position. The batter
        # index changing marks a new plate appearance, which is what the real data
        # says; the simulation does not get to decide when a PA ends except by
        # strikeout, walk or a ball in play.
        nseq_b = sw.shape[0]
        for i in range(nseq_b):
            t = 0
            while t < T:
                if not valid[i, t]:
                    t += 1
                    continue
                # A real plate appearance is a maximal run of one batter. PA
                # boundaries come from the DATA, and the simulation resolves at
                # most one outcome inside each. An earlier version let a
                # simulated PA that ended early start a second PA on the same
                # batter, which inflated the denominator and pushed the K rate to
                # a third of its real value.
                b_id = int(bidx[i, t])
                end = t
                while end < T and valid[i, end] and int(bidx[i, end]) == b_id:
                    end += 1

                balls = strikes = 0
                outcome = None
                # A simulated PA may need more pitches than the real one used. It
                # continues past the real end by REUSING the distributions at the
                # PA's last real position, so the count is no longer fed back into
                # the conditioning for those extra pitches. That approximation is
                # confined to the tail of long PAs, and it is preferable to
                # truncating, which would bias both K and BB downward.
                for j in range(MAX_PITCHES_PER_PA):
                    u = min(t + j, end - 1)
                    if sw[i, u]:
                        if not ct[i, u]:
                            strikes += 1
                        elif fl[i, u]:
                            if strikes < 2:
                                strikes += 1
                        else:
                            outcome = "inplay"
                            break
                    else:
                        if cs[i, u]:
                            strikes += 1
                        else:
                            balls += 1
                    if strikes >= 3:
                        outcome = "K"
                        break
                    if balls >= 4:
                        outcome = "BB"
                        break

                sim_pa[b_id] = sim_pa.get(b_id, 0) + 1
                if outcome == "K":
                    sim_k[b_id] = sim_k.get(b_id, 0) + 1
                elif outcome == "BB":
                    sim_bb[b_id] = sim_bb.get(b_id, 0) + 1
                t = end

        if (start // bs_eval) % 20 == 0:
            print(f"  {min(start + bs_eval, n_seq)}/{n_seq}", flush=True)

    # Real rates, per batter, from the parquet.
    pa_real = (te.filter(pl.col("pa_terminal"))
                 .group_by("batter_id")
                 .agg([pl.len().alias("pa"),
                       (pl.col("pa_outcome") == "K").sum().alias("k"),
                       (pl.col("pa_outcome") == "BB").sum().alias("bb")]))
    inv = {v: k for k, v in maps["batter"].items()}

    rows = []
    for b_id, npa in sim_pa.items():
        if b_id == 0 or npa < args.min_pa:
            continue                      # index 0 is the UNKNOWN sink, excluded
        real_id = inv.get(b_id)
        if real_id is None:
            continue
        r = pa_real.filter(pl.col("batter_id") == real_id)
        if r.height == 0 or r["pa"][0] < args.min_pa:
            continue
        rows.append((sim_k.get(b_id, 0) / npa, sim_bb.get(b_id, 0) / npa,
                     r["k"][0] / r["pa"][0], r["bb"][0] / r["pa"][0]))

    arr = np.array(rows)
    res = {
        "n_batters": len(rows),
        "K_corr": float(np.corrcoef(arr[:, 0], arr[:, 2])[0, 1]),
        "BB_corr": float(np.corrcoef(arr[:, 1], arr[:, 3])[0, 1]),
        "sim_K_rate": float(arr[:, 0].mean()), "real_K_rate": float(arr[:, 2].mean()),
        "sim_BB_rate": float(arr[:, 1].mean()), "real_BB_rate": float(arr[:, 3].mean()),
        "sim_in_zone_rate": zone_hits / max(zone_n, 1),
        "real_in_zone_rate": 0.477,
        "reference_v16": {"K": 0.792, "BB": 0.651},
    }
    print(json.dumps(res, indent=2))
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
