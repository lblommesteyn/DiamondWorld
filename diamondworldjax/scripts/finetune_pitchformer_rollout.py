"""Self-conditioned fine-tuning for a pretrained, separate A--D stack.

This is scheduled sampling for the pitch-level models: a configurable fraction
of half-inning sequences receives a generated pitch/game-state history, while
the observed next-pitch labels remain the supervised targets.  The unit is a
half-inning because that is the sequence unit the current pitchformer was
trained on; a full-game roster/lineup rollout remains a subsequent extension.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import polars as pl

from diamondworldjax.data.pitch_seq import build_id_maps, load_seasons, make_sequences
from diamondworldjax.model.pitchformer import TransformerA, TransformerB, loss_a, loss_b
from diamondworldjax.model.transformer_c import TransformerC, loss_c
from diamondworldjax.model.transformer_d import TransformerD, loss_d
from diamondworldjax.sim.rules_engine import EmpiricalEngine
from diamondworldjax.sim.c_transition_engine import CTransitionEngine
from diamondworldjax.simulate.pitchformer_rollout import PitchformerHeads, rollout_batch


def _load(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _save(path: Path, params) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(jax.device_get(params), f)


def _step_fn(model, loss_fn, opt):
    @jax.jit
    def step(params, state, batch, key):
        def objective(p):
            return loss_fn(model.apply(p, batch, train=True, rngs={"dropout": key}), batch)[0]
        loss, grads = jax.value_and_grad(objective)(params)
        updates, state = opt.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss
    return step


def _mixed_batches(batch: dict[str, np.ndarray], rolled: dict[str, np.ndarray], use: np.ndarray):
    """Build head-specific scheduled-sampling inputs without changing labels."""
    def replace(names):
        out = {name: value.copy() for name, value in batch.items()}
        for name in names:
            # Rollout deliberately has a padded decode extension so generated
            # PAs are not capped by the observed pitch count.  Fine-tuning has
            # supervised labels only in the original fixed-width sequence,
            # therefore inject just that overlapping history window.
            n = min(out[name].shape[1], rolled[name].shape[1])
            out[name][use, :n] = rolled[name][use, :n]
        return {name: jnp.asarray(value) for name, value in out.items()}
    # A conditions only on generated state.  Later heads additionally receive
    # generated upstream samples, while each still scores the recorded target.
    return (
        replace(("ctx",)),
        replace(("ctx", "pitch_type", "stuff")),
        replace(("ctx", "pitch_type", "stuff", "swing", "contact", "foul")),
        replace(("ctx", "pitch_type", "stuff", "launch")),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-tag", required=True, help="Existing teacher-forced A--D checkpoint tag.")
    ap.add_argument("--tag", required=True, help="Output fine-tuned checkpoint tag.")
    ap.add_argument("--params-dir", default="checkpoints/pitchformer")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--self-conditioned-fraction", type=float, default=0.15)
    ap.add_argument("--ss-warmup", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--game-context", default="data/processed/game_context.parquet")
    ap.add_argument("--events", default="data/processed/events.parquet")
    ap.add_argument("--no-env", action="store_true")
    ap.add_argument("--no-geom", action="store_true")
    args = ap.parse_args()
    if args.steps < 1 or args.batch < 1 or not 0 <= args.self_conditioned_fraction <= 1:
        raise SystemExit("steps/batch must be positive and self-conditioned fraction must be in [0, 1]")

    years = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]
    train = load_seasons(years)
    maps = build_id_maps(train)
    context_path = Path(args.game_context)
    gctx = None if args.no_env or not context_path.exists() else pl.read_parquet(context_path)
    event_path = Path(args.events)
    if not event_path.exists():
        raise SystemExit(f"C fine-tuning requires extracted event labels: {event_path}")
    seqs = make_sequences(pl.concat(train), maps, args.max_len,
                          events=pl.read_parquet(event_path), game_ctx=gctx)
    if args.no_geom:
        seqs["geom"][:] = 0.0
    root = Path(args.params_dir)
    kw = dict(n_pitchers=maps["n_pitcher"], n_batters=maps["n_batter"],
              n_parks=maps["n_park"], d_model=args.d_model, n_layers=args.layers, n_heads=args.heads)
    models = (TransformerA(**kw), TransformerB(**kw), TransformerC(**kw), TransformerD(**kw))
    params = [_load(root / f"{letter}_{args.init_tag}_params.pkl") for letter in "ABCD"]
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(args.lr, weight_decay=1e-4))
    states = [opt.init(p) for p in params]
    steps = [_step_fn(model, loss, opt) for model, loss in zip(models, (loss_a, loss_b, loss_c, loss_d))]
    engine = EmpiricalEngine().fit(pl.concat(train).filter(pl.col("pa_terminal")))
    c_engine = CTransitionEngine().fit(pl.concat(train), pl.read_parquet(event_path))
    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    nseq = len(seqs["valid"])
    log = []

    for i in range(args.steps):
        take = rng.integers(0, nseq, size=args.batch)
        raw = {name: np.asarray(value[take]).copy() for name, value in seqs.items()}
        rate = args.self_conditioned_fraction * min(1.0, (i + 1) / max(args.ss_warmup, 1))
        use = rng.random(args.batch) < rate
        if use.any():
            heads = PitchformerHeads(*models, *params)
            rolled = rollout_batch(heads, raw, seed=args.seed + i, engine=engine,
                                   c_engine=c_engine)
            batches = _mixed_batches(raw, rolled, use)
        else:
            base = {name: jnp.asarray(value) for name, value in raw.items()}
            batches = (base, base, base, base)
        losses = []
        for j in range(4):
            key, step_key = jax.random.split(key)
            params[j], states[j], loss = steps[j](params[j], states[j], batches[j], step_key)
            losses.append(float(loss))
        if (i + 1) % 50 == 0 or i == 0:
            row = {"step": i + 1, "ss_rate": rate, "sampled_sequences": int(use.sum()),
                   **{letter: loss for letter, loss in zip("ABCD", losses)}}
            log.append(row)
            print(row, flush=True)

    for letter, p in zip("ABCD", params):
        _save(root / f"{letter}_{args.tag}_params.pkl", p)
    out = Path("data/eval2") / f"pitchformer_finetune_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "log": log}, indent=2) + "\n")
    print(f"saved checkpoints with tag {args.tag}; log -> {out}")


if __name__ == "__main__":
    main()
