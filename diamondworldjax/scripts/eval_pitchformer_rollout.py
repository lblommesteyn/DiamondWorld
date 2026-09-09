"""Evaluate a generated-state A--D pitchformer rollout on player-rate gates.

Unlike ``sim_pa_pitchlevel.py``, this evaluator never reads recorded prior pitch
state after a rollout starts.  Player, pitcher, and park schedules stay exogenous;
all counts, bases, score state, pitch packages, and PA outcomes are generated.

It is intentionally a bounded, diagnostic evaluator.  Autoregressive decoding
uses a KV cache and one compiled loop per batch; ``--limit-seqs`` remains useful
for smoke tests, while a full 2024 run is now practical on a GPU.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl

from diamondworldjax.model.pitchformer_checkpoint import (restore_metadata, head_kwargs, add_skill_season,
    trainable_optimizer, export_shared_head, save_metadata)
from diamondworldjax.data.pitch_seq import build_id_maps, load_seasons, make_sequences
from diamondworldjax.model.pitchformer import TransformerA, TransformerB
from diamondworldjax.model.transformer_c import TransformerC
from diamondworldjax.model.transformer_d import TransformerD
from diamondworldjax.sim.rules_engine import EmpiricalEngine
from diamondworldjax.sim.c_transition_engine import CTransitionEngine
from diamondworldjax.simulate.pitchformer_rollout import PitchformerHeads, rollout_batch


def _load(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _outcome_counts(rows: list[tuple[int, int]]) -> dict[int, np.ndarray]:
    """Outcome tuples -> PA and rate numerator counts per batter."""
    counts: dict[int, np.ndarray] = {}
    for batter, outcome in rows:
        if batter <= 0 or outcome < 0:
            continue
        x = counts.setdefault(batter, np.zeros(7, dtype=np.float64))
        x[0] += 1  # PA
        x[1] += outcome == 0
        x[2] += outcome == 1
        x[3] += outcome in (3, 4, 5, 6)
        x[4] += outcome == 6
        x[5] += outcome not in (1, 2)  # AB
        x[6] += outcome in (3, 4, 5, 6)  # hits
    return counts


def _rates_from_counts(counts: dict[int, np.ndarray], min_pa: int) -> dict[int, np.ndarray]:
    """Counts -> K, BB, Hit, HR, AVG rate vector for eligible batters."""
    out = {}
    for batter, x in counts.items():
        if x[0] >= min_pa and x[5] > 0:
            out[batter] = np.array([x[1] / x[0], x[2] / x[0], x[3] / x[0],
                                    x[4] / x[0], x[6] / x[5]])
    return out


def _rates(rows: list[tuple[int, int]], min_pa: int) -> dict[int, np.ndarray]:
    return _rates_from_counts(_outcome_counts(rows), min_pa)


def _coverage(counts: dict[int, np.ndarray], min_pa: int) -> dict[str, int]:
    """Report how much simulated PA support underlies a rate correlation."""
    return {
        "terminal_pas": int(sum(x[0] for x in counts.values())),
        "batters_with_pa": len(counts),
        "batters_at_min_pa": int(
            sum(x[0] >= min_pa and x[5] > 0 for x in counts.values())
        ),    }


def _correlations(sim: dict[int, np.ndarray], real: dict[int, np.ndarray]) -> dict[str, float]:
    ids = sorted(set(sim) & set(real))
    names = ("K", "BB", "Hit", "HR", "AVG")
    if len(ids) < 3:
        return {"n_batters": len(ids), **{f"{name}_corr": float("nan") for name in names}}
    s = np.stack([sim[i] for i in ids])
    r = np.stack([real[i] for i in ids])
    corr = {}
    for j, name in enumerate(names):
        # A collapsed simulated outcome distribution has zero variance, so a
        # correlation is undefined—not zero.  Avoid np.corrcoef's warning and
        # keep that distinction explicit in the JSON result.
        corr[f"{name}_corr"] = (float(np.corrcoef(s[:, j], r[:, j])[0, 1])
                                 if np.std(s[:, j]) > 0 and np.std(r[:, j]) > 0
                                 else float("nan"))
    return {"n_batters": len(ids), **corr}


def _finite_mean(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def _finite_sd(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.std(ddof=1)) if len(values) > 1 else (0.0 if len(values) == 1 else float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--params-dir", default="checkpoints/pitchformer")
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32,
                    help="Half-inning sequences per compiled rollout batch; lower only if GPU memory requires it.")
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--min-pa", type=int, default=150)
    ap.add_argument("--limit-seqs", type=int, default=None,
                    help="Smoke-test only: output is explicitly marked non-comparable.")
    ap.add_argument("--game-context", default="data/processed/game_context.parquet")
    ap.add_argument("--events", default="data/processed/events.parquet")
    ap.add_argument("--c-events", action="store_true",
                    help="Apply C's sampled non-terminal event transitions through the empirical adapter.")
    ap.add_argument("--no-env", action="store_true", help="Must match training.")
    ap.add_argument("--no-geom", action="store_true", help="Must match training.")
    ap.add_argument("--out", default=None)
    ap.add_argument('--skill-mode', choices=['auto', 'mean', 'sample'], default='auto',
                    help='auto samples native Bayesian skills once per rep; keeps exported worlds and ordinary checkpoints fixed')
    args = ap.parse_args()
    if args.reps < 1:
        ap.error('--reps must be positive')
    metadata = restore_metadata(args, args.tag)
    if args.reps < 1 or args.batch < 1 or args.min_pa < 1:
        raise SystemExit("--reps, --batch, and --min-pa must be positive")

    train_years = metadata["train_years"] if metadata else [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]
    train = load_seasons(train_years)
    maps = metadata["maps"] if metadata else build_id_maps(train)
    test = load_seasons([args.season])[0]
    context_path = Path(args.game_context)
    if args.no_env:
        gctx = None
    elif context_path.exists():
        gctx = pl.read_parquet(context_path)
    else:
        # Match train_pitchformer.py: absence of the optional enrichment must
        # not turn a checkpoint evaluation into a file-not-found failure.
        print("game context file absent; evaluating with environment features zeroed", flush=True)
        gctx = None
    seqs = add_skill_season(make_sequences(test, maps, args.max_len, game_ctx=gctx), metadata)
    if args.no_geom:
        seqs["geom"][:] = 0.0
    if args.limit_seqs:
        seqs = {name: value[:args.limit_seqs] for name, value in seqs.items()}

    kw = head_kwargs(args, maps, metadata)
    root = Path(args.params_dir)
    def head(letter: str, cls):
        path = root / f"{letter}_{args.tag}_params.pkl"
        return (cls(**kw), _load(path)) if path.exists() else (None, None)
    a, ap_ = head("A", TransformerA)
    b, bp = head("B", TransformerB)
    c, cp = head("C", TransformerC)
    d, dp = head("D", TransformerD)
    if a is None or b is None:
        raise SystemExit("A and B checkpoints are required for a pitch rollout")
    heads = PitchformerHeads(a, b, c, d, ap_, bp, cp, dp)
    from diamondworldjax.eval.pitchformer_worlds import PitchformerWorlds
    worlds = PitchformerWorlds(heads, args.params_dir, args.tag, metadata,
                              args.skill_mode, args.seed)
    engine = EmpiricalEngine().fit(pl.concat(train).filter(pl.col("pa_terminal")))
    c_engine = None
    if args.c_events:
        event_path = Path(args.events)
        if c is None or not event_path.exists():
            raise SystemExit("--c-events requires a C checkpoint and extracted events parquet")
        c_engine = CTransitionEngine(event_mode=getattr(c, "c_event_mode", "legacy")).fit(pl.concat(train), pl.read_parquet(event_path))

    # Full-slate real rates are comparable only when every sequence is rolled.
    terminal = test.filter(pl.col("pa_terminal") & pl.col("pa_outcome").is_not_null())
    outcome_map = {"K": 0, "BB": 1, "HBP": 2, "1B": 3, "2B": 4, "3B": 5, "HR": 6, "out": 7, "E": 8}
    real_rows = [(maps["batter"].get(row[0], 0), outcome_map[row[1]])
                 for row in terminal.select(["batter_id", "pa_outcome"]).iter_rows()
                 if row[1] in outcome_map]
    real_counts = _outcome_counts(real_rows)
    real = _rates_from_counts(real_counts, args.min_pa)

    rep_results = []
    nseq = len(seqs["valid"])
    for rep in range(args.reps):
        heads = worlds.for_rep(rep)
        sampled: list[tuple[int, int]] = []
        event_counts = np.zeros(8, dtype=np.int64)
        for start in range(0, nseq, args.batch):
            end = min(start + args.batch, nseq)
            batch = {name: value[start:end] for name, value in seqs.items()}
            rolled = rollout_batch(heads, batch, seed=args.seed + rep * 1_000_003 + start,
                                   engine=engine, c_engine=c_engine)
            event_counts += rolled["event"].sum(axis=(0, 1))
            terminal_mask = rolled["pa_terminal"]
            # A generated terminal may occur on a different pitch index from
            # the recorded PA.  Attribute it to the generated PA schedule, not
            # to the recorded batter occupying that pitch position.
            batters = rolled["batter_idx"]
            for batter, outcome in zip(batters[terminal_mask], rolled["pa_outcome"][terminal_mask]):
                sampled.append((int(batter), int(outcome)))
            if start % (args.batch * 50) == 0:
                print(f"rep {rep + 1}/{args.reps}: {end}/{nseq}", flush=True)
        sim_counts = _outcome_counts(sampled)
        result = _correlations(_rates_from_counts(sim_counts, args.min_pa), real)
        result["sim_coverage"] = _coverage(sim_counts, args.min_pa)
        result["events"] = event_counts.tolist()
        rep_results.append(result)

    keys = [key for key in rep_results[0] if key.endswith("_corr")]
    result: dict[str, object] = {
        "tag": args.tag, "season": args.season, "reps": args.reps, "c_events": args.c_events,
        "min_pa": args.min_pa, "real_coverage": _coverage(real_counts, args.min_pa),
        "limit_seqs": args.limit_seqs, "comparable_full_slate": args.limit_seqs is None,
        "per_rep": rep_results,
        "skill_policy": worlds.report(args.reps),
        "mean": {key: _finite_mean([r[key] for r in rep_results]) for key in keys},
        "sd": {key: _finite_sd([r[key] for r in rep_results]) if args.reps > 1 else 0.0 for key in keys},
        "note": ("Generated pitch/state history; pitcher/batter/park schedule remains exogenous. "
                 "C events are sampled and counted, while the empirical engine retains base advancement."),
    }
    out = Path(args.out or f"data/eval2/pitchformer_rollout_{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    #print(result)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
