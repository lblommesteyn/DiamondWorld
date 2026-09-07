"""Observed-schedule full-game evaluation for the autoregressive A--D rollout.

Lineup order, pitcher changes, and parks are read from the held-out game.  Every
pitch outcome and all game state are generated; this is explicitly *not* a
pre-game roster/staff simulator.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl

from diamondworldjax.data.pitch_seq import build_id_maps, load_seasons, make_sequences
from diamondworldjax.model.pitchformer import TransformerA, TransformerB
from diamondworldjax.model.transformer_c import TransformerC
from diamondworldjax.model.transformer_d import TransformerD
from diamondworldjax.sim.c_transition_engine import CTransitionEngine
from diamondworldjax.sim.rules_engine import EmpiricalEngine
from diamondworldjax.simulate.pitchformer_rollout import PitchformerHeads, rollout_batch


def _load(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _states(n_games: int) -> dict[str, np.ndarray]:
    """One independently evolving state row per observed-schedule game."""
    z = np.zeros(n_games, np.int32)
    return {"balls": z.copy(), "strikes": z.copy(), "outs": z.copy(), "base": z.copy(),
            "home_score": z.copy(), "away_score": z.copy(), "inning": np.ones(n_games, np.int32),
            "half": z.copy(), "tto": np.ones(n_games, np.int32), "pitch_count": z.copy(),
            "pa_slot": z.copy(), "ended": np.zeros(n_games, bool)}


def _start_half(state: dict[str, np.ndarray], rows: np.ndarray, inning: int, half: int) -> None:
    """Reset half-inning state but retain the game score and long-lived fields."""
    state["balls"][rows] = 0
    state["strikes"][rows] = 0
    state["outs"][rows] = 0
    state["base"][rows] = 0
    state["inning"][rows] = inning
    state["half"][rows] = half
    state["pa_slot"][rows] = 0
    state["ended"][rows] = False


def _stack_chunk(items, chunk: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Stack one same-phase chunk from several games into a rollout batch."""
    rows = np.asarray([row for row, _ in items], np.int32)
    names = items[0][1].keys()
    batch = {
        name: np.concatenate([seqs[name][chunk:chunk + 1] for _, seqs in items], axis=0)
        for name in names
    }
    return rows, batch


def _decode_bucket(batch: dict[str, np.ndarray], max_len: int) -> int:
    """Bucket the padded 1.5x decode horizon, not just observed pitches."""
    used = int(np.flatnonzero(batch["valid"].any(axis=0))[-1] + 1)
    target = int(np.ceil(used * 1.5))
    return ((target + 31) // 32) * 32


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--params-dir", default="checkpoints/pitchformer")
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--limit-games", type=int, default=None)
    ap.add_argument("--batch-games", type=int, default=32,
                    help="Active same-inning half-innings per A--D rollout batch.")
    ap.add_argument("--events", default="data/processed/events.parquet")
    ap.add_argument("--c-events", action="store_true")
    ap.add_argument("--game-context", default="data/processed/game_context.parquet")
    ap.add_argument("--no-env", action="store_true", help="Must match training.")
    ap.add_argument("--no-geom", action="store_true", help="Must match training.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.batch_games < 1:
        raise SystemExit("--batch-games must be positive")

    years = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]
    train = load_seasons(years)
    maps = build_id_maps(train)
    test = load_seasons([args.season])[0]
    context_path = Path(args.game_context)
    gctx = None if args.no_env or not context_path.exists() else pl.read_parquet(context_path)
    kw = dict(n_pitchers=maps["n_pitcher"], n_batters=maps["n_batter"], n_parks=maps["n_park"],
              d_model=args.d_model, n_layers=args.layers, n_heads=args.heads)
    root = Path(args.params_dir)
    def load_head(letter, cls):
        path = root / f"{letter}_{args.tag}_params.pkl"
        return (cls(**kw), _load(path)) if path.exists() else (None, None)
    a, ap_ = load_head("A", TransformerA)
    b, bp = load_head("B", TransformerB)
    c, cp = load_head("C", TransformerC)
    d, dp = load_head("D", TransformerD)
    if a is None or b is None:
        raise SystemExit("A and B checkpoints are required")
    heads = PitchformerHeads(a, b, c, d, ap_, bp, cp, dp)
    engine = EmpiricalEngine().fit(pl.concat(train).filter(pl.col("pa_terminal")))
    c_engine = None
    if args.c_events:
        if c is None or not Path(args.events).exists():
            raise SystemExit("--c-events requires C checkpoint and extracted events")
        c_engine = CTransitionEngine().fit(pl.concat(train), pl.read_parquet(args.events))

    games = test["game_pk"].unique().sort().to_list()
    if args.limit_games:
        games = games[:args.limit_games]
    # Limit the raw table once.  Each inning/side is then partitioned by game
    # and the compatible half-innings are stacked into GPU batches below.
    test = test.filter(pl.col("game_pk").is_in(games))
    game_row = {int(game_pk): i for i, game_pk in enumerate(games)}
    state = _states(len(games))
    done = np.zeros(len(games), bool)
    truncations, event_counts, rollout_calls = 0, np.zeros(8, np.int64), 0
    scheduled_pas, generated_pas = 0, 0
    outcome_counts = np.zeros(9, np.int64)
    innings = sorted(test["inning"].unique().to_list())

    for inning in innings:
        for half_name, half in (("top", 0), ("bot", 1)):
            phase = test.filter((pl.col("inning") == inning) & (pl.col("half") == half_name))
            if not phase.height:
                continue
            # A home team already leading after the top of the ninth does not
            # bat.  Mark it complete before materialising its observed rows.
            if half == 1 and inning >= 9:
                won_before_batting = ~done & (state["home_score"] > state["away_score"])
                done |= won_before_batting
            candidates = []
            for portion in phase.partition_by("game_pk", maintain_order=True):
                gi = game_row[int(portion["game_pk"][0])]
                if done[gi]:
                    continue
                seqs = make_sequences(portion, maps, args.max_len, game_ctx=gctx)
                if args.no_geom:
                    seqs["geom"][:] = 0.0
                candidates.append((gi, seqs))
            if not candidates:
                continue
            rows_this_half = np.asarray([row for row, _ in candidates], np.int32)
            _start_half(state, rows_this_half, int(inning), half)
            max_chunks = max(len(seqs["valid"]) for _, seqs in candidates)
            for chunk in range(max_chunks):
                active_items = [(gi, seqs) for gi, seqs in candidates
                                if (chunk < len(seqs["valid"])
                                and not state["ended"][gi]
                                # A generated third out advances the rollout
                                # state to the next half.  The outer evaluator
                                # supplies that next observed schedule later;
                                # do not keep decoding stale chunks here.
                                and state["inning"][gi] == inning
                                and state["half"][gi] == half)]
                for start in range(0, len(active_items), args.batch_games):
                    rows_batch, batch = _stack_chunk(active_items[start:start + args.batch_games], chunk)
                    initial = {name: value[rows_batch].copy() for name, value in state.items()}
                    # The scalar seed makes this batch deterministic.  Batching
                    # changes individual draws versus the old B=1 evaluator,
                    # but not the model or transition distributions.
                    batch_seed = args.seed + int(inning) * 10_000 + half * 1_000 + chunk * 100 + start
                    rolled = rollout_batch(heads, batch, seed=batch_seed, engine=engine,
                                           c_engine=c_engine, initial_state=initial,
                                           stop_when_decided=True,
                                           decode_len=_decode_bucket(batch, args.max_len))
                    event_counts += rolled["event"].sum(axis=(0, 1))
                    scheduled_pas += int(batch["pa_start"].sum())
                    terminal = rolled["pa_terminal"]
                    generated_pas += int(terminal.sum())
                    outcome_counts += np.bincount(
                        rolled["pa_outcome"][terminal], minlength=len(outcome_counts)
                    )[:len(outcome_counts)]
                    for name, value in rolled["final_state"].items():
                        state[name][rows_batch] = value
                    rollout_calls += 1
            # A row remaining in this inning/side after its observed chunks
            # were exhausted genuinely needed more than the padded horizon.
            still_this_half = (~state["ended"][rows_this_half]
                               & (state["inning"][rows_this_half] == inning)
                               & (state["half"][rows_this_half] == half))
            truncations += int(still_this_half.sum())
            if half == 1 and inning >= 9:
                done |= state["home_score"] > state["away_score"]
            print(f"{half_name} {inning}: {len(rows_this_half)} half-innings", flush=True)

    rows = [(int(game_pk), int(state["home_score"][gi]), int(state["away_score"][gi]))
            for gi, game_pk in enumerate(games)]

    arr = np.asarray(rows, np.int64)
    result = {
        "tag": args.tag, "season": args.season, "c_events": args.c_events,
        "games": len(rows), "truncated_half_innings": truncations,
        "batch_games": args.batch_games, "rollout_calls": rollout_calls,
        "scheduled_pas": scheduled_pas, "generated_pas": generated_pas,
        "pa_outcomes": {name: int(outcome_counts[i]) for i, name in enumerate(
            ("K", "BB", "HBP", "1B", "2B", "3B", "HR", "out", "E"))},
        "home_mean": float(arr[:, 1].mean()), "away_mean": float(arr[:, 2].mean()),
        "total_mean": float((arr[:, 1] + arr[:, 2]).mean()),
        "home_win_rate": float((arr[:, 1] > arr[:, 2]).mean()),
        "events": event_counts.tolist(),
        "note": "Generated game state on observed lineup/staff schedule; not pre-game roster selection.",
    }
    out = Path(args.out or f"data/eval2/pitchformer_games_{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**result, "scores": rows}, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
