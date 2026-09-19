"""Observed-schedule full-game evaluation for the autoregressive A--D rollout.

Lineup order, pitcher changes, and parks are read from the held-out game.  Every
pitch outcome and all game state are generated; this is explicitly *not* a
pre-game roster/staff simulator.

v2: outputs the same metrics as the PA model's simulate_games.py —
  runs/game (real vs sim), score distribution shape, marginal outcome calibration,
  occupancy, runs-by-inning fraction, extras/walk-offs, per-player rate-stat
  reproduction (AVG, OBP, SLG, K%, BB%, HR% with MAE and cross-player correlation),
  multi-rep confidence intervals, PIT calibration (composed from B+D heads),
  and coverage (empirical prediction intervals from multi-rep).
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
from diamondworldjax.model.superstate import load_geometry
from diamondworldjax.model.pitchformer import TransformerA, TransformerB
from diamondworldjax.model.transformer_c import TransformerC
from diamondworldjax.model.transformer_d import TransformerD
from diamondworldjax.sim.c_transition_engine import CTransitionEngine
from diamondworldjax.sim.rules_engine import EmpiricalEngine
from diamondworldjax.simulate.pitchformer_rollout import PitchformerHeads, rollout_batch

# PA outcome index order: K=0, BB=1, HBP=2, 1B=3, 2B=4, 3B=5, HR=6, out=7, E=8
_K, _BB, _HBP, _1B, _2B, _3B, _HR, _OUT, _E = range(9)
_OUTCOME_NAMES = ("K", "BB", "HBP", "1B", "2B", "3B", "HR", "out", "E")
_OUTCOME_MAP = {name: i for i, name in enumerate(_OUTCOME_NAMES)}

# D-head outcome order: out=0, 1B=1, 2B=2, 3B=3, HR=4, E=5
# Maps D-head index -> PA outcome index
_D_TO_PA_IDX = np.array([_OUT, _1B, _2B, _3B, _HR, _E], dtype=np.int32)


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


def _entered_extra_mask(state: dict[str, np.ndarray], done: np.ndarray) -> np.ndarray:
    """Games tied at the regulation boundary and therefore entering extras."""
    return (~np.asarray(done, bool)
            & (np.asarray(state["home_score"]) == np.asarray(state["away_score"])))


def _start_half(state: dict[str, np.ndarray], rows: np.ndarray, inning: int, half: int,
                base_override: int | None = None) -> None:
    """Reset half-inning state but retain the game score and long-lived fields."""
    state["balls"][rows] = 0
    state["strikes"][rows] = 0
    state["outs"][rows] = 0
    state["base"][rows] = 0 if base_override is None else base_override
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


def _prepare_observed_schedule(
    test: pl.DataFrame, game_row: dict[int, int], maps: dict, args, gctx,
) -> dict[tuple[int, int], list[tuple[int, dict[str, np.ndarray]]]]:
    """Build immutable held-out half-inning inputs once for every rollout rep.

    ``make_sequences`` joins context, encodes every pitch, and allocates padded
    arrays. Those values are exogenous to a generated game, so rebuilding them
    for every world draw was pure CPU/Polars overhead. The returned arrays are
    read-only by this evaluator; per-rep state and model caches remain separate.
    """
    schedules: dict[tuple[int, int], list[tuple[int, dict[str, np.ndarray]]]] = {}
    geometry_table = load_geometry()
    for inning in sorted(test["inning"].unique().to_list()):
        for half_name, half in (("top", 0), ("bot", 1)):
            phase = test.filter((pl.col("inning") == inning) & (pl.col("half") == half_name))
            if not len(phase):
                continue
            # ``make_sequences`` already respects game boundaries. Building all
            # games in a phase together replaces thousands of tiny Polars/JAX
            # conversions (one per game-half) with one vectorised build, then
            # slices the immutable result for the rollout scheduler.
            seqs = add_skill_season(
                make_sequences(phase, maps, args.max_len, game_ctx=gctx,
                               geometry_table=geometry_table,
                               include_game_pk=True),
                args.checkpoint_metadata,
            )
            if args.no_geom:
                seqs["geom"][:] = 0.0
            sequence_game = np.asarray(seqs["game_pk"])[:, 0]
            items = []
            for game_pk in phase["game_pk"].unique(maintain_order=True).to_list():
                rows = np.flatnonzero(sequence_game == game_pk)
                if len(rows):
                    items.append((game_row[int(game_pk)],
                                  # Provenance is only needed to split this
                                  # vectorised build; do not carry it into the
                                  # rollout/JIT model batch.
                                  {name: np.asarray(value)[rows]
                                   for name, value in seqs.items()
                                   if name != "game_pk"}))
            if items:
                schedules[(int(inning), half)] = items
    return schedules


MAX_EXTRA_INNINGS = 20
_PITCHES_PER_PA = 6
_PAS_PER_EXTRA_HALF = 12


def _build_extra_schedule(
    lineup: list[int],
    lineup_ptr: int,
    pitcher_idx: int,
    park_idx: int,
    inning: int,
    half: int,
    score_diff: float,
    base_state: int,
    env_ctx: np.ndarray | None,
    geom: np.ndarray | None,
    season: int,
    max_len: int,
    stand_throws: np.ndarray | None,
    metadata: dict | None,
) -> tuple[dict[str, np.ndarray], int]:
    """Build a synthetic pitch schedule for one extra half-inning.

    Returns (seqs_dict, new_lineup_ptr).  The dict has shape (1, T, ...) matching
    make_sequences output so it can be fed to rollout_batch via _stack_chunk.
    """
    from diamondworldjax.data.pitch_seq import D_CTX_TOTAL, N_GEOMETRY
    from diamondworldjax.model.pitchformer_checkpoint import add_skill_season

    n_pas = _PAS_PER_EXTRA_HALF
    T = min(n_pas * _PITCHES_PER_PA, max_len)

    out = {
        "pa_outcome": np.full((1, T), -1, np.int32),
        "pa_terminal": np.zeros((1, T), bool),
        "season": np.full((1, T), season, np.int32),
        "pitcher_idx": np.full((1, T), pitcher_idx, np.int32),
        "batter_idx": np.zeros((1, T), np.int32),
        "park_idx": np.full((1, T), park_idx, np.int32),
        "ctx": np.zeros((1, T, D_CTX_TOTAL), np.float32),
        "geom": np.zeros((1, T, N_GEOMETRY if geom is None else geom.shape[-1]), np.float32),
        "pitch_type": np.zeros((1, T), np.int32),
        "type_valid": np.zeros((1, T), np.float32),
        "stuff": np.zeros((1, T, 5), np.float32),
        "stuff_valid": np.zeros((1, T), np.float32),
        "swing": np.zeros((1, T), np.float32),
        "contact": np.zeros((1, T), np.float32),
        "foul": np.zeros((1, T), np.float32),
        "hbp": np.zeros((1, T), np.float32),
        "valid": np.ones((1, T), np.float32),
        "pa_start": np.zeros((1, T), bool),
        "launch": np.zeros((1, T, 2), np.float32),
        "launch_valid": np.zeros((1, T), np.float32),
        "batted_out": np.zeros((1, T), np.int32),
        "batted_valid": np.zeros((1, T), np.float32),
    }

    if geom is not None:
        out["geom"][0, :] = geom

    on1 = float((base_state & 1) > 0)
    on2 = float((base_state & 2) > 0)
    on3 = float((base_state & 4) > 0)
    half_f = float(half)
    norm_inning = (inning - 5.0) / 4.0
    leverage = float(inning >= 7 and abs(score_diff) <= 1)
    norm_score = np.clip(score_diff, -10, 10) / 10.0

    ptr = lineup_ptr
    for pa in range(n_pas):
        slot = pa * _PITCHES_PER_PA
        if slot >= T:
            break
        batter = lineup[ptr % len(lineup)]
        out["pa_start"][0, slot] = True
        end = min(slot + _PITCHES_PER_PA, T)
        out["batter_idx"][0, slot:end] = batter

        # stand/throws: use per-batter info if available, else default 1.0 (R)
        stand = 1.0
        throws = 1.0
        if stand_throws is not None and batter < len(stand_throws):
            stand = float(stand_throws[batter, 0])
            throws = float(stand_throws[batter, 1])

        for t in range(slot, end):
            ctx = out["ctx"][0, t]
            ctx[0] = 0.0           # balls/3
            ctx[1] = 0.0           # strikes/2
            ctx[2] = 0.0           # outs/2
            ctx[3] = on1           # on1
            ctx[4] = on2           # on2
            ctx[5] = on3           # on3
            ctx[6] = max(on2, on3) # scoring position
            ctx[7] = norm_score
            ctx[8] = norm_inning
            ctx[9] = half_f
            ctx[10] = 1.0 - half_f
            ctx[11] = leverage
            ctx[12] = 0.0          # tto/4 (reset for extras)
            ctx[13] = stand
            ctx[14] = throws
            ctx[15] = float(stand == throws)

        if env_ctx is not None:
            out["ctx"][0, slot:end, 16:16 + len(env_ctx)] = env_ctx

        ptr += 1

    out = add_skill_season(out, metadata)
    return out, ptr


def _extract_lineup_info(test: "pl.DataFrame", game_row: dict, maps: dict, G: int):
    """Extract batting orders and last pitcher per game/half from observed data."""
    batting_orders = [[[], []] for _ in range(G)]
    last_pitcher = [[0, 0] for _ in range(G)]
    lineup_ptrs = [[0, 0] for _ in range(G)]
    park_indices = [0] * G
    env_contexts = [None] * G
    geom_per_game = [None] * G

    bcol = "batter_id" if "batter_id" in test.columns else "batter_idx"
    pcol = "pitcher_id" if "pitcher_id" in test.columns else "pitcher_idx"

    for portion in test.partition_by("game_pk", maintain_order=True):
        gpk = int(portion["game_pk"][0])
        if gpk not in game_row:
            continue
        gi = game_row[gpk]
        if "park_id" in portion.columns:
            pid = portion["park_id"].to_list()[0]
            park_indices[gi] = maps.get("park", {}).get(pid, 0)

        for half, half_name in enumerate(("top", "bot")):
            h = portion.filter(pl.col("half") == half_name)
            if h.height == 0:
                continue
            seen, order = set(), []
            for b in h[bcol].to_list():
                b = int(b)
                idx = maps.get("batter", {}).get(b, 0)
                if idx not in seen:
                    seen.add(idx)
                    order.append(idx)
            batting_orders[gi][half] = order if order else [0]

            pitchers = h[pcol].to_list()
            last_p = maps.get("pitcher", {}).get(int(pitchers[-1]), 0) if pitchers else 0
            last_pitcher[gi][half] = last_p

            n_pas = len(set(h["at_bat_number"].to_list())) if "at_bat_number" in h.columns else 0
            lineup_ptrs[gi][half] = n_pas

    return batting_orders, last_pitcher, lineup_ptrs, park_indices, env_contexts, geom_per_game


def _rate_stats(c: np.ndarray) -> dict[str, float] | None:
    """length-9 outcome counts -> rate stats. Order K,BB,HBP,1B,2B,3B,HR,out,E."""
    pa = c.sum()
    if pa == 0:
        return None
    h = c[_1B] + c[_2B] + c[_3B] + c[_HR]
    ab = max(pa - c[_BB] - c[_HBP], 1)
    tb = c[_1B] + 2 * c[_2B] + 3 * c[_3B] + 4 * c[_HR]
    return {"AVG": h / ab, "OBP": (h + c[_BB] + c[_HBP]) / pa, "SLG": tb / ab,
            "K%": c[_K] / pa, "BB%": c[_BB] / pa, "HR%": c[_HR] / pa}


# ---------------------------------------------------------------------------
# PIT calibration: compose implied PA outcome distribution from B+D heads
# ---------------------------------------------------------------------------

def _compose_pa_probs(
    swing_prob: np.ndarray,
    contact_prob: np.ndarray,
    foul_prob: np.ndarray,
    hbp_prob: np.ndarray,
    called_strike_prob: np.ndarray,
    d_outcome_probs: np.ndarray,
    balls: np.ndarray,
    strikes: np.ndarray,
) -> np.ndarray:
    """Compose per-pitch B+D probabilities into a 9-way PA outcome distribution.

    Only meaningful at terminal pitches (where the PA actually ends). At non-
    terminal pitches the distribution includes mass on "PA continues" events
    that cannot produce a PA outcome.

    Parameters
    ----------
    swing_prob, contact_prob, foul_prob, hbp_prob, called_strike_prob : (N,) float
        B-head sigmoid probabilities for each terminal pitch.
    d_outcome_probs : (N, 6) float
        D-head softmax probabilities [out, 1B, 2B, 3B, HR, E].
    balls, strikes : (N,) int
        Count state at the time of the pitch (before the pitch is resolved).

    Returns
    -------
    pa_probs : (N, 9) float
        Implied probabilities over [K, BB, HBP, 1B, 2B, 3B, HR, out, E].
    """
    from diamondworldjax.eval.pitch_calibration import pitch_resolution_probs
    probabilities = pitch_resolution_probs(swing_prob, contact_prob, foul_prob,
        hbp_prob, d_outcome_probs, called_strike_prob, balls, strikes)[..., :9]
    return probabilities / np.maximum(probabilities.sum(-1, keepdims=True), 1e-10)


def _pit_values(pa_probs: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Compute PIT (probability integral transform) values.

    For discrete outcomes, PIT uses the randomised version:
        U ~ Uniform(F(y-), F(y))
    where F(y-) = P(Y < y) and F(y) = P(Y <= y).

    Under a well-calibrated model, U ~ Uniform(0, 1).
    """
    N = len(observed)
    rng = np.random.default_rng(42)
    pit = np.zeros(N, dtype=np.float64)
    for i in range(N):
        oc = observed[i]
        f_lower = pa_probs[i, :oc].sum()  # P(Y < y)
        f_upper = f_lower + pa_probs[i, oc]  # P(Y <= y)
        pit[i] = rng.uniform(f_lower, f_upper)
    return pit


def _pit_summary(pit_values: np.ndarray, n_bins: int = 10) -> dict:
    """Summarise PIT values: histogram + KS test for uniformity."""
    from scipy.stats import kstest
    hist, edges = np.histogram(pit_values, bins=n_bins, range=(0, 1))
    expected = len(pit_values) / n_bins
    chi2 = float(((hist - expected) ** 2 / expected).sum())
    ks_stat, ks_p = kstest(pit_values, "uniform")
    return {
        "n": len(pit_values),
        "histogram": hist.tolist(),
        "bin_edges": edges.tolist(),
        "chi2": chi2,
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_p),
        "mean": float(pit_values.mean()),
        "std": float(pit_values.std()),
    }


# ---------------------------------------------------------------------------
# Single-rep game simulation
# ---------------------------------------------------------------------------

def _run_one_rep(
    heads, engine, c_engine, test, games, game_row, maps, gctx, args,
    seed: int, n_batters: int,
    lineup_info: tuple | None = None,
    phase_schedules: dict[tuple[int, int], list[tuple[int, dict[str, np.ndarray]]]] | None = None,
    terminal_outcome_sampler=None,
    rate_batter_index: np.ndarray | None = None,
    n_rate_players: int = 0,
    rate_max_pa_per_game: int = 0,
) -> dict:
    """Run one complete game evaluation pass. Returns per-rep metrics."""
    if phase_schedules is None:
        # Keep this helper usable by focused tests and external callers. Normal
        # evaluation supplies the shared cache from main(), so this runs once.
        phase_schedules = _prepare_observed_schedule(test, game_row, maps, args, gctx)
    G = len(games)
    state = _states(G)
    done = np.zeros(G, bool)
    from diamondworldjax.simulate.pitchformer_rollout import GameHistory
    histories = GameHistory(getattr(args, "history_reset", "legacy"))
    completion_faults = {name: np.zeros(G, bool) for name in
                         ("regulation_truncated", "extra_truncated", "unresolved_tie", "skipped_extra_innings")}
    truncations = 0
    event_counts = np.zeros(8, np.int64)
    rollout_calls = 0
    scheduled_pas, generated_pas = 0, 0
    outcome_counts = np.zeros(9, np.int64)
    pcounts = np.zeros((n_batters, 9), dtype=np.float64)
    rate_pcounts = (np.zeros((n_rate_players, 9), dtype=np.float64)
                    if rate_batter_index is not None else None)
    # Rate exports may reproduce a historical cohort without truncating game
    # state evolution or the score-based benchmark arrays.
    rate_pa_seen = np.zeros(G, dtype=np.int32) if rate_pcounts is not None else None
    runs_by_inning = np.zeros(9, dtype=np.float64)
    extra_runs = 0.0
    n_walkoffs = 0
    score_before = np.zeros(G, dtype=np.float64)

    def hybrid_game_batch(batch: dict[str, np.ndarray], rows: np.ndarray) -> dict[str, np.ndarray]:
        """Attach stable game keys for a sequential PA carry, only in hybrid mode."""
        if terminal_outcome_sampler is None:
            return batch
        out = dict(batch)
        out["hybrid_game_index"] = np.broadcast_to(
            np.asarray(rows, np.int32)[:, None], batch["valid"].shape,
        ).copy()
        return out

    def add_rate_outcomes(rolled, terminal, rows_batch) -> None:
        """Accumulate the selected observed-schedule terminal PAs by batter."""
        if rate_pcounts is None:
            return
        # Score rollouts may need continuation PAs after exhausting the
        # observed half's source schedule.  They are necessary to complete the
        # simulated inning, but do not belong in the fixed observed-PA cohort
        # used by the paired v16 player-rate comparison.
        scheduled = rolled.get("scheduled_pa", np.ones_like(terminal, bool))
        # Count unknown batters too: the cap applies to the game schedule, not
        # just players represented in the shared rate table. Batch rows are
        # distinct games, so only this small outer loop is needed; outcome
        # accumulation remains vectorized over terminal pitches.
        for batch_i, game_i in enumerate(rows_batch):
            times = np.flatnonzero(terminal[batch_i] & scheduled[batch_i])
            if rate_max_pa_per_game:
                remaining = rate_max_pa_per_game - rate_pa_seen[game_i]
                times = times[:max(remaining, 0)]
            rate_pa_seen[game_i] += len(times)
            if not len(times):
                continue
            batters = rolled["batter_idx"][batch_i, times]
            outcomes = rolled["pa_outcome"][batch_i, times]
            valid = (batters >= 0) & (batters < len(rate_batter_index))
            rate_idx = np.full(len(batters), -1, dtype=np.int32)
            rate_idx[valid] = rate_batter_index[batters[valid]]
            known = rate_idx >= 0
            np.add.at(rate_pcounts, (rate_idx[known], outcomes[known]), 1.0)

    # PIT calibration accumulators
    pit_swing_prob = []
    pit_contact_prob = []
    pit_foul_prob = []
    pit_hbp_prob = []
    pit_called_strike_prob = []
    pit_d_probs = []
    pit_zone = []
    pit_balls = []
    pit_strikes = []
    pit_outcome = []

    innings = sorted({inning for inning, _ in phase_schedules})

    for inning in innings:
        for half_name, half in (("top", 0), ("bot", 1)):
            scheduled_items = phase_schedules.get((inning, half), ())
            if not scheduled_items:
                continue
            if half == 1 and inning >= 9:
                won_before_batting = ~done & (state["home_score"] > state["away_score"])
                done |= won_before_batting
            candidates = [(gi, seqs) for gi, seqs in scheduled_items if not done[gi]]
            if not candidates:
                continue
            rows_this_half = np.asarray([row for row, _ in candidates], np.int32)
            _start_half(state, rows_this_half, int(inning), half)

            # Snapshot scores before this half for runs-by-inning
            score_before[rows_this_half] = (state["home_score"][rows_this_half].astype(np.float64)
                                            + state["away_score"][rows_this_half].astype(np.float64))

            max_chunks = max(len(seqs["valid"]) for _, seqs in candidates)
            for chunk in range(max_chunks):
                active_items = [(gi, seqs) for gi, seqs in candidates
                                if (chunk < len(seqs["valid"])
                                and not state["ended"][gi]
                                and state["inning"][gi] == inning
                                and state["half"][gi] == half)]
                for start in range(0, len(active_items), args.batch_games):
                    rows_batch, batch = _stack_chunk(active_items[start:start + args.batch_games], chunk)
                    batch = hybrid_game_batch(batch, rows_batch)
                    initial = {name: value[rows_batch].copy() for name, value in state.items()}
                    batch_seed = seed + int(inning) * 10_000 + half * 1_000 + chunk * 100 + start
                    history_inning = int(initial["inning"][0])
                    prior_cache = histories.get(heads, batch, rows_batch, history_inning, half)
                    rolled = rollout_batch(heads, batch, seed=batch_seed, engine=engine,
                                           c_engine=c_engine, initial_state=initial, initial_cache=prior_cache,
                                           stop_when_decided=True,
                                           decode_len=_decode_bucket(batch, args.max_len),
                                           terminal_outcome_sampler=terminal_outcome_sampler)
                    histories.put(rows_batch, history_inning, half, rolled.get("final_cache"))
                    event_counts += rolled["event"].sum(axis=(0, 1))
                    scheduled_pas += int(batch["pa_start"].sum())
                    terminal = rolled["pa_terminal"]
                    generated_pas += int(terminal.sum())
                    outcome_counts += np.bincount(
                        rolled["pa_outcome"][terminal], minlength=len(outcome_counts)
                    )[:len(outcome_counts)]

                    # Per-player outcome accumulation
                    batter_idx = rolled["batter_idx"]
                    flat_bidx = batter_idx[terminal]
                    flat_oc = rolled["pa_outcome"][terminal]
                    known = (flat_bidx >= 0) & (flat_bidx < n_batters)
                    np.add.at(pcounts, (flat_bidx[known], flat_oc[known]), 1.0)
                    add_rate_outcomes(rolled, terminal, rows_batch)

                    # PIT: collect B/D probabilities at terminal pitches
                    if terminal_outcome_sampler is None and "swing_prob" in rolled:
                        pit_swing_prob.append(rolled["swing_prob"][terminal])
                        pit_contact_prob.append(rolled["contact_prob"][terminal])
                        pit_foul_prob.append(rolled["foul_prob"][terminal])
                        pit_hbp_prob.append(rolled["hbp_prob"][terminal])
                        pit_called_strike_prob.append(rolled["called_strike_prob"][terminal])
                        pit_d_probs.append(rolled["d_outcome_probs"][terminal])
                        pit_zone.append(rolled["zone"][terminal])
                        pit_balls.append(rolled["balls"][terminal])
                        pit_strikes.append(rolled["strikes"][terminal])
                        pit_outcome.append(rolled["pa_outcome"][terminal])

                    for name, value in rolled["final_state"].items():
                        state[name][rows_batch] = value
                    rollout_calls += 1

            # Truncation tracking
            still_this_half = (~state["ended"][rows_this_half]
                               & (state["inning"][rows_this_half] == inning)
                               & (state["half"][rows_this_half] == half))
            # The observed pitch segment is only a source of matchup/context
            # rows.  A stochastic inning can take longer than that segment,
            # so keep decoding its cyclic schedule until it reaches three outs
            # (or the explicitly bounded safety budget is exhausted).  Without
            # this, every long half was silently scored as an incomplete game.
            if still_this_half.any():
                by_game = {gi: seqs for gi, seqs in candidates}
                unfinished_items = [(int(gi), by_game[int(gi)])
                                    for gi in rows_this_half[still_this_half]]
                for group_start in range(0, len(unfinished_items), args.batch_games):
                    pending_rows, batch = _stack_chunk(
                        unfinished_items[group_start:group_start + args.batch_games], 0)
                    batch = hybrid_game_batch(batch, pending_rows)
                    for continuation in range(args.max_half_continuations):
                        initial = {name: value[pending_rows].copy() for name, value in state.items()}
                        history_inning = int(initial["inning"][0])
                        cache = histories.get(heads, batch, pending_rows, history_inning, half)
                        rolled = rollout_batch(
                            heads, batch,
                            seed=(seed + int(inning) * 10_000 + half * 1_000
                                  + 700_000 + group_start * 100 + continuation),
                            engine=engine, c_engine=c_engine, initial_state=initial,
                            initial_cache=cache, stop_when_decided=True,
                            decode_len=_decode_bucket(batch, args.max_len),
                            terminal_outcome_sampler=terminal_outcome_sampler,
                        )
                        histories.put(pending_rows, history_inning, half, rolled.get("final_cache"))
                        event_counts += rolled["event"].sum(axis=(0, 1))
                        terminal = rolled["pa_terminal"]
                        generated_pas += int(terminal.sum())
                        outcome_counts += np.bincount(
                            rolled["pa_outcome"][terminal], minlength=len(outcome_counts)
                        )[:len(outcome_counts)]
                        batter_idx = rolled["batter_idx"]
                        flat_bidx = batter_idx[terminal]
                        flat_oc = rolled["pa_outcome"][terminal]
                        known = (flat_bidx >= 0) & (flat_bidx < n_batters)
                        np.add.at(pcounts, (flat_bidx[known], flat_oc[known]), 1.0)
                        add_rate_outcomes(rolled, terminal, pending_rows)
                        if terminal_outcome_sampler is None and "swing_prob" in rolled:
                            pit_swing_prob.append(rolled["swing_prob"][terminal])
                            pit_contact_prob.append(rolled["contact_prob"][terminal])
                            pit_foul_prob.append(rolled["foul_prob"][terminal])
                            pit_hbp_prob.append(rolled["hbp_prob"][terminal])
                            pit_called_strike_prob.append(rolled["called_strike_prob"][terminal])
                            pit_d_probs.append(rolled["d_outcome_probs"][terminal])
                            pit_zone.append(rolled["zone"][terminal])
                            pit_balls.append(rolled["balls"][terminal])
                            pit_strikes.append(rolled["strikes"][terminal])
                            pit_outcome.append(rolled["pa_outcome"][terminal])
                        for name, value in rolled["final_state"].items():
                            state[name][pending_rows] = value
                        rollout_calls += 1

                        still = (~state["ended"][pending_rows]
                                 & (state["inning"][pending_rows] == inning)
                                 & (state["half"][pending_rows] == half))
                        if not still.any():
                            break
                        pending_rows = pending_rows[still]
                        batch = {name: value[still] for name, value in batch.items()}
            still_this_half = (~state["ended"][rows_this_half]
                               & (state["inning"][rows_this_half] == inning)
                               & (state["half"][rows_this_half] == half))
            truncations += int(still_this_half.sum())
            completion_faults["regulation_truncated"][rows_this_half] |= still_this_half

            # Runs from continuation pitches count in the same half-inning.
            score_after = (state["home_score"][rows_this_half].astype(np.float64)
                           + state["away_score"][rows_this_half].astype(np.float64))
            half_runs = (score_after - score_before[rows_this_half]).sum()
            if inning <= 9:
                runs_by_inning[inning - 1] += half_runs
            else:
                extra_runs += half_runs

            # Walk-off tracking
            if half == 1 and inning >= 9:
                walkoff_mask = (~done[rows_this_half]
                                & (state["home_score"][rows_this_half] > state["away_score"][rows_this_half]))
                n_walkoffs += int(walkoff_mask.sum())
                done |= state["home_score"] > state["away_score"]

    # -----------------------------------------------------------------
    # Extra innings for tied games
    # -----------------------------------------------------------------
    n_extra_half_innings = 0
    n_unresolved = 0
    # This is the extra-inning metric: a game entered extras only if it was
    # tied after regulation.  Do not infer it from final ``state['inning']``:
    # completing a normal bottom ninth advances that state to inning 10 too.
    entered_extras = np.zeros(G, dtype=bool)
    if lineup_info is not None:
        batting_orders, last_pitcher, lineup_ptrs, park_indices, env_contexts, geom_per_game = lineup_info
        tied = _entered_extra_mask(state, done)
        entered_extras = tied.copy()
        last_observed = max(innings) if innings else 9
        for gpk, last in test.group_by('game_pk').agg(pl.col('inning').max()).iter_rows():
            gi = game_row[int(gpk)]
            completion_faults['skipped_extra_innings'][gi] = bool(tied[gi] and int(last) < last_observed)
        for extra_inning in range(last_observed + 1, MAX_EXTRA_INNINGS + 1):
            if not tied.any():
                break
            for half_name, half in (("top", 0), ("bot", 1)):
                active_gi = np.where(tied & ~done)[0]
                if len(active_gi) == 0:
                    continue
                ghost_base = 2 if extra_inning >= 10 else 0
                _start_half(state, active_gi, extra_inning, half, base_override=ghost_base)

                score_before[active_gi] = (state["home_score"][active_gi].astype(np.float64)
                                           + state["away_score"][active_gi].astype(np.float64))

                candidates = []
                for gi in active_gi:
                    order = batting_orders[gi][half]
                    if not order:
                        order = [0]
                    pitcher = last_pitcher[gi][half]
                    park = park_indices[gi]
                    diff = float(state["away_score"][gi] - state["home_score"][gi]) if half == 0 else float(state["home_score"][gi] - state["away_score"][gi])
                    synth, new_ptr = _build_extra_schedule(
                        lineup=order,
                        lineup_ptr=lineup_ptrs[gi][half],
                        pitcher_idx=pitcher,
                        park_idx=park,
                        inning=extra_inning,
                        half=half,
                        score_diff=diff,
                        base_state=ghost_base,
                        env_ctx=env_contexts[gi],
                        geom=geom_per_game[gi],
                        season=args.season,
                        max_len=args.max_len,
                        stand_throws=None,
                        metadata=args.checkpoint_metadata,
                    )
                    lineup_ptrs[gi][half] = new_ptr
                    candidates.append((gi, synth))

                if not candidates:
                    continue
                rows_this_half = np.asarray([row for row, _ in candidates], np.int32)

                for start in range(0, len(candidates), args.batch_games):
                    batch_items = candidates[start:start + args.batch_games]
                    rows_batch, batch = _stack_chunk(batch_items, 0)
                    batch = hybrid_game_batch(batch, rows_batch)
                    initial = {name: value[rows_batch].copy() for name, value in state.items()}
                    batch_seed = seed + extra_inning * 10_000 + half * 1_000 + start + 500_000
                    history_inning = int(initial["inning"][0])
                    prior_cache = histories.get(heads, batch, rows_batch, history_inning, half)
                    rolled = rollout_batch(heads, batch, seed=batch_seed, engine=engine,
                                           c_engine=c_engine, initial_state=initial, initial_cache=prior_cache,
                                           stop_when_decided=True,
                                           decode_len=_decode_bucket(batch, args.max_len),
                                           terminal_outcome_sampler=terminal_outcome_sampler)
                    histories.put(rows_batch, history_inning, half, rolled.get("final_cache"))
                    event_counts += rolled["event"].sum(axis=(0, 1))
                    terminal = rolled["pa_terminal"]
                    generated_pas += int(terminal.sum())
                    outcome_counts += np.bincount(
                        rolled["pa_outcome"][terminal], minlength=len(outcome_counts)
                    )[:len(outcome_counts)]

                    batter_idx = rolled["batter_idx"]
                    flat_bidx = batter_idx[terminal]
                    flat_oc = rolled["pa_outcome"][terminal]
                    known = (flat_bidx >= 0) & (flat_bidx < n_batters)
                    np.add.at(pcounts, (flat_bidx[known], flat_oc[known]), 1.0)
                    # Synthetic extra innings have no observed PA rows, so
                    # they are excluded from the paired-rate artifact.

                    for name, value in rolled["final_state"].items():
                        state[name][rows_batch] = value
                    rollout_calls += 1

                incomplete = (~state['ended'][rows_this_half] &
                              (state['inning'][rows_this_half] == extra_inning) &
                              (state['half'][rows_this_half] == half))
                completion_faults['extra_truncated'][rows_this_half] |= incomplete
                n_extra_half_innings += 1

                score_after = (state["home_score"][rows_this_half].astype(np.float64)
                               + state["away_score"][rows_this_half].astype(np.float64))
                extra_runs += (score_after - score_before[rows_this_half]).sum()

                if half == 1 and extra_inning >= 9:
                    walkoff_mask = (~done[rows_this_half]
                                    & (state["home_score"][rows_this_half] > state["away_score"][rows_this_half]))
                    n_walkoffs += int(walkoff_mask.sum())
                    done |= state["home_score"] > state["away_score"]

            tied = ~done & (state["home_score"] == state["away_score"])
        n_unresolved = int(tied.sum())

    completion_faults["unresolved_tie"] = state["home_score"] == state["away_score"]
    # Build per-game scores
    home = np.array([int(state["home_score"][gi]) for gi in range(G)], dtype=np.float64)
    away = np.array([int(state["away_score"][gi]) for gi in range(G)], dtype=np.float64)
    total = home + away
    margin = np.abs(home - away)

    # PIT computation
    pit_result = None
    if pit_swing_prob:
        all_sp = np.concatenate(pit_swing_prob)
        all_cp = np.concatenate(pit_contact_prob)
        all_fp = np.concatenate(pit_foul_prob)
        all_hp = np.concatenate(pit_hbp_prob)
        all_csp = np.concatenate(pit_called_strike_prob)
        all_dp = np.concatenate(pit_d_probs)
        all_z = np.concatenate(pit_zone)
        all_b = np.concatenate(pit_balls)
        all_s = np.concatenate(pit_strikes)
        all_oc = np.concatenate(pit_outcome)
        pa_probs = _compose_pa_probs(all_sp, all_cp, all_fp, all_hp, all_csp,
                                     all_dp, all_b, all_s)
        pv = _pit_values(pa_probs, all_oc)
        try:
            pit_result = _pit_summary(pv)
        except ImportError:
            # scipy not available; report raw stats
            pit_result = {"n": len(pv), "mean": float(pv.mean()), "std": float(pv.std())}

    return {
        "home": home,
        "away": away,
        "total": total,
        "margin": margin,
        "outcome_counts": outcome_counts,
        "pcounts": pcounts,
        "rate_pcounts": rate_pcounts,
        "runs_by_inning": runs_by_inning,
        "extra_runs": extra_runs,
        "n_walkoffs": n_walkoffs,
        "n_ties": int((home == away).sum()),
        "sim_extra_games": int(entered_extras.sum()),
        "truncations": truncations,
        "event_counts": event_counts,
        "rollout_calls": rollout_calls,
        "scheduled_pas": scheduled_pas,
        "generated_pas": generated_pas,
        "simulation_consistency": pit_result,
        "n_extra_half_innings": n_extra_half_innings,
        "completion_faults": completion_faults,
        "n_unresolved": n_unresolved,
    }


def completion_report(reps, game_row):
    """Read-only audit: existing simulation and headline metrics are unchanged."""
    ids = [gpk for gpk, row in sorted(game_row.items(), key=lambda pair: pair[1])]
    reports = []
    for rep in reps:
        flags = rep['completion_faults']
        bad = np.logical_or.reduce(list(flags.values()))
        clean = ~bad
        reports.append(dict(
            flagged_games=int(bad.sum()), no_detected_completion_fault=int(clean.sum()),
            flagged_fraction=float(bad.mean()),
            reasons={name: [int(ids[i]) for i in np.flatnonzero(mask)] for name, mask in flags.items()},
            mean_total_all=float(np.mean(rep['total'])),
            mean_total_no_detected_fault=float(np.mean(rep['total'][clean])) if clean.any() else None))
    return dict(per_rep=reports, simulation_changed=False,
        headline_metrics_include_flagged_games=True,
        note='Diagnostic only. No detected fault is not certification of legal completion; '
             'observed lineup/staff policy and other modelling limitations remain. '
             'Filtered means are a sensitivity check, not an unbiased replacement benchmark.')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--params-dir", default="checkpoints/pitchformer")
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=1,
                    help="Number of independent replications for confidence intervals.")
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--limit-games", type=int, default=None)
    ap.add_argument("--batch-games", type=int, default=32,
                    help="Active same-inning half-innings per A--D rollout batch.")
    ap.add_argument("--max-half-continuations", type=int, default=6,
                    help="Additional cyclic schedule decodes allowed to finish an incomplete half-inning.")
    ap.add_argument("--events", default="data/processed/events.parquet")
    ap.add_argument("--c-events", action="store_true")
    ap.add_argument("--game-context", default="data/processed/game_context.parquet")
    ap.add_argument("--no-env", action="store_true", help="Must match training.")
    ap.add_argument("--no-geom", action="store_true", help="Must match training.")
    ap.add_argument("--min-pa", type=int, default=150,
                    help="Only report player-level stats for batters with >= this many real test PAs.")
    ap.add_argument("--player-stats", action="store_true",
                    help="Report per-player rate-stat reproduction (AVG, OBP, SLG, K%%, BB%%, HR%%).")
    ap.add_argument("--rates-out", type=Path, default=None,
                    help="Write PA-ladder-compatible per-batter K/BB/Hit/HR rate totals. "
                         "Uses generated ABCD outcomes but the observed 2024 PA denominator.")
    ap.add_argument("--rates-max-pa-per-game", type=int, default=0,
                    help="Optional cap for only the paired-rate cohort; 0 keeps every observed PA. "
                         "The historical v16 artifact used 90.")
    ap.add_argument("--arrays-out", type=Path, default=None,
                    help="Write benchmark arrays (game x replication). These use the observed "
                         "lineup/staff schedule, cycling a half's sources only when needed to finish it; "
                         "they are not pre-game roster/staff simulations.")
    ap.add_argument("--out", default=None)
    ap.add_argument('--skill-mode', choices=['auto', 'mean', 'sample'], default='auto',
                    help='auto samples native Bayesian skills once per rep; keeps exported worlds and ordinary checkpoints fixed')
    ap.add_argument("--hybrid-pa-ckpt", type=Path, default=None,
                    help="Model 7 PA export: evaluate A/B/C pitches and use PA for generated in-play outcomes (D is omitted)")
    ap.add_argument("--hybrid-pa-skill-mode", choices=["mean", "sample"], default="mean",
                    help="Posterior policy for the hybrid PA head (default: mean)")
    args = ap.parse_args()
    if args.reps < 1:
        ap.error('--reps must be positive')
    if args.rates_max_pa_per_game < 0:
        ap.error('--rates-max-pa-per-game must be nonnegative')
    metadata = restore_metadata(args, args.tag)
    args.checkpoint_metadata = metadata
    args.history_reset = (metadata or {}).get("config", {}).get("history_reset", "legacy")
    if args.batch_games < 1:
        raise SystemExit("--batch-games must be positive")
    if args.max_half_continuations < 0:
        raise SystemExit("--max-half-continuations must be nonnegative")

    years = metadata["train_years"] if metadata else [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]
    train = load_seasons(years)
    maps = metadata["maps"] if metadata else build_id_maps(train)
    test = load_seasons([args.season])[0]
    context_path = Path(args.game_context)
    gctx = None if args.no_env or not context_path.exists() else pl.read_parquet(context_path)
    kw = head_kwargs(args, maps, metadata)
    root = Path(args.params_dir)
    def load_head(letter, cls):
        path = root / f"{letter}_{args.tag}_params.pkl"
        return (cls(**kw), _load(path)) if path.exists() else (None, None)
    a, ap_ = load_head("A", TransformerA)
    b, bp = load_head("B", TransformerB)
    c, cp = load_head("C", TransformerC)
    # A direct PA-on-ABC hybrid has no D dependency, including at checkpoint
    # load time.  Native evaluation still loads D normally.
    d, dp = ((None, None) if args.hybrid_pa_ckpt is not None
             else load_head("D", TransformerD))
    if a is None or b is None:
        raise SystemExit("A and B checkpoints are required")
    terminal_outcome_sampler = None
    if args.hybrid_pa_ckpt is not None:
        from diamondworldjax.simulate.pa_abcd_hybrid import PAInPlayOutcomeSampler
        terminal_outcome_sampler = PAInPlayOutcomeSampler.from_paths(
            args.hybrid_pa_ckpt, metadata, season=args.season,
            skill_mode=args.hybrid_pa_skill_mode, seed=args.seed,
        )
        print("Hybrid rollout: A/B/C generate pitches; the Model 7 PA head replaces D for in-play outcomes.",
              flush=True)
    heads_obj = PitchformerHeads(a, b, c, d, ap_, bp, cp, dp)
    from diamondworldjax.eval.pitchformer_worlds import PitchformerWorlds
    worlds = PitchformerWorlds(heads_obj, args.params_dir, args.tag, metadata,
                              args.skill_mode, args.seed)
    engine = EmpiricalEngine().fit(pl.concat(train).filter(pl.col("pa_terminal")))
    c_engine = None
    if args.c_events:
        if c is None or not Path(args.events).exists():
            raise SystemExit("--c-events requires C checkpoint and extracted events")
        c_engine = CTransitionEngine(event_mode=getattr(c, "c_event_mode", "legacy")).fit(pl.concat(train), pl.read_parquet(args.events))

    games = test["game_pk"].unique().sort().to_list()
    if args.limit_games:
        games = games[:args.limit_games]
    test = test.filter(pl.col("game_pk").is_in(games))
    game_row = {int(game_pk): i for i, game_pk in enumerate(games)}
    G = len(games)
    n_batters = maps["n_batter"]

    # The PA ladder indexes its player table by the sorted union of training
    # pitcher and batter IDs.  Retain the native ABCD role-local counts for its
    # diagnostics, but build this second map only when exporting a paired-rate
    # artifact so the bootstrap can compare the same real batters to v16.
    rate_player_ids = None
    rate_batter_index = None
    if args.rates_out is not None:
        raw_ids = list(maps.get("pitcher", {})) + list(maps.get("batter", {}))
        rate_player_ids = np.unique(np.asarray(raw_ids, dtype=np.int64))
        rate_lookup = {int(player_id): i for i, player_id in enumerate(rate_player_ids)}
        rate_batter_index = np.full(n_batters, -1, np.int32)
        for player_id, local_index in maps["batter"].items():
            rate_batter_index[int(local_index)] = rate_lookup[int(player_id)]

    # =====================================================================
    # Load real comparison data from the PA pipeline
    # =====================================================================
    has_real = False
    real_pa = None
    try:
        from diamondworldjax.paths import processed_root
        from diamondworldjax.data.pipeline import load_seasons as load_pa_seasons
        real_pa = load_pa_seasons([args.season], data_root=processed_root()).filter(pl.col("pa_terminal"))
        if args.limit_games:
            real_pa = real_pa.filter(pl.col("game_pk").is_in([int(g) for g in games]))
        has_real = True
    except Exception:
        pass

    # The shared game benchmark schema needs real home/away totals in precisely
    # the game order used by the generated replications.
    real_home = np.full(G, np.nan, dtype=np.float64)
    real_away = np.full(G, np.nan, dtype=np.float64)
    if has_real and "runs_scored" in real_pa.columns:
        half_col = "half_bin" if "half_bin" in real_pa.columns else "half"
        sides = (real_pa.with_columns(
            pl.when(pl.col(half_col) == 0).then(pl.lit("away")).otherwise(pl.lit("home")).alias("side"))
            .group_by(["game_pk", "side"]).agg(pl.col("runs_scored").sum().alias("runs"))
            .pivot(on="side", values="runs", index="game_pk").fill_null(0))
        for row in sides.iter_rows(named=True):
            gi = game_row.get(int(row["game_pk"]))
            if gi is not None:
                real_home[gi] = float(row.get("home", 0))
                real_away[gi] = float(row.get("away", 0))

    # Real runs/game
    real_total = float("nan")
    if has_real and "runs_scored" in real_pa.columns:
        half_col = "half_bin" if "half_bin" in real_pa.columns else "half"
        rg = real_pa.group_by(["game_pk", half_col]).agg(pl.col("runs_scored").sum().alias("r"))
        real_total = rg["r"].sum() / real_pa["game_pk"].n_unique()

    # Real occupancy
    real_occ = float("nan")
    if has_real and "base_state" in real_pa.columns:
        real_occ = float(real_pa.with_columns(
            (pl.col("base_state") > 0).cast(pl.Int32).alias("o"))["o"].mean()) * 100

    # Real extras rate
    real_extra_rate = float("nan")
    if has_real:
        real_n_games = real_pa["game_pk"].n_unique()
        real_extra_ct = (real_pa.group_by("game_pk")
                         .agg(pl.col("inning").max().alias("mi"))
                         .filter(pl.col("mi") >= 10)["mi"].len())
        real_extra_rate = real_extra_ct / real_n_games * 100

    # Real runs-by-inning
    real_frac = np.full(9, np.nan)
    if has_real and "runs_scored" in real_pa.columns:
        rdf = (real_pa.filter((pl.col("inning") >= 1) & (pl.col("inning") <= 9))
               .group_by("inning").agg(pl.col("runs_scored").sum().alias("r")).sort("inning"))
        real_rbi = np.zeros(9)
        for row in rdf.iter_rows(named=True):
            real_rbi[int(row["inning"]) - 1] = row["r"]
        real_frac = real_rbi / max(real_rbi.sum(), 1)

    # Real marginal outcome frequencies
    real_outcome_counts = np.zeros(9, dtype=np.float64)
    if has_real and "pa_outcome" in real_pa.columns:
        for row in real_pa.select("pa_outcome").iter_rows():
            oc = row[0]
            if oc in _OUTCOME_MAP:
                real_outcome_counts[_OUTCOME_MAP[oc]] += 1

    # Real per-batter outcome counts
    batter_map = maps.get("batter", {})
    real_pcounts = np.zeros((n_batters, 9), dtype=np.float64)
    real_rate_pcounts = (np.zeros((len(rate_player_ids), 9), dtype=np.float64)
                         if rate_player_ids is not None else None)
    if has_real:
        # Match prod_playercorr exactly: rows without an observed terminal
        # outcome are excluded *before* the historical per-game PA cap.  Capping
        # first lets a null label consume one of the 90 slots and creates a
        # different paired-bootstrap denominator for that game's later batter.
        real_rate_pa = real_pa.filter(pl.col("pa_outcome").is_not_null())
        if args.rates_max_pa_per_game:
            real_rate_pa = (real_rate_pa.sort(["game_pk", "at_bat_number"])
                             .with_columns(pl.int_range(pl.len()).over("game_pk").alias("_pa_pos"))
                             .filter(pl.col("_pa_pos") < args.rates_max_pa_per_game)
                             .drop("_pa_pos"))
        bcol = "batter_id" if "batter_id" in real_pa.columns else "batter_idx"
        for row in real_pa.select([bcol, "pa_outcome"]).iter_rows():
            bid, oc = row
            if oc in _OUTCOME_MAP and int(bid) in batter_map:
                real_pcounts[batter_map[int(bid)], _OUTCOME_MAP[oc]] += 1
        for row in real_rate_pa.select([bcol, "pa_outcome"]).iter_rows():
            bid, oc = row
            if oc in _OUTCOME_MAP and rate_player_ids is not None:
                rate_idx = rate_lookup.get(int(bid))
                if rate_idx is not None:
                    real_rate_pcounts[rate_idx, _OUTCOME_MAP[oc]] += 1
    else:
        # Fall back to pitch-level test data
        terminal = test.filter(pl.col("pa_terminal") & pl.col("pa_outcome").is_not_null())
        if terminal.height:
            for row in terminal.select(["batter_id", "pa_outcome"]).iter_rows():
                bid, oc = row
                if oc in _OUTCOME_MAP and int(bid) in batter_map:
                    real_pcounts[batter_map[int(bid)], _OUTCOME_MAP[oc]] += 1

    # Real score distribution
    real_score_std = float("nan")
    real_score_median = float("nan")
    real_shutout_rate = float("nan")
    real_blowout_rate = float("nan")
    if has_real and "runs_scored" in real_pa.columns:
        real_game_runs = (real_pa.group_by("game_pk")
                          .agg(pl.col("runs_scored").sum().alias("r")))
        rv = real_game_runs["r"].to_numpy().astype(np.float64)
        real_score_std = float(rv.std())
        real_score_median = float(np.median(rv))
        # Shutout: one side scored 0
        real_half_runs = (real_pa.group_by(["game_pk", "half_bin" if "half_bin" in real_pa.columns else "half"])
                          .agg(pl.col("runs_scored").sum().alias("r")))
        shutouts = (real_half_runs.filter(pl.col("r") == 0).group_by("game_pk").len()
                    .filter(pl.col("len") >= 1))
        real_shutout_rate = float(shutouts.height / real_n_games * 100)
        # Blowout: margin >= 5
        game_sides = (real_pa.with_columns(
            pl.when(pl.col("half_bin" if "half_bin" in real_pa.columns else "half") == 0)
            .then(pl.lit("away")).otherwise(pl.lit("home")).alias("side"))
            .group_by(["game_pk", "side"])
            .agg(pl.col("runs_scored").sum().alias("r"))
            .pivot(on="side", values="r", index="game_pk")
            .fill_null(0))
        if "home" in game_sides.columns and "away" in game_sides.columns:
            margins = np.abs(game_sides["home"].to_numpy() - game_sides["away"].to_numpy())
            real_blowout_rate = float((margins >= 5).mean() * 100)

    # =====================================================================
    # Run replications
    # =====================================================================
    import time
    t0 = time.time()
    print("Preparing immutable observed pitch schedules...", flush=True)
    phase_schedules = _prepare_observed_schedule(test, game_row, maps, args, gctx)
    lineup_info = _extract_lineup_info(test, game_row, maps, G)

    # Pre-populate extra-inning context from the first cached observed chunk
    # for each game. This avoids a second full make_sequences pass.
    for items in phase_schedules.values():
        for gi, seqs in items:
            if lineup_info[4][gi] is not None:
                continue
            first_valid = int(seqs["valid"][0].argmax())
            lineup_info[4][gi] = seqs["ctx"][0, first_valid, 16:].copy()
            lineup_info[5][gi] = (np.zeros_like(seqs["geom"][0, first_valid])
                                  if args.no_geom else seqs["geom"][0, first_valid].copy())

    rep_results = []
    for rep in range(args.reps):
        heads_obj = worlds.for_rep(rep)
        rep_seed = args.seed + rep * 1_000_003
        print(f"\n--- Rep {rep + 1}/{args.reps} (seed={rep_seed}) ---", flush=True)
        if terminal_outcome_sampler is not None and hasattr(terminal_outcome_sampler, "reset"):
            terminal_outcome_sampler.reset()
        import copy
        li = copy.deepcopy(lineup_info)
        res = _run_one_rep(
            heads_obj, engine, c_engine, test, games, game_row, maps, gctx, args,
            seed=rep_seed, n_batters=n_batters,
            lineup_info=li, phase_schedules=phase_schedules,
            terminal_outcome_sampler=terminal_outcome_sampler,
            rate_batter_index=rate_batter_index,
            n_rate_players=0 if rate_player_ids is None else len(rate_player_ids),
            rate_max_pa_per_game=args.rates_max_pa_per_game,
        )
        rep_results.append(res)
    elapsed = time.time() - t0

    # =====================================================================
    # Aggregate across reps
    # =====================================================================
    n_reps = len(rep_results)

    # Per-rep scalar metrics
    totals = np.array([r["total"].mean() for r in rep_results])
    homes = np.array([r["home"].mean() for r in rep_results])
    aways = np.array([r["away"].mean() for r in rep_results])
    home_win_rates = np.array([np.mean(r["home"] > r["away"]) for r in rep_results])
    walkoff_rates = np.array([r["n_walkoffs"] / G for r in rep_results])
    extra_rates = np.array([r["sim_extra_games"] / G for r in rep_results])
    tie_counts = np.array([r["n_ties"] for r in rep_results])
    extra_half_innings = np.array([r["n_extra_half_innings"] for r in rep_results])
    unresolved_counts = np.array([r["n_unresolved"] for r in rep_results])

    # Score distribution across reps (pool all game scores)
    all_totals = np.concatenate([r["total"] for r in rep_results])
    all_margins = np.concatenate([r["margin"] for r in rep_results])
    all_homes = np.concatenate([r["home"] for r in rep_results])

    # Aggregate outcome counts
    total_outcome_counts = sum(r["outcome_counts"] for r in rep_results).astype(np.float64)
    total_outcome_counts /= n_reps  # average per rep

    # Aggregate runs-by-inning
    avg_rbi = sum(r["runs_by_inning"] for r in rep_results) / n_reps
    sim_frac = avg_rbi / max(avg_rbi.sum(), 1)

    # Aggregate player counts (average across reps)
    avg_pcounts = sum(r["pcounts"] for r in rep_results) / n_reps
    avg_rate_pcounts = (sum(r["rate_pcounts"] for r in rep_results) / n_reps
                        if rate_player_ids is not None else None)

    # Generated targets check sampling consistency, not held-out calibration.
    pit_result = rep_results[0].get("simulation_consistency")

    if terminal_outcome_sampler is None:
        from diamondworldjax.eval.pitch_calibration import score_heldout
        calibration_arrays = add_skill_season(
            make_sequences(test, maps, args.max_len, game_ctx=gctx, history_reset=args.history_reset if args.history_reset != "legacy" else "batting_side", context_len=(metadata or {}).get("config", {}).get("context_len", 0)), metadata)
        if args.no_geom:
            calibration_arrays["geom"][:] = 0
        heldout_calibration = score_heldout(worlds.calibration_heads(), calibration_arrays,
                                           batch_size=args.batch_games, seed=args.seed)
    else:
        heldout_calibration = {
            "n": 0,
            "scope": "not_applicable_pa_replaces_d",
            "reason": "Native B+D held-out resolution is not a score for an A/B/C+PA hybrid.",
        }
    # =====================================================================
    # Print results
    # =====================================================================

    def _fmt_ci(values, fmt=".2f"):
        if n_reps == 1:
            return f"{values[0]:{fmt}}"
        return f"{values.mean():{fmt}} +/- {values.std():{fmt}}"

    # This is the pitch-level held-out score.  It conditions on the observed
    # pitch and prior history, so it evaluates B+D's next-pitch resolution—not
    # A's pitch-generation density or an unconstrained full-game rollout.
    print("\n=== Held-out next-pitch resolution (B+D) ===", flush=True)
    if heldout_calibration.get("n", 0):
        print(f"  n={heldout_calibration['n']}  NLL={heldout_calibration['nll']:.4f}  "
              f"Brier={heldout_calibration['brier']:.4f}", flush=True)
        print("  conditioning: observed pitch + prior history; launch marginalized; "
              "includes continuation", flush=True)
    else:
        print("  not reported for the A/B/C + PA hybrid" if terminal_outcome_sampler is not None
              else "  no eligible held-out pitch-resolution rows", flush=True)

    # --- Marginal outcome calibration ---
    print(f"\n=== Marginal outcome calibration ===", flush=True)
    sim_total_pa = total_outcome_counts.sum()
    real_total_pa = real_outcome_counts.sum()
    print(f"  {'outcome':>6s} {'real%':>7s} {'sim%':>7s} {'ratio':>7s}   {'recal':>8s}", flush=True)
    for i, name in enumerate(_OUTCOME_NAMES):
        real_pct = real_outcome_counts[i] / max(real_total_pa, 1) * 100
        sim_pct = total_outcome_counts[i] / max(sim_total_pa, 1) * 100
        ratio = (sim_pct / real_pct) if real_pct > 0 else float("nan")
        recal = -np.log(ratio) if ratio > 0 and not np.isnan(ratio) else float("nan")
        if not np.isnan(real_pct):
            print(f"  {name:>6s} {real_pct:7.2f} {sim_pct:7.2f} {ratio:7.3f}   {recal:8.4f}", flush=True)
        else:
            print(f"  {name:>6s}     n/a {sim_pct:7.2f}", flush=True)

    # --- Score distribution ---
    print(f"\n=== Score distribution ===", flush=True)
    print(f"  {'':12s} {'real':>8s} {'sim':>8s}", flush=True)
    print(f"  {'mean':12s} {real_total:8.2f} {_fmt_ci(totals):>8s}", flush=True)
    print(f"  {'std':12s} {real_score_std:8.2f} {all_totals.std():8.2f}", flush=True)
    print(f"  {'median':12s} {real_score_median:8.1f} {np.median(all_totals):8.1f}", flush=True)
    sim_shutout = float(np.any(np.stack([np.column_stack([r["home"], r["away"]]) for r in rep_results])
                               .reshape(-1, 2) == 0, axis=1).mean() * 100)
    sim_blowout = float((all_margins >= 5).mean() * 100)
    print(f"  {'shutout%':12s} {real_shutout_rate:8.1f} {sim_shutout:8.1f}", flush=True)
    print(f"  {'blowout%':12s} {real_blowout_rate:8.1f} {sim_blowout:8.1f}", flush=True)

    # --- Runs-by-inning fraction ---
    print(f"\n=== Runs-by-inning fraction (fatigue shape test) ===", flush=True)
    print(f"  inning   1    2    3    4    5    6    7    8    9   | late(6-9)", flush=True)
    if not np.isnan(real_frac[0]):
        print(f"  real  " + " ".join(f"{x*100:4.1f}" for x in real_frac)
              + f"  | {real_frac[5:].sum()*100:.1f}%", flush=True)
    print(f"  sim   " + " ".join(f"{x*100:4.1f}" for x in sim_frac)
          + f"  | {sim_frac[5:].sum()*100:.1f}%", flush=True)

    # --- Main summary ---
    eval_label = "PA + ABCD HYBRID GAME EVAL" if terminal_outcome_sampler is not None else "ABCD PITCHFORMER GAME EVAL"
    print(f"\n=== {eval_label} ({G} games, {n_reps} rep{'s' if n_reps > 1 else ''}, "
          f"{elapsed:.0f}s) ===", flush=True)
    if not np.isnan(real_total):
        print(f"  runs/game   real {real_total:.2f}   sim {_fmt_ci(totals)}", flush=True)
    else:
        print(f"  runs/game   sim {_fmt_ci(totals)}", flush=True)
    print(f"  home runs/g {_fmt_ci(homes)}   away {_fmt_ci(aways)}   "
          f"home-win {_fmt_ci(home_win_rates * 100, '.1f')}%  (ties {tie_counts.mean():.0f})", flush=True)
    if not np.isnan(real_extra_rate):
        print(f"  extras      real {real_extra_rate:.1f}%   sim {_fmt_ci(extra_rates * 100, '.1f')}%   "
              f"walk-offs {_fmt_ci(walkoff_rates * 100, '.1f')}%", flush=True)
    else:
        print(f"  extras      sim {_fmt_ci(extra_rates * 100, '.1f')}%   "
              f"walk-offs {_fmt_ci(walkoff_rates * 100, '.1f')}%", flush=True)
    print(f"  truncated half-innings: {sum(r['truncations'] for r in rep_results) / n_reps:.0f}", flush=True)
    if extra_half_innings.sum() > 0:
        print(f"  extra half-innings: {extra_half_innings.mean():.1f}   unresolved: {unresolved_counts.mean():.1f}", flush=True)

    # --- PIT calibration ---
    if pit_result:
        print(f"\n=== Sampling consistency only (generated B+D, {pit_result['n']} terminal PAs) ===", flush=True)
        print(f"  mean={pit_result['mean']:.4f}  std={pit_result['std']:.4f}  "
              f"(ideal: mean=0.5, std={1/12**0.5:.4f})", flush=True)
        if "ks_stat" in pit_result:
            print(f"  KS stat={pit_result['ks_stat']:.4f}  p={pit_result['ks_pvalue']:.4g}", flush=True)
        if "histogram" in pit_result:
            n_bins = len(pit_result["histogram"])
            expected = pit_result["n"] / n_bins
            bars = pit_result["histogram"]
            print(f"  PIT histogram ({n_bins} bins, expected {expected:.0f} each):", flush=True)
            print(f"    " + "  ".join(f"{x:5d}" for x in bars), flush=True)
            print(f"    " + "  ".join(f"{x/expected:5.2f}" for x in bars), flush=True)

    # --- Player-level rate-stat reproduction ---
    player_result = {}
    if args.player_stats:
        real_pa_total = real_pcounts.sum(axis=1)
        keep_idx = np.where(real_pa_total >= args.min_pa)[0]

        mets = ["AVG", "OBP", "SLG", "K%", "BB%", "HR%"]
        rv = {m: [] for m in mets}
        sv = {m: [] for m in mets}
        for i in keep_idx:
            rs = _rate_stats(real_pcounts[i])
            ss = _rate_stats(avg_pcounts[i])
            if rs and ss:
                for m in mets:
                    rv[m].append(rs[m])
                    sv[m].append(ss[m])

        print(f"\n=== Player-level via ABCD game eval ({len(rv['AVG'])} batters >= {args.min_pa} PA) ===",
              flush=True)
        print(f"  {'stat':5s} {'real':>8s} {'sim':>8s} {'MAE':>8s} {'corr':>7s}", flush=True)
        for m in mets:
            a_arr, b_arr = np.array(rv[m]), np.array(sv[m])
            corr = np.corrcoef(a_arr, b_arr)[0, 1] if len(a_arr) > 1 else float("nan")
            print(f"  {m:5s} {a_arr.mean():8.4f} {b_arr.mean():8.4f} "
                  f"{np.abs(a_arr - b_arr).mean():8.4f} {corr:7.3f}", flush=True)

        player_result = {
            "n_batters": len(rv["AVG"]),
            "min_pa": args.min_pa,
            "stats": {
                m: {"real_mean": float(np.array(rv[m]).mean()),
                    "sim_mean": float(np.array(sv[m]).mean()),
                    "mae": float(np.abs(np.array(rv[m]) - np.array(sv[m])).mean()),
                    "corr": float(np.corrcoef(np.array(rv[m]), np.array(sv[m]))[0, 1])
                          if len(rv[m]) > 1 else float("nan")}
                for m in mets
            },
        }

        # --- Coverage (multi-rep empirical prediction intervals) ---
        if n_reps >= 3:
            # Collect per-player rate stats across reps
            rep_player_stats = []
            for r in rep_results:
                rep_rates = {}
                for i in keep_idx:
                    rs = _rate_stats(r["pcounts"][i])
                    if rs:
                        rep_rates[i] = rs
                rep_player_stats.append(rep_rates)

            coverage_levels = [50, 90, 95]
            print(f"\n=== Coverage ({n_reps} reps, {len(keep_idx)} batters) ===", flush=True)
            print(f"  {'stat':5s}" + "".join(f" {'cov'+str(l)+'%':>8s}" for l in coverage_levels), flush=True)
            coverage_result = {}
            for m in mets:
                cov_by_level = {}
                for level in coverage_levels:
                    lo_q = (100 - level) / 200
                    hi_q = 1 - lo_q
                    in_interval = 0
                    total_checked = 0
                    for i in keep_idx:
                        rs = _rate_stats(real_pcounts[i])
                        if rs is None:
                            continue
                        rep_vals = [rps[i][m] for rps in rep_player_stats if i in rps and m in rps[i]]
                        if len(rep_vals) < n_reps * 0.5:
                            continue
                        lo = np.quantile(rep_vals, lo_q)
                        hi = np.quantile(rep_vals, hi_q)
                        if lo <= rs[m] <= hi:
                            in_interval += 1
                        total_checked += 1
                    cov = in_interval / max(total_checked, 1) * 100
                    cov_by_level[level] = cov
                coverage_result[m] = cov_by_level
                print(f"  {m:5s}" + "".join(f" {cov_by_level[l]:8.1f}" for l in coverage_levels), flush=True)
            player_result["coverage"] = coverage_result
        elif n_reps > 1:
            print(f"\n  (coverage requires >= 3 reps; got {n_reps})", flush=True)

    rate_export = None
    if args.rates_out is not None:
        if not has_real or real_rate_pcounts is None:
            raise SystemExit("--rates-out requires held-out PA rows with terminal outcomes")
        real_cnt = real_rate_pcounts.sum(axis=1)
        sim_cnt = avg_rate_pcounts.sum(axis=1)
        sim_rates = np.divide(avg_rate_pcounts, sim_cnt[:, None],
                              out=np.zeros_like(avg_rate_pcounts), where=sim_cnt[:, None] > 0)
        args.rates_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.rates_out,
            # Match prod_playercorr's contract: sum* / cnt is the model's
            # expected rate on the real held-out PA denominator.
            sumK=sim_rates[:, _K] * real_cnt,
            sumBB=(sim_rates[:, _BB] + sim_rates[:, _HBP]) * real_cnt,
            sumHit=(sim_rates[:, _1B] + sim_rates[:, _2B] + sim_rates[:, _3B] + sim_rates[:, _HR]) * real_cnt,
            sumHR=sim_rates[:, _HR] * real_cnt,
            rK=real_rate_pcounts[:, _K],
            rBB=real_rate_pcounts[:, _BB] + real_rate_pcounts[:, _HBP],
            rHit=(real_rate_pcounts[:, _1B] + real_rate_pcounts[:, _2B]
                  + real_rate_pcounts[:, _3B] + real_rate_pcounts[:, _HR]),
            rHR=real_rate_pcounts[:, _HR],
            cnt=real_cnt,
            player_ids=rate_player_ids,
        )
        rate_export = dict(path=str(args.rates_out), players=int((real_cnt > 0).sum()),
                           max_pa_per_game=args.rates_max_pa_per_game,
                           note="Generated-outcome rates on the selected observed 2024 PA denominator.")
        print(f"saved paired-rate artifact -> {args.rates_out}", flush=True)

    arrays_export = None
    if args.arrays_out is not None:
        if not np.isfinite(real_home).all() or not np.isfinite(real_away).all():
            raise SystemExit("--arrays-out requires real home and away scores for every evaluated game")
        args.arrays_out.parent.mkdir(parents=True, exist_ok=True)
        sim_home = np.stack([r["home"] for r in rep_results], axis=1)
        sim_away = np.stack([r["away"] for r in rep_results], axis=1)
        np.savez(args.arrays_out, sim_home=sim_home, sim_away=sim_away,
                 sim_total=sim_home + sim_away, real_home=real_home,
                 real_away=real_away, real_total=real_home + real_away,
                 game_pk=np.asarray(games, dtype=np.int64),
                 evaluation_mode=np.asarray("observed_schedule"))
        arrays_export = dict(
            path=str(args.arrays_out), games=G, reps=n_reps,
            note="Observed lineup/staff schedule; source rows cycle only if a generated half outlives "
                 "its observed PA schedule. Not a pre-game roster/staff simulation.",
        )
        print(f"saved benchmark arrays -> {args.arrays_out}", flush=True)

    # =====================================================================
    # Build result dict
    # =====================================================================
    result = {
        "tag": args.tag, "season": args.season, "c_events": args.c_events,
        "games": G, "reps": n_reps, "elapsed_s": elapsed,
        "truncated_half_innings": int(sum(r["truncations"] for r in rep_results) / n_reps),
        "batch_games": args.batch_games,
        # Runs / scoring
        "total_mean": float(totals.mean()),
        "total_std": float(totals.std()) if n_reps > 1 else None,
        "home_mean": float(homes.mean()), "away_mean": float(aways.mean()),
        "home_win_rate": float(home_win_rates.mean()),
        "real_total_mean": float(real_total) if not np.isnan(real_total) else None,
        # Score distribution
        "score_dist": {
            "sim_std": float(all_totals.std()),
            "sim_median": float(np.median(all_totals)),
            "sim_shutout_rate": sim_shutout,
            "sim_blowout_rate": sim_blowout,
            "real_std": float(real_score_std) if not np.isnan(real_score_std) else None,
            "real_median": float(real_score_median) if not np.isnan(real_score_median) else None,
            "real_shutout_rate": float(real_shutout_rate) if not np.isnan(real_shutout_rate) else None,
            "real_blowout_rate": float(real_blowout_rate) if not np.isnan(real_blowout_rate) else None,
        },
        # Marginal outcome calibration
        "pa_outcomes_sim": {name: float(total_outcome_counts[i]) for i, name in enumerate(_OUTCOME_NAMES)},
        "pa_outcomes_real": {name: float(real_outcome_counts[i]) for i, name in enumerate(_OUTCOME_NAMES)}
                           if real_outcome_counts.sum() > 0 else None,
        # Runs-by-inning
        "runs_by_inning_frac_sim": sim_frac.tolist(),
        "runs_by_inning_frac_real": real_frac.tolist() if not np.isnan(real_frac[0]) else None,
        # Extras / walk-offs
        "extra_runs_per_game": float(sum(r["extra_runs"] for r in rep_results) / n_reps / G),
        "sim_extra_rate": float(extra_rates.mean() * 100),
        "real_extra_rate": float(real_extra_rate) if not np.isnan(real_extra_rate) else None,
        "walkoff_rate": float(walkoff_rates.mean() * 100),
        "n_ties_mean": float(tie_counts.mean()),
        "n_extra_half_innings_mean": float(extra_half_innings.mean()),
        "n_unresolved_mean": float(unresolved_counts.mean()),
        "events": (sum(r["event_counts"] for r in rep_results) / n_reps).tolist(),
        # PIT
        "simulation_consistency": pit_result,
        "heldout_pitch_calibration": heldout_calibration,
        "skill_policy": worlds.report(args.reps),
        "completion_diagnostics": completion_report(rep_results, game_row),
        "rate_export": rate_export,
        "arrays_export": arrays_export,
        "note": ("Generated game state on observed lineup/staff schedule; a half that outlives its "
                 "observed PA sources cycles those sources until a generated third out. Continuation PAs "
                 "are excluded from the paired fixed-cohort rate export; this is not pre-game roster selection. "
                 "Hybrid mode generates pitches with A/B/C and replaces D's in-play outcome with the Model 7 PA head."
                 if terminal_outcome_sampler is not None else
                 "Generated game state on observed lineup/staff schedule; a half that outlives its "
                 "observed PA sources cycles those sources until a generated third out. Continuation PAs "
                 "are excluded from the paired fixed-cohort rate export; not a pre-game roster/staff simulation."),
    }
    if args.hybrid_pa_ckpt is not None:
        result["hybrid_pa_checkpoint"] = str(args.hybrid_pa_ckpt)
        result["hybrid_pa_skill_mode"] = args.hybrid_pa_skill_mode
    if player_result:
        result["player_stats"] = player_result

    # Per-rep detail (for downstream analysis)
    if n_reps > 1:
        result["per_rep"] = {
            "total_mean": totals.tolist(),
            "home_win_rate": home_win_rates.tolist(),
            "walkoff_rate": walkoff_rates.tolist(),
            "extra_rate": extra_rates.tolist(),
        }

    # Per-game scores from first rep (backward compat)
    scores = [(int(game_pk), int(rep_results[0]["home"][gi]), int(rep_results[0]["away"][gi]))
              for gi, game_pk in enumerate(games)]
    out = Path(args.out or f"data/eval2/pitchformer_games_{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**result, "scores": scores}, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
