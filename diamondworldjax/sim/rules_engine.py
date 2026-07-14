"""Deterministic baseball state-transition engine (Phase 1 of the world-model plan).

The core thesis: runs_scored and base_state_after are almost entirely DETERMINED
by (base_state, outs, pa_outcome) plus standard baserunning rules. They are
arithmetic, not predictions. Letting independent neural heads sample them (as the
v1-v5 pa_model does) injects per-PA inconsistency that compounds into the
variance blowup that loses to a curve fit in free-rollout.

This module ports the validated `_deterministic_transition` logic from the B0
Markov baseline (diamondworld/baselines/markov_re24.py, KL=0.0083) into a
vectorized form usable inside a JAX rollout.

Encodings (must match the data pipeline):
  base_state : 3-bit mask, bit0=runner on 1B, bit1=2B, bit2=3B  (0..7)
  outcomes   : ["K","BB","HBP","1B","2B","3B","HR","out","E"]   (0..8)
  outs       : 0,1,2 (a 3rd out ends the half-inning)

A PA adds an out only for K and out (in-play out). All other outcomes reach base
or score. `E` (reached-on-error) is treated like a single: batter reaches 1B and
runners advance one base. This is an approximation; the empirical transition
table captures the true stochastic spread and is used in the stochastic engine.
"""
from __future__ import annotations

import numpy as np

# Outcome index convention (matches diamondworld/baselines/base.py)
PA_OUTCOMES = ["K", "BB", "HBP", "1B", "2B", "3B", "HR", "out", "E"]
PA_OUTCOME_IDX = {o: i for i, o in enumerate(PA_OUTCOMES)}
N_OUTCOMES = len(PA_OUTCOMES)
N_BASE_STATES = 8

# Outcomes that record an out (the only ones that increment the out counter).
_OUT_OUTCOMES = {"K", "out"}


def _deterministic_transition(bs: int, outcome: str) -> tuple[int, int]:
    """(base_state, outcome) -> (base_state_after, runs_scored).

    Ported verbatim from MarkovRE24Simulator._deterministic_transition, with
    `E` mapped to single-style advancement. Does not depend on outs.
    """
    on_1b = bool(bs & 1)
    on_2b = bool(bs & 2)
    on_3b = bool(bs & 4)
    count_runners = int(on_1b) + int(on_2b) + int(on_3b)

    if outcome == "K":
        return (bs, 0)

    if outcome in ("BB", "HBP"):
        # Force advancement only.
        if bs == 7:      # bases loaded -> forces a run
            return (7, 1)
        elif bs == 3:    # 1B+2B
            return (7, 0)
        elif bs == 1:    # 1B only
            return (3, 0)
        elif bs == 6:    # 2B+3B (no force on the batter past 1B)
            return (7, 0)
        elif bs == 2:    # 2B only
            return (3, 0)
        elif bs == 4:    # 3B only
            return (5, 0)
        else:            # empty
            return (1, 0)

    if outcome == "HR":
        return (0, count_runners + 1)

    if outcome == "3B":
        return (4, count_runners)

    if outcome == "2B":
        runs = 0
        new_bs = 0
        if on_2b:
            runs += 1
        if on_3b:
            runs += 1
        if on_1b:
            if not on_2b and not on_3b:
                new_bs |= 4   # 1B -> 3B
            else:
                runs += 1     # 1B runner scores
        new_bs |= 2           # batter on 2B
        return (new_bs, runs)

    if outcome in ("1B", "E"):
        runs = 0
        new_bs = 0
        if on_3b:
            runs += 1
        if on_2b:
            new_bs |= 4       # 2B -> 3B
        if on_1b:
            new_bs |= 2       # 1B -> 2B
        new_bs |= 1           # batter on 1B
        return (new_bs, runs)

    if outcome == "out":
        return (bs, 0)

    return (bs, 0)


def build_lookup_tables() -> dict[str, np.ndarray]:
    """Precompute static lookup tables indexed by [base_state, outcome_idx].

    Returns dict with:
      bs_after : (8, 9) int   resulting base_state
      runs     : (8, 9) int   runs scored
      out_inc  : (9,)   int   1 if outcome records an out, else 0
    """
    bs_after = np.zeros((N_BASE_STATES, N_OUTCOMES), dtype=np.int32)
    runs = np.zeros((N_BASE_STATES, N_OUTCOMES), dtype=np.int32)
    out_inc = np.zeros(N_OUTCOMES, dtype=np.int32)

    for bs in range(N_BASE_STATES):
        for oi, oc in enumerate(PA_OUTCOMES):
            after, r = _deterministic_transition(bs, oc)
            bs_after[bs, oi] = after
            runs[bs, oi] = r
    for oi, oc in enumerate(PA_OUTCOMES):
        out_inc[oi] = 1 if oc in _OUT_OUTCOMES else 0

    return {"bs_after": bs_after, "runs": runs, "out_inc": out_inc}


# Module-level singletons (cheap; 8x9 tables).
_TABLES = build_lookup_tables()
BS_AFTER = _TABLES["bs_after"]   # (8, 9)
RUNS = _TABLES["runs"]           # (8, 9)
OUT_INC = _TABLES["out_inc"]     # (9,)


def step(base_state: np.ndarray, outcome: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized deterministic transition.

    Parameters
    ----------
    base_state : int array, values 0..7
    outcome    : int array (same shape), values 0..8

    Returns dict with arrays (same shape as inputs):
      runs       : runs scored on the PA
      bs_after   : resulting base_state
      out_inc    : 1 if the PA recorded an out, else 0
    """
    bs = np.asarray(base_state, dtype=np.int64)
    oc = np.asarray(outcome, dtype=np.int64)
    return {
        "runs": RUNS[bs, oc],
        "bs_after": BS_AFTER[bs, oc],
        "out_inc": OUT_INC[oc],
    }


def _key(bs: int, outs: int, oc: int) -> int:
    """Flatten (base_state 0..7, outs 0..2, outcome 0..8) into a single index."""
    return (bs * 3 + outs) * N_OUTCOMES + oc


N_KEYS = N_BASE_STATES * 3 * N_OUTCOMES  # 8 * 3 * 9 = 216


class EmpiricalEngine:
    """Samples (base_state_after, runs) from the real transition distribution.

    Fits P(bs_after, runs | base_state, outs, outcome) from training data, keyed
    by (base_state, outs, outcome). This captures the true stochastic baserunning
    (runner scoring from 2B on a single, tagging up on a fly out, taking the extra
    base) that the deterministic rule misses, and is unbiased in aggregate because
    it reproduces the observed transition frequencies.

    Falls back to the deterministic transition for any key never seen in training.
    """

    def __init__(self) -> None:
        # Per key: arrays of observed (bs_after, runs); sampling draws a row.
        self._bsa: dict[int, np.ndarray] = {}
        self._runs: dict[int, np.ndarray] = {}
        # Expected runs per key (for unbiased deterministic-expectation use / bias checks).
        self._exp_runs = np.full(N_KEYS, np.nan, dtype=np.float64)

    def fit(self, pa_df) -> "EmpiricalEngine":
        import numpy as _np
        import polars as pl

        df = pa_df.filter(
            pl.col("pa_outcome").is_not_null()
            & pl.col("base_state").is_not_null()
            & pl.col("base_state_after").is_not_null()
        )
        bs = df["base_state"].to_numpy().astype(_np.int64)
        outs = _np.clip(df["outs"].to_numpy().astype(_np.int64), 0, 2)
        oc_str = df["pa_outcome"].to_list()
        runs = df["runs_scored"].to_numpy().astype(_np.int64)
        bsa = df["base_state_after"].to_numpy().astype(_np.int64)

        oc = _np.array([PA_OUTCOME_IDX.get(o, -1) for o in oc_str], dtype=_np.int64)
        valid = oc >= 0
        bs, outs, oc, runs, bsa = bs[valid], outs[valid], oc[valid], runs[valid], bsa[valid]

        keys = (bs * 3 + outs) * N_OUTCOMES + oc
        order = _np.argsort(keys, kind="stable")
        keys_s, bsa_s, runs_s = keys[order], bsa[order], runs[order]
        uniq, starts = _np.unique(keys_s, return_index=True)
        ends = _np.append(starts[1:], len(keys_s))
        for k, s, e in zip(uniq, starts, ends):
            self._bsa[int(k)] = bsa_s[s:e]
            self._runs[int(k)] = runs_s[s:e]
            self._exp_runs[int(k)] = runs_s[s:e].mean()
        return self

    def expected_runs(self, base_state, outs, outcome) -> np.ndarray:
        """Vectorized expected runs per (bs, outs, outcome); NaN-free via det fallback."""
        bs = np.asarray(base_state, np.int64)
        ou = np.clip(np.asarray(outs, np.int64), 0, 2)
        oc = np.asarray(outcome, np.int64)
        keys = (bs * 3 + ou) * N_OUTCOMES + oc
        exp = self._exp_runs[keys]
        miss = np.isnan(exp)
        if miss.any():
            exp = exp.copy()
            exp[miss] = RUNS[bs[miss], oc[miss]]
        return exp

    def sample(self, base_state, outs, outcome, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Sample (bs_after, runs) per element from the empirical distribution.

        Vectorized by unique key: at most 216 keys, so this is a short Python loop
        over the keys present in the batch, each doing one vectorized draw.
        """
        bs = np.asarray(base_state, np.int64)
        ou = np.clip(np.asarray(outs, np.int64), 0, 2)
        oc = np.asarray(outcome, np.int64)
        keys = (bs * 3 + ou) * N_OUTCOMES + oc

        out_runs = np.empty(len(bs), np.int64)
        out_bsa = np.empty(len(bs), np.int64)

        for k in np.unique(keys):
            m = keys == k
            k = int(k)
            if k in self._bsa and len(self._bsa[k]):
                idx = rng.integers(0, len(self._bsa[k]), size=int(m.sum()))
                out_runs[m] = self._runs[k][idx]
                out_bsa[m] = self._bsa[k][idx]
            else:
                # Deterministic fallback for unseen keys.
                b = bs[m]
                o = oc[m]
                out_runs[m] = RUNS[b, o]
                out_bsa[m] = BS_AFTER[b, o]

        return {"runs": out_runs, "bs_after": out_bsa, "out_inc": OUT_INC[oc]}
