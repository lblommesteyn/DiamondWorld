"""Empirical, legal state transitions for Transformer C event types.

Transformer C predicts *which* rare event occurred but not the runner or target
base.  This adapter learns the observed state change conditional on event type,
base state, and outs.  It is only applied after non-terminal pitches; terminal
PA transitions remain owned by :class:`EmpiricalEngine`, avoiding double counts
when an event and PA result share a pitch record.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from diamondworldjax.model.transformer_c import EVENT_FLAGS


class CTransitionEngine:
    def __init__(self, event_mode="legacy") -> None:
        self.event_mode = event_mode
        self._base: dict[tuple[int, int, int], np.ndarray] = {}
        self._outs: dict[tuple[int, int, int], np.ndarray] = {}
        self._runs: dict[tuple[int, int, int], np.ndarray] = {}

    def fit(self, pitches: pl.DataFrame, events: pl.DataFrame) -> "CTransitionEngine":
        if not events.height:
            return self
        cols = ["game_pk", "at_bat_number", "pitch_number", "base_state", "outs",
                "pa_terminal", "home_score", "away_score"]
        have = [c for c in cols if c in pitches.columns]
        required = {"game_pk", "at_bat_number", "pitch_number", "base_state", "outs", "pa_terminal"}
        if not required.issubset(have):
            return self
        d = pitches.select(have).sort(["game_pk", "at_bat_number", "pitch_number"])
        next_cols = [
            pl.col("base_state").shift(-1).over("game_pk").alias("next_base"),
            pl.col("outs").shift(-1).over("game_pk").alias("next_outs"),
        ]
        if {"home_score", "away_score"}.issubset(have):
            next_cols += [
                (pl.col("home_score").shift(-1).over("game_pk") - pl.col("home_score")
                 + pl.col("away_score").shift(-1).over("game_pk") - pl.col("away_score")).alias("next_runs")
            ]
        else:
            next_cols += [pl.lit(0).alias("next_runs")]
        d = d.with_columns(next_cols)
        flag_cols = [f for f in EVENT_FLAGS if f in events.columns]
        if not flag_cols:
            return self
        d = d.join(events.select(["game_pk", "at_bat_number", "pitch_number", *flag_cols]),
                   on=["game_pk", "at_bat_number", "pitch_number"], how="inner")
        # The next recorded pitch exposes the result of an event after this pitch.
        # Terminal PA rows conflate that event with ordinary PA advancement.
        d = d.filter(~pl.col("pa_terminal").cast(pl.Boolean)
                     & pl.col("next_base").is_not_null() & pl.col("next_outs").is_not_null())
        if self.event_mode == 'bundles':
            d = d.with_columns(sum(pl.col(f).fill_null(0).cast(pl.Int32) * (1 << i)
                                  for i, f in enumerate(EVENT_FLAGS) if f in d.columns).alias('_bundle'))
            event_columns = [(int(k)-1, int(k)) for k in d['_bundle'].unique().to_list() if k]
        else:
            event_columns = list(enumerate(EVENT_FLAGS))
        for event_idx, flag in event_columns:
            if self.event_mode != "bundles" and flag not in d.columns:
                continue
            e = d.filter(pl.col("_bundle") == flag) if self.event_mode == "bundles" else d.filter(pl.col(flag) > 0)
            if not e.height:
                continue
            for row in e.select(["base_state", "outs", "next_base", "next_outs", "next_runs"]).iter_rows():
                base, outs, nxt_base, nxt_outs, runs = map(int, row)
                if nxt_outs < outs:
                    nxt_outs, nxt_base = 3, 0
                key = (event_idx, int(np.clip(base, 0, 7)), int(np.clip(outs, 0, 2)))
                self._base.setdefault(key, []).append(int(np.clip(nxt_base, 0, 7)))
                self._outs.setdefault(key, []).append(int(np.clip(nxt_outs, 0, 3)))
                self._runs.setdefault(key, []).append(max(0, runs))
        for table in (self._base, self._outs, self._runs):
            for key, values in table.items():
                table[key] = np.asarray(values, dtype=np.int32)
        return self

    def support(self):
        """Explicit model support: none plus bundles with a fitted transition."""
        result = np.zeros((256, 24), bool)
        result[0] = True
        for event, base, outs in self._base:
            result[event + 1, base * 3 + outs] = True
        return tuple(result.ravel().tolist())

    def sample(self, event: np.ndarray, base_state: np.ndarray, outs: np.ndarray,
               rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Sample an observed legal transition; unsupported cells are no-ops."""
        event = np.asarray(event, np.int32)
        base = np.asarray(base_state, np.int32)
        before_outs = np.asarray(outs, np.int32)
        after_base, after_outs, runs = base.copy(), before_outs.copy(), np.zeros(len(base), np.int32)
        for i in range(len(base)):
            key = (int(event[i]), int(np.clip(base[i], 0, 7)), int(np.clip(before_outs[i], 0, 2)))
            choices = self._base.get(key)
            if choices is None or not len(choices):
                continue
            j = int(rng.integers(len(choices)))
            after_base[i] = self._base[key][j]
            after_outs[i] = self._outs[key][j]
            runs[i] = self._runs[key][j]
        return {"base": after_base, "outs": after_outs, "runs": runs}
