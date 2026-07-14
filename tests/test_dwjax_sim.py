"""Tests for the DWJAX rules engine and true-simulator extraction helpers.

These cover the pure numpy/polars layer only (no JAX required), so they run
on the Windows dev venv as well as the WSL training env.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from diamondworldjax.sim.rules_engine import (
    BS_AFTER,
    OUT_INC,
    PA_OUTCOME_IDX,
    RUNS,
    step,
)
from diamondworldjax.sim.game_extract import (
    MAX_STAFF,
    cap_walkoff_runs,
    extract_games,
    fit_hook_dists,
    pad_staffs,
)


def _oc(name: str) -> int:
    return PA_OUTCOME_IDX[name]


class TestRulesEngine:
    def test_walk_forces(self):
        # Bases loaded walk forces in exactly one run, bases stay loaded.
        assert RUNS[7, _oc("BB")] == 1
        assert BS_AFTER[7, _oc("BB")] == 7
        # Runner on 2B only: batter to 1B, no force of the runner.
        assert RUNS[2, _oc("BB")] == 0
        assert BS_AFTER[2, _oc("BB")] == 3
        # Runner on 3B only: batter to 1B, runner holds.
        assert BS_AFTER[4, _oc("HBP")] == 5

    def test_home_run_clears_bases(self):
        for bs in range(8):
            n_runners = bin(bs).count("1")
            assert RUNS[bs, _oc("HR")] == n_runners + 1
            assert BS_AFTER[bs, _oc("HR")] == 0

    def test_single_advancement(self):
        # Runner on 3B scores, 2B -> 3B, 1B -> 2B, batter to 1B.
        assert RUNS[7, _oc("1B")] == 1
        assert BS_AFTER[7, _oc("1B")] == 7
        assert RUNS[4, _oc("1B")] == 1
        assert BS_AFTER[4, _oc("1B")] == 1
        # Error behaves like a single for base advancement.
        assert RUNS[4, _oc("E")] == 1
        assert BS_AFTER[4, _oc("E")] == 1

    def test_double_advancement(self):
        # Lone runner on 1B goes to 3B on a double.
        assert RUNS[1, _oc("2B")] == 0
        assert BS_AFTER[1, _oc("2B")] == 2 | 4
        # Runners on 2B and 3B both score.
        assert RUNS[6, _oc("2B")] == 2
        assert BS_AFTER[6, _oc("2B")] == 2

    def test_outs_only_on_k_and_out(self):
        expected = {o: (1 if o in ("K", "out") else 0) for o in PA_OUTCOME_IDX}
        for name, i in PA_OUTCOME_IDX.items():
            assert OUT_INC[i] == expected[name], name

    def test_step_vectorized_matches_tables(self):
        rng = np.random.default_rng(0)
        bs = rng.integers(0, 8, size=100)
        oc = rng.integers(0, 9, size=100)
        out = step(bs, oc)
        np.testing.assert_array_equal(out["runs"], RUNS[bs, oc])
        np.testing.assert_array_equal(out["bs_after"], BS_AFTER[bs, oc])
        np.testing.assert_array_equal(out["out_inc"], OUT_INC[oc])


def _mk_pa_df(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "game_pk": pl.Int64,
            "half_bin": pl.Int8,
            "at_bat_number": pl.Int64,
            "batter_id": pl.Int64,
            "pitcher_id": pl.Int64,
            "park_id": pl.Utf8,  # venue code string, e.g. "HOU"
        },
    )


def _rows(game_pk, half_bin, batters, pitchers, park_id="TOR", ab_start=1):
    step = 2  # interleave top/bottom at-bat numbers loosely
    return [
        {
            "game_pk": game_pk,
            "half_bin": half_bin,
            "at_bat_number": ab_start + i * step,
            "batter_id": b,
            "pitcher_id": p,
            "park_id": park_id,
        }
        for i, (b, p) in enumerate(zip(batters, pitchers))
    ]


class TestExtractGames:
    def setup_method(self):
        # Player ids 100-129 known; idx = id - 90 so idx values are distinct.
        self.id_to_idx = {i: i - 90 for i in range(100, 140)}

    def test_staffs_assigned_to_fielding_team(self):
        # Top half (away batting): pitchers 120, 121 belong to the HOME staff.
        # Bottom half (home batting): pitcher 130 belongs to the AWAY staff.
        away_bats = list(range(100, 109)) * 2
        home_bats = list(range(110, 119)) * 2
        top = _rows(1, 0, away_bats, [120] * 12 + [121] * 6, ab_start=1)
        bot = _rows(1, 1, home_bats, [130] * 18, ab_start=2)
        games = extract_games(_mk_pa_df(top + bot), self.id_to_idx)
        assert len(games) == 1
        g = games[0]
        assert g["home_staff"] == [120 - 90, 121 - 90]
        assert g["away_staff"] == [130 - 90]
        assert g["away_lineup"] == [i - 90 for i in range(100, 109)]
        assert g["home_lineup"] == [i - 90 for i in range(110, 119)]

    def test_unknown_players_map_to_zero(self):
        top = _rows(1, 0, [999] * 9, [120] * 9)
        bot = _rows(1, 1, list(range(110, 119)), [888] * 9, ab_start=2)
        games = extract_games(_mk_pa_df(top + bot), self.id_to_idx)
        g = games[0]
        assert g["away_lineup"][0] == 0
        assert g["away_staff"] == [0]

    def test_park_mapping(self):
        top = _rows(1, 0, list(range(100, 109)), [120] * 9, park_id="COL")
        bot = _rows(1, 1, list(range(110, 119)), [130] * 9, park_id="COL", ab_start=2)
        df = _mk_pa_df(top + bot)
        games = extract_games(df, self.id_to_idx, park_map={"COL": 7})
        assert games[0]["park"] == 7
        # Unknown park -> 0; no map -> 0.
        games = extract_games(df, self.id_to_idx, park_map={"NYY": 3})
        assert games[0]["park"] == 0
        games = extract_games(df, self.id_to_idx)
        assert games[0]["park"] == 0

    def test_game_missing_half_skipped(self):
        top = _rows(1, 0, list(range(100, 109)), [120] * 9)
        games = extract_games(_mk_pa_df(top), self.id_to_idx)
        assert games == []


class TestPadStaffs:
    def test_pads_with_last_pitcher(self):
        games = [{"home_staff": [5, 6]}, {"home_staff": [7]}]
        out, lens = pad_staffs(games, "home_staff")
        assert out.shape == (2, MAX_STAFF)
        assert lens.tolist() == [2, 1]
        assert out[0, :3].tolist() == [5, 6, 6]
        assert (out[0, 1:] == 6).all()
        assert (out[1] == 7).all()

    def test_empty_staff_falls_back_to_unknown(self):
        out, lens = pad_staffs([{"home_staff": []}], "home_staff")
        assert lens[0] == 1
        assert (out[0] == 0).all()


class TestCapWalkoffRuns:
    def test_nonhr_capped_at_winning_run(self):
        # Home tied 3-3, bases-loaded double would score 2: only 1 counts.
        bat = np.array([3.0])
        fld = np.array([3.0])
        runs = np.array([2.0])
        out = cap_walkoff_runs(bat, fld, runs, np.array([False]))
        assert out[0] == 1.0

    def test_hr_counts_in_full(self):
        # Walk-off grand slam down 2: all 4 runs count.
        out = cap_walkoff_runs(
            np.array([1.0]), np.array([3.0]), np.array([4.0]), np.array([True])
        )
        assert out[0] == 4.0

    def test_non_winning_runs_untouched(self):
        # Down 3, single scores 1: not a walk-off, runs unchanged.
        out = cap_walkoff_runs(
            np.array([0.0]), np.array([3.0]), np.array([1.0]), np.array([False])
        )
        assert out[0] == 1.0

    def test_down_two_triple_scoring_three_capped(self):
        # Home down 2 in the 9th; play would score 3 -> capped at 3 (fld+1-bat).
        out = cap_walkoff_runs(
            np.array([2.0]), np.array([4.0]), np.array([3.0]), np.array([False])
        )
        assert out[0] == 3.0


class TestHookDists:
    def test_starter_vs_reliever_split(self):
        rows = []
        # Game 1 top half: starter 120 faces 18 PA, reliever 121 faces 6.
        rows += _rows(1, 0, [100] * 18, [120] * 18, ab_start=1)
        rows += _rows(1, 0, [100] * 6, [121] * 6, ab_start=100)
        # Game 1 bottom half: single pitcher 130 faces 27.
        rows += _rows(1, 1, [110] * 27, [130] * 27, ab_start=2)
        starters, relievers = fit_hook_dists(_mk_pa_df(rows))
        assert sorted(starters.tolist()) == [18, 27]
        assert relievers.tolist() == [6]

    def test_starter_identified_by_first_at_bat_not_count(self):
        rows = []
        # Opener faces 3 PA first; bulk guy faces 20 after. Opener = starter.
        rows += _rows(1, 0, [100] * 3, [120] * 3, ab_start=1)
        rows += _rows(1, 0, [100] * 20, [121] * 20, ab_start=50)
        rows += _rows(1, 1, [110] * 9, [130] * 9, ab_start=2)
        starters, relievers = fit_hook_dists(_mk_pa_df(rows))
        assert 3 in starters.tolist()
        assert 20 in relievers.tolist()
