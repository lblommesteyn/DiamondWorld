"""Pure-python helpers for the true game simulator (no JAX imports).

Kept separate from scripts/simulate_games.py so the extraction and
hook-distribution logic is unit-testable without the JAX/NumPyro stack.
"""
from __future__ import annotations

import numpy as np
import polars as pl

MAX_STAFF = 12  # max pitchers per side per game


def fit_hook_dists(train_pa: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Empirical PAs-faced distributions for starters and relievers.

    Per (game, half) the pitcher with the earliest at-bat is the starter;
    everyone else is a reliever. Returns (starter_pas, reliever_pas) arrays
    to sample hook thresholds from.
    """
    g = (
        train_pa.group_by(["game_pk", "half_bin", "pitcher_id"])
        .agg(pl.len().alias("n"), pl.col("at_bat_number").min().alias("fab"))
        .with_columns(
            pl.col("fab").rank("ordinal").over(["game_pk", "half_bin"]).alias("rk")
        )
    )
    starters = g.filter(pl.col("rk") == 1)["n"].to_numpy().astype(np.int64)
    relievers = g.filter(pl.col("rk") > 1)["n"].to_numpy().astype(np.int64)
    return starters, relievers


def extract_games(test_pa: pl.DataFrame, id_to_idx: dict, park_map: dict | None = None) -> list[dict]:
    """Per game: lineups (9 batter idx each), pitching staffs in appearance
    order, park idx.

    The staff a lineup FACES belongs to the fielding team: half_bin 0 (top,
    away batting) is pitched by the HOME staff and vice versa. (The v1
    extractor tagged these crossed, so every lineup faced its own starter.)

    Unknown players (not in 2015-22 training) map to index 0 to keep full
    game coverage; index 0 is excluded from player-stat REPORTING so its
    stats aren't corrupted. (Requiring 9 known starters skipped ~63% of
    2023-24 games and biased the sample.)
    """
    bcol = "batter_id" if "batter_id" in test_pa.columns else "batter_idx"
    pcol = "pitcher_id" if "pitcher_id" in test_pa.columns else "pitcher_idx"
    games = []
    df = test_pa.sort(["game_pk", "at_bat_number"])
    for gid, gdf in df.group_by("game_pk", maintain_order=True):
        rec = {}
        ok = True
        for half, batting, fielding in [(0, "away", "home"), (1, "home", "away")]:
            h = gdf.filter(pl.col("half_bin") == half)
            if len(h) == 0:
                ok = False
                break
            bats = [id_to_idx.get(int(b), 0) for b in h[bcol].to_list()]
            seen, lineup = set(), []
            for b in bats:
                if b not in seen:
                    seen.add(b)
                    lineup.append(b)
                if len(lineup) == 9:
                    break
            while len(lineup) < 9:
                lineup.append(lineup[0] if lineup else 0)
            rec[f"{batting}_lineup"] = lineup

            staff, pseen = [], set()
            for p in h[pcol].to_list():
                p = int(p)
                if p not in pseen:
                    pseen.add(p)
                    staff.append(id_to_idx.get(p, 0))
                if len(staff) == MAX_STAFF:
                    break
            rec[f"{fielding}_staff"] = staff
        if not ok:
            continue
        park = 0
        if park_map is not None and "park_id" in gdf.columns:
            pid = gdf["park_id"].to_list()[0]
            if pid is not None:
                park = park_map.get(pid, 0)
        rec["park"] = park
        games.append(rec)
    return games


def cap_walkoff_runs(
    bat_score: np.ndarray,
    fld_score: np.ndarray,
    runs: np.ndarray,
    is_hr: np.ndarray,
) -> np.ndarray:
    """Cap sampled runs on walk-off plays.

    The game ends the moment the winning run scores; on non-HR plays only the
    winning run counts (MLB rule 7.01(g)(3)). Home runs count in full.
    """
    would = bat_score + runs
    nonhr_walkoff = (would > fld_score) & ~is_hr
    return np.where(nonhr_walkoff, fld_score + 1 - bat_score, runs)


def pad_staffs(games: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    """(G, MAX_STAFF) staff idx padded with the last pitcher + (G,) lengths."""
    G = len(games)
    out = np.zeros((G, MAX_STAFF), dtype=np.int64)
    lens = np.zeros(G, dtype=np.int64)
    for i, g in enumerate(games):
        s = g[key] or [0]
        lens[i] = len(s)
        out[i, : len(s)] = s
        out[i, len(s):] = s[-1]
    return out, lens
