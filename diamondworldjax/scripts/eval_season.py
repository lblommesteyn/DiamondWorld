"""Season-level evaluation from independent, fully generated game seasons."""
from __future__ import annotations

import argparse
import pickle
import time
from functools import partial
from pathlib import Path

import numpy as np
import polars as pl

from diamondworldjax.paths import processed_root, checkpoints_root
from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.sim.game_evaluation import simulate_game_draws
from diamondworldjax.sim.game_extract import extract_games, fit_hook_dists
from diamondworldjax.sim.rules_engine import EmpiricalEngine
from diamondworldjax.scripts.train_pa import _build_player_table, _build_park_index, apply_park_idx

TRAIN_SEASONS = list(range(2015, 2023))
TEST_SEASONS = [2023, 2024]


def _real_team_games(test_pa, game_pks: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return real team-game runs/hits in the same game order as the simulator."""
    with_hit = test_pa.with_columns(
        pl.col("pa_outcome").is_in(["1B", "2B", "3B", "HR"]).cast(pl.Int32).alias("_hit")
    )
    grouped = with_hit.group_by(["game_pk", "half_bin"]).agg(
        pl.col("runs_scored").sum().alias("runs"), pl.col("_hit").sum().alias("hits")
    )
    values: dict[tuple[int, int], tuple[float, float]] = {
        (int(row["game_pk"]), int(row["half_bin"])): (float(row["runs"]), float(row["hits"]))
        for row in grouped.iter_rows(named=True)
    }
    runs, hits = [], []
    for game_pk in game_pks:
        for half in (0, 1):
            run, hit = values.get((game_pk, half), (0.0, 0.0))
            runs.append(run)
            hits.append(hit)
    return np.asarray(runs), np.asarray(hits)


def _season_metrics(runs: np.ndarray, hits: np.ndarray) -> dict[str, float]:
    return {
        "team_runs": float(runs.mean()),
        "shutout%": float(np.mean(runs == 0) * 100),
        "nohit%": float(np.mean(hits == 0) * 100),
        "blowout%": float(np.mean(runs >= 10) * 100),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path,
                        default=checkpoints_root() / "dwjax_pa_v6" / "dwjax_step_0050000.pkl")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seasons", type=int, default=20,
                        help="Number of independent alternate histories.")
    parser.add_argument("--limit-games", type=int, default=0)
    parser.add_argument("--outcome-only", action="store_true")
    parser.add_argument("--fatigue", action="store_true")
    parser.add_argument("--platoon", action="store_true")
    parser.add_argument("--pitchformer", action="store_true")
    parser.add_argument("--pitchformer-dim", type=int, default=128)
    parser.add_argument("--pitchformer-layers", type=int, default=2)
    parser.add_argument("--pitchformer-heads", type=int, default=4)
    parser.add_argument("--pitchformer-dropout", type=float, default=0.0)
    parser.add_argument("--use-park", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    from diamondworldjax.model.pa_model import pa_model

    with open(args.ckpt, "rb") as f:
        params = pickle.load(f)["params"]
    train = load_seasons(TRAIN_SEASONS, data_root=processed_root())
    ptab = _build_player_table(train)
    park_map = _build_park_index(train)
    train_pa = train.filter(pl.col("pa_terminal"))
    engine, hooks = EmpiricalEngine().fit(train_pa), fit_hook_dists(train_pa)
    del train, train_pa

    test_pa = load_seasons(TEST_SEASONS, data_root=processed_root()).filter(pl.col("pa_terminal"))
    if args.limit_games > 0:
        keep = test_pa["game_pk"].unique().sort().to_numpy()[:args.limit_games]
        test_pa = test_pa.filter(pl.col("game_pk").is_in(keep.tolist()))
    if args.use_park:
        test_pa = apply_park_idx(test_pa, park_map)
    games = extract_games(test_pa, ptab["id_to_idx"], park_map=park_map if args.use_park else None,
                          unknown_idx=ptab["unknown_index"])
    if not games:
        raise SystemExit("No complete games could be extracted from the requested data.")
    real_runs, real_hits = _real_team_games(test_pa, [game["game_pk"] for game in games])
    real_metrics = _season_metrics(real_runs, real_hits)

    pt = {
        "stats": jnp.array(ptab["stats"]), "league": jnp.array(ptab["league"]),
        "hand": jnp.array(ptab["hand"]), "unknown_index": ptab["unknown_index"],
        "bat_hand": np.asarray(ptab["bat_hand"]), "pit_hand": np.asarray(ptab["pit_hand"]),
        "_engine": engine, "_hook_dists": hooks,
    }
    model_kwargs = {name: True for name, value in {
        "outcome_only": args.outcome_only, "fatigue": args.fatigue, "platoon": args.platoon,
    }.items() if value}
    if args.pitchformer:
        model_kwargs.update(
            pitchformer=True,
            pitchformer_dim=args.pitchformer_dim,
            pitchformer_layers=args.pitchformer_layers,
            pitchformer_heads=args.pitchformer_heads,
            pitchformer_dropout=args.pitchformer_dropout,
        )
    model_fn = partial(pa_model, **model_kwargs) if model_kwargs else pa_model
    chunks = [games[i:i + args.batch] for i in range(0, len(games), args.batch)]

    season_metrics = {name: [] for name in real_metrics}
    master_key = jax.random.PRNGKey(args.seed)
    t0 = time.time()
    for season in range(args.seasons):
        all_away, all_home, all_away_hits, all_home_hits = [], [], [], []
        for batch_number, game_chunk in enumerate(chunks):
            draws = simulate_game_draws(
                model_fn, params, pt, game_chunk,
                jax.random.fold_in(master_key, season * len(chunks) + batch_number), 1,
                seed=args.seed + season * len(chunks) + batch_number,
                platoon=args.platoon,
                pitchformer=args.pitchformer,
            )
            all_away.append(draws["away"][0]); all_home.append(draws["home"][0])
            all_away_hits.append(draws["away_hits"][0]); all_home_hits.append(draws["home_hits"][0])
        runs = np.concatenate([np.concatenate(all_away), np.concatenate(all_home)])
        hits = np.concatenate([np.concatenate(all_away_hits), np.concatenate(all_home_hits)])
        metrics = _season_metrics(runs, hits)
        for name, value in metrics.items():
            season_metrics[name].append(value)
        print(f"  season {season + 1}/{args.seasons} elapsed={time.time() - t0:.0f}s", flush=True)

    print(f"\n=== Season-level generated structure ({args.seasons} alternate histories) ===", flush=True)
    print(f"  {'metric':10s} {'real':>8s} {'sim_mean':>9s} {'sim_std':>8s} {'in_range':>9s}", flush=True)
    for name, real_value in real_metrics.items():
        values = np.asarray(season_metrics[name])
        low, high = values.mean() - 2 * values.std(), values.mean() + 2 * values.std()
        in_range = "yes" if low <= real_value <= high else "NO"
        print(f"  {name:10s} {real_value:8.3f} {values.mean():9.3f} {values.std():8.3f} {in_range:>9s}", flush=True)


if __name__ == "__main__":
    main()
