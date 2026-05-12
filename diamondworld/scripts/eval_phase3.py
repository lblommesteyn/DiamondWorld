"""Evaluate the Phase 3 no-memory MLP and print full comparison table."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from diamondworld.baselines.base import game_logs_to_frame
from diamondworld.eval.metrics import evaluate_game_logs
from diamondworld.models.no_memory_mlp import NoMemoryMLP, TemperatureScaler
from diamondworld.models.registry import PlayerRegistry
from diamondworld.simulate.game_simulator import GameSimulator
from diamondworld.simulate.pa_simulator import PASimulator, PitchSampler

_DATA_ROOT = Path("/scratch/lblommes/diamondworld/data/processed")
_CKPT_DIR = Path("/scratch/lblommes/diamondworld/checkpoints")
_RESULTS_DIR = Path("/scratch/lblommes/diamondworld/eval/results")

TEST_SEASONS = [2023, 2024]
# Use 2020-2022 for pitch sampling: recent enough to match test-era pitch mix
SAMPLER_SEASONS = [2020, 2021, 2022]


def load_model(path: Path, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    registry = PlayerRegistry.load(Path(ckpt["registry_path"]))
    model = NoMemoryMLP(registry).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    scaler = TemperatureScaler().to(device)
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    print(f"  Loaded epoch {ckpt['epoch']}, best val PA NLL = {ckpt['best_val_pa_nll']:.4f}")
    return model, scaler, registry


def main() -> None:
    print("Loading model...")
    model, scaler, registry = load_model(_CKPT_DIR / "no_memory_mlp" / "final_model.pt")

    print("Loading transition table + HBP rate from B0 checkpoint...")
    with open(_CKPT_DIR / "baselines" / "B0_markov_re24.pkl", "rb") as f:
        b0 = pickle.load(f)
    transition_table = b0._transitions
    hbp_rate = b0._hbp_rate
    print(f"  HBP rate: {hbp_rate:.4f}")

    print(f"Building pitch sampler from {SAMPLER_SEASONS} (~1.7M pitches)...")
    sampler_df = pl.concat([
        pl.read_parquet(_DATA_ROOT / f"pitches_{yr}.parquet") for yr in SAMPLER_SEASONS
    ])
    pitch_sampler = PitchSampler()
    pitch_sampler.fit(sampler_df, registry)
    del sampler_df
    print(f"  Done. {sum(len(v) for v in pitch_sampler._pitcher_pool.values()):,} pitch features indexed.")

    print("Loading test data (2023-2024)...")
    test_df = pl.concat([
        pl.read_parquet(_DATA_ROOT / f"pitches_{yr}.parquet") for yr in TEST_SEASONS
    ])
    print(f"  {len(test_df):,} pitches.")

    # All known player indices (0 = UNK, skip it)
    pitcher_pool = list(range(1, registry.n_pitchers))
    batter_pool = list(range(1, registry.n_batters))
    umpire_pool = list(range(1, registry.n_umpires))
    park_idx = 1  # fixed park; park embedding will be the same for all games

    pa_sim = PASimulator(
        model=model, scaler=scaler, registry=registry,
        transition_table=transition_table, hbp_rate=hbp_rate, device="cpu",
    )
    game_sim = GameSimulator(
        pa_sim=pa_sim, pitch_sampler=pitch_sampler,
        transition_table=transition_table,
        rng=np.random.default_rng(42),
    )

    print("Simulating 2430 games (pitch-by-pitch)...")
    logs = []
    for i in tqdm(range(2430), unit="game"):
        logs.append(game_sim.simulate_game(
            pitcher_pool, batter_pool, umpire_pool, park_idx, game_id=i,
        ))
    sim_df = game_logs_to_frame(logs)

    print("Evaluating...")
    report = evaluate_game_logs(test_df, sim_df, skip_pa_length_kl=True)
    metrics = report.as_dict()

    out_path = _RESULTS_DIR / "phase3_no_memory_mlp.json"
    out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Saved → {out_path}")

    # ---- Full comparison table ----
    all_results: dict[str, dict] = {}
    for p in sorted(_RESULTS_DIR.glob("baselines_*.json")):
        all_results[p.stem.replace("baselines_", "")] = json.loads(p.read_text())
    all_results["Phase3_MLP"] = metrics

    headers = [
        "name", "kl_run_distribution", "wasserstein_runs", "mean_rg_error",
        "variance_error", "p0_error", "p5_plus_error", "p8_plus_error", "crooked_kl",
    ]
    W = 16
    print("\n=== FULL RESULTS TABLE ===")
    print(" | ".join(h.ljust(W) for h in headers))
    print("-" * (W * len(headers) + 3 * (len(headers) - 1)))
    for name, m in all_results.items():
        row = [name] + [f"{m.get(h, float('nan')):.5f}" for h in headers[1:]]
        print(" | ".join(v.ljust(W) for v in row))


if __name__ == "__main__":
    main()
