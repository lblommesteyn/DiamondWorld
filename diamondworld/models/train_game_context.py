"""Train the Phase 4 GameContextTransformer.

The key difference from train_no_memory.py: each pitch sample includes a
variable-length prefix of completed PAs from the same game (game history).
We build this history once during dataset init (one sorted pass over data)
and look it up per sample at __getitem__ time using a pre-sorted index.

Usage:
    python -m diamondworld.models.train_game_context
    python -m diamondworld.models.train_game_context --epochs 50 --device cuda
"""
from __future__ import annotations

import argparse
import bisect
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from diamondworld.baselines.base import PA_OUTCOME_IDX, PA_OUTCOMES
from diamondworld.models.game_context_transformer import (
    MAX_HISTORY,
    _N_OUTCOMES,
    GameContextTransformer,
    encode_pa_token_floats,
    encode_pa_token_ints,
)
from diamondworld.models.no_memory_mlp import TemperatureScaler
from diamondworld.models.registry import PlayerRegistry

_DATA_ROOT = Path("/scratch/lblommes/diamondworld/data/processed")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GameSequenceDataset(Dataset):
    """Pitch-level dataset augmented with per-game PA history.

    During init, pitches are sorted by (game_pk, at_bat_number, pitch_number)
    and terminal PAs are indexed per game so each __getitem__ call can quickly
    retrieve the prefix history for that pitch's PA.
    """

    def __init__(
        self,
        pitches: pl.DataFrame,
        registry: PlayerRegistry,
        max_history: int = MAX_HISTORY,
    ) -> None:
        self._registry = registry
        self._max_history = max_history

        # Sort: all pitches ordered within each game
        pitches_s = pitches.sort(["game_pk", "at_bat_number", "pitch_number"])
        self._rows: list[dict] = pitches_s.to_dicts()

        # Build game histories: game_pk → (sorted at_bat_numbers, list of (ints, floats))
        self._game_ab_nums: dict[Any, list[int]] = {}
        self._game_tokens: dict[Any, list[tuple[list[int], list[float]]]] = {}

        for row in self._rows:
            if row.get("pa_terminal"):
                gp = row.get("game_pk")
                ab = int(row.get("at_bat_number") or 0)
                ints = encode_pa_token_ints(row)
                floats = encode_pa_token_floats(row)
                if gp not in self._game_ab_nums:
                    self._game_ab_nums[gp] = []
                    self._game_tokens[gp] = []
                self._game_ab_nums[gp].append(ab)
                self._game_tokens[gp].append((ints, floats))

    def _get_history(self, game_pk: Any, ab_num: int) -> list[tuple[list[int], list[float]]]:
        """Return the last max_history terminal PAs strictly before ab_num."""
        ab_nums = self._game_ab_nums.get(game_pk)
        if not ab_nums:
            return []
        cutoff = bisect.bisect_left(ab_nums, ab_num)
        start = max(0, cutoff - self._max_history)
        return self._game_tokens[game_pk][start:cutoff]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._rows[idx]
        reg = self._registry

        def _f(v, default=0.0):
            return float(v) if v is not None else default

        def _i(v, default=0):
            return int(v) if v is not None else default

        # Targets
        pa_outcome_raw = row.get("pa_outcome")
        pa_outcome_target = PA_OUTCOME_IDX.get(pa_outcome_raw, -1) if pa_outcome_raw else -1
        runs_raw = row.get("runs_scored")
        runs_scored_target = min(int(runs_raw), 4) if runs_raw is not None else -1

        stand_enc = -1.0 if row.get("stand") == "L" else 1.0
        p_throws_enc = -1.0 if row.get("p_throws") == "L" else 1.0
        half_enc = 1.0 if row.get("half") == "bot" else 0.0

        plate_x = _f(row.get("plate_x"), 0.0)
        plate_z = _f(row.get("plate_z"), 0.0)
        release_speed_norm = _f(row.get("release_speed"), 92.0)
        pfx_x = _f(row.get("pfx_x"), 0.0)
        pfx_z = _f(row.get("pfx_z"), 0.0)

        raw_ls = row.get("launch_speed")
        raw_la = row.get("launch_angle")
        launch_speed_norm = (_f(raw_ls, 80.0) - 80.0) / 30.0 if raw_ls is not None else 0.0
        launch_angle_norm = _f(raw_la, 0.0) / 45.0 if raw_la is not None else 0.0
        score_diff_norm = float(np.clip(_i(row.get("score_diff"), 0), -10, 10))

        # Game history for this pitch
        game_pk = row.get("game_pk")
        ab_num = _i(row.get("at_bat_number"), 0)
        history = self._get_history(game_pk, ab_num)

        return {
            # Integer inputs (same as PitchDataset)
            "pitcher_id": reg.pitcher(row.get("pitcher_id")),
            "batter_id": reg.batter(row.get("batter_id")),
            "umpire_id": reg.umpire(row.get("umpire_id")),
            "park_id": reg.park(row.get("park_id")),
            "pitch_type_idx": reg.pitch_type(row.get("pitch_type")),
            "balls": _i(row.get("balls"), 0),
            "strikes": _i(row.get("strikes"), 0),
            "outs": _i(row.get("outs"), 0),
            "base_state": _i(row.get("base_state"), 0),
            "inning": _i(row.get("inning"), 1),
            "tto": _i(row.get("tto"), 1),
            "pitch_count_game": _i(row.get("pitch_count_game"), 0),
            "pitch_count_inning": _i(row.get("pitch_count_inning"), 0),
            "runs_scored_target": runs_scored_target,
            "pa_outcome_target": pa_outcome_target,
            # Float inputs
            "plate_x": plate_x,
            "plate_z": plate_z,
            "release_speed_norm": release_speed_norm,
            "pfx_x": pfx_x,
            "pfx_z": pfx_z,
            "score_diff_norm": score_diff_norm,
            "half_enc": half_enc,
            "stand_enc": stand_enc,
            "p_throws_enc": p_throws_enc,
            "tracking_era": float(_i(row.get("tracking_era"), 0)),
            "launch_speed_norm": launch_speed_norm,
            "launch_angle_norm": launch_angle_norm,
            # Bool inputs
            "swing": bool(row.get("swing", False)),
            "contact": bool(row.get("contact", False)),
            "foul": bool(row.get("foul", False)),
            "in_play": bool(row.get("in_play", False)),
            "pa_terminal": bool(row.get("pa_terminal", False)),
            # Game history
            "hist_ints": [t[0] for t in history],    # list of [3] int lists
            "hist_floats": [t[1] for t in history],  # list of [5] float lists
            "hist_len": len(history),
        }


def game_sequence_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Tensor]:
    """Collate variable-length game histories + fixed pitch features into tensors."""
    int_keys = [
        "pitcher_id", "batter_id", "umpire_id", "park_id", "pitch_type_idx",
        "balls", "strikes", "outs", "base_state", "inning", "tto",
        "pitch_count_game", "pitch_count_inning", "runs_scored_target", "pa_outcome_target",
    ]
    float_keys = [
        "plate_x", "plate_z", "release_speed_norm", "pfx_x", "pfx_z",
        "score_diff_norm", "half_enc", "stand_enc", "p_throws_enc",
        "tracking_era", "launch_speed_norm", "launch_angle_norm",
    ]
    bool_keys = ["swing", "contact", "foul", "in_play", "pa_terminal"]

    result: dict[str, Tensor] = {}
    for k in int_keys:
        result[k] = torch.tensor([s[k] for s in batch], dtype=torch.long)
    for k in float_keys:
        result[k] = torch.tensor([s[k] for s in batch], dtype=torch.float32)
    for k in bool_keys:
        result[k] = torch.tensor([s[k] for s in batch], dtype=torch.bool)

    # Game history tensors
    B = len(batch)
    max_hist = max(s["hist_len"] for s in batch)
    max_hist = max(max_hist, 1)  # need at least 1 for valid tensor shape

    hist_outcome = torch.full((B, max_hist), _N_OUTCOMES, dtype=torch.long)  # padding_idx
    hist_bs_before = torch.zeros(B, max_hist, dtype=torch.long)
    hist_bs_after = torch.zeros(B, max_hist, dtype=torch.long)
    hist_float = torch.zeros(B, max_hist, 5, dtype=torch.float32)
    hist_len = torch.tensor([s["hist_len"] for s in batch], dtype=torch.long)

    for i, sample in enumerate(batch):
        hl = sample["hist_len"]
        if hl == 0:
            continue
        for j, (ints, floats) in enumerate(zip(sample["hist_ints"], sample["hist_floats"])):
            hist_outcome[i, j] = ints[0]
            hist_bs_before[i, j] = ints[1]
            hist_bs_after[i, j] = ints[2]
            hist_float[i, j] = torch.tensor(floats, dtype=torch.float32)

    result["hist_outcome"] = hist_outcome
    result["hist_bs_before"] = hist_bs_before
    result["hist_bs_after"] = hist_bs_after
    result["hist_float"] = hist_float
    result["hist_len"] = hist_len
    return result


# ---------------------------------------------------------------------------
# Training helpers (same logic as train_no_memory.py)
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device: str) -> dict[str, float]:
    model.train()
    agg: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(batch)
        loss, loss_dict = model.compute_loss(batch, outputs)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        for k, v in loss_dict.items():
            agg[k] = agg.get(k, 0.0) + v
        count += 1
    return {k: v / max(count, 1) for k, v in agg.items()}


def eval_epoch(model, loader, device: str) -> dict[str, float]:
    model.eval()
    agg: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch)
            _, loss_dict = model.compute_loss(batch, outputs)
            for k, v in loss_dict.items():
                agg[k] = agg.get(k, 0.0) + v
            count += 1
    return {k: v / max(count, 1) for k, v in agg.items()}


def calibrate_temperature(model, scaler, val_loader, device: str) -> TemperatureScaler:
    model.eval()
    scaler = scaler.to(device)
    optimizer = torch.optim.LBFGS(scaler.parameters(), lr=0.01, max_iter=50)

    all_pa_logits, all_pa_targets = [], []
    all_runs_logits, all_runs_targets = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch)
            pa_mask = batch["pa_terminal"] & (batch["pa_outcome_target"] >= 0)
            runs_mask = batch["pa_terminal"] & (batch["runs_scored_target"] >= 0)
            if pa_mask.any():
                all_pa_logits.append(outputs["pa_outcome"][pa_mask].cpu())
                all_pa_targets.append(batch["pa_outcome_target"][pa_mask].cpu())
            if runs_mask.any():
                all_runs_logits.append(outputs["runs_scored"][runs_mask].cpu())
                all_runs_targets.append(batch["runs_scored_target"][runs_mask].cpu())

    if not all_pa_logits:
        return scaler

    pa_logits_all = torch.cat(all_pa_logits).to(device)
    pa_targets_all = torch.cat(all_pa_targets).to(device)
    runs_logits_all = torch.cat(all_runs_logits).to(device) if all_runs_logits else None
    runs_targets_all = torch.cat(all_runs_targets).to(device) if all_runs_logits else None

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(pa_logits_all / scaler.pa_outcome_temp, pa_targets_all)
        if runs_logits_all is not None:
            loss = loss + F.cross_entropy(runs_logits_all / scaler.runs_scored_temp, runs_targets_all)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GameContextTransformer.")
    parser.add_argument("--train-seasons", nargs="+", type=int, default=list(range(2015, 2022)))
    parser.add_argument("--val-season", type=int, default=2022)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/game_context_transformer"))
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading training data...")
    train_frames = [
        pl.read_parquet(_DATA_ROOT / f"pitches_{s}.parquet") for s in args.train_seasons
    ]
    train_df = pl.concat(train_frames)
    print(f"  {len(train_df):,} train pitches across seasons {args.train_seasons}")

    registry = PlayerRegistry()
    registry.fit(train_df)
    registry_path = args.output_dir / "registry.json"
    registry.save(registry_path)
    print(f"  Registry: {registry.n_pitchers} pitchers, {registry.n_batters} batters, "
          f"{registry.n_umpires} umpires, {registry.n_parks} parks")

    print("Building GameSequenceDataset (sorting + indexing game histories)...")
    train_dataset = GameSequenceDataset(train_df, registry)
    del train_df

    val_df = pl.read_parquet(_DATA_ROOT / f"pitches_{args.val_season}.parquet")
    print(f"  {len(val_df):,} val pitches (season {args.val_season})")
    val_dataset = GameSequenceDataset(val_df, registry)
    del val_df

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=game_sequence_collate_fn, num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        collate_fn=game_sequence_collate_fn, num_workers=2,
        pin_memory=(args.device == "cuda"),
    )

    model = GameContextTransformer(registry).to(args.device)
    scaler = TemperatureScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    best_val_nll = float("inf")
    no_improve = 0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, args.device)
        val_metrics = eval_epoch(model, val_loader, args.device)
        scheduler.step()

        val_nll = val_metrics.get("pa_outcome", float("inf"))
        print(
            f"Epoch {epoch:3d} | train_pa_nll={train_metrics.get('pa_outcome', 0):.4f} "
            f"| val_pa_nll={val_nll:.4f} | val_total={val_metrics.get('total', 0):.4f}"
        )

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "registry_path": str(registry_path),
                "epoch": epoch,
                "val_metrics": val_metrics,
            }, args.output_dir / "best_model.pt")
            print(f"  --> New best (epoch {epoch}, val_pa_nll={val_nll:.4f})")
        else:
            no_improve += 1
            if no_improve >= 10:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Calibrating temperature on best model (epoch {best_epoch})...")
    best_ckpt = torch.load(args.output_dir / "best_model.pt", map_location=args.device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    scaler = calibrate_temperature(model, scaler, val_loader, args.device)

    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "registry_path": str(registry_path),
        "epoch": best_epoch,
        "val_metrics": best_ckpt["val_metrics"],
        "best_val_pa_nll": best_val_nll,
    }, args.output_dir / "final_model.pt")
    print(f"Done. Best val PA NLL: {best_val_nll:.4f}")


if __name__ == "__main__":
    main()
