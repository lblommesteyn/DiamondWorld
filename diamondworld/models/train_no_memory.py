from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from diamondworld.baselines.base import PA_OUTCOMES, PA_OUTCOME_IDX
from diamondworld.models.no_memory_mlp import NoMemoryMLP, TemperatureScaler
from diamondworld.models.registry import PlayerRegistry

_DATA_ROOT = Path("/scratch/lblommes/diamondworld/data/processed")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PitchDataset(Dataset):
    """Torch Dataset wrapping a Polars DataFrame of pitch rows."""

    def __init__(self, pitches: pl.DataFrame, registry: PlayerRegistry) -> None:
        self._rows = pitches.to_dicts()
        self._registry = registry

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._rows[idx]
        reg = self._registry

        def _f(v, default=0.0):
            return float(v) if v is not None else default

        def _i(v, default=0):
            return int(v) if v is not None else default

        # PA outcome: -1 if null (masked)
        pa_outcome_raw = row.get("pa_outcome")
        pa_outcome_target = PA_OUTCOME_IDX.get(pa_outcome_raw, -1) if pa_outcome_raw else -1

        # Runs scored: min(v, 4), -1 if null
        runs_raw = row.get("runs_scored")
        runs_scored_target = min(int(runs_raw), 4) if runs_raw is not None else -1

        # Stand/p_throws encoding: L=-1, R=1
        stand_enc = -1.0 if row.get("stand") == "L" else 1.0
        p_throws_enc = -1.0 if row.get("p_throws") == "L" else 1.0

        # Half encoding: top=0, bot=1
        half_enc = 1.0 if row.get("half") == "bot" else 0.0

        # Normalized continuous features (nulls -> 0.0; masks handle missing)
        plate_x = _f(row.get("plate_x"), 0.0)
        plate_z = _f(row.get("plate_z"), 0.0)
        release_speed_norm = _f(row.get("release_speed"), 92.0)
        pfx_x = _f(row.get("pfx_x"), 0.0)
        pfx_z = _f(row.get("pfx_z"), 0.0)
        # Normalize targets to match model prediction space
        # launch_speed: subtract 80 / 30; launch_angle: / 45
        raw_ls = row.get("launch_speed")
        raw_la = row.get("launch_angle")
        launch_speed_norm = (_f(raw_ls, 80.0) - 80.0) / 30.0 if raw_ls is not None else 0.0
        launch_angle_norm = _f(raw_la, 0.0) / 45.0 if raw_la is not None else 0.0
        score_diff_norm = float(np.clip(_i(row.get("score_diff"), 0), -10, 10))

        return {
            # Integer (Long) inputs
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
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Tensor]:
    """Collate a list of sample dicts into batched tensors."""
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
    return result


def build_dataset(
    parquet_paths: list[Path], registry: PlayerRegistry
) -> PitchDataset:
    """Load parquet files and build a PitchDataset."""
    frames = [pl.read_parquet(p) for p in parquet_paths]
    pitches = pl.concat(frames)
    return PitchDataset(pitches, registry)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: NoMemoryMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> dict[str, float]:
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


def eval_epoch(
    model: NoMemoryMLP,
    loader: DataLoader,
    device: str,
) -> dict[str, float]:
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


def calibrate_temperature(
    model: NoMemoryMLP,
    scaler: TemperatureScaler,
    val_loader: DataLoader,
    device: str,
) -> TemperatureScaler:
    """Optimize temperature parameters on the validation set."""
    model.eval()
    scaler = scaler.to(device)
    optimizer = torch.optim.LBFGS(scaler.parameters(), lr=0.01, max_iter=50)

    # Collect all pa_terminal logits and targets
    all_pa_logits: list[Tensor] = []
    all_pa_targets: list[Tensor] = []
    all_runs_logits: list[Tensor] = []
    all_runs_targets: list[Tensor] = []

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

    if all_pa_logits:
        pa_logits_all = torch.cat(all_pa_logits).to(device)
        pa_targets_all = torch.cat(all_pa_targets).to(device)
    else:
        return scaler

    if all_runs_logits:
        runs_logits_all = torch.cat(all_runs_logits).to(device)
        runs_targets_all = torch.cat(all_runs_targets).to(device)
    else:
        runs_logits_all = None
        runs_targets_all = None

    def closure():
        optimizer.zero_grad()
        pa_loss = F.cross_entropy(pa_logits_all / scaler.pa_outcome_temp, pa_targets_all)
        loss = pa_loss
        if runs_logits_all is not None:
            runs_loss = F.cross_entropy(
                runs_logits_all / scaler.runs_scored_temp, runs_targets_all
            )
            loss = loss + runs_loss
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler


# ---------------------------------------------------------------------------
# Main training script
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NoMemoryMLP on DiamondWorld pitch data.")
    parser.add_argument(
        "--train-seasons", nargs="+", type=int,
        default=list(range(2015, 2022)),
        help="Seasons to use for training (default: 2015-2021)",
    )
    parser.add_argument("--val-season", type=int, default=2022)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("checkpoints/no_memory_mlp"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load training data for registry
    print("Loading training data for registry...")
    train_paths = [_DATA_ROOT / f"pitches_{s}.parquet" for s in args.train_seasons]
    val_path = _DATA_ROOT / f"pitches_{args.val_season}.parquet"

    train_frames = [pl.read_parquet(p) for p in train_paths]
    train_df = pl.concat(train_frames)

    registry = PlayerRegistry()
    registry.fit(train_df)
    registry_path = args.output_dir / "registry.json"
    registry.save(registry_path)
    print(f"Registry: {registry.n_pitchers} pitchers, {registry.n_batters} batters")

    # Build datasets
    print("Building datasets...")
    train_dataset = PitchDataset(train_df, registry)
    del train_df  # free memory

    val_df = pl.read_parquet(val_path)
    val_dataset = PitchDataset(val_df, registry)
    del val_df

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    # Build model
    model = NoMemoryMLP(registry).to(args.device)
    scaler = TemperatureScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

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
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "registry_path": str(registry_path),
                "epoch": epoch,
                "val_metrics": val_metrics,
            }
            torch.save(checkpoint, args.output_dir / "best_model.pt")
            print(f"  --> New best model saved (epoch {epoch})")
        else:
            no_improve += 1
            if no_improve >= 10:
                print(f"Early stopping at epoch {epoch} (no improvement for 10 epochs)")
                break

    # Calibrate temperature on validation set using best model
    print(f"Calibrating temperature using best model (epoch {best_epoch})...")
    best_ckpt = torch.load(args.output_dir / "best_model.pt", map_location=args.device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    scaler = calibrate_temperature(model, scaler, val_loader, args.device)

    final_checkpoint = {
        "model_state_dict": model.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "registry_path": str(registry_path),
        "epoch": best_epoch,
        "val_metrics": best_ckpt["val_metrics"],
        "best_val_pa_nll": best_val_nll,
    }
    torch.save(final_checkpoint, args.output_dir / "final_model.pt")
    print(f"Training complete. Best val pa_outcome NLL: {best_val_nll:.4f}")


if __name__ == "__main__":
    main()
