from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from diamondworld.models.registry import PlayerRegistry

PA_OUTCOMES = ["K", "BB", "HBP", "1B", "2B", "3B", "HR", "out"]
PA_OUTCOME_IDX = {o: i for i, o in enumerate(PA_OUTCOMES)}


def _mlp(dims: list[int], use_layer_norm: bool = False) -> nn.Sequential:
    """Build a simple MLP with ReLU activations and optional LayerNorm."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:  # not the last layer
            if use_layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class NoMemoryMLP(nn.Module):
    """No-memory MLP world model for pitch-by-pitch baseball simulation.

    Predicts multiple targets per pitch (swing, contact, foul, launch_speed,
    launch_angle, pa_outcome, runs_scored) from context and action features.
    """

    def __init__(self, registry: PlayerRegistry) -> None:
        super().__init__()
        self.registry = registry

        # ---- Embeddings ----
        self.park_emb = nn.Embedding(registry.n_parks, 16, padding_idx=0)
        self.pitcher_emb = nn.Embedding(registry.n_pitchers, 32, padding_idx=0)
        self.batter_emb = nn.Embedding(registry.n_batters, 32, padding_idx=0)
        self.umpire_emb = nn.Embedding(registry.n_umpires, 16, padding_idx=0)
        self.pitch_type_emb = nn.Embedding(registry.n_pitch_types, 16, padding_idx=0)

        # Context MLP: 13 scalars + 16 + 32 + 32 + 16 = 109
        self.context_mlp = _mlp([109, 256, 256, 128], use_layer_norm=True)

        # Action MLP: pitch_type_emb(16) + plate_x(1) + plate_z(1) + release_speed_norm(1) + pfx_x(1) + pfx_z(1) = 21
        self.action_mlp = _mlp([21, 64, 64], use_layer_norm=False)

        # Trunk: LayerNorm(concat(h_context=128, h_action=64)) = 192, then MLP
        self.trunk_norm = nn.LayerNorm(192)
        self.trunk_mlp = _mlp([192, 192], use_layer_norm=False)

        # Output heads
        self.head_swing = nn.Linear(192, 1)
        self.head_contact = nn.Linear(192, 1)
        self.head_foul = nn.Linear(192, 1)
        self.head_launch_speed = nn.Linear(192, 1)
        self.head_launch_angle = nn.Linear(192, 1)
        self.head_pa_outcome = nn.Linear(192, 8)
        self.head_runs_scored = nn.Linear(192, 5)

        # Normalization constants (stored as buffers, not learnable)
        self.register_buffer("_release_speed_mean", torch.tensor(92.0))
        self.register_buffer("_release_speed_std", torch.tensor(4.0))
        self.register_buffer("_plate_x_scale", torch.tensor(1.5))
        self.register_buffer("_plate_z_mean", torch.tensor(2.5))
        self.register_buffer("_plate_z_std", torch.tensor(0.75))
        self.register_buffer("_pfx_scale", torch.tensor(12.0))
        self.register_buffer("_launch_speed_mean", torch.tensor(80.0))
        self.register_buffer("_launch_speed_std", torch.tensor(30.0))
        self.register_buffer("_launch_angle_scale", torch.tensor(45.0))
        self.register_buffer("_pitch_count_game_scale", torch.tensor(100.0))
        self.register_buffer("_pitch_count_inning_scale", torch.tensor(30.0))
        self.register_buffer("_score_diff_clip", torch.tensor(10.0))
        self.register_buffer("_score_diff_scale", torch.tensor(5.0))

    def _normalize_scalars(self, batch: dict[str, Tensor]) -> Tensor:
        """Normalize and concatenate the 13 scalar context features."""
        # Integer scalars normalized to float
        balls = batch["balls"].float() / 3.0
        strikes = batch["strikes"].float() / 2.0
        outs = batch["outs"].float() / 2.0
        base_state = batch["base_state"].float() / 7.0
        score_diff = torch.clamp(batch["score_diff_norm"], -self._score_diff_clip, self._score_diff_clip) / self._score_diff_scale
        inning = (batch["inning"].float() - 1.0) / 8.0
        half_enc = batch["half_enc"]  # already encoded as float
        tto = (batch["tto"].float() - 1.0) / 2.0
        pitch_count_game = batch["pitch_count_game"].float() / self._pitch_count_game_scale
        pitch_count_inning = batch["pitch_count_inning"].float() / self._pitch_count_inning_scale
        stand_enc = batch["stand_enc"]
        p_throws_enc = batch["p_throws_enc"]
        tracking_era = batch["tracking_era"].float()

        return torch.stack([
            balls, strikes, outs, base_state, score_diff, inning, half_enc,
            tto, pitch_count_game, pitch_count_inning, stand_enc, p_throws_enc, tracking_era
        ], dim=-1)  # (B, 13)

    def _normalize_action(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Normalize pitch action features; returns (pitch_type_idx, continuous_feats)."""
        plate_x = batch["plate_x"] / self._plate_x_scale
        plate_z = (batch["plate_z"] - self._plate_z_mean) / self._plate_z_std
        release_speed = (batch["release_speed_norm"] - self._release_speed_mean) / self._release_speed_std
        pfx_x = batch["pfx_x"] / self._pfx_scale
        pfx_z = batch["pfx_z"] / self._pfx_scale
        cont = torch.stack([plate_x, plate_z, release_speed, pfx_x, pfx_z], dim=-1)  # (B, 5)
        return batch["pitch_type_idx"], cont

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        # Context encoding
        scalars = self._normalize_scalars(batch)  # (B, 13)
        park_e = self.park_emb(batch["park_id"])       # (B, 16)
        pitcher_e = self.pitcher_emb(batch["pitcher_id"])  # (B, 32)
        batter_e = self.batter_emb(batch["batter_id"])    # (B, 32)
        umpire_e = self.umpire_emb(batch["umpire_id"])    # (B, 16)

        ctx_raw = torch.cat([scalars, park_e, pitcher_e, batter_e, umpire_e], dim=-1)  # (B, 109)
        h_context = self.context_mlp(ctx_raw)  # (B, 128)

        # Action encoding
        pitch_type_idx, cont_feats = self._normalize_action(batch)
        pt_e = self.pitch_type_emb(pitch_type_idx)  # (B, 16)
        action_raw = torch.cat([pt_e, cont_feats], dim=-1)  # (B, 21)
        h_action = self.action_mlp(action_raw)  # (B, 64)

        # Trunk
        h = torch.cat([h_context, h_action], dim=-1)  # (B, 192)
        h = self.trunk_norm(h)
        h = self.trunk_mlp(h)  # (B, 192)
        h = F.relu(h)

        return {
            "swing": self.head_swing(h).squeeze(-1),          # (B,)
            "contact": self.head_contact(h).squeeze(-1),      # (B,)
            "foul": self.head_foul(h).squeeze(-1),            # (B,)
            "launch_speed": self.head_launch_speed(h).squeeze(-1),  # (B,)
            "launch_angle": self.head_launch_angle(h).squeeze(-1),  # (B,)
            "pa_outcome": self.head_pa_outcome(h),             # (B, 8)
            "runs_scored": self.head_runs_scored(h),           # (B, 5)
        }

    def compute_loss(
        self, batch: dict[str, Tensor], outputs: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute multi-head loss with masking. Returns (total_loss, per_head_dict)."""
        weights = {
            "swing": 1.0,
            "contact": 1.0,
            "foul": 0.5,
            "launch_speed": 0.3,
            "launch_angle": 0.3,
            "pa_outcome": 2.0,
            "runs_scored": 1.5,
        }
        loss_dict: dict[str, float] = {}
        total = torch.tensor(0.0, device=outputs["swing"].device)

        # Swing: all pitches
        swing_target = batch["swing"].float()
        swing_loss = F.binary_cross_entropy_with_logits(outputs["swing"], swing_target)
        loss_dict["swing"] = swing_loss.item()
        total = total + weights["swing"] * swing_loss

        # Contact: only where swing==True
        contact_mask = batch["swing"]
        if contact_mask.any():
            contact_target = batch["contact"][contact_mask].float()
            contact_loss = F.binary_cross_entropy_with_logits(
                outputs["contact"][contact_mask], contact_target
            )
        else:
            contact_loss = torch.tensor(0.0, device=total.device)
        loss_dict["contact"] = contact_loss.item()
        total = total + weights["contact"] * contact_loss

        # Foul: only where contact==True
        foul_mask = batch["contact"]
        if foul_mask.any():
            foul_target = batch["foul"][foul_mask].float()
            foul_loss = F.binary_cross_entropy_with_logits(
                outputs["foul"][foul_mask], foul_target
            )
        else:
            foul_loss = torch.tensor(0.0, device=total.device)
        loss_dict["foul"] = foul_loss.item()
        total = total + weights["foul"] * foul_loss

        # Launch speed/angle: in_play==True pitches
        in_play_mask = batch["in_play"]
        if in_play_mask.any():
            ls_target = batch["launch_speed_norm"][in_play_mask]
            ls_pred = outputs["launch_speed"][in_play_mask]
            la_target = batch["launch_angle_norm"][in_play_mask]
            la_pred = outputs["launch_angle"][in_play_mask]
            launch_speed_loss = F.mse_loss(ls_pred, ls_target)
            launch_angle_loss = F.mse_loss(la_pred, la_target)
        else:
            launch_speed_loss = torch.tensor(0.0, device=total.device)
            launch_angle_loss = torch.tensor(0.0, device=total.device)
        loss_dict["launch_speed"] = launch_speed_loss.item()
        loss_dict["launch_angle"] = launch_angle_loss.item()
        total = total + weights["launch_speed"] * launch_speed_loss
        total = total + weights["launch_angle"] * launch_angle_loss

        # PA outcome: pa_terminal==True and pa_outcome_target != -1
        pa_mask = batch["pa_terminal"] & (batch["pa_outcome_target"] >= 0)
        if pa_mask.any():
            pa_logits = outputs["pa_outcome"][pa_mask]
            pa_target = batch["pa_outcome_target"][pa_mask]
            pa_loss = F.cross_entropy(pa_logits, pa_target)
        else:
            pa_loss = torch.tensor(0.0, device=total.device)
        loss_dict["pa_outcome"] = pa_loss.item()
        total = total + weights["pa_outcome"] * pa_loss

        # Runs scored: pa_terminal==True and runs_scored_target != -1
        runs_mask = batch["pa_terminal"] & (batch["runs_scored_target"] >= 0)
        if runs_mask.any():
            runs_logits = outputs["runs_scored"][runs_mask]
            runs_target = batch["runs_scored_target"][runs_mask]
            runs_loss = F.cross_entropy(runs_logits, runs_target)
        else:
            runs_loss = torch.tensor(0.0, device=total.device)
        loss_dict["runs_scored"] = runs_loss.item()
        total = total + weights["runs_scored"] * runs_loss

        loss_dict["total"] = total.item()
        return total, loss_dict


class TemperatureScaler(nn.Module):
    """Per-head temperature scaling for calibration after training."""

    def __init__(self) -> None:
        super().__init__()
        self.swing_temp = nn.Parameter(torch.ones(1))
        self.contact_temp = nn.Parameter(torch.ones(1))
        self.foul_temp = nn.Parameter(torch.ones(1))
        self.pa_outcome_temp = nn.Parameter(torch.ones(1))
        self.runs_scored_temp = nn.Parameter(torch.ones(1))

    def scale(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        return {
            "swing": outputs["swing"] / self.swing_temp,
            "contact": outputs["contact"] / self.contact_temp,
            "foul": outputs["foul"] / self.foul_temp,
            "launch_speed": outputs["launch_speed"],
            "launch_angle": outputs["launch_angle"],
            "pa_outcome": outputs["pa_outcome"] / self.pa_outcome_temp,
            "runs_scored": outputs["runs_scored"] / self.runs_scored_temp,
        }
