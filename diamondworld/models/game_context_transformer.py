"""Phase 4: GameContextTransformer — pitch-level world model with game memory.

Architecture
------------
Each pitch sees a CLS-based summary of all completed prior PAs in the game
(base states, outcomes, pitch mix, fatigue signals), enabling the model to
learn big-inning effects, fatigue, and time-through-order patterns.

Context encoder  : scalars(13) + embs(park16+pitcher32+batter32+umpire16=96)
                   → MLP(109→256→256→128, LayerNorm+ReLU) → 128-dim
Action encoder   : pitch_type_emb(16) + 5 scalars → MLP(21→64→64) → 64-dim
History encoder  : CLS + per-PA tokens (outcome_emb16 + bs_before4 + bs_after4 + cont5)
                   → hist_proj(29→64) → 2-layer TransformerEncoder → CLS → game_ctx(64)
Trunk            : LayerNorm(256) → MLP(256→256→256)
Output heads     : 7 heads (same as Phase 3) reading from 256-dim trunk
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from diamondworld.baselines.base import PA_OUTCOMES
from diamondworld.models.registry import PlayerRegistry

PA_OUTCOME_IDX = {o: i for i, o in enumerate(PA_OUTCOMES)}

# PA summary feature dimension (28 continuous/categorical + 1 base_state_after = 29 total)
PA_SUMMARY_DIM = 29

HOW_ON_ENC = {None: 0, "hit": 1, "walk": 2, "hbp": 3, "error": 4, "fc": 5}
FASTBALL_TYPES = {"FF", "SI", "FA", "FC"}
BREAKING_TYPES = {"SL", "CU", "KC", "CS", "SV", "ST"}
CHANGE_TYPES = {"CH", "FS", "SC"}

# Indices into the 29-dim PA summary for the history encoder's 5 continuous features
_HIST_CONT_IDXS = [0, 8, 12, 13, 18]  # pitch_count_game/100, runs_inning/5, (inn-1)/8, half, n_pitches/10


def _mlp(dims: list[int], use_layer_norm: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            if use_layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


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


class GameContextTransformer(nn.Module):
    """Pitch-level world model with game-history context.

    Batch keys — per pitch (B, N):
      Long:  pitcher_id, batter_id, umpire_id, park_id, pitch_type_idx,
             balls, strikes, outs, base_state, inning, tto,
             pitch_count_game, pitch_count_inning,
             runs_scored_target, pa_outcome_target
      Float: plate_x, plate_z, release_speed_norm, pfx_x, pfx_z,
             score_diff_norm, half_enc, stand_enc, p_throws_enc,
             tracking_era, launch_speed_norm, launch_angle_norm
      Bool:  swing, contact, foul, in_play, pa_terminal
      Long:  pa_idx, within_pa_pos

    Batch keys — game history:
      Float (B, M, 29): game_memory
      Bool  (B, M):     mem_pad_mask   (True = padded PA)
    """

    def __init__(
        self,
        registry: PlayerRegistry,
        hist_d_model: int = 64,
        hist_n_heads: int = 4,
        hist_n_layers: int = 2,
        hist_ffn_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.registry = registry

        # ---- Context encoder: scalars(13) + park(16) + pitcher(32) + batter(32) + umpire(16) = 109 → 128 ----
        self.park_emb = nn.Embedding(registry.n_parks, 16, padding_idx=0)
        self.pitcher_emb = nn.Embedding(registry.n_pitchers, 32, padding_idx=0)
        self.batter_emb = nn.Embedding(registry.n_batters, 32, padding_idx=0)
        self.umpire_emb = nn.Embedding(registry.n_umpires, 16, padding_idx=0)
        self.context_mlp = _mlp([109, 256, 256, 128], use_layer_norm=True)

        # ---- Action encoder: pitch_type_emb(16) + 5 scalars = 21 → 64 ----
        self.pitch_type_emb = nn.Embedding(registry.n_pitch_types, 16, padding_idx=0)
        self.action_mlp = _mlp([21, 64, 64])

        # ---- History encoder ----
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hist_d_model))
        # Discrete embeddings for per-PA outcome and base states
        self.hist_outcome_emb = nn.Embedding(9, 16)    # 8 PA outcomes + 1 unknown
        self.hist_bs_before_emb = nn.Embedding(8, 4)   # base state before PA
        self.hist_bs_after_emb = nn.Embedding(8, 4)    # base state after PA
        # hist_proj: (outcome16 + bs_before4 + bs_after4 + 5 continuous) = 29 → hist_d_model
        self.hist_proj = nn.Linear(29, hist_d_model, bias=False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hist_d_model,
            nhead=hist_n_heads,
            dim_feedforward=hist_ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=hist_n_layers)
        self.game_ctx_norm = nn.LayerNorm(hist_d_model)

        # ---- Trunk: cat(context128, action64, game_ctx64) = 256 ----
        self.trunk_norm = nn.LayerNorm(256)
        self.trunk_mlp = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256))

        # ---- Output heads ----
        self.head_swing = nn.Linear(256, 1)
        self.head_contact = nn.Linear(256, 1)
        self.head_foul = nn.Linear(256, 1)
        self.head_launch_speed = nn.Linear(256, 1)
        self.head_launch_angle = nn.Linear(256, 1)
        self.head_pa_outcome = nn.Linear(256, 8)
        self.head_runs_scored = nn.Linear(256, 5)

        # ---- Normalization buffers ----
        self.register_buffer("_release_speed_mean", torch.tensor(92.0))
        self.register_buffer("_release_speed_std", torch.tensor(4.0))
        self.register_buffer("_plate_x_scale", torch.tensor(1.5))
        self.register_buffer("_plate_z_mean", torch.tensor(2.5))
        self.register_buffer("_plate_z_std", torch.tensor(0.75))
        self.register_buffer("_pfx_scale", torch.tensor(12.0))
        self.register_buffer("_pitch_count_game_scale", torch.tensor(100.0))
        self.register_buffer("_pitch_count_inning_scale", torch.tensor(30.0))
        self.register_buffer("_score_diff_clip", torch.tensor(10.0))
        self.register_buffer("_score_diff_scale", torch.tensor(5.0))

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _normalize_scalars(self, batch: dict[str, Tensor]) -> Tensor:
        """Normalize and stack the 13 scalar context features."""
        balls = batch["balls"].float() / 3.0
        strikes = batch["strikes"].float() / 2.0
        outs = batch["outs"].float() / 2.0
        base_state = batch["base_state"].float() / 7.0
        score_diff = torch.clamp(
            batch["score_diff_norm"], -self._score_diff_clip, self._score_diff_clip
        ) / self._score_diff_scale
        inning = (batch["inning"].float() - 1.0) / 8.0
        half_enc = batch["half_enc"]
        tto = (batch["tto"].float() - 1.0) / 2.0
        pitch_count_game = batch["pitch_count_game"].float() / self._pitch_count_game_scale
        pitch_count_inning = batch["pitch_count_inning"].float() / self._pitch_count_inning_scale
        stand_enc = batch["stand_enc"]
        p_throws_enc = batch["p_throws_enc"]
        tracking_era = batch["tracking_era"].float()
        return torch.stack([
            balls, strikes, outs, base_state, score_diff, inning, half_enc,
            tto, pitch_count_game, pitch_count_inning, stand_enc, p_throws_enc, tracking_era,
        ], dim=-1)  # (..., 13)

    def _encode_context(self, batch: dict[str, Tensor]) -> Tensor:
        """Context encoder: (B, N) → (B, N, 128)."""
        scalars = self._normalize_scalars(batch)
        park_e = self.park_emb(batch["park_id"])
        pitcher_e = self.pitcher_emb(batch["pitcher_id"])
        batter_e = self.batter_emb(batch["batter_id"])
        umpire_e = self.umpire_emb(batch["umpire_id"])
        ctx_raw = torch.cat([scalars, park_e, pitcher_e, batter_e, umpire_e], dim=-1)
        return self.context_mlp(ctx_raw)

    def _encode_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Action encoder: (B, N) → (B, N, 64)."""
        plate_x = batch["plate_x"] / self._plate_x_scale
        plate_z = (batch["plate_z"] - self._plate_z_mean) / self._plate_z_std
        release_speed = (batch["release_speed_norm"] - self._release_speed_mean) / self._release_speed_std
        pfx_x = batch["pfx_x"] / self._pfx_scale
        pfx_z = batch["pfx_z"] / self._pfx_scale
        cont = torch.stack([plate_x, plate_z, release_speed, pfx_x, pfx_z], dim=-1)
        pt_e = self.pitch_type_emb(batch["pitch_type_idx"])
        action_raw = torch.cat([pt_e, cont], dim=-1)
        return self.action_mlp(action_raw)

    def _encode_history(
        self, game_memory: Tensor, mem_pad_mask: Tensor | None
    ) -> Tensor:
        """History encoder: (B, M, 29) → (B, 64) game context vector.

        PA summary layout (29 dims):
          [0]   pitch_count_game/100
          [1]   pitch_count_inning/30
          [2]   velo_delta/10
          [3]   prior_batter_reached
          [4]   prior_two_reached
          [5-7] how_on_1b/2b/3b (divided by 5)
          [8]   runs_this_inning/5
          [9]   base_state_before/7
          [10]  outs/2
          [11]  tto/3
          [12]  (inning-1)/8
          [13]  half (0/1)
          [14]  fastball_rate
          [15]  breaking_rate
          [16]  changeup_rate
          [17]  zone_rate
          [18]  n_pitches/10
          [19]  stand_enc (-1/1)
          [20:28] pa_outcome one-hot (8 dims)
          [28]  base_state_after/7
        """
        B, M, _ = game_memory.shape

        # Extract discrete indices for embedding lookup
        outcome_idx = game_memory[:, :, 20:28].argmax(-1).long().clamp(0, 8)  # (B, M)
        bs_before_idx = (game_memory[:, :, 9] * 7).round().long().clamp(0, 7)  # (B, M)
        bs_after_idx = (game_memory[:, :, 28] * 7).round().long().clamp(0, 7)  # (B, M)

        outcome_e = self.hist_outcome_emb(outcome_idx)      # (B, M, 16)
        bs_before_e = self.hist_bs_before_emb(bs_before_idx)  # (B, M, 4)
        bs_after_e = self.hist_bs_after_emb(bs_after_idx)     # (B, M, 4)

        # 5 continuous features at fixed indices
        cont_idxs = torch.tensor(_HIST_CONT_IDXS, device=game_memory.device)
        cont = game_memory[:, :, cont_idxs]  # (B, M, 5)

        hist_input = torch.cat([outcome_e, bs_before_e, bs_after_e, cont], dim=-1)  # (B, M, 29)
        hist_tokens = self.hist_proj(hist_input)  # (B, M, 64)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, 64)
        seq = torch.cat([cls, hist_tokens], dim=1)  # (B, M+1, 64)

        # Transformer encoder with key-padding mask (CLS is never padded)
        if mem_pad_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=game_memory.device)
            src_key_padding_mask = torch.cat([cls_mask, mem_pad_mask], dim=1)  # (B, M+1)
        else:
            src_key_padding_mask = None

        out = self.transformer(seq, src_key_padding_mask=src_key_padding_mask)  # (B, M+1, 64)
        game_ctx = self.game_ctx_norm(out[:, 0])  # (B, 64) — CLS output
        return game_ctx

    # ------------------------------------------------------------------

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        h_context = self._encode_context(batch)   # (B, N, 128)
        h_action = self._encode_action(batch)     # (B, N, 64)
        B, N, _ = h_context.shape

        # Encode game history
        game_memory = batch.get("game_memory")    # (B, M, 29) or None
        mem_pad_mask = batch.get("mem_pad_mask")  # (B, M) bool or None
        if game_memory is not None and game_memory.size(1) > 0:
            game_ctx = self._encode_history(game_memory, mem_pad_mask)  # (B, 64)
        else:
            game_ctx = torch.zeros(B, 64, device=h_context.device, dtype=h_context.dtype)

        # Broadcast game_ctx to all pitches and concat
        game_ctx_expanded = game_ctx.unsqueeze(1).expand(-1, N, -1)  # (B, N, 64)
        trunk_in = torch.cat([h_context, h_action, game_ctx_expanded], dim=-1)  # (B, N, 256)
        trunk_in = self.trunk_norm(trunk_in)
        trunk_out = self.trunk_mlp(trunk_in)  # (B, N, 256)

        return {
            "swing": self.head_swing(trunk_out).squeeze(-1),
            "contact": self.head_contact(trunk_out).squeeze(-1),
            "foul": self.head_foul(trunk_out).squeeze(-1),
            "launch_speed": self.head_launch_speed(trunk_out).squeeze(-1),
            "launch_angle": self.head_launch_angle(trunk_out).squeeze(-1),
            "pa_outcome": self.head_pa_outcome(trunk_out),
            "runs_scored": self.head_runs_scored(trunk_out),
        }

    def compute_loss(
        self, batch: dict[str, Tensor], outputs: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, float]]:
        """Multi-head loss with masking."""
        weights = {
            "swing": 1.0, "contact": 1.0, "foul": 0.5,
            "launch_speed": 0.3, "launch_angle": 0.3,
            "pa_outcome": 2.0, "runs_scored": 1.5,
        }
        loss_dict: dict[str, float] = {}
        total = torch.tensor(0.0, device=outputs["swing"].device)

        pad_mask = batch.get("pitch_pad_mask")

        def _valid(t: Tensor) -> Tensor:
            flat = t.reshape(-1) if t.dim() > 1 else t
            if pad_mask is not None:
                return flat[~pad_mask.reshape(-1)]
            return flat

        def _valid2d(t: Tensor) -> Tensor:
            B, N, C = t.shape
            flat = t.reshape(B * N, C)
            if pad_mask is not None:
                return flat[~pad_mask.reshape(-1)]
            return flat

        swing_pred = _valid(outputs["swing"])
        swing_tgt = _valid(batch["swing"].float())
        swing_loss = F.binary_cross_entropy_with_logits(swing_pred, swing_tgt)
        loss_dict["swing"] = swing_loss.item()
        total = total + weights["swing"] * swing_loss

        swing_mask_flat = _valid(batch["swing"])
        contact_pred = _valid(outputs["contact"])
        contact_tgt_flat = _valid(batch["contact"].float())
        if swing_mask_flat.any():
            contact_loss = F.binary_cross_entropy_with_logits(
                contact_pred[swing_mask_flat], contact_tgt_flat[swing_mask_flat]
            )
        else:
            contact_loss = torch.tensor(0.0, device=total.device)
        loss_dict["contact"] = contact_loss.item()
        total = total + weights["contact"] * contact_loss

        contact_mask_flat = _valid(batch["contact"])
        foul_pred = _valid(outputs["foul"])
        foul_tgt_flat = _valid(batch["foul"].float())
        if contact_mask_flat.any():
            foul_loss = F.binary_cross_entropy_with_logits(
                foul_pred[contact_mask_flat], foul_tgt_flat[contact_mask_flat]
            )
        else:
            foul_loss = torch.tensor(0.0, device=total.device)
        loss_dict["foul"] = foul_loss.item()
        total = total + weights["foul"] * foul_loss

        in_play_flat = _valid(batch["in_play"])
        ls_pred = _valid(outputs["launch_speed"])
        ls_tgt = _valid(batch["launch_speed_norm"])
        la_pred = _valid(outputs["launch_angle"])
        la_tgt = _valid(batch["launch_angle_norm"])
        if in_play_flat.any():
            launch_speed_loss = F.mse_loss(ls_pred[in_play_flat], ls_tgt[in_play_flat])
            launch_angle_loss = F.mse_loss(la_pred[in_play_flat], la_tgt[in_play_flat])
        else:
            launch_speed_loss = torch.tensor(0.0, device=total.device)
            launch_angle_loss = torch.tensor(0.0, device=total.device)
        loss_dict["launch_speed"] = launch_speed_loss.item()
        loss_dict["launch_angle"] = launch_angle_loss.item()
        total = total + weights["launch_speed"] * launch_speed_loss
        total = total + weights["launch_angle"] * launch_angle_loss

        pa_terminal_flat = _valid(batch["pa_terminal"])
        pa_outcome_tgt_flat = _valid(batch["pa_outcome_target"])
        pa_outcome_pred = _valid2d(outputs["pa_outcome"])
        pa_mask = pa_terminal_flat & (pa_outcome_tgt_flat >= 0)
        if pa_mask.any():
            pa_loss = F.cross_entropy(pa_outcome_pred[pa_mask], pa_outcome_tgt_flat[pa_mask])
        else:
            pa_loss = torch.tensor(0.0, device=total.device)
        loss_dict["pa_outcome"] = pa_loss.item()
        total = total + weights["pa_outcome"] * pa_loss

        runs_tgt_flat = _valid(batch["runs_scored_target"])
        runs_pred = _valid2d(outputs["runs_scored"])
        runs_mask = pa_terminal_flat & (runs_tgt_flat >= 0)
        if runs_mask.any():
            runs_loss = F.cross_entropy(runs_pred[runs_mask], runs_tgt_flat[runs_mask])
        else:
            runs_loss = torch.tensor(0.0, device=total.device)
        loss_dict["runs_scored"] = runs_loss.item()
        total = total + weights["runs_scored"] * runs_loss

        loss_dict["total"] = total.item()
        return total, loss_dict
