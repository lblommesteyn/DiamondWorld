"""Train the Phase 4 GameContextTransformer (RoPE decoder with game memory).

Usage:
    python -m diamondworld.models.train_phase4
    python -m diamondworld.models.train_phase4 --epochs 50 --batch-size 8 --device cuda
"""
from __future__ import annotations

import argparse
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
from diamondworld.models.game_context_transformer import (
    GameContextTransformer,
    TemperatureScaler,
    FASTBALL_TYPES,
    BREAKING_TYPES,
    CHANGE_TYPES,
    HOW_ON_ENC,
    PA_SUMMARY_DIM,
)
from diamondworld.models.registry import PlayerRegistry

_DATA_ROOT = Path("/scratch/lblommes/diamondworld/data/processed")


# ---------------------------------------------------------------------------
# PA Summary computation
# ---------------------------------------------------------------------------

def compute_pa_summaries(pitches: pl.DataFrame) -> list[np.ndarray]:
    """Compute PA summary vectors for all completed PAs in one game.

    pitches: all pitches from one game, sorted by (at_bat_number, pitch_number).
    Returns a list of 29-dim float32 arrays, one per completed PA in PA order.

    Feature layout (29 dims):
      [0]  pitch_count_game / 100
      [1]  pitch_count_inning / 30
      [2]  clip(velo_delta, -10, 10) / 10  (0 if null)
      [3]  float(prior_batter_reached)
      [4]  float(prior_two_reached)
      [5]  how_on_1b_enc / 5
      [6]  how_on_2b_enc / 5
      [7]  how_on_3b_enc / 5
      [8]  runs_this_inning / 5
      [9]  base_state / 7             (base state BEFORE PA)
      [10] outs / 2
      [11] tto / 3
      [12] (inning - 1) / 8
      [13] float(half == "bot")
      [14] fastball_rate
      [15] breaking_rate
      [16] changeup_rate
      [17] zone_rate
      [18] n_pitches / 10
      [19] stand_enc  (-1=L, 1=R)
      [20:28] pa_outcome one-hot (8 dims, PA_OUTCOMES order)
      [28] base_state_after / 7      (base state AFTER PA)
    """
    rows = pitches.to_dicts()

    # Group by at_bat_number, preserving order
    pa_order: list[int] = []
    pa_groups: dict[int, list[dict]] = {}
    for row in rows:
        ab = int(row.get("at_bat_number") or 0)
        if ab not in pa_groups:
            pa_order.append(ab)
            pa_groups[ab] = []
        pa_groups[ab].append(row)

    summaries: list[np.ndarray] = []

    for ab in pa_order:
        pa_rows = pa_groups[ab]
        if not pa_rows:
            continue

        # Use FIRST pitch for game state features
        first = pa_rows[0]

        def _i(v, default=0):
            return int(v) if v is not None else default

        def _f(v, default=0.0):
            return float(v) if v is not None else default

        # [0] pitch_count_game / 100
        feat_pcg = _i(first.get("pitch_count_game"), 0) / 100.0

        # [1] pitch_count_inning / 30
        feat_pci = _i(first.get("pitch_count_inning"), 0) / 30.0

        # [2] velo_delta
        velo_delta = first.get("velo_delta")
        if velo_delta is not None:
            feat_velo = float(np.clip(float(velo_delta), -10.0, 10.0)) / 10.0
        else:
            feat_velo = 0.0

        # [3] prior_batter_reached
        feat_pbr = float(bool(first.get("prior_batter_reached", False)))

        # [4] prior_two_reached
        feat_ptr = float(bool(first.get("prior_two_reached", False)))

        # [5-7] how_on_1b/2b/3b encoding
        how_on_1b = first.get("how_on_1b")
        how_on_2b = first.get("how_on_2b")
        how_on_3b = first.get("how_on_3b")
        feat_h1 = HOW_ON_ENC.get(how_on_1b, 0) / 5.0
        feat_h2 = HOW_ON_ENC.get(how_on_2b, 0) / 5.0
        feat_h3 = HOW_ON_ENC.get(how_on_3b, 0) / 5.0

        # [8] runs_this_inning / 5
        feat_ri = _i(first.get("runs_this_inning"), 0) / 5.0

        # [9] base_state / 7
        feat_bs = _i(first.get("base_state"), 0) / 7.0

        # [10] outs / 2
        feat_outs = _i(first.get("outs"), 0) / 2.0

        # [11] tto / 3
        feat_tto = _i(first.get("tto"), 1) / 3.0

        # [12] (inning - 1) / 8
        feat_inn = (_i(first.get("inning"), 1) - 1) / 8.0

        # [13] half == "bot"
        feat_half = 1.0 if first.get("half") == "bot" else 0.0

        # Pitch mix rates from ALL pitches in PA
        n_pitches = len(pa_rows)
        fb_count = 0
        br_count = 0
        ch_count = 0
        zone_count = 0
        for r in pa_rows:
            pt = r.get("pitch_type") or ""
            if pt in FASTBALL_TYPES:
                fb_count += 1
            elif pt in BREAKING_TYPES:
                br_count += 1
            elif pt in CHANGE_TYPES:
                ch_count += 1
            # zone_rate
            px = r.get("plate_x")
            pz = r.get("plate_z")
            if px is not None and pz is not None:
                if abs(float(px)) < 0.83 and 1.5 < float(pz) < 3.5:
                    zone_count += 1

        feat_fb_rate = fb_count / n_pitches if n_pitches > 0 else 0.0
        feat_br_rate = br_count / n_pitches if n_pitches > 0 else 0.0
        feat_ch_rate = ch_count / n_pitches if n_pitches > 0 else 0.0
        feat_zone_rate = zone_count / n_pitches if n_pitches > 0 else 0.0

        # [18] n_pitches / 10
        feat_np = n_pitches / 10.0

        # [19] stand encoding
        stand = first.get("stand", "R")
        feat_stand = -1.0 if stand == "L" else 1.0

        # [20:28] pa_outcome one-hot (from the terminal pitch)
        terminal_row = pa_rows[-1]
        for r in pa_rows:
            if r.get("pa_terminal"):
                terminal_row = r
                break
        pa_outcome_raw = terminal_row.get("pa_outcome")
        pa_outcome_idx = PA_OUTCOME_IDX.get(pa_outcome_raw, -1) if pa_outcome_raw else -1
        outcome_onehot = np.zeros(8, dtype=np.float32)
        if 0 <= pa_outcome_idx < 8:
            outcome_onehot[pa_outcome_idx] = 1.0

        # [28] base_state_after / 7 (base state AFTER this PA)
        bs_after_raw = terminal_row.get("base_state_after")
        feat_bs_after = _i(bs_after_raw, 0) / 7.0 if bs_after_raw is not None else 0.0

        vec = np.array([
            feat_pcg, feat_pci, feat_velo, feat_pbr, feat_ptr,
            feat_h1, feat_h2, feat_h3, feat_ri, feat_bs,
            feat_outs, feat_tto, feat_inn, feat_half,
            feat_fb_rate, feat_br_rate, feat_ch_rate, feat_zone_rate,
            feat_np, feat_stand,
        ], dtype=np.float32)
        vec = np.concatenate([vec, outcome_onehot, [feat_bs_after]])  # (29,)
        summaries.append(vec)

    return summaries


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GameDataset(Dataset):
    """One sample = one full game (all pitches as a sequence).

    Returns dicts with (B=1)-less per-game tensors. The collate fn pads
    and stacks to (B, N, ...).
    """

    def __init__(
        self,
        game_pks: list[int],
        pitches_df: pl.DataFrame,
        registry: PlayerRegistry,
        max_pitches: int = 500,
        max_memory: int = 150,
    ) -> None:
        self._game_pks = game_pks
        self._registry = registry
        self._max_pitches = max_pitches
        self._max_memory = max_memory

        # Build game_pk → sorted list of row dicts
        print(f"  Indexing {len(game_pks)} games...")
        pitches_sorted = pitches_df.sort(["game_pk", "at_bat_number", "pitch_number"])
        rows = pitches_sorted.to_dicts()

        self._game_rows: dict[int, list[dict]] = {gp: [] for gp in game_pks}
        for row in rows:
            gp = row.get("game_pk")
            if gp in self._game_rows:
                self._game_rows[gp].append(row)

    def __len__(self) -> int:
        return len(self._game_pks)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        gp = self._game_pks[idx]
        rows = self._game_rows.get(gp, [])

        if not rows:
            return self._empty_sample()

        reg = self._registry

        def _i(v, default=0):
            return int(v) if v is not None else default

        def _f(v, default=0.0):
            return float(v) if v is not None else default

        # Build PA index mapping: at_bat_number → sequential 0-indexed PA counter
        unique_abs = sorted(set(_i(r.get("at_bat_number"), 0) for r in rows))
        ab_to_pa_idx = {ab: i for i, ab in enumerate(unique_abs)}

        # Compute PA summaries for completed PAs (as a polars-based call)
        import polars as pl
        game_df = pl.from_dicts(rows, infer_schema_length=None)
        pa_summaries = compute_pa_summaries(game_df)
        # pa_summaries[i] is summary for the i-th PA (in order)

        # Build per-pitch features
        pitchers_ids = []
        batter_ids = []
        umpire_ids = []
        park_ids = []
        pitch_type_idxs = []
        balls_l, strikes_l, outs_l, base_state_l = [], [], [], []
        inning_l, tto_l, pcg_l, pci_l = [], [], [], []
        runs_scored_target_l, pa_outcome_target_l = [], []
        plate_x_l, plate_z_l, rs_l, pfx_x_l, pfx_z_l = [], [], [], [], []
        sd_l, half_l, stand_l, p_throws_l, tracking_l = [], [], [], [], []
        ls_l, la_l = [], []
        swing_l, contact_l, foul_l, in_play_l, pa_terminal_l = [], [], [], [], []
        pa_idx_l, within_pa_pos_l = [], []

        # Track within-PA position counter
        pa_pitch_counter: dict[int, int] = {}

        for row in rows:
            ab = _i(row.get("at_bat_number"), 0)
            pa_i = ab_to_pa_idx.get(ab, 0)
            wpp = pa_pitch_counter.get(pa_i, 0)
            pa_pitch_counter[pa_i] = wpp + 1

            pa_outcome_raw = row.get("pa_outcome")
            pa_outcome_target = PA_OUTCOME_IDX.get(pa_outcome_raw, -1) if pa_outcome_raw else -1
            runs_raw = row.get("runs_scored")
            runs_scored_target = min(int(runs_raw), 4) if runs_raw is not None else -1

            stand_enc = -1.0 if row.get("stand") == "L" else 1.0
            p_throws_enc = -1.0 if row.get("p_throws") == "L" else 1.0
            half_enc = 1.0 if row.get("half") == "bot" else 0.0

            raw_ls = row.get("launch_speed")
            raw_la = row.get("launch_angle")
            launch_speed_norm = (_f(raw_ls, 80.0) - 80.0) / 30.0 if raw_ls is not None else 0.0
            launch_angle_norm = _f(raw_la, 0.0) / 45.0 if raw_la is not None else 0.0
            score_diff_norm = float(np.clip(_i(row.get("score_diff"), 0), -10, 10))

            pitchers_ids.append(reg.pitcher(row.get("pitcher_id")))
            batter_ids.append(reg.batter(row.get("batter_id")))
            umpire_ids.append(reg.umpire(row.get("umpire_id")))
            park_ids.append(reg.park(row.get("park_id")))
            pitch_type_idxs.append(reg.pitch_type(row.get("pitch_type")))
            balls_l.append(_i(row.get("balls"), 0))
            strikes_l.append(_i(row.get("strikes"), 0))
            outs_l.append(_i(row.get("outs"), 0))
            base_state_l.append(_i(row.get("base_state"), 0))
            inning_l.append(_i(row.get("inning"), 1))
            tto_l.append(_i(row.get("tto"), 1))
            pcg_l.append(_i(row.get("pitch_count_game"), 0))
            pci_l.append(_i(row.get("pitch_count_inning"), 0))
            runs_scored_target_l.append(runs_scored_target)
            pa_outcome_target_l.append(pa_outcome_target)
            plate_x_l.append(_f(row.get("plate_x"), 0.0))
            plate_z_l.append(_f(row.get("plate_z"), 0.0))
            rs_l.append(_f(row.get("release_speed"), 92.0))
            pfx_x_l.append(_f(row.get("pfx_x"), 0.0))
            pfx_z_l.append(_f(row.get("pfx_z"), 0.0))
            sd_l.append(score_diff_norm)
            half_l.append(half_enc)
            stand_l.append(stand_enc)
            p_throws_l.append(p_throws_enc)
            tracking_l.append(float(_i(row.get("tracking_era"), 0)))
            ls_l.append(launch_speed_norm)
            la_l.append(launch_angle_norm)
            swing_l.append(bool(row.get("swing", False)))
            contact_l.append(bool(row.get("contact", False)))
            foul_l.append(bool(row.get("foul", False)))
            in_play_l.append(bool(row.get("in_play", False)))
            pa_terminal_l.append(bool(row.get("pa_terminal", False)))
            pa_idx_l.append(pa_i)
            within_pa_pos_l.append(wpp)

        # Truncate to max_pitches
        N = min(len(rows), self._max_pitches)
        def trunc(lst):
            return lst[:N]

        pitchers_ids = trunc(pitchers_ids)
        batter_ids = trunc(batter_ids)
        umpire_ids = trunc(umpire_ids)
        park_ids = trunc(park_ids)
        pitch_type_idxs = trunc(pitch_type_idxs)
        balls_l = trunc(balls_l)
        strikes_l = trunc(strikes_l)
        outs_l = trunc(outs_l)
        base_state_l = trunc(base_state_l)
        inning_l = trunc(inning_l)
        tto_l = trunc(tto_l)
        pcg_l = trunc(pcg_l)
        pci_l = trunc(pci_l)
        runs_scored_target_l = trunc(runs_scored_target_l)
        pa_outcome_target_l = trunc(pa_outcome_target_l)
        plate_x_l = trunc(plate_x_l)
        plate_z_l = trunc(plate_z_l)
        rs_l = trunc(rs_l)
        pfx_x_l = trunc(pfx_x_l)
        pfx_z_l = trunc(pfx_z_l)
        sd_l = trunc(sd_l)
        half_l = trunc(half_l)
        stand_l = trunc(stand_l)
        p_throws_l = trunc(p_throws_l)
        tracking_l = trunc(tracking_l)
        ls_l = trunc(ls_l)
        la_l = trunc(la_l)
        swing_l = trunc(swing_l)
        contact_l = trunc(contact_l)
        foul_l = trunc(foul_l)
        in_play_l = trunc(in_play_l)
        pa_terminal_l = trunc(pa_terminal_l)
        pa_idx_l = trunc(pa_idx_l)
        within_pa_pos_l = trunc(within_pa_pos_l)

        # Game memory: completed PA summaries (up to max_memory)
        # pa_summaries[i] is ready after PA i completes; pitch at pa_idx=k
        # can see summaries for PAs 0..k-1
        mem_summaries = pa_summaries[:self._max_memory]  # list of (28,) arrays
        if mem_summaries:
            game_memory = np.stack(mem_summaries, axis=0).astype(np.float32)  # (n_pas, 28)
        else:
            game_memory = np.zeros((0, PA_SUMMARY_DIM), dtype=np.float32)

        return {
            # Integer (N,)
            "pitcher_id": pitchers_ids,
            "batter_id": batter_ids,
            "umpire_id": umpire_ids,
            "park_id": park_ids,
            "pitch_type_idx": pitch_type_idxs,
            "balls": balls_l,
            "strikes": strikes_l,
            "outs": outs_l,
            "base_state": base_state_l,
            "inning": inning_l,
            "tto": tto_l,
            "pitch_count_game": pcg_l,
            "pitch_count_inning": pci_l,
            "runs_scored_target": runs_scored_target_l,
            "pa_outcome_target": pa_outcome_target_l,
            # Float (N,)
            "plate_x": plate_x_l,
            "plate_z": plate_z_l,
            "release_speed_norm": rs_l,
            "pfx_x": pfx_x_l,
            "pfx_z": pfx_z_l,
            "score_diff_norm": sd_l,
            "half_enc": half_l,
            "stand_enc": stand_l,
            "p_throws_enc": p_throws_l,
            "tracking_era": tracking_l,
            "launch_speed_norm": ls_l,
            "launch_angle_norm": la_l,
            # Bool (N,)
            "swing": swing_l,
            "contact": contact_l,
            "foul": foul_l,
            "in_play": in_play_l,
            "pa_terminal": pa_terminal_l,
            # Position (N,)
            "pa_idx": pa_idx_l,
            "within_pa_pos": within_pa_pos_l,
            # Memory (n_pas, 29) — variable length
            "game_memory": game_memory,
        }

    def _empty_sample(self) -> dict[str, Any]:
        """Return a minimal sample for games with no pitches."""
        return {
            "pitcher_id": [0], "batter_id": [0], "umpire_id": [0], "park_id": [0],
            "pitch_type_idx": [0], "balls": [0], "strikes": [0], "outs": [0],
            "base_state": [0], "inning": [1], "tto": [1],
            "pitch_count_game": [0], "pitch_count_inning": [0],
            "runs_scored_target": [-1], "pa_outcome_target": [-1],
            "plate_x": [0.0], "plate_z": [2.5], "release_speed_norm": [92.0],
            "pfx_x": [0.0], "pfx_z": [0.0], "score_diff_norm": [0.0],
            "half_enc": [0.0], "stand_enc": [1.0], "p_throws_enc": [1.0],
            "tracking_era": [1.0], "launch_speed_norm": [0.0], "launch_angle_norm": [0.0],
            "swing": [False], "contact": [False], "foul": [False],
            "in_play": [False], "pa_terminal": [False],
            "pa_idx": [0], "within_pa_pos": [0],
            "game_memory": np.zeros((0, PA_SUMMARY_DIM), dtype=np.float32),
        }


def collate_fn_phase4(batch: list[dict[str, Any]]) -> dict[str, Tensor]:
    """Collate a list of game samples into padded batch tensors."""
    int_keys = [
        "pitcher_id", "batter_id", "umpire_id", "park_id", "pitch_type_idx",
        "balls", "strikes", "outs", "base_state", "inning", "tto",
        "pitch_count_game", "pitch_count_inning", "runs_scored_target", "pa_outcome_target",
        "pa_idx", "within_pa_pos",
    ]
    float_keys = [
        "plate_x", "plate_z", "release_speed_norm", "pfx_x", "pfx_z",
        "score_diff_norm", "half_enc", "stand_enc", "p_throws_enc",
        "tracking_era", "launch_speed_norm", "launch_angle_norm",
    ]
    bool_keys = ["swing", "contact", "foul", "in_play", "pa_terminal"]

    B = len(batch)
    max_N = max(len(s["pitcher_id"]) for s in batch)
    max_M = max(s["game_memory"].shape[0] for s in batch)
    max_M = max(max_M, 1)  # ensure at least dim=1 for valid tensor shape

    result: dict[str, Tensor] = {}

    # Pad per-pitch tensors to (B, max_N)
    for k in int_keys:
        pad_val = -1 if k in ("runs_scored_target", "pa_outcome_target") else 0
        rows = []
        for s in batch:
            lst = s[k]
            n = len(lst)
            if n < max_N:
                lst = list(lst) + [pad_val] * (max_N - n)
            rows.append(lst[:max_N])
        result[k] = torch.tensor(rows, dtype=torch.long)

    for k in float_keys:
        rows = []
        for s in batch:
            lst = list(s[k])
            n = len(lst)
            if n < max_N:
                lst = lst + [0.0] * (max_N - n)
            rows.append(lst[:max_N])
        result[k] = torch.tensor(rows, dtype=torch.float32)

    for k in bool_keys:
        rows = []
        for s in batch:
            lst = list(s[k])
            n = len(lst)
            if n < max_N:
                lst = lst + [False] * (max_N - n)
            rows.append(lst[:max_N])
        result[k] = torch.tensor(rows, dtype=torch.bool)

    # Pitch pad mask: True for padded positions
    pitch_pad_mask = torch.zeros(B, max_N, dtype=torch.bool)
    for i, s in enumerate(batch):
        n = len(s["pitcher_id"])
        if n < max_N:
            pitch_pad_mask[i, n:] = True
    result["pitch_pad_mask"] = pitch_pad_mask

    # Game memory: pad to (B, max_M, 28)
    game_memory = torch.zeros(B, max_M, PA_SUMMARY_DIM, dtype=torch.float32)
    mem_pad_mask = torch.zeros(B, max_M, dtype=torch.bool)
    for i, s in enumerate(batch):
        mem = s["game_memory"]
        m = mem.shape[0]
        if m > 0:
            game_memory[i, :m] = torch.from_numpy(mem[:max_M])
            m_actual = min(m, max_M)
        else:
            m_actual = 0
        if m_actual < max_M:
            mem_pad_mask[i, m_actual:] = True
    result["game_memory"] = game_memory
    result["mem_pad_mask"] = mem_pad_mask

    # Cross-attention mask: (B, max_N, max_M) bool
    # Pitch i blocks PA summary k if k >= pa_idx[i]
    # (i.e., can only attend to PAs that completed BEFORE this PA)
    pa_idx_tensor = result["pa_idx"]   # (B, max_N)
    k_idx = torch.arange(max_M).unsqueeze(0).unsqueeze(0)  # (1, 1, max_M)
    # Block if k >= pa_idx[i] (k is 0-indexed PA index in memory)
    cross_attn_mask = k_idx >= pa_idx_tensor.unsqueeze(2)  # (B, max_N, max_M)
    # Also block padded memory positions (mem_pad_mask handles that in model)
    result["cross_attn_mask"] = cross_attn_mask

    return result


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: GameContextTransformer,
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
    model: GameContextTransformer,
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
    model: GameContextTransformer,
    scaler: TemperatureScaler,
    val_loader: DataLoader,
    device: str,
) -> TemperatureScaler:
    """Optimize temperature parameters on the validation set."""
    model.eval()
    scaler = scaler.to(device)
    optimizer = torch.optim.LBFGS(scaler.parameters(), lr=0.01, max_iter=50)

    all_pa_logits: list[Tensor] = []
    all_pa_targets: list[Tensor] = []
    all_runs_logits: list[Tensor] = []
    all_runs_targets: list[Tensor] = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch)

            pad_mask = batch.get("pitch_pad_mask")

            def _valid(t: Tensor) -> Tensor:
                flat = t.reshape(-1)
                if pad_mask is not None:
                    return flat[~pad_mask.reshape(-1)]
                return flat

            def _valid2d(t: Tensor) -> Tensor:
                B, N, C = t.shape
                flat = t.reshape(B * N, C)
                if pad_mask is not None:
                    return flat[~pad_mask.reshape(-1)]
                return flat

            pa_terminal_flat = _valid(batch["pa_terminal"])
            pa_outcome_tgt_flat = _valid(batch["pa_outcome_target"])
            pa_mask = pa_terminal_flat & (pa_outcome_tgt_flat >= 0)
            runs_tgt_flat = _valid(batch["runs_scored_target"])
            runs_mask = pa_terminal_flat & (runs_tgt_flat >= 0)

            if pa_mask.any():
                all_pa_logits.append(_valid2d(outputs["pa_outcome"])[pa_mask].cpu())
                all_pa_targets.append(pa_outcome_tgt_flat[pa_mask].cpu())
            if runs_mask.any():
                all_runs_logits.append(_valid2d(outputs["runs_scored"])[runs_mask].cpu())
                all_runs_targets.append(runs_tgt_flat[runs_mask].cpu())

    if not all_pa_logits:
        return scaler

    pa_logits_all = torch.cat(all_pa_logits).to(device)
    pa_targets_all = torch.cat(all_pa_targets).to(device)
    runs_logits_all = torch.cat(all_runs_logits).to(device) if all_runs_logits else None
    runs_targets_all = torch.cat(all_runs_targets).to(device) if all_runs_logits else None

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
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase 4 GameContextTransformer.")
    parser.add_argument(
        "--train-seasons", nargs="+", type=int,
        default=list(range(2015, 2022)),
    )
    parser.add_argument("--val-season", type=int, default=2022)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("checkpoints/phase4_transformer"),
    )
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-pitches", type=int, default=500)
    parser.add_argument("--max-memory", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading training data...")
    train_paths = [_DATA_ROOT / f"pitches_{s}.parquet" for s in args.train_seasons]
    val_path = _DATA_ROOT / f"pitches_{args.val_season}.parquet"

    train_frames = [pl.read_parquet(p) for p in train_paths]
    train_df = pl.concat(train_frames)
    print(f"  {len(train_df):,} train pitches across seasons {args.train_seasons}")

    # Fit registry
    registry = PlayerRegistry()
    registry.fit(train_df)
    registry_path = args.output_dir / "registry.json"
    registry.save(registry_path)
    print(f"  Registry: {registry.n_pitchers} pitchers, {registry.n_batters} batters, "
          f"{registry.n_umpires} umpires, {registry.n_parks} parks")

    # Build game-level datasets
    print("Building GameDataset for train...")
    train_game_pks = train_df["game_pk"].unique().to_list()
    train_dataset = GameDataset(
        train_game_pks, train_df, registry,
        max_pitches=args.max_pitches, max_memory=args.max_memory,
    )
    del train_df

    print("Building GameDataset for val...")
    val_df = pl.read_parquet(val_path)
    print(f"  {len(val_df):,} val pitches (season {args.val_season})")
    val_game_pks = val_df["game_pk"].unique().to_list()
    val_dataset = GameDataset(
        val_game_pks, val_df, registry,
        max_pitches=args.max_pitches, max_memory=args.max_memory,
    )
    del val_df

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn_phase4, num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn_phase4, num_workers=2,
        pin_memory=(args.device == "cuda"),
    )

    # Build model — use defaults (hist_d_model=64, n_heads=4, n_layers=2)
    # matching the checkpoint architecture from the original training run
    model = GameContextTransformer(registry).to(args.device)
    scaler = TemperatureScaler()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-2,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

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

    print(f"Calibrating temperature using best model (epoch {best_epoch})...")
    best_ckpt = torch.load(
        args.output_dir / "best_model.pt", map_location=args.device, weights_only=False
    )
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
    print(f"Training complete. Best val PA NLL: {best_val_nll:.4f}")


if __name__ == "__main__":
    main()
