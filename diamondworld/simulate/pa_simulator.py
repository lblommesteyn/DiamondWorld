from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from diamondworld.baselines.base import PA_OUTCOMES, PA_OUTCOME_IDX, IN_PLAY_OUTCOMES
from diamondworld.models.no_memory_mlp import NoMemoryMLP, TemperatureScaler
from diamondworld.models.registry import PlayerRegistry

IN_PLAY_INDICES = [PA_OUTCOME_IDX[o] for o in ["1B", "2B", "3B", "HR", "out"]]
IN_PLAY_OUTCOMES_LIST = ["1B", "2B", "3B", "HR", "out"]

# Strike zone bounds (feet from center)
_ZONE_X = 0.83
_ZONE_Z_LO = 1.5
_ZONE_Z_HI = 3.5


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _in_strike_zone(plate_x: float, plate_z: float) -> bool:
    return abs(plate_x) < _ZONE_X and _ZONE_Z_LO < plate_z < _ZONE_Z_HI


class PitchSampler:
    """Samples pitch feature dicts from an empirical pool, stratified by pitcher."""

    def fit(self, pitches, registry: PlayerRegistry) -> None:
        """Build pitcher-specific and global pitch feature pools."""
        import polars as pl
        feat_cols = [
            "pitcher_id", "pitch_type", "plate_x", "plate_z",
            "release_speed", "pfx_x", "pfx_z",
        ]
        available = [c for c in feat_cols if c in pitches.columns]

        self._registry = registry
        self._pitcher_pool: dict[int, list[dict[str, Any]]] = {}
        global_pool: list[dict[str, Any]] = []

        for row in pitches.select(available).iter_rows(named=True):
            if row.get("plate_x") is None or row.get("plate_z") is None:
                continue
            pitcher_id = row.get("pitcher_id")
            pitcher_idx = registry.pitcher(pitcher_id)
            pitch_type_idx = registry.pitch_type(row.get("pitch_type"))
            feat = {
                "pitch_type_idx": pitch_type_idx,
                "plate_x": float(row.get("plate_x") or 0.0),
                "plate_z": float(row.get("plate_z") or 0.0),
                "release_speed_norm": float(row.get("release_speed") or 92.0),
                "pfx_x": float(row.get("pfx_x") or 0.0),
                "pfx_z": float(row.get("pfx_z") or 0.0),
            }
            if pitcher_idx not in self._pitcher_pool:
                self._pitcher_pool[pitcher_idx] = []
            self._pitcher_pool[pitcher_idx].append(feat)
            global_pool.append(feat)

        self._global_pool = global_pool

    def sample(self, pitcher_idx: int, rng: np.random.Generator) -> dict[str, Any]:
        """Sample a pitch feature dict for the given pitcher."""
        pool = self._pitcher_pool.get(pitcher_idx) or self._global_pool
        if not pool:
            return {
                "pitch_type_idx": 0, "plate_x": 0.0, "plate_z": 2.5,
                "release_speed_norm": 92.0, "pfx_x": 0.0, "pfx_z": 0.0,
            }
        idx = int(rng.integers(len(pool)))
        return pool[idx]


def _empirical_transition(
    bs: int, outs: int, outcome: str,
    transition_table: dict,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Sample (bs_after, runs) from empirical transition table or deterministic fallback."""
    from diamondworld.baselines.markov_re24 import MarkovRE24Simulator
    key = (bs, outs, outcome)
    entries = transition_table.get(key)
    if entries:
        chosen = entries[int(rng.integers(len(entries)))]
        return chosen
    return MarkovRE24Simulator._deterministic_transition(bs, outs, outcome)


class PASimulator:
    """Pitch-by-pitch PA simulator using the NoMemoryMLP model."""

    def __init__(
        self,
        model: NoMemoryMLP,
        scaler: TemperatureScaler,
        registry: PlayerRegistry,
        transition_table: dict,
        hbp_rate: float,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.scaler = scaler
        self.registry = registry
        self.transition_table = transition_table
        self.hbp_rate = hbp_rate
        self.device = device
        self.model.eval()

    def _build_batch(
        self,
        pitcher_idx: int,
        batter_idx: int,
        umpire_idx: int,
        park_idx: int,
        game_state: dict[str, Any],
        pitch_feats: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Construct a single-item batch dict for model forward."""
        half_enc = 1.0 if game_state.get("half") == "bot" else 0.0
        stand = game_state.get("stand", "R")
        p_throws = game_state.get("p_throws", "R")
        stand_enc = -1.0 if stand == "L" else 1.0
        p_throws_enc = -1.0 if p_throws == "L" else 1.0

        score_diff = float(np.clip(game_state.get("score_diff", 0), -10, 10))

        batch = {
            # Integers (Long)
            "pitcher_id": torch.tensor([pitcher_idx], dtype=torch.long),
            "batter_id": torch.tensor([batter_idx], dtype=torch.long),
            "umpire_id": torch.tensor([umpire_idx], dtype=torch.long),
            "park_id": torch.tensor([park_idx], dtype=torch.long),
            "pitch_type_idx": torch.tensor([pitch_feats["pitch_type_idx"]], dtype=torch.long),
            "balls": torch.tensor([game_state.get("balls", 0)], dtype=torch.long),
            "strikes": torch.tensor([game_state.get("strikes", 0)], dtype=torch.long),
            "outs": torch.tensor([game_state.get("outs", 0)], dtype=torch.long),
            "base_state": torch.tensor([game_state.get("base_state", 0)], dtype=torch.long),
            "inning": torch.tensor([game_state.get("inning", 5)], dtype=torch.long),
            "tto": torch.tensor([game_state.get("tto", 1)], dtype=torch.long),
            "pitch_count_game": torch.tensor([game_state.get("pitch_count_game", 0)], dtype=torch.long),
            "pitch_count_inning": torch.tensor([game_state.get("pitch_count_inning", 0)], dtype=torch.long),
            "runs_scored_target": torch.tensor([-1], dtype=torch.long),
            "pa_outcome_target": torch.tensor([-1], dtype=torch.long),
            # Floats
            "plate_x": torch.tensor([pitch_feats["plate_x"]], dtype=torch.float32),
            "plate_z": torch.tensor([pitch_feats["plate_z"]], dtype=torch.float32),
            "release_speed_norm": torch.tensor([pitch_feats["release_speed_norm"]], dtype=torch.float32),
            "pfx_x": torch.tensor([pitch_feats["pfx_x"]], dtype=torch.float32),
            "pfx_z": torch.tensor([pitch_feats["pfx_z"]], dtype=torch.float32),
            "score_diff_norm": torch.tensor([score_diff], dtype=torch.float32),
            "half_enc": torch.tensor([half_enc], dtype=torch.float32),
            "stand_enc": torch.tensor([stand_enc], dtype=torch.float32),
            "p_throws_enc": torch.tensor([p_throws_enc], dtype=torch.float32),
            "tracking_era": torch.tensor([float(game_state.get("tracking_era", 1))], dtype=torch.float32),
            "launch_speed_norm": torch.tensor([0.0], dtype=torch.float32),
            "launch_angle_norm": torch.tensor([0.0], dtype=torch.float32),
            # Bools
            "swing": torch.tensor([False], dtype=torch.bool),
            "contact": torch.tensor([False], dtype=torch.bool),
            "foul": torch.tensor([False], dtype=torch.bool),
            "in_play": torch.tensor([False], dtype=torch.bool),
            "pa_terminal": torch.tensor([False], dtype=torch.bool),
        }
        return {k: v.to(self.device) for k, v in batch.items()}

    def simulate_pa(
        self,
        pitcher_idx: int,
        batter_idx: int,
        umpire_idx: int,
        park_idx: int,
        game_state: dict[str, Any],
        pitch_sampler: PitchSampler,
        *,
        rng: np.random.Generator,
        forced_pitch: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Simulate one plate appearance. Returns outcome dict."""
        balls = game_state.get("balls", 0)
        strikes = game_state.get("strikes", 0)
        bs = game_state.get("base_state", 0)
        outs = game_state.get("outs", 0)

        # HBP: apply once per PA using the empirical per-PA rate
        if rng.random() < self.hbp_rate:
            bs_after, runs = _empirical_transition(bs, outs, "HBP", self.transition_table, rng)
            return {
                "pa_outcome": "HBP",
                "runs_scored": runs,
                "n_pitches": 1,
                "base_state_after": bs_after,
            }

        for n in range(20):
            # Get pitch features
            pitch_feats = forced_pitch if forced_pitch is not None else pitch_sampler.sample(pitcher_idx, rng)

            # Build batch and run model
            state = dict(game_state)
            state["balls"] = balls
            state["strikes"] = strikes
            state["base_state"] = bs
            state["outs"] = outs

            with torch.no_grad():
                batch = self._build_batch(pitcher_idx, batter_idx, umpire_idx, park_idx, state, pitch_feats)
                outputs = self.model(batch)
                scaled = self.scaler.scale(outputs)

            swing_logit = float(scaled["swing"][0].cpu().item())
            contact_logit = float(scaled["contact"][0].cpu().item())
            foul_logit = float(scaled["foul"][0].cpu().item())
            pa_outcome_logits = scaled["pa_outcome"][0].cpu().numpy()
            pa_outcome_temp = float(self.scaler.pa_outcome_temp.cpu().item())

            plate_x = pitch_feats["plate_x"]
            plate_z = pitch_feats["plate_z"]

            # Swing decision
            did_swing = rng.random() < _sigmoid(swing_logit)

            if did_swing:
                # Contact decision
                did_contact = rng.random() < _sigmoid(contact_logit)
                if did_contact:
                    # Foul decision
                    is_foul = rng.random() < _sigmoid(foul_logit)
                    if is_foul:
                        # Foul: strikes can increase up to 2
                        strikes = min(strikes + 1, 2)
                        continue
                    else:
                        # Ball in play: sample in-play outcome from model, runs from transition table
                        in_play_logits = pa_outcome_logits[IN_PLAY_INDICES]
                        in_play_probs = F.softmax(
                            torch.tensor(in_play_logits / pa_outcome_temp), dim=0
                        ).numpy()
                        in_play_probs = in_play_probs / in_play_probs.sum()
                        outcome_idx = int(rng.choice(len(IN_PLAY_OUTCOMES_LIST), p=in_play_probs))
                        outcome = IN_PLAY_OUTCOMES_LIST[outcome_idx]

                        bs_after, runs = _empirical_transition(bs, outs, outcome, self.transition_table, rng)
                        return {
                            "pa_outcome": outcome,
                            "runs_scored": runs,
                            "n_pitches": n + 1,
                            "base_state_after": bs_after,
                        }
                else:
                    # Swinging strike
                    strikes += 1
                    if strikes >= 3:
                        bs_after, runs = _empirical_transition(bs, outs, "K", self.transition_table, rng)
                        return {
                            "pa_outcome": "K",
                            "runs_scored": runs,
                            "n_pitches": n + 1,
                            "base_state_after": bs_after,
                        }
            else:
                # Take: strike or ball based on zone
                if _in_strike_zone(plate_x, plate_z):
                    strikes += 1
                    if strikes >= 3:
                        bs_after, runs = _empirical_transition(bs, outs, "K", self.transition_table, rng)
                        return {
                            "pa_outcome": "K",
                            "runs_scored": runs,
                            "n_pitches": n + 1,
                            "base_state_after": bs_after,
                        }
                else:
                    balls += 1
                    if balls >= 4:
                        bs_after, runs = _empirical_transition(bs, outs, "BB", self.transition_table, rng)
                        return {
                            "pa_outcome": "BB",
                            "runs_scored": runs,
                            "n_pitches": n + 1,
                            "base_state_after": bs_after,
                        }

        # Fallback: walk after 20 pitches (shouldn't happen in practice)
        bs_after, runs = _empirical_transition(bs, outs, "BB", self.transition_table, rng)
        return {
            "pa_outcome": "BB",
            "runs_scored": runs,
            "n_pitches": 20,
            "base_state_after": bs_after,
        }
