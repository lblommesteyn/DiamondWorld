"""PA-on-ABCD hybrid outcome sampling for Model 7 exports.

The pitchformer remains responsible for every generated pitch: A samples pitch
type/stuff, B samples the swing/contact path, and D continues to generate
launch data.  Once that path produces an in-play ball, this adapter samples the
conditional in-play result from Model 7's exported PA head.  It is therefore a
useful rollout comparison, not a second trainer or a replacement for either
native PA or native ABCD evaluation.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping

import jax
import numpy as np

from diamondworldjax.domain import PAOutcome
from diamondworldjax.model.pa_checkpoint import posterior_params
from diamondworldjax.model.pa_inference import build_pa_inference


class PAInPlayOutcomeSampler:
    """Map an ABCD rollout's PA starts to Model 7 PA in-play outcomes.

    The PA checkpoint owns the global player registry while ABCD uses separate
    local registries.  Those maps are reconciled once on construction, keeping
    the hot callback to lookup + one compiled PA-head forward pass.
    """

    def __init__(
        self,
        pa_checkpoint: Mapping[str, Any],
        abcd_metadata: Mapping[str, Any],
        *,
        season: int,
        skill_mode: str = "mean",
        seed: int = 0,
    ) -> None:
        pa_metadata = pa_checkpoint.get("pa_metadata")
        if not pa_metadata:
            raise ValueError("Hybrid rollout requires a PA checkpoint with pa_metadata")
        if "joint_source" not in pa_checkpoint:
            raise ValueError("Hybrid rollout requires the PA export produced by Model 7 joint training")
        if not abcd_metadata.get("joint"):
            raise ValueError("Hybrid rollout requires Model 7 ABCD metadata")
        config = pa_metadata["config"]
        if config.get("pitchformer", False):
            raise ValueError("Hybrid rollout currently requires the non-sequence PA export")
        if skill_mode not in {"mean", "sample"}:
            raise ValueError("Hybrid PA skill mode must be 'mean' or 'sample'")

        table = pa_metadata["player_table"]
        params = posterior_params(
            pa_checkpoint["params"], config.get("skill_prior", "iso"), skill_mode, seed
        )
        self.inference = build_pa_inference(
            params, table,
            outcome_only=config.get("outcome_only", False),
            fatigue=config.get("fatigue", False),
            platoon=config.get("platoon", False),
            nested=config.get("nested", False),
            bilinear_rank=config.get("bilinear_rank", 0),
            skill_prior=config.get("skill_prior", "iso"),
            season_base=min(pa_metadata["train_seasons"]),
            n_seasons=len(pa_metadata["train_seasons"])
            if config.get("skill_prior", "iso") == "walk" else 1,
            simulation_season=season,
        )
        maps = abcd_metadata["maps"]
        id_to_idx = table["id_to_idx"]
        unknown = int(table.get("unknown_index", len(table["stats"])))
        self.pitcher_to_pa = self._player_map(
            maps["pitcher"], int(maps["n_pitcher"]), id_to_idx, unknown
        )
        self.batter_to_pa = self._player_map(
            maps["batter"], int(maps["n_batter"]), id_to_idx, unknown
        )
        self.park_to_pa = self._park_map(
            maps["park"], int(maps["n_park"]), pa_metadata.get("park_map", {})
        )

    @staticmethod
    def _player_map(local_map: Mapping[Any, int], size: int,
                    global_map: Mapping[Any, int], unknown: int) -> np.ndarray:
        result = np.full(size, unknown, np.int32)
        for raw_id, local_idx in local_map.items():
            result[int(local_idx)] = int(global_map.get(raw_id, unknown))
        return result

    @staticmethod
    def _park_map(local_map: Mapping[Any, int], size: int,
                  global_map: Mapping[Any, int]) -> np.ndarray:
        result = np.zeros(size, np.int32)
        for raw_id, local_idx in local_map.items():
            result[int(local_idx)] = int(global_map.get(raw_id, 0))
        return result

    @classmethod
    def from_paths(cls, pa_checkpoint: Path | str, abcd_metadata: Mapping[str, Any], **kwargs):
        with Path(pa_checkpoint).open("rb") as f:
            checkpoint = pickle.load(f)
        return cls(checkpoint, abcd_metadata, **kwargs)

    @staticmethod
    def _lookup(local_to_pa: np.ndarray, values: np.ndarray) -> np.ndarray:
        safe = np.clip(np.asarray(values, np.int32), 0, len(local_to_pa) - 1)
        return local_to_pa[safe]

    def __call__(
        self,
        state: dict[str, np.ndarray],
        source: np.ndarray,
        in_play: np.ndarray,
        original: dict[str, np.ndarray],
        key: jax.Array,
    ) -> np.ndarray:
        """Return a full PA outcome vector; only ``in_play`` entries are used."""
        B = len(source)
        rows = np.arange(B)
        source = np.clip(np.asarray(source, np.int32), 0, original["valid"].shape[1] - 1)
        pitcher_local = np.asarray(original["pitcher_idx"])[rows, source]
        batter_local = np.asarray(original["batter_idx"])[rows, source]
        park_local = np.asarray(original["park_idx"])[rows, source]
        ctx = np.asarray(original["ctx"])[rows, source]
        half = np.asarray(state["half"], np.float32)
        score_diff = np.where(half == 0,
                              state["away_score"] - state["home_score"],
                              state["home_score"] - state["away_score"])

        logits = self.inference.logits(
            inning=(np.asarray(state["inning"], np.float32) - 1.0) / 8.0,
            half=half,
            outs=np.asarray(state["outs"], np.float32) / 2.0,
            base_state=np.asarray(state["base"], np.float32) / 7.0,
            score_diff=np.clip(score_diff, -10, 10).astype(np.float32) / 10.0,
            tto=np.clip(np.asarray(state["tto"], np.float32), 0, 3) / 3.0,
            shift_restricted=np.zeros(B, np.float32),
            pitch_clock=np.zeros(B, np.float32),
            pitch_count_game=np.asarray(state["pitch_count"], np.float32) / 120.0,
            pitcher_ids=self._lookup(self.pitcher_to_pa, pitcher_local),
            batter_ids=self._lookup(self.batter_to_pa, batter_local),
            park_ids=self._lookup(self.park_to_pa, park_local),
            bat_side=ctx[:, 13] if ctx.shape[1] > 13 else np.full(B, .5, np.float32),
            pit_hand=ctx[:, 14] if ctx.shape[1] > 14 else np.full(B, .5, np.float32),
        )
        # PA outcomes are K, BB, HBP, 1B, 2B, 3B, HR, out, E.  A/B have
        # already ruled out the first three, so normalise only the legal
        # in-play suffix rather than allowing the PA model to alter pitch timing.
        sampled = np.asarray(jax.random.categorical(key, logits[:, 3:9], axis=-1), np.int32) + 3
        outcome = np.full(B, int(PAOutcome.OUT), np.int32)
        outcome[np.asarray(in_play, bool)] = sampled[np.asarray(in_play, bool)]
        return outcome
