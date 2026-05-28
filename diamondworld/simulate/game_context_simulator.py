"""Phase 4 simulators: PA and game simulators with game-context (cross-attention) memory.

GameContextPASimulator wraps GameContextTransformer and maintains a running list of
PA summaries (game_memory) that gets passed as cross-attention context on every pitch.
GameContextGameSimulator drives it for full 9-inning games.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from diamondworld.baselines.base import PA_OUTCOME_IDX, PA_OUTCOMES, GameLog
from diamondworld.models.game_context_transformer import (
    GameContextTransformer,
    TemperatureScaler,
    PA_SUMMARY_DIM,
    HOW_ON_ENC,
    FASTBALL_TYPES,
    BREAKING_TYPES,
    CHANGE_TYPES,
)
from diamondworld.models.registry import PlayerRegistry
from diamondworld.simulate.pa_simulator import (
    PitchSampler,
    _empirical_transition,
    _in_strike_zone,
    _sigmoid,
    IN_PLAY_INDICES,
    IN_PLAY_OUTCOMES_LIST,
)


class GameContextPASimulator:
    """Pitch-by-pitch PA simulator using GameContextTransformer with game memory.

    Maintains self.game_memory: list of 29-dim PA summary arrays accumulated
    during the current game. Call reset_game() at the start of each new game.
    """

    def __init__(
        self,
        model: GameContextTransformer,
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
        # Game memory: list of 29-dim numpy arrays, one per completed PA
        self.game_memory: list[np.ndarray] = []
        # Half-inning tracking (reset on inning/half change)
        self._current_inning_half: tuple[int, str] | None = None
        self._runs_this_inning: int = 0
        self._reach_history: list[bool] = []
        self._runner_how_on: dict[int, str | None] = {1: None, 2: None, 3: None}
        # Per-pitcher baseline velo for velo_delta (first pitch seen in game)
        self._pitcher_baseline_velo: dict[int, float] = {}
        # Velo of the first pitch of the previous PA (for velo_delta on current PA's first pitch)
        self._last_pa_first_pitch_velo: float | None = None

    def reset_game(self) -> None:
        """Clear all per-game tracking state at the start of a new game."""
        self.game_memory = []
        self._current_inning_half = None
        self._runs_this_inning = 0
        self._reach_history = []
        self._runner_how_on = {1: None, 2: None, 3: None}
        self._pitcher_baseline_velo = {}
        self._last_pa_first_pitch_velo = None

    def _reset_half_inning_if_changed(self, inning: int, half: str) -> None:
        """Clear per-half-inning state when inning or half changes."""
        key = (int(inning), str(half))
        if self._current_inning_half != key:
            self._current_inning_half = key
            self._runs_this_inning = 0
            self._reach_history = []
            self._runner_how_on = {1: None, 2: None, 3: None}

    def _update_runner_how_on(
        self,
        bs_after: int,
        outcome: str,
        runner_how_on_before: dict[int, str | None],
    ) -> dict[int, str | None]:
        """Approximate post-PA runner provenance per base.

        Maps outcome to the batter's how_on category and propagates existing
        runners forward where bases remain occupied.  This is approximate —
        the empirical transition table is a black box that doesn't tell us
        which runner ended up where — but matches the right distribution
        well enough for the model to use the signal it was trained on.
        """
        outcome_to_how_on = {
            "BB": "walk", "HBP": "hbp",
            "1B": "hit", "2B": "hit", "3B": "hit", "HR": "hit",
            "E": "error",
        }
        batter_how_on = outcome_to_how_on.get(outcome)

        if outcome in {"BB", "HBP", "1B", "E"}:
            batter_to = 1
        elif outcome == "2B":
            batter_to = 2
        elif outcome == "3B":
            batter_to = 3
        else:
            batter_to = None  # HR scored / K / out

        new_how_on: dict[int, str | None] = {1: None, 2: None, 3: None}
        if batter_how_on is not None and batter_to is not None:
            new_how_on[batter_to] = batter_how_on

        # For bases occupied after the PA but not filled by the batter, inherit
        # the provenance of whoever was on that base or a lower base before.
        bases_after = {b: bool(bs_after & (1 << (b - 1))) for b in (1, 2, 3)}
        for b in (1, 2, 3):
            if bases_after[b] and new_how_on[b] is None:
                for src in (b, b - 1, b - 2):
                    if src >= 1 and runner_how_on_before.get(src):
                        new_how_on[b] = runner_how_on_before[src]
                        break

        # Drop entries for bases that are now empty (defensive cleanup).
        for b in (1, 2, 3):
            if not bases_after[b]:
                new_how_on[b] = None
        return new_how_on

    def _build_batch(
        self,
        pitcher_idx: int,
        batter_idx: int,
        umpire_idx: int,
        park_idx: int,
        game_state: dict[str, Any],
        pitch_feats: dict[str, Any],
        within_pa_pos: int,
        pa_game_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Construct a single-pitch batch dict (B=1, N=1) for the transformer."""
        half_enc = 1.0 if game_state.get("half") == "bot" else 0.0
        stand = game_state.get("stand", "R")
        p_throws = game_state.get("p_throws", "R")
        stand_enc = -1.0 if stand == "L" else 1.0
        p_throws_enc = -1.0 if p_throws == "L" else 1.0
        score_diff = float(np.clip(game_state.get("score_diff", 0), -10, 10))

        M = len(self.game_memory)
        if M > 0:
            mem_arr = np.stack(self.game_memory, axis=0).astype(np.float32)  # (M, 29)
            game_memory_t = torch.from_numpy(mem_arr).unsqueeze(0)           # (1, M, 29)
        else:
            game_memory_t = torch.zeros(1, 0, PA_SUMMARY_DIM, dtype=torch.float32)

        # cross_attn_mask: (1, 1, M) — False for all (can attend all prior PAs)
        cross_attn_mask = torch.zeros(1, 1, max(M, 0), dtype=torch.bool)

        # mem_pad_mask: (1, M) — all False (no padding in simulation)
        mem_pad_mask = torch.zeros(1, max(M, 0), dtype=torch.bool)

        batch = {
            # Long (1, 1)
            "pitcher_id": torch.tensor([[pitcher_idx]], dtype=torch.long),
            "batter_id": torch.tensor([[batter_idx]], dtype=torch.long),
            "umpire_id": torch.tensor([[umpire_idx]], dtype=torch.long),
            "park_id": torch.tensor([[park_idx]], dtype=torch.long),
            "pitch_type_idx": torch.tensor([[pitch_feats["pitch_type_idx"]]], dtype=torch.long),
            "balls": torch.tensor([[game_state.get("balls", 0)]], dtype=torch.long),
            "strikes": torch.tensor([[game_state.get("strikes", 0)]], dtype=torch.long),
            "outs": torch.tensor([[game_state.get("outs", 0)]], dtype=torch.long),
            "base_state": torch.tensor([[game_state.get("base_state", 0)]], dtype=torch.long),
            "inning": torch.tensor([[game_state.get("inning", 5)]], dtype=torch.long),
            "tto": torch.tensor([[game_state.get("tto", 1)]], dtype=torch.long),
            "pitch_count_game": torch.tensor([[game_state.get("pitch_count_game", 0)]], dtype=torch.long),
            "pitch_count_inning": torch.tensor([[game_state.get("pitch_count_inning", 0)]], dtype=torch.long),
            "runs_scored_target": torch.tensor([[-1]], dtype=torch.long),
            "pa_outcome_target": torch.tensor([[-1]], dtype=torch.long),
            "pa_idx": torch.tensor([[pa_game_idx]], dtype=torch.long),
            "within_pa_pos": torch.tensor([[within_pa_pos]], dtype=torch.long),
            # Float (1, 1)
            "plate_x": torch.tensor([[pitch_feats["plate_x"]]], dtype=torch.float32),
            "plate_z": torch.tensor([[pitch_feats["plate_z"]]], dtype=torch.float32),
            "release_speed_norm": torch.tensor([[pitch_feats["release_speed_norm"]]], dtype=torch.float32),
            "pfx_x": torch.tensor([[pitch_feats["pfx_x"]]], dtype=torch.float32),
            "pfx_z": torch.tensor([[pitch_feats["pfx_z"]]], dtype=torch.float32),
            "score_diff_norm": torch.tensor([[score_diff]], dtype=torch.float32),
            "half_enc": torch.tensor([[half_enc]], dtype=torch.float32),
            "stand_enc": torch.tensor([[stand_enc]], dtype=torch.float32),
            "p_throws_enc": torch.tensor([[p_throws_enc]], dtype=torch.float32),
            "tracking_era": torch.tensor([[float(game_state.get("tracking_era", 1))]], dtype=torch.float32),
            "launch_speed_norm": torch.tensor([[0.0]], dtype=torch.float32),
            "launch_angle_norm": torch.tensor([[0.0]], dtype=torch.float32),
            # Bool (1, 1)
            "swing": torch.tensor([[False]], dtype=torch.bool),
            "contact": torch.tensor([[False]], dtype=torch.bool),
            "foul": torch.tensor([[False]], dtype=torch.bool),
            "in_play": torch.tensor([[False]], dtype=torch.bool),
            "pa_terminal": torch.tensor([[False]], dtype=torch.bool),
            "pitch_pad_mask": torch.zeros(1, 1, dtype=torch.bool),
            # Memory
            "game_memory": game_memory_t,
            "cross_attn_mask": cross_attn_mask,
            "mem_pad_mask": mem_pad_mask,
        }
        return {k: v.to(self.device) for k, v in batch.items()}

    def _compute_pa_summary(
        self,
        pa_pitches: list[dict[str, Any]],
        pa_outcome: str,
        game_state_start: dict[str, Any],
        bs_after: int = 0,
    ) -> np.ndarray:
        """Compute the 29-dim PA summary for a just-completed PA.

        pa_pitches: list of pitch feature dicts from this PA (accumulated during simulate_pa)
        pa_outcome: the outcome string
        game_state_start: the game state at the START of this PA (before any pitch)
        bs_after: base state AFTER this PA completed
        """
        n_pitches = len(pa_pitches)
        if n_pitches == 0:
            n_pitches = 1

        gs = game_state_start

        def _i(v, default=0):
            return int(v) if v is not None else default

        feat_pcg = _i(gs.get("pitch_count_game"), 0) / 100.0
        feat_pci = _i(gs.get("pitch_count_inning"), 0) / 30.0

        # velo_delta: first pitch's release_speed minus the pitcher's baseline velo
        # (first pitch we saw from them this game). Falls back to 0 if unknown.
        velo_delta = gs.get("velo_delta", 0.0)
        feat_velo = float(np.clip(velo_delta, -10.0, 10.0)) / 10.0

        # prior_batter_reached / prior_two_reached: from this half-inning's history
        feat_pbr = 1.0 if (len(self._reach_history) >= 1 and self._reach_history[-1]) else 0.0
        feat_ptr = (
            1.0
            if (
                len(self._reach_history) >= 2
                and self._reach_history[-1]
                and self._reach_history[-2]
            )
            else 0.0
        )

        # how_on_*: provenance of runners currently on each base
        feat_h1 = HOW_ON_ENC.get(self._runner_how_on.get(1), 0) / 5.0
        feat_h2 = HOW_ON_ENC.get(self._runner_how_on.get(2), 0) / 5.0
        feat_h3 = HOW_ON_ENC.get(self._runner_how_on.get(3), 0) / 5.0

        # runs_this_inning: accumulated runs in current half-inning before this PA
        feat_ri = float(self._runs_this_inning) / 5.0

        feat_bs = _i(gs.get("base_state"), 0) / 7.0
        feat_outs = _i(gs.get("outs"), 0) / 2.0
        feat_tto = _i(gs.get("tto"), 1) / 3.0
        feat_inn = (_i(gs.get("inning"), 1) - 1) / 8.0
        feat_half = 1.0 if gs.get("half") == "bot" else 0.0

        # Pitch mix rates
        fb_count = 0
        br_count = 0
        ch_count = 0
        zone_count = 0
        for pf in pa_pitches:
            pt = pf.get("pitch_type") or ""
            if pt in FASTBALL_TYPES:
                fb_count += 1
            elif pt in BREAKING_TYPES:
                br_count += 1
            elif pt in CHANGE_TYPES:
                ch_count += 1
            px = pf.get("plate_x", 0.0)
            pz = pf.get("plate_z", 2.5)
            if abs(float(px)) < 0.83 and 1.5 < float(pz) < 3.5:
                zone_count += 1

        feat_fb_rate = fb_count / n_pitches
        feat_br_rate = br_count / n_pitches
        feat_ch_rate = ch_count / n_pitches
        feat_zone_rate = zone_count / n_pitches
        feat_np = n_pitches / 10.0

        stand = gs.get("stand", "R")
        feat_stand = -1.0 if stand == "L" else 1.0

        pa_outcome_idx = PA_OUTCOME_IDX.get(pa_outcome, -1)
        outcome_onehot = np.zeros(8, dtype=np.float32)
        if 0 <= pa_outcome_idx < 8:
            outcome_onehot[pa_outcome_idx] = 1.0

        feat_bs_after = float(bs_after) / 7.0

        vec = np.array([
            feat_pcg, feat_pci, feat_velo, feat_pbr, feat_ptr,
            feat_h1, feat_h2, feat_h3, feat_ri, feat_bs,
            feat_outs, feat_tto, feat_inn, feat_half,
            feat_fb_rate, feat_br_rate, feat_ch_rate, feat_zone_rate,
            feat_np, feat_stand,
        ], dtype=np.float32)
        vec = np.concatenate([vec, outcome_onehot, [feat_bs_after]])  # (29,)
        return vec

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
        pa_game_idx: int = 0,
    ) -> dict[str, Any]:
        """Simulate one plate appearance with game memory context.

        Returns outcome dict. Also appends a PA summary to self.game_memory.
        """
        balls = game_state.get("balls", 0)
        strikes = game_state.get("strikes", 0)
        bs = game_state.get("base_state", 0)
        outs = game_state.get("outs", 0)

        # Detect inning/half changes and reset per-half-inning tracking
        self._reset_half_inning_if_changed(
            game_state.get("inning", 1), game_state.get("half", "top")
        )

        # Save starting game state for PA summary computation
        game_state_start = dict(game_state)

        # HBP: apply once per PA. No pitch sampled → velo_delta unknown, leave at 0.
        if rng.random() < self.hbp_rate:
            game_state_start["velo_delta"] = 0.0
            bs_after, runs = _empirical_transition(bs, outs, "HBP", self.transition_table, rng)
            pa_summary = self._compute_pa_summary([], "HBP", game_state_start, bs_after)
            self.game_memory.append(pa_summary)
            return self._finalize_pa("HBP", bs_after, runs, n_pitches=1)

        pa_pitches: list[dict[str, Any]] = []

        for n in range(20):
            pitch_feats = forced_pitch if forced_pitch is not None else pitch_sampler.sample(pitcher_idx, rng)
            pa_pitches.append(pitch_feats)

            # On the first pitch of the PA, set velo_delta vs the pitcher's
            # baseline velocity (their first observed pitch this game).
            if n == 0:
                first_velo = float(pitch_feats.get("release_speed_norm", 92.0))
                if pitcher_idx not in self._pitcher_baseline_velo:
                    self._pitcher_baseline_velo[pitcher_idx] = first_velo
                game_state_start["velo_delta"] = (
                    first_velo - self._pitcher_baseline_velo[pitcher_idx]
                )

            state = dict(game_state)
            state["balls"] = balls
            state["strikes"] = strikes
            state["base_state"] = bs
            state["outs"] = outs

            within_pa_pos = n

            with torch.no_grad():
                batch = self._build_batch(
                    pitcher_idx, batter_idx, umpire_idx, park_idx,
                    state, pitch_feats, within_pa_pos, pa_game_idx,
                )
                outputs = self.model(batch)
                scaled = self.scaler.scale(outputs)

            # Outputs are (B=1, N=1) — take [0, 0]
            swing_logit = float(scaled["swing"][0, 0].cpu().item())
            contact_logit = float(scaled["contact"][0, 0].cpu().item())
            foul_logit = float(scaled["foul"][0, 0].cpu().item())
            pa_outcome_logits = scaled["pa_outcome"][0, 0].cpu().numpy()
            pa_outcome_temp = float(self.scaler.pa_outcome_temp.cpu().item())

            plate_x = pitch_feats["plate_x"]
            plate_z = pitch_feats["plate_z"]

            if rng.random() < _sigmoid(swing_logit):
                if rng.random() < _sigmoid(contact_logit):
                    if rng.random() < _sigmoid(foul_logit):
                        strikes = min(strikes + 1, 2)
                        continue
                    # Ball in play
                    ip_logits = pa_outcome_logits[IN_PLAY_INDICES]
                    ip_probs = F.softmax(
                        torch.tensor(ip_logits / pa_outcome_temp), dim=0
                    ).numpy()
                    ip_probs = ip_probs / ip_probs.sum()
                    outcome = IN_PLAY_OUTCOMES_LIST[int(rng.choice(len(IN_PLAY_OUTCOMES_LIST), p=ip_probs))]
                    bs_after, runs = _empirical_transition(bs, outs, outcome, self.transition_table, rng)
                    pa_summary = self._compute_pa_summary(pa_pitches, outcome, game_state_start, bs_after)
                    self.game_memory.append(pa_summary)
                    return self._finalize_pa(outcome, bs_after, runs, n_pitches=n + 1)
                else:
                    strikes += 1
                    if strikes >= 3:
                        bs_after, runs = _empirical_transition(bs, outs, "K", self.transition_table, rng)
                        pa_summary = self._compute_pa_summary(pa_pitches, "K", game_state_start, bs_after)
                        self.game_memory.append(pa_summary)
                        return self._finalize_pa("K", bs_after, runs, n_pitches=n + 1)
            else:
                if _in_strike_zone(plate_x, plate_z):
                    strikes += 1
                    if strikes >= 3:
                        bs_after, runs = _empirical_transition(bs, outs, "K", self.transition_table, rng)
                        pa_summary = self._compute_pa_summary(pa_pitches, "K", game_state_start, bs_after)
                        self.game_memory.append(pa_summary)
                        return self._finalize_pa("K", bs_after, runs, n_pitches=n + 1)
                else:
                    balls += 1
                    if balls >= 4:
                        bs_after, runs = _empirical_transition(bs, outs, "BB", self.transition_table, rng)
                        pa_summary = self._compute_pa_summary(pa_pitches, "BB", game_state_start, bs_after)
                        self.game_memory.append(pa_summary)
                        return self._finalize_pa("BB", bs_after, runs, n_pitches=n + 1)

        # Fallback: walk after 20 pitches
        bs_after, runs = _empirical_transition(bs, outs, "BB", self.transition_table, rng)
        pa_summary = self._compute_pa_summary(pa_pitches, "BB", game_state_start, bs_after)
        self.game_memory.append(pa_summary)
        return self._finalize_pa("BB", bs_after, runs, n_pitches=20)

    def _finalize_pa(
        self,
        outcome: str,
        bs_after: int,
        runs: int,
        n_pitches: int,
    ) -> dict[str, Any]:
        """Update tracking state after a PA completes and return the result dict."""
        self._runs_this_inning += int(runs)
        reached = outcome in {"BB", "HBP", "1B", "2B", "3B", "HR", "E"}
        self._reach_history.append(reached)
        self._runner_how_on = self._update_runner_how_on(
            bs_after, outcome, self._runner_how_on
        )
        return {
            "pa_outcome": outcome,
            "runs_scored": runs,
            "n_pitches": n_pitches,
            "base_state_after": bs_after,
        }


class GameContextGameSimulator:
    """Simulate full 9-inning games using GameContextPASimulator with live game memory."""

    def __init__(
        self,
        pa_sim: GameContextPASimulator,
        pitch_sampler: PitchSampler,
        transition_table: dict,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.pa_sim = pa_sim
        self.pitch_sampler = pitch_sampler
        self.transition_table = transition_table
        self.rng = rng if rng is not None else np.random.default_rng()

    def simulate_game(
        self,
        pitcher_pool: list[int],
        batter_pool: list[int],
        umpire_pool: list[int],
        park_idx: int,
        game_id: int = 0,
    ) -> GameLog:
        """Simulate one 9-inning game with game memory accumulation."""
        inning_runs_away: list[int] = []
        inning_runs_home: list[int] = []

        pitcher_pool = pitcher_pool or [0]
        batter_pool = batter_pool or [0]
        umpire_pool = umpire_pool or [0]

        # Reset game memory at the start of each game
        self.pa_sim.reset_game()

        tto_tracker: dict[int, int] = {}
        pa_game_counter = 0  # sequential PA index across the entire game

        for inning in range(1, 10):
            for half in ["top", "bot"]:
                base_state = 0
                outs = 0
                runs = 0
                ab_num = 0
                score_diff = sum(inning_runs_away) - sum(inning_runs_home)
                pitch_count_game = 0

                while outs < 3:
                    pitcher_idx = int(self.rng.choice(pitcher_pool))
                    batter_idx = int(self.rng.choice(batter_pool))
                    umpire_idx = int(self.rng.choice(umpire_pool))

                    tto_tracker[pitcher_idx] = tto_tracker.get(pitcher_idx, 0) + 1
                    tto = min(3, tto_tracker[pitcher_idx] // 9 + 1)

                    game_state: dict[str, Any] = {
                        "base_state": base_state,
                        "outs": outs,
                        "inning": inning,
                        "half": half,
                        "score_diff": score_diff,
                        "tto": tto,
                        "balls": 0,
                        "strikes": 0,
                        "pitch_count_game": pitch_count_game,
                        "pitch_count_inning": ab_num,
                        "tracking_era": 1,
                        "stand": "R",
                        "p_throws": "R",
                    }

                    result = self.pa_sim.simulate_pa(
                        pitcher_idx, batter_idx, umpire_idx, park_idx,
                        game_state, self.pitch_sampler, rng=self.rng,
                        pa_game_idx=pa_game_counter,
                    )
                    pa_game_counter += 1

                    bs_after = result.get("base_state_after", base_state)
                    pa_runs = result.get("runs_scored", 0)
                    outcome = result.get("pa_outcome", "out")

                    runs += pa_runs
                    score_diff += pa_runs
                    pitch_count_game += result.get("n_pitches", 1)

                    if outcome in ("K", "out"):
                        outs += 1

                    base_state = bs_after
                    ab_num += 1

                if half == "top":
                    inning_runs_away.append(runs)
                else:
                    inning_runs_home.append(runs)

        return GameLog(
            game_id=game_id,
            home_runs=sum(inning_runs_home),
            away_runs=sum(inning_runs_away),
            inning_runs_home=inning_runs_home,
            inning_runs_away=inning_runs_away,
        )

    def simulate_season(
        self,
        pitcher_pool: list[int],
        batter_pool: list[int],
        umpire_pool: list[int],
        park_idx: int,
        n_games: int = 2430,
    ) -> "pl.DataFrame":
        """Simulate a full season and return a DataFrame for the eval harness."""
        from diamondworld.baselines.base import game_logs_to_frame
        logs = [
            self.simulate_game(
                pitcher_pool, batter_pool, umpire_pool, park_idx, game_id=i
            )
            for i in range(n_games)
        ]
        return game_logs_to_frame(logs)


# Aliases expected by eval_phase4.py (already uses these names)
Phase4PASimulator = GameContextPASimulator
Phase4GameSimulator = GameContextGameSimulator
