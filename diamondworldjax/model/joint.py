"""Joint NumPyro model: wires all DiamondWorldJAX components into one probabilistic program.

Call signature
--------------
    diamondworld_model(batch, player_table, teacher_force=True)

In teacher_force=True mode (SVI training) every sample site receives the
corresponding observed value from `batch` and scores its log-likelihood.

In teacher_force=False mode (posterior predictive / free rollout) all obs
are None and the model samples freely from its learned distributions.
"""
from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpyro

from .embeddings import encode_players_numpyro
from .fatigue import fatigue_rollout
from .manager import manager_decisions_numpyro
from .pitch_transformer import pitch_transformer_numpyro
from .hurdle import hurdle_numpyro
from .batted_ball import batted_ball_numpyro
from .transition import transition_numpyro

# History window for pitch transformer
H = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_history(
    pitch_type: jnp.ndarray,    # (B, T) int32
    plate_x: jnp.ndarray,       # (B, T) float32
    plate_z: jnp.ndarray,       # (B, T) float32
    release_speed: jnp.ndarray, # (B, T) float32
    game_state_8: jnp.ndarray,  # (B, T, 8) float32
    valid_mask: jnp.ndarray,    # (B, T) bool
    window: int = H,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Build sliding-window history tensors.

    For each position t the history is the `window` pitches immediately
    preceding it (oldest → newest).  Positions before t=0 are zero-padded.

    Returns
    -------
    hist_pitch_type : (B, T, H) int32
    hist_location   : (B, T, H, 4) float32  — (plate_x, plate_z, rel_spd, 0)
    hist_outcome    : (B, T, H) int32        — reuses pitch_type as outcome token
    hist_game_state : (B, T, H, 8) float32
    history_mask    : (B, T, H) bool         — True = valid (non-padding) token
    """
    B, T = valid_mask.shape

    def _pad(arr: jnp.ndarray) -> jnp.ndarray:
        """Prepend `window` zero-slices along axis=1."""
        zeros = jnp.zeros((B, window) + arr.shape[2:], dtype=arr.dtype)
        return jnp.concatenate([zeros, arr], axis=1)  # (B, T+H, ...)

    pt_pad = _pad(pitch_type)                               # (B, T+H)
    px_pad = _pad(plate_x)
    pz_pad = _pad(plate_z)
    rs_pad = _pad(release_speed)
    gs_pad = _pad(game_state_8)                             # (B, T+H, 8)
    vm_pad = _pad(valid_mask.astype(jnp.float32))           # (B, T+H)

    # For position t, history indices into the padded array are [t, t+1, ..., t+H-1]
    idx_t = jnp.arange(T)[:, None]        # (T, 1)
    idx_h = jnp.arange(window)[None, :]   # (1, H)
    idx   = idx_t + idx_h                 # (T, H)  values in [0, T+H-1]

    hist_pt  = pt_pad[:, idx].astype(jnp.int32)    # (B, T, H)
    hist_px  = px_pad[:, idx]                       # (B, T, H)
    hist_pz  = pz_pad[:, idx]                       # (B, T, H)
    hist_rs  = rs_pad[:, idx]                       # (B, T, H)
    hist_gs  = gs_pad[:, idx]                       # (B, T, H, 8)
    hist_vm  = vm_pad[:, idx].astype(bool)          # (B, T, H)

    # Location: (plate_x, plate_z, release_speed, zero-pad)
    hist_location = jnp.stack(
        [hist_px, hist_pz, hist_rs, jnp.zeros_like(hist_rs)], axis=-1
    )  # (B, T, H, 4)

    return hist_pt, hist_location, hist_pt, hist_gs, hist_vm  # outcome ≡ pitch_type


def _make_pitch_execution(batch: dict) -> jnp.ndarray:
    """
    Assemble 8-dim pitch execution features from batch.

    Uses: release_speed, pfx_x, pfx_z, plate_x, plate_z,
          pitch_count_game, tto, base_state  (all pre-normalised by batching.py)
    """
    return jnp.stack([
        batch["release_speed"],
        batch["pfx_x"],
        batch["pfx_z"],
        batch["plate_x"],
        batch["plate_z"],
        batch["pitch_count_game"],
        batch["tto"],
        batch["base_state"],
    ], axis=-1)  # (B, T, 8)


def _make_pitch_pkg_12(batch: dict) -> jnp.ndarray:
    """
    Assemble 12-dim pitch package features for the FatigueCell.

    Extends the 8-dim execution features with 4 count/state features.
    """
    exec8 = _make_pitch_execution(batch)                 # (B, T, 8)
    extra = jnp.stack([
        batch["balls"],
        batch["strikes"],
        batch["outs"],
        batch["half"],
    ], axis=-1)                                          # (B, T, 4)
    return jnp.concatenate([exec8, extra], axis=-1)      # (B, T, 12)


def _make_game_state_8(batch: dict) -> jnp.ndarray:
    """8-dim normalised game-state vector for pitch history tokens."""
    return jnp.stack([
        batch["balls"],
        batch["strikes"],
        batch["outs"],
        batch["base_state"],
        batch["score_diff"],
        batch["tto"],
        batch["inning"],
        batch["half"],
    ], axis=-1)  # (B, T, 8)


def _derive_outcome_type(batch: dict) -> jnp.ndarray:
    """
    Map pa_outcome integer to transition outcome_type code.

    0 = strikeout, 1 = walk/HBP, 2 = in_play, 3 = no_outcome
    """
    pa_out   = batch["pa_outcome"]      # -1 for non-terminal
    in_play  = batch["in_play_mask"]

    is_k      = (pa_out == 0)
    is_bb_hbp = (pa_out == 1) | (pa_out == 2)
    is_ip     = in_play.astype(bool)

    return jnp.where(is_k, 0,
           jnp.where(is_bb_hbp, 1,
           jnp.where(is_ip, 2, 3))).astype(jnp.int32)


def _as_int_field(normalised: jnp.ndarray, scale: float) -> jnp.ndarray:
    """Recover integer field from batch float. E.g. base_state: ×7 → round."""
    return jnp.round(normalised * scale).astype(jnp.int32)


def _obs_or_none(arr: jnp.ndarray, sentinel: int = -1) -> Optional[jnp.ndarray]:
    """Replace sentinel values with 0 and return, or None if arr is None.

    Pass None explicitly to leave a site unobserved.  We do NOT use
    jnp.all() as a Python branch here because that would fail under JAX
    JIT tracing.  Instead, sentinel (-1) positions are replaced with 0
    (a valid in-distribution value) so they contribute a fixed but harmless
    log-prob contribution; padded positions are further down-weighted by the
    pitch_valid mask outside this function.
    """
    if arr is None:
        return None
    return jnp.where(arr == sentinel, 0, arr)


# ---------------------------------------------------------------------------
# Master model
# ---------------------------------------------------------------------------

def diamondworld_model(
    batch: dict,
    player_table: dict,
    teacher_force: bool = True,
) -> None:
    """
    Full DiamondWorldJAX NumPyro generative model.

    Parameters
    ----------
    batch : dict
        Output of build_batch().  Shape (B, T, ...).
    player_table : dict
        Keys: 'stats' float32 (P, F), 'league' int32 (P,), 'hand' int32 (P,).
    teacher_force : bool
        True  → score observed pitches (SVI training).
        False → free rollout from the posterior predictive.
    """
    B, T = batch["pitch_valid"].shape

    # ------------------------------------------------------------------ #
    # 1. Player embeddings  →  pitcher_z, batter_z  (B, T, 64)           #
    # ------------------------------------------------------------------ #
    pitcher_z, batter_z = encode_players_numpyro(
        player_table["stats"],
        player_table["league"],
        player_table["hand"],
        pitcher_ids = batch["pitcher_ids"],
        batter_ids  = batch["batter_ids"],
    )  # (B, T, 64) each

    # ------------------------------------------------------------------ #
    # 2. Fatigue rollout  →  fatigue_state  (B, T, 16)                    #
    # ------------------------------------------------------------------ #
    pitch_pkg_12     = _make_pitch_pkg_12(batch)                # (B, T, 12)
    game_state_scalar = batch["pitch_count_game"][..., None]    # (B, T, 1) proxy

    # pitch_count_plate_appearance: 0 triggers reset.  Use a zeros proxy so
    # fatigue resets every pitch — conservative but safe for v0.
    pc_pa_int = jnp.zeros((B, T), dtype=jnp.int32)

    fatigue_state = fatigue_rollout(
        pitcher_z                    = pitcher_z,
        pitch_package_features       = pitch_pkg_12,
        game_state_scalar            = game_state_scalar,
        pitch_count_plate_appearance = pc_pa_int,
        obs_fatigue                  = None,    # latent: never observed
    )  # (B, T, 16)

    # ------------------------------------------------------------------ #
    # 3. Manager decisions  →  ManagerDecisions                           #
    # ------------------------------------------------------------------ #
    base_state_int = _as_int_field(batch["base_state"], 7.0)   # (B, T) int
    base_state_oh  = jax.nn.one_hot(
        jnp.clip(base_state_int, 0, 7), num_classes=8
    )                                                           # (B, T, 8)

    steal_valid = (base_state_int > 0)  # runner on base

    # ------------------------------------------------------------------ #
    # 4. Pitch transformer  →  shared_context  (B, T, 128)                #
    # ------------------------------------------------------------------ #
    game_state_8 = _make_game_state_8(batch)                   # (B, T, 8)
    pitch_type_safe = jnp.clip(batch["pitch_type"], 0, 7).astype(jnp.int32)

    hist_pt, hist_loc, hist_oc, hist_gs, hist_mask = _build_history(
        pitch_type    = pitch_type_safe,
        plate_x       = batch["plate_x"],
        plate_z       = batch["plate_z"],
        release_speed = batch["release_speed"],
        game_state_8  = game_state_8,
        valid_mask    = batch["pitch_valid"],
    )

    # ------------------------------------------------------------------ #
    # 5. Pitch execution features  (B, T, 8)                              #
    # ------------------------------------------------------------------ #
    pitch_execution = _make_pitch_execution(batch)
    in_play_mask    = batch["in_play_mask"]
    outs_int        = _as_int_field(batch["outs"], 2.0)
    outcome_type    = _derive_outcome_type(batch)

    # ------------------------------------------------------------------ #
    # All (B, T) sample sites live inside these plates so that            #
    # AutoGuides can distinguish batch from event dimensions.             #
    # ------------------------------------------------------------------ #
    with numpyro.plate("games", B, dim=-2), numpyro.plate("pitches", T, dim=-1):

        # Manager decisions (sampled before transformer since mgr_vec feeds it)
        mgr = manager_decisions_numpyro(
            inning            = batch["inning"][..., None],
            outs              = batch["outs"][..., None],
            score_diff        = batch["score_diff"][..., None],
            pitch_count_game  = batch["pitch_count_game"][..., None],
            base_state_onehot = base_state_oh,
            pitcher_z         = pitcher_z,
            batter_z          = batter_z,
            steal_valid_mask  = steal_valid,
            obs_pitching_change = _obs_or_none(batch.get("mgr_pitch_change")) if teacher_force else None,
            obs_steal           = _obs_or_none(batch.get("mgr_steal"))        if teacher_force else None,
            obs_runner_send     = None,
            obs_alignment       = None,
        )

        # Use soft probabilities (not sampled ints) so shape is stable under
        # funsor enumeration which adds extra leading dims to discrete samples.
        mgr_vec = jnp.stack([
            mgr.pc_prob,
            mgr.st_prob,
            mgr.rs_prob,
            mgr.al_probs[..., 0],  # P(standard alignment)
        ], axis=-1)  # (B, T, 4)

        shared_context = pitch_transformer_numpyro(
            hist_pitch_type  = hist_pt,
            hist_location    = hist_loc,
            hist_outcome     = hist_oc,
            hist_game_state  = hist_gs,
            history_mask     = hist_mask,
            fatigue_state    = fatigue_state,
            manager_decision = mgr_vec,
        )  # (B, T, 128)

        # Hurdle: pitch_type, location, swing/contact tree
        hurdle = hurdle_numpyro(
            shared_context  = shared_context,
            pitch_execution = pitch_execution,
            obs_pitch_type    = _obs_or_none(batch["pitch_type"])           if teacher_force else None,
            obs_plate_x       = batch["plate_x"]                            if teacher_force else None,
            obs_plate_z       = batch["plate_z"]                            if teacher_force else None,
            obs_release_speed = batch["release_speed"]                      if teacher_force else None,
            obs_swing         = _obs_or_none(batch["obs_swing"])            if teacher_force else None,
            obs_called_strike = _obs_or_none(batch.get("obs_called_strike")) if teacher_force else None,
            obs_contact       = _obs_or_none(batch["obs_contact"])          if teacher_force else None,
            obs_foul          = _obs_or_none(batch["obs_foul"])             if teacher_force else None,
        )

        # Batted-ball physics
        bb = batted_ball_numpyro(
            shared_context    = shared_context,
            pitch_execution   = pitch_execution,
            batter_z          = batter_z,
            pitcher_z         = pitcher_z,
            park_id           = batch["park_ids"],
            in_play_mask      = in_play_mask,
            obs_launch_speed  = batch["launch_speed"]  if teacher_force else None,
            obs_launch_angle  = batch["launch_angle"]  if teacher_force else None,
            obs_spray_angle   = batch["spray_angle"]   if teacher_force else None,
            obs_hit_distance  = batch["hit_distance"]  if teacher_force else None,
        )

        # State transition
        batted_ball_features = jnp.stack([
            bb.launch_speed, bb.launch_angle, bb.spray_angle, bb.hit_distance
        ], axis=-1)  # (B, T, 4)

        transition_numpyro(
            shared_context       = shared_context,
            batted_ball_features = batted_ball_features,
            base_state           = base_state_int,
            outs                 = outs_int,
            outcome_type         = outcome_type,
            in_play_mask         = in_play_mask,
            obs_runs_scored      = _obs_or_none(batch["runs_scored"])       if teacher_force else None,
            obs_base_state_after = _obs_or_none(batch["base_state_after"])  if teacher_force else None,
            obs_error_flag       = None,
            obs_outs_added       = None,
            obs_wild_pitch       = None,
            obs_passed_ball      = None,
            obs_balk             = None,
        )
