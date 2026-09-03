"""Joint NumPyro model: wires all DiamondWorldJAX components into one probabilistic program.

Call signature
--------------
    diamondworld_model(batch, player_table, teacher_force=True)

In teacher_force=True mode (SVI training) every sample site receives the
corresponding observed value from `batch` and scores its log-likelihood.

In teacher_force=False mode all targets are unobserved. Calling this function
directly remains a conditional posterior-predictive draw over the supplied batch;
use ``autoregressive_joint_rollout_samples`` for a generated pitch history.
"""
from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .embeddings import encode_players_numpyro, SKILL_DIM
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


def _make_pre_pitch_features(batch: dict) -> jnp.ndarray:
    """Features known before the current pitch is selected and thrown."""
    return jnp.stack([
        batch["pitch_count_game"],
        batch["tto"],
        batch["base_state"],
    ], axis=-1)


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
    player_skills_override: jnp.ndarray | None = None,
    direct_player_context: bool = False,
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
        False → unobserved conditional draw. Use the autoregressive rollout
                 utility for a generated pitch-and-state trajectory.
    """
    B, T = batch["pitch_valid"].shape
    P = player_table["stats"].shape[0]

    # ------------------------------------------------------------------ #
    # 0. Per-player latent skill vectors  (P, SKILL_DIM)                  #
    #    Sampled once per forward pass outside the pitch plates.           #
    #    Prior: N(0, I).  Guide provides the variational posterior.        #
    # ------------------------------------------------------------------ #
    if player_skills_override is None:
        player_skills = numpyro.sample(
            "player_skills",
            dist.Normal(
                jnp.zeros((P, SKILL_DIM)),
                jnp.ones((P, SKILL_DIM)),
            ).to_event(2),
        )
    else:
        if player_skills_override.shape != (P, SKILL_DIM):
            raise ValueError("player_skills_override must have shape (P, SKILL_DIM)")
        player_skills = player_skills_override

    # ------------------------------------------------------------------ #
    # 1. Player embeddings  →  pitcher_z, batter_z  (B, T, 64)           #
    # ------------------------------------------------------------------ #
    pitcher_z, batter_z = encode_players_numpyro(
        player_table["stats"],
        player_table["league"],
        player_table["hand"],
        pitcher_ids   = batch["pitcher_ids"],
        batter_ids    = batch["batter_ids"],
        player_skills = player_skills,
    )  # (B, T, 64) each

    # ------------------------------------------------------------------ #
    # 2. Fatigue rollout  →  fatigue_state  (B, T, 16)                    #
    # ------------------------------------------------------------------ #
    pitch_pkg_12     = _make_pitch_pkg_12(batch)                # (B, T, 12)
    game_state_scalar = batch["pitch_count_game"][..., None]    # (B, T, 1) proxy

    # A zero pre-pitch count marks a genuine PA boundary; it must not reset
    # fatigue on every pitch.
    pc_pa_int = _as_int_field(batch["pitch_count_pa"], 10.0)

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
    pre_pitch = _make_pre_pitch_features(batch)
    outs_int        = _as_int_field(batch["outs"], 2.0)

    # ------------------------------------------------------------------ #
    # All (B, T) sample sites live inside these plates so that            #
    # AutoGuides can distinguish batch from event dimensions.             #
    # ------------------------------------------------------------------ #
    with (numpyro.plate("games", B, dim=-2), numpyro.plate("pitches", T, dim=-1),
          numpyro.handlers.mask(mask=batch["pitch_valid"])):

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
            pitching_change_mask = batch.get("mgr_pitch_change_mask") if teacher_force else None,
            steal_mask           = batch.get("mgr_steal_mask") if teacher_force else None,
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
            pitcher_z         = pitcher_z if direct_player_context else None,
            batter_z          = batter_z if direct_player_context else None,
        )  # (B, T, 128)

        # Hurdle: pitch_type, location, swing/contact tree
        hurdle = hurdle_numpyro(
            shared_context  = shared_context,
            pre_pitch       = pre_pitch,
            obs_pitch_type    = _obs_or_none(batch["pitch_type"])           if teacher_force else None,
            obs_plate_x       = batch["plate_x"]                            if teacher_force else None,
            obs_plate_z       = batch["plate_z"]                            if teacher_force else None,
            obs_release_speed = batch["release_speed"]                      if teacher_force else None,
            obs_swing         = _obs_or_none(batch["obs_swing"])            if teacher_force else None,
            obs_called_strike = _obs_or_none(batch.get("obs_called_strike")) if teacher_force else None,
            obs_contact       = _obs_or_none(batch["obs_contact"])          if teacher_force else None,
            obs_foul          = _obs_or_none(batch["obs_foul"])             if teacher_force else None,
            pitch_type_mask   = batch["pitch_type"] >= 0 if teacher_force else None,
            plate_x_mask      = batch["pitch_valid"] if teacher_force else None,
            plate_z_mask      = batch["pitch_valid"] if teacher_force else None,
            release_speed_mask = batch["pitch_valid"] if teacher_force else None,
            swing_mask        = batch.get("swing_mask") if teacher_force else None,
            called_strike_mask = batch.get("called_strike_mask") if teacher_force else None,
            contact_mask      = batch.get("contact_mask") if teacher_force else None,
            foul_mask         = batch.get("foul_mask") if teacher_force else None,
        )

        # Batted-ball physics sees the realised generated pitch. pfx_x/pfx_z
        # are not generated by this model, so keep their slots neutral rather
        # than leaking their observed values into a free rollout.
        batted_ball_execution = jnp.stack([
            hurdle.release_speed,
            jnp.zeros_like(hurdle.release_speed),
            jnp.zeros_like(hurdle.release_speed),
            hurdle.plate_x,
            hurdle.plate_z,
            batch["pitch_count_game"],
            batch["tto"],
            batch["base_state"],
        ], axis=-1)
        in_play_mask = batch["in_play_mask"] if teacher_force else hurdle.in_play.astype(bool)

        # Batted-ball physics
        bb = batted_ball_numpyro(
            shared_context    = shared_context,
            pitch_execution   = batted_ball_execution,
            batter_z          = batter_z,
            pitcher_z         = pitcher_z,
            park_id           = batch["park_ids"],
            in_play_mask      = in_play_mask,
            obs_launch_speed  = batch["launch_speed"]  if teacher_force else None,
            obs_launch_angle  = batch["launch_angle"]  if teacher_force else None,
            obs_spray_angle   = batch["spray_angle"]   if teacher_force else None,
            obs_hit_distance  = batch["hit_distance"]  if teacher_force else None,
            launch_speed_mask = batch.get("launch_speed_mask") if teacher_force else None,
            launch_angle_mask = batch.get("launch_angle_mask") if teacher_force else None,
            spray_angle_mask  = batch.get("spray_angle_mask") if teacher_force else None,
            hit_distance_mask = batch.get("hit_distance_mask") if teacher_force else None,
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
            in_play_mask         = in_play_mask,
            obs_pa_outcome       = _obs_or_none(batch["pa_outcome"]) if teacher_force else None,
            obs_runs_scored      = _obs_or_none(batch["runs_scored"])       if teacher_force else None,
            obs_base_state_after = _obs_or_none(batch["base_state_after"])  if teacher_force else None,
            obs_outs_added       = _obs_or_none(batch["outs_added"]) if teacher_force else None,
            pa_outcome_mask      = batch.get("pa_outcome_mask") if teacher_force else None,
            runs_mask            = batch.get("runs_mask") if teacher_force else None,
            base_state_after_mask = batch.get("base_state_after_mask") if teacher_force else None,
            outs_added_mask       = batch.get("outs_added_mask") if teacher_force else None,
        )
