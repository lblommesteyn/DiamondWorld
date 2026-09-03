"""Build padded JAX game-sequence batches from pitch DataFrames.

Output tensor shapes follow spec.md section 8:
  B = batch (games), T = max padded pitches per game
"""
from __future__ import annotations
from typing import Any
import numpy as np
import jax.numpy as jnp
import polars as pl

# Max pitches to keep per game (clips very long games)
MAX_T: int = 350


def _safe_float(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
    out = arr.astype(np.float32)
    out[np.isnan(out)] = fill
    return out


def _safe_int(arr: np.ndarray, fill: int = 0) -> np.ndarray:
    mask = np.isnan(arr.astype(np.float64))
    out = arr.astype(np.int32)
    out[mask] = fill
    return out


def build_batch(df: pl.DataFrame, max_t: int = MAX_T) -> dict[str, Any]:
    """Convert a polars DataFrame of pitches into a padded JAX batch.

    Groups by game_pk, sorts by (at_bat_number, pitch_number), pads to max_t.
    Returns a dict of arrays with leading (B, T) dimensions.
    """
    def _col(name: str, default=None):
        if name in df.columns:
            return df[name].to_numpy()
        if default is not None:
            return np.full(len(df), default)
        return np.zeros(len(df))

    game_ids = df["game_pk"].to_numpy()
    unique_games = np.unique(game_ids)
    B = len(unique_games)

    # Pre-allocate arrays
    def _alloc_f(fill=0.0):  return np.full((B, max_t), fill, dtype=np.float32)
    def _alloc_i(fill=0):    return np.full((B, max_t), fill, dtype=np.int32)
    def _alloc_b(fill=False): return np.full((B, max_t), fill, dtype=bool)

    pitch_valid   = _alloc_b()
    terminal_mask = _alloc_b()
    in_play_mask  = _alloc_b()
    swing_mask    = _alloc_b()
    called_strike_mask = _alloc_b()
    contact_mask  = _alloc_b()
    foul_mask     = _alloc_b()
    mgr_pitch_change_mask = _alloc_b()
    mgr_steal_mask = _alloc_b()
    batted_mask   = _alloc_b()
    runs_mask     = _alloc_b()
    base_after_mask = _alloc_b()
    outs_added_mask = _alloc_b()
    pa_outcome_mask = _alloc_b()
    launch_speed_mask = _alloc_b()
    launch_angle_mask = _alloc_b()
    spray_angle_mask = _alloc_b()
    hit_distance_mask = _alloc_b()

    # State
    inning       = _alloc_f()
    half_bin     = _alloc_f()
    balls        = _alloc_f()
    strikes      = _alloc_f()
    outs_arr     = _alloc_f()
    base_state   = _alloc_f()
    score_diff   = _alloc_f()
    pc_game      = _alloc_f()
    pc_inning    = _alloc_f()
    pc_pa        = _alloc_f()
    tto_arr      = _alloc_f()
    shift_restr  = _alloc_f()
    pitch_clock  = _alloc_f()

    # Pitch package
    pitch_type   = _alloc_i(-1)
    release_spd  = _alloc_f()
    plate_x      = _alloc_f()
    plate_z      = _alloc_f()
    pfx_x        = _alloc_f()
    pfx_z        = _alloc_f()

    # Hurdle
    swing_obs    = _alloc_i(-1)
    cs_obs       = _alloc_i(-1)
    contact_obs  = _alloc_i(-1)
    foul_obs     = _alloc_i(-1)
    in_play_obs  = _alloc_i(-1)

    # Batted
    launch_spd   = _alloc_f()
    launch_ang   = _alloc_f()
    spray_ang    = _alloc_f()
    hit_dist     = _alloc_f()

    # Transition
    pa_outcome   = _alloc_i(-1)
    runs_scored  = _alloc_i(0)
    bs_after     = _alloc_i(0)
    outs_added   = _alloc_i(-1)

    # Player IDs
    pitcher_ids  = _alloc_i(0)
    batter_ids   = _alloc_i(0)
    park_ids     = _alloc_i(0)

    # Manager (optional)
    mgr_pitch_change = _alloc_i(-1)
    mgr_steal        = _alloc_i(-1)

    def _get(col, default=0.0): return _col(col, default)

    def _observed(*names: str, missing: float = -1.0) -> np.ndarray:
        """Return the first available observed label, preserving null as NaN."""
        for name in names:
            if name in df.columns:
                return df[name].to_numpy().astype(np.float64)
        return np.full(len(df), missing, dtype=np.float64)

    raw_inning   = _get("inning", 1)
    raw_half     = _col("half_bin", 0) if "half_bin" in df.columns else np.zeros(len(df))
    raw_balls    = _get("balls")
    raw_strikes  = _get("strikes")
    raw_outs     = _get("outs")
    raw_bs       = _get("base_state")
    raw_sd       = _get("score_diff") if "score_diff" in df.columns else (
        _get("home_score") - _get("away_score")
    )
    raw_pcg      = _get("pitch_count_game")
    raw_pci      = _get("pitch_count_inning")
    raw_tto      = _get("tto", 1)
    raw_sr       = _get("shift_restricted")
    raw_pc       = _get("pitch_clock")
    raw_pt       = _col("pitch_type_idx", -1)
    raw_rspd     = _get("release_speed")
    raw_px       = _get("plate_x")
    raw_pz       = _get("plate_z")
    raw_pfxx     = _get("pfx_x")
    raw_pfxz     = _get("pfx_z")
    raw_ls       = _get("launch_speed")
    raw_la       = _get("launch_angle")
    raw_sa       = _get("spray_angle") if "spray_angle" in df.columns else np.zeros(len(df))
    raw_hd       = _get("hit_distance") if "hit_distance" in df.columns else np.zeros(len(df))
    raw_pao      = _col("pa_outcome_idx", -1)
    raw_runs     = _get("runs_scored")
    raw_bsa      = _get("base_state_after")
    raw_outs_added = _observed("outs_added")
    raw_pit      = _get("pitcher_id") if "pitcher_id" in df.columns else _get("pitcher_idx")
    raw_bat      = _get("batter_id") if "batter_id" in df.columns else _get("batter_idx")
    if "park_idx" in df.columns:
        raw_park = _get("park_idx")
    elif "park_id" in df.columns and np.issubdtype(df["park_id"].to_numpy().dtype, np.number):
        raw_park = _get("park_id")
    else:
        raw_park = np.zeros(len(df))
    raw_terminal = _col("pa_terminal", False).astype(bool)
    raw_pitch_number = _get("pitch_number", 1)
    raw_pc_pa = _get("pitch_count_pa") if "pitch_count_pa" in df.columns else raw_pitch_number - 1
    # Hurdle observations (encoded int8 by pipeline.py; -1 = missing).
    raw_swing_obs = _observed("swing_obs", "swing")
    raw_cs_obs = _observed("called_strike_obs", "called_strike")
    raw_contact_obs = _observed("contact_obs", "contact")
    raw_foul_obs = _observed("foul_obs", "foul")
    raw_in_play_obs = _observed("in_play_obs", "in_play")
    raw_mgr_pitch_change = _observed("mgr_pitch_change")
    raw_mgr_steal = _observed("mgr_steal")

    sort_key1 = _get("at_bat_number") if "at_bat_number" in df.columns else _get("game_pk")
    sort_key2 = _get("pitch_number")

    for b_idx, gid in enumerate(unique_games):
        mask = game_ids == gid
        order = np.lexsort((sort_key2[mask], sort_key1[mask]))
        n = min(int(mask.sum()), max_t)

        def _fill(dst, src, fill=0.0):
            vals = src[mask][order][:n]
            if np.issubdtype(vals.dtype, np.floating):
                nan_mask = np.isnan(vals)
                if nan_mask.any():
                    vals = vals.copy()
                    vals[nan_mask] = fill
            dst[b_idx, :n] = vals

        pitch_valid[b_idx, :n] = True
        _fill(inning,      (raw_inning - 1) / 8.0)
        _fill(half_bin,    raw_half)
        _fill(balls,       raw_balls / 3.0)
        _fill(strikes,     raw_strikes / 2.0)
        _fill(outs_arr,    raw_outs / 2.0)
        _fill(base_state,  raw_bs / 7.0)
        _fill(score_diff,  raw_sd / 10.0)
        _fill(pc_game,     raw_pcg / 100.0)
        _fill(pc_inning,   raw_pci / 30.0)
        _fill(pc_pa,       raw_pc_pa / 10.0)
        _fill(tto_arr,     raw_tto / 3.0)
        _fill(shift_restr, raw_sr)
        _fill(pitch_clock, raw_pc)
        _fill(pitch_type,  raw_pt, fill=-1)
        _fill(release_spd, (raw_rspd - 90.0) / 10.0)
        _fill(plate_x,     raw_px)
        _fill(plate_z,     raw_pz)
        _fill(pfx_x,       raw_pfxx)
        _fill(pfx_z,       raw_pfxz)
        _fill(launch_spd,  raw_ls / 100.0)
        _fill(launch_ang,  raw_la / 45.0)
        _fill(spray_ang,   raw_sa / 90.0)
        _fill(hit_dist,    raw_hd / 400.0)
        _fill(pa_outcome,  raw_pao, fill=-1)
        _fill(runs_scored, raw_runs)
        _fill(bs_after,    raw_bsa)
        _fill(outs_added,  raw_outs_added, fill=-1)
        _fill(pitcher_ids, raw_pit, fill=0)
        _fill(batter_ids,  raw_bat, fill=0)
        _fill(park_ids,    raw_park, fill=0)

        terminal_slice = raw_terminal[mask][order][:n]
        swing_slice = raw_swing_obs[mask][order][:n]
        contact_slice = raw_contact_obs[mask][order][:n]
        called_strike_slice = raw_cs_obs[mask][order][:n]
        foul_slice = raw_foul_obs[mask][order][:n]
        mgr_pitch_change_slice = raw_mgr_pitch_change[mask][order][:n]
        mgr_steal_slice = raw_mgr_steal[mask][order][:n]
        in_play_slice = raw_in_play_obs[mask][order][:n]
        outcome_slice = raw_pao[mask][order][:n]
        runs_slice = raw_runs[mask][order][:n]
        base_after_slice = raw_bsa[mask][order][:n]
        outs_added_slice = raw_outs_added[mask][order][:n]
        launch_speed_slice = raw_ls[mask][order][:n]
        launch_angle_slice = raw_la[mask][order][:n]
        spray_angle_slice = raw_sa[mask][order][:n]
        hit_distance_slice = raw_hd[mask][order][:n]

        terminal_mask[b_idx, :n] = terminal_slice
        swing_mask[b_idx, :n] = np.isfinite(swing_slice) & (swing_slice >= 0)
        called_strike_mask[b_idx, :n] = (
            (swing_slice == 0) & np.isfinite(called_strike_slice) & (called_strike_slice >= 0)
        )
        contact_mask[b_idx, :n] = (
            (swing_slice == 1) & np.isfinite(contact_slice) & (contact_slice >= 0)
        )
        foul_mask[b_idx, :n] = (
            (contact_slice == 1) & np.isfinite(foul_slice) & (foul_slice >= 0)
        )
        mgr_pitch_change_mask[b_idx, :n] = (
            np.isfinite(mgr_pitch_change_slice) & (mgr_pitch_change_slice >= 0)
        )
        mgr_steal_mask[b_idx, :n] = np.isfinite(mgr_steal_slice) & (mgr_steal_slice >= 0)
        in_play_mask[b_idx, :n] = np.isfinite(in_play_slice) & (in_play_slice == 1)
        batted_mask[b_idx, :n] = in_play_mask[b_idx, :n]
        runs_mask[b_idx, :n] = terminal_slice & np.isfinite(runs_slice)
        base_after_mask[b_idx, :n] = terminal_slice & np.isfinite(base_after_slice)
        outs_added_mask[b_idx, :n] = (
            terminal_slice
            & np.isfinite(outs_added_slice)
            & (outs_added_slice >= 0)
            & (outs_added_slice <= 3)
        )
        pa_outcome_mask[b_idx, :n] = terminal_slice & np.isfinite(outcome_slice) & (outcome_slice >= 0)
        launch_speed_mask[b_idx, :n] = batted_mask[b_idx, :n] & np.isfinite(launch_speed_slice)
        launch_angle_mask[b_idx, :n] = batted_mask[b_idx, :n] & np.isfinite(launch_angle_slice)
        spray_angle_mask[b_idx, :n] = batted_mask[b_idx, :n] & np.isfinite(spray_angle_slice)
        hit_distance_mask[b_idx, :n] = batted_mask[b_idx, :n] & np.isfinite(hit_distance_slice)

        # Fill hurdle obs from data (pipeline encodes swing/contact/foul as int8).
        _fill(swing_obs,   raw_swing_obs,   fill=-1)
        _fill(cs_obs,      raw_cs_obs,      fill=-1)
        _fill(contact_obs, raw_contact_obs, fill=-1)
        _fill(foul_obs,    raw_foul_obs,    fill=-1)
        _fill(in_play_obs, raw_in_play_obs, fill=-1)
        _fill(mgr_pitch_change, raw_mgr_pitch_change, fill=-1)
        _fill(mgr_steal, raw_mgr_steal, fill=-1)

    return {
        # Masks
        "pitch_valid":   jnp.array(pitch_valid),
        "terminal_mask": jnp.array(terminal_mask),
        "in_play_mask":  jnp.array(in_play_mask),
        "swing_mask":    jnp.array(swing_mask),
        "called_strike_mask": jnp.array(called_strike_mask),
        "contact_mask":  jnp.array(contact_mask),
        "foul_mask":     jnp.array(foul_mask),
        "mgr_pitch_change_mask": jnp.array(mgr_pitch_change_mask),
        "mgr_steal_mask": jnp.array(mgr_steal_mask),
        "batted_mask":   jnp.array(batted_mask),
        "runs_mask":     jnp.array(runs_mask),
        "base_state_after_mask": jnp.array(base_after_mask),
        "outs_added_mask": jnp.array(outs_added_mask),
        "pa_outcome_mask": jnp.array(pa_outcome_mask),
        "launch_speed_mask": jnp.array(launch_speed_mask),
        "launch_angle_mask": jnp.array(launch_angle_mask),
        "spray_angle_mask": jnp.array(spray_angle_mask),
        "hit_distance_mask": jnp.array(hit_distance_mask),
        # State
        "inning":        jnp.array(inning),
        "half":          jnp.array(half_bin),
        "balls":         jnp.array(balls),
        "strikes":       jnp.array(strikes),
        "outs":          jnp.array(outs_arr),
        "base_state":    jnp.array(base_state),
        "score_diff":    jnp.array(score_diff),
        "pitch_count_game":    jnp.array(pc_game),
        "pitch_count_inning":  jnp.array(pc_inning),
        "pitch_count_pa":      jnp.array(pc_pa),
        "tto":           jnp.array(tto_arr),
        "shift_restricted": jnp.array(shift_restr),
        "pitch_clock":   jnp.array(pitch_clock),
        # Pitch package
        "pitch_type":    jnp.array(pitch_type),
        "release_speed": jnp.array(release_spd),
        "plate_x":       jnp.array(plate_x),
        "plate_z":       jnp.array(plate_z),
        "pfx_x":         jnp.array(pfx_x),
        "pfx_z":         jnp.array(pfx_z),
        "obs_swing":     jnp.array(swing_obs),
        "obs_called_strike": jnp.array(cs_obs),
        "obs_contact":   jnp.array(contact_obs),
        "obs_foul":      jnp.array(foul_obs),
        "obs_in_play":   jnp.array(in_play_obs),
        # Batted ball
        "launch_speed":  jnp.array(launch_spd),
        "launch_angle":  jnp.array(launch_ang),
        "spray_angle":   jnp.array(spray_ang),
        "hit_distance":  jnp.array(hit_dist),
        # Transition
        "pa_outcome":    jnp.array(pa_outcome),
        "runs_scored":   jnp.array(runs_scored),
        "base_state_after": jnp.array(bs_after),
        "outs_added":    jnp.array(outs_added),
        # Player IDs
        "pitcher_ids":   jnp.array(pitcher_ids),
        "batter_ids":    jnp.array(batter_ids),
        "park_ids":      jnp.array(park_ids),
        # Game IDs for reference
        "game_ids":      unique_games,
    }
