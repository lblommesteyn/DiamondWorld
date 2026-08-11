"""Build PA-level batches from plate-appearance-terminal pitch rows.

Each row in the output represents one complete plate appearance.
Input must already be filtered to pa_terminal=True rows.
Output tensor shapes: (B, T_pa) where T_pa = MAX_PA.
"""
from __future__ import annotations
import numpy as np
import jax.numpy as jnp
import polars as pl

MAX_PA: int = 90  # covers extra-inning games comfortably


def build_pa_batch(pa_df: pl.DataFrame, max_pa: int = MAX_PA) -> dict:
    """
    Convert PA-terminal rows into a padded JAX batch.

    Groups by game_pk, sorts by at_bat_number, pads to max_pa.
    Returns a dict of arrays with shape (B, T_pa).
    """
    def _col(name, default=0.0):
        if name in pa_df.columns:
            return pa_df[name].to_numpy()
        return np.full(len(pa_df), default)

    game_ids_arr = pa_df["game_pk"].to_numpy()
    unique_games = np.unique(game_ids_arr)
    B = len(unique_games)

    def _f():  return np.zeros((B, max_pa), dtype=np.float32)
    def _i(v=0): return np.full((B, max_pa), v, dtype=np.int32)
    def _b():  return np.zeros((B, max_pa), dtype=bool)

    pa_valid         = _b()
    arr_inning       = _f()
    arr_half         = _f()
    arr_outs         = _f()
    arr_base_state   = _f()
    arr_score_diff   = _f()
    arr_tto          = _f()
    arr_shift        = _f()
    arr_clock        = _f()
    arr_pa_outcome   = _i(-1)
    arr_runs_scored  = _i(0)
    arr_bs_after     = _i(0)
    arr_pitcher_ids  = _i(0)
    arr_batter_ids   = _i(0)
    arr_park_ids     = _i(0)
    arr_pitch_count  = _f()   # Phase-4 fatigue: pitcher's cumulative game pitch count
    arr_bat_side     = _f()   # platoon: batter side this PA (R=1, L=0); 0.5 unknown
    arr_pit_hand     = _f()   # platoon: pitcher throw hand (R=1, L=0); 0.5 unknown
    arr_season       = _i(0)  # calendar season of this PA; consumed only by the
                              # random-walk skill prior (--skill-prior walk). Emitted
                              # unconditionally because an extra batch key is inert
                              # for every existing consumer.

    raw_inning   = _col("inning", 1.0)
    raw_half     = _col("half_bin", 0.0)
    raw_outs     = _col("outs", 0.0)
    raw_bs       = _col("base_state", 0.0)
    # score_diff: compute from home/away or use directly
    if "score_diff" in pa_df.columns:
        raw_sd = _col("score_diff")
    else:
        raw_sd = _col("home_score", 0.0) - _col("away_score", 0.0)
    raw_tto      = _col("tto", 1.0)
    raw_shift    = _col("shift_restricted", 0.0)
    raw_clock    = _col("pitch_clock", 0.0)
    raw_pao      = _col("pa_outcome_idx", -1.0)
    raw_runs     = _col("runs_scored", 0.0)
    raw_bsa      = _col("base_state_after", 0.0)
    raw_pc_game  = _col("pitch_count_game", 0.0)
    raw_at_bat   = _col("at_bat_number", 0.0)

    def _hand_col(*names):
        # 'R'/'L' strings; encode R=1.0, L=0.0, unknown=0.5. DWJAX schema renames
        # raw stand/p_throws -> batter_hand/pitcher_hand (try both).
        for name in names:
            if name in pa_df.columns:
                s = pa_df[name].to_numpy()
                out = np.full(len(pa_df), 0.5, dtype=np.float32)
                out[s == "R"] = 1.0
                out[s == "L"] = 0.0
                return out
        return np.full(len(pa_df), 0.5, dtype=np.float32)
    raw_bat_side = _hand_col("batter_hand", "stand")
    raw_pit_hand = _hand_col("pitcher_hand", "p_throws")
    raw_pitcher  = (_col("pitcher_id") if "pitcher_id" in pa_df.columns
                    else _col("pitcher_idx"))
    raw_batter   = (_col("batter_id")  if "batter_id"  in pa_df.columns
                    else _col("batter_idx"))
    raw_park     = _col("park_idx", 0.0)
    raw_season   = _col("season", 0.0)

    for b_idx, gid in enumerate(unique_games):
        mask  = game_ids_arr == gid
        order = np.argsort(raw_at_bat[mask])
        n     = min(int(mask.sum()), max_pa)

        def _ff(dst, src, scale=1.0):
            vals = (src[mask][order][:n] / scale).astype(np.float32)
            nan_m = np.isnan(vals)
            if nan_m.any():
                vals = vals.copy(); vals[nan_m] = 0.0
            dst[b_idx, :n] = vals

        def _fi(dst, src, fill=0):
            raw = src[mask][order][:n].astype(np.float64)
            nan_m = np.isnan(raw)
            raw[nan_m] = fill
            dst[b_idx, :n] = raw.astype(np.int32)

        pa_valid[b_idx, :n] = True
        _ff(arr_inning,     raw_inning - 1.0, scale=8.0)
        _ff(arr_half,       raw_half)
        _ff(arr_outs,       raw_outs, scale=2.0)
        _ff(arr_base_state, raw_bs, scale=7.0)
        _ff(arr_score_diff, raw_sd, scale=10.0)
        _ff(arr_tto,        raw_tto, scale=3.0)
        _ff(arr_shift,      raw_shift)
        _ff(arr_clock,      raw_clock)
        _ff(arr_pitch_count, raw_pc_game, scale=120.0)  # ~120 = typical starter pull point
        _ff(arr_bat_side,   raw_bat_side)
        _ff(arr_pit_hand,   raw_pit_hand)
        _fi(arr_pa_outcome, raw_pao, fill=-1)
        _fi(arr_runs_scored, raw_runs)
        _fi(arr_bs_after,   raw_bsa)
        _fi(arr_pitcher_ids, raw_pitcher)
        _fi(arr_batter_ids,  raw_batter)
        _fi(arr_park_ids,    raw_park)
        _fi(arr_season,      raw_season)

    return {
        "pa_valid":         jnp.array(pa_valid),
        "inning":           jnp.array(arr_inning),
        "half":             jnp.array(arr_half),
        "outs":             jnp.array(arr_outs),
        "base_state":       jnp.array(arr_base_state),
        "score_diff":       jnp.array(arr_score_diff),
        "tto":              jnp.array(arr_tto),
        "shift_restricted": jnp.array(arr_shift),
        "pitch_clock":      jnp.array(arr_clock),
        "pa_outcome":       jnp.array(arr_pa_outcome),
        "runs_scored":      jnp.array(arr_runs_scored),
        "base_state_after": jnp.array(arr_bs_after),
        "pitch_count_game": jnp.array(arr_pitch_count),
        "bat_side":         jnp.array(arr_bat_side),
        "pit_hand":         jnp.array(arr_pit_hand),
        "pitcher_ids":      jnp.array(arr_pitcher_ids),
        "batter_ids":       jnp.array(arr_batter_ids),
        "season":           jnp.array(arr_season),
        "park_ids":         jnp.array(arr_park_ids),
        "game_ids":         unique_games,
    }
