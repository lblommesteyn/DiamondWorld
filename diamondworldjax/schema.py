"""Column names, types, and mask definitions for DiamondWorldJAX.

Maps DiamondWorld Statcast parquet columns to the expanded schema
defined in spec.md sections 7.1-7.8.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Identity / indexing
# ---------------------------------------------------------------------------
ID_COLS = [
    "game_pk", "season", "inning", "half", "at_bat_number",
    "pitch_number", "pitcher_id", "batter_id", "umpire_id",
    "home_team_id", "away_team_id", "park_id",
]

# ---------------------------------------------------------------------------
# Pre-pitch game state  (spec 7.2)
# ---------------------------------------------------------------------------
STATE_COLS = [
    "balls", "strikes", "outs",
    "base_state",          # 3-bit bitmask: 1b=bit0, 2b=bit1, 3b=bit2
    "home_score", "away_score", "score_diff",
    "pitch_count_game", "pitch_count_inning", "pitch_count_pa",
    "tto",                 # times through order
    "inning_norm",         # (inning-1)/8
    "half_bin",            # 0=top 1=bot
    "tracking_era",
]

# ---------------------------------------------------------------------------
# Rule-era indicators  (spec 7.3)
# ---------------------------------------------------------------------------
RULE_COLS = [
    "shift_restricted",    # 1 if season >= 2023
    "pitch_clock",         # 1 if season >= 2023
]

# ---------------------------------------------------------------------------
# Manager decisions  (spec 7.4) — inferred / masked where unobserved
# ---------------------------------------------------------------------------
MANAGER_COLS = [
    "pitching_change",
    "steal_attempt",
    "runner_send",
    "defensive_alignment_proxy",  # 0=standard 1=shift 2=extreme
]

# ---------------------------------------------------------------------------
# Pitch package  (spec 7.5)
# ---------------------------------------------------------------------------
PITCH_COLS = [
    "pitch_type",
    "plate_x", "plate_z",
    "release_speed",
    "pfx_x", "pfx_z",
    "release_pos_x", "release_pos_y", "release_pos_z",
    "release_extension",
    "spin_rate",
]

# ---------------------------------------------------------------------------
# Hurdle path  (spec 7.6)
# ---------------------------------------------------------------------------
HURDLE_COLS = [
    "swing", "take",
    "called_strike", "called_ball",
    "contact", "whiff",
    "foul", "in_play",
    "pa_terminal",
]

# ---------------------------------------------------------------------------
# Batted-ball physics  (spec 7.7)
# ---------------------------------------------------------------------------
BATTED_COLS = [
    "launch_speed", "launch_angle", "spray_angle",
    "hc_x", "hc_y", "hit_distance",
]

# ---------------------------------------------------------------------------
# Transition state  (spec 7.8)
# ---------------------------------------------------------------------------
TRANSITION_COLS = [
    "pa_outcome",
    "runs_scored", "base_state_after",
    "balls_after", "strikes_after", "outs_after",
    "error_flag", "steal_flag", "wild_pitch_flag",
    "passed_ball_flag", "balk_flag",
]

SCHEMA_COLS = (
    ID_COLS + STATE_COLS + RULE_COLS + MANAGER_COLS
    + PITCH_COLS + HURDLE_COLS + BATTED_COLS + TRANSITION_COLS
)

# ---------------------------------------------------------------------------
# Mask column names
# ---------------------------------------------------------------------------
MASK_COLS = [
    "pitch_valid_mask",
    "swing_mask",
    "take_mask",
    "contact_mask",
    "foul_mask",
    "in_play_mask",
    "batted_ball_mask",
    "terminal_pitch_mask",
    "steal_valid_mask",
]

# ---------------------------------------------------------------------------
# DiamondWorld → DiamondWorldJAX column remap
# ---------------------------------------------------------------------------
DW_REMAP: dict[str, str] = {
    "game_pk":           "game_pk",
    "at_bat_number":     "at_bat_number",
    "pitch_number":      "pitch_number",
    "inning":            "inning",
    "half":              "half",
    "balls":             "balls",
    "strikes":           "strikes",
    "outs":              "outs",
    "base_state":        "base_state",
    "runs_scored":       "runs_scored",
    "base_state_after":  "base_state_after",
    "pa_terminal":       "pa_terminal",
    "pa_outcome":        "pa_outcome",
    "pitch_type":        "pitch_type",
    "release_speed":     "release_speed",
    "plate_x":           "plate_x",
    "plate_z":           "plate_z",
    "pfx_x":             "pfx_x",
    "pfx_z":             "pfx_z",
    "launch_speed":      "launch_speed",
    "launch_angle":      "launch_angle",
    "hc_x":              "hc_x",
    "hc_y":              "hc_y",
    "pitch_count_game":  "pitch_count_game",
    "pitch_count_inning":"pitch_count_inning",
    "tto":               "tto",
    "tracking_era":      "tracking_era",
    "stand":             "batter_hand",
    "p_throws":          "pitcher_hand",
}

# Pitch type string → integer index (K=8 including unknown)
PITCH_TYPE_IDX: dict[str, int] = {
    "FF": 0, "SI": 1, "FC": 2,   # fastballs
    "SL": 3, "CU": 4, "KC": 5,   # breaking
    "CH": 6,                      # change
    "UNK": 7,                     # unknown / other
}
N_PITCH_TYPES = len(PITCH_TYPE_IDX)

# PA outcome → integer index
PA_OUTCOME_IDX: dict[str, int] = {
    "K": 0, "BB": 1, "HBP": 2,
    "1B": 3, "2B": 4, "3B": 5, "HR": 6,
    "out": 7, "E": 8,
}
N_PA_OUTCOMES = len(PA_OUTCOME_IDX)
