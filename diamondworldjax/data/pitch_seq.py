"""Pitch sequences for the A/B transformers.

SEQUENCE UNIT

One sequence is one (game, half): the pitches thrown by one team's staff in that
half of that game, in at_bat_number then pitch_number order. That is the unit the
design's "recent pitches, recent swings" refers to, and it keeps continuity across
batters, which per-PA windows destroy exactly where sequence effects live.

INDEX 0 IS RESERVED, DELIBERATELY

The external review found that embedding index 0 was the first real player in the
training table, so every unseen player silently inherited a real player's learned
representation. Here index 0 is an explicit UNKNOWN slot for every id space, and
real ids start at 1. An unseen player in a later season gets a genuine unknown
embedding instead of impersonating somebody.

GEOMETRY

Loaded through model/superstate.py, which returns zeros plus has_geometry = 0 for
any park with no row. data/parks/geometry.csv does not exist yet, so today every
park takes the has_geometry = 0 path and the geometry block is inert. That is the
intended degradation: the flag means a head can tell "no data" from "a park whose
dimensions happen to be zero", so populating the table later changes behaviour
without invalidating anything trained before it.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ..model.superstate import geometry_features, load_geometry, N_GEOMETRY

UNKNOWN = 0          # reserved in every id space

# Statcast records 18 distinct pitch codes, but the model's vocabulary is 8 and
# the tail is genuinely thin: everything below FS is under 2% and the last six
# codes together are under 0.3%. The seven most common get their own class and
# the rest share class 7. The mapping is FIXED here rather than derived per run,
# because a frequency-derived vocabulary would silently renumber classes between
# seasons and make two runs incomparable.
#   0 FF four-seam   1 SI sinker     2 SL slider    3 CH change
#   4 FC cutter      5 ST sweeper    6 CU curve     7 everything else
PITCH_CODE_TO_CLASS = {
    "FF": 0, "SI": 1, "SL": 2, "CH": 3, "FC": 4, "ST": 5, "CU": 6,
    "FS": 7, "KC": 7, "SV": 7, "KN": 7, "FA": 7, "EP": 7, "SC": 7,
    "FO": 7, "CS": 7, "PO": 7,
}
OTHER_CLASS = 7

STUFF_COLS = ("release_speed", "pfx_x", "pfx_z", "plate_x", "plate_z")
# Empirical centres and scales (2024), so the Gaussian heads see standardised
# targets. Fixed constants rather than statistics recomputed per run: recomputing
# would change the units of nll_stuff between runs and make two runs' numbers
# quietly incomparable.
STUFF_CENTRE = np.array([89.15, -0.10, 0.59, 0.06, 2.30], dtype=np.float32)
STUFF_SCALE = np.array([5.98, 0.90, 0.71, 0.84, 0.97], dtype=np.float32)


def build_id_maps(dfs: list[pl.DataFrame]) -> dict:
    """id -> index, with 0 reserved for unknown, built from TRAINING data only.

    park_id is a string code ("LAA"), pitcher/batter ids are integers, so keys
    are kept in their native type rather than coerced.
    """
    maps = {}
    for col, key in (("pitcher_id", "pitcher"), ("batter_id", "batter"),
                     ("park_id", "park")):
        vals = pl.concat([d.select(col) for d in dfs])[col].drop_nulls().unique().sort().to_list()
        maps[key] = {v: i + 1 for i, v in enumerate(vals)}        # 1-based
        maps[f"n_{key}"] = len(vals) + 1                          # + UNKNOWN
    return maps


def _ctx(df: pl.DataFrame) -> np.ndarray:
    """Game state + external state, standardised. Shape (N, D_CTX)."""
    balls = df["balls"].to_numpy().astype(np.float32)
    strikes = df["strikes"].to_numpy().astype(np.float32)
    outs = df["outs"].to_numpy().astype(np.float32)
    bs = df["base_state"].to_numpy().astype(np.int64)
    score = df["score_diff"].to_numpy().astype(np.float32)
    inning = df["inning"].to_numpy().astype(np.float32)
    # half is the string "top"/"bot". Bottom means the home team is batting.
    half = (df["half"].to_numpy() == "bot").astype(np.float32)
    tto = df["tto"].to_numpy().astype(np.float32)
    stand = (df["stand"].to_numpy() == "R").astype(np.float32)
    throws = (df["p_throws"].to_numpy() == "R").astype(np.float32)

    on1 = (bs & 1 > 0).astype(np.float32)
    on2 = (bs & 2 > 0).astype(np.float32)
    on3 = (bs & 4 > 0).astype(np.float32)

    return np.stack([
        balls / 3.0, strikes / 2.0, outs / 2.0,
        on1, on2, on3, np.maximum(on2, on3),
        np.clip(score, -10, 10) / 10.0,
        (inning - 5.0) / 4.0,
        half,                                  # 1 = bottom, so home is batting
        1.0 - half,                            # explicit away-batting flag
        ((inning >= 7) & (np.abs(score) <= 1)).astype(np.float32),
        np.clip(tto, 0, 4) / 4.0,
        stand, throws,
        (stand == throws).astype(np.float32),  # platoon: same-handed matchup
    ], axis=-1).astype(np.float32)


D_CTX = 16
D_ENV = 8          # temp, elevation, roof, wind (3), air density, env_known
D_CTX_TOTAL = D_CTX + D_ENV

ENV_COLS = ("temp_f", "elevation_ft", "roof_closed", "wind_mph", "wind_out",
            "wind_cross")


def _env(df: pl.DataFrame) -> np.ndarray:
    """Per-pitch environment block. Zeros + env_known=0 when unavailable.

    Only two of these have a plausible path to helping the CURRENT heads. Air
    density affects how much a pitch breaks, which is why breaking balls misbehave
    at altitude, and that acts on A's stuff model. Wind and roof act on batted-ball
    flight, which no head in this stack models, so they are carried for the
    batted-ball head that does not exist yet rather than expected to help now.
    """
    n = df.height
    if "temp_f" not in df.columns:
        return np.zeros((n, D_ENV), np.float32)

    temp = df["temp_f"].fill_null(72.0).to_numpy().astype(np.float32)
    elev = df["elevation_ft"].fill_null(500.0).to_numpy().astype(np.float32)
    roof = df["roof_closed"].fill_null(0).to_numpy().astype(np.float32)
    wmph = df["wind_mph"].fill_null(0.0).to_numpy().astype(np.float32)
    wout = df["wind_out"].fill_null(0.0).to_numpy().astype(np.float32)
    wcrs = df["wind_cross"].fill_null(0.0).to_numpy().astype(np.float32)
    known = df["temp_f"].is_not_null().to_numpy().astype(np.float32)

    # Air density relative to sea level at 15C, as a physical quantity rather than
    # leaving a linear head to discover the interaction between two raw numbers:
    # pressure falls with altitude, density falls with temperature.
    t_kelvin = (temp - 32.0) * 5.0 / 9.0 + 273.15
    density = np.exp(-elev / 29000.0) * (288.15 / np.maximum(t_kelvin, 200.0))

    # A closed roof means the weather columns describe indoor air, so wind is
    # zeroed rather than left to imply a breeze that cannot reach the field.
    wmph = wmph * (1.0 - roof)
    wout = wout * (1.0 - roof)
    wcrs = wcrs * (1.0 - roof)

    return np.stack([
        (temp - 72.0) / 15.0,
        (elev - 500.0) / 900.0,
        roof,
        wmph / 12.0,
        wout, wcrs,
        (density - 1.0) / 0.12,
        known,
    ], axis=-1).astype(np.float32)


EVENT_FLAGS = ("wild_pitch", "passed_ball", "balk", "steal", "caught_stealing",
               "pickoff", "error", "defensive_indiff")


def make_sequences(df: pl.DataFrame, maps: dict, max_len: int = 160,
                   geometry_table=None, events: pl.DataFrame | None = None,
                   game_ctx: pl.DataFrame | None = None):
    """(game, half) sequences -> padded arrays for the A/B/C transformers.

    `events` is the table from data/extract_events.py. It is LEFT-joined, so a
    pitch with no event row gets zeros: absence of an event row means the event
    did not happen, not that the label is missing. That is only true because the
    extractor emits a row for every pitch that had any event at all, which the
    join-rate check in extract_events.verify is what establishes.
    """
    if geometry_table is None:
        geometry_table = load_geometry()

    if events is not None and events.height:
        df = df.join(events, on=["game_pk", "at_bat_number", "pitch_number"],
                     how="left").with_columns(
            [pl.col(f).fill_null(0).cast(pl.Int8) for f in EVENT_FLAGS])

    if game_ctx is not None and game_ctx.height:
        keep = ["game_pk"] + [c for c in ENV_COLS if c in game_ctx.columns]
        df = df.join(game_ctx.select(keep), on="game_pk", how="left")

    df = df.sort(["game_pk", "half", "at_bat_number", "pitch_number"])
    ctx_all = np.concatenate([_ctx(df), _env(df)], axis=-1)

    pit = np.array([maps["pitcher"].get(v, UNKNOWN)
                    for v in df["pitcher_id"].to_list()], dtype=np.int32)
    bat = np.array([maps["batter"].get(v, UNKNOWN)
                    for v in df["batter_id"].to_list()], dtype=np.int32)
    prk = np.array([maps["park"].get(v, UNKNOWN)
                    for v in df["park_id"].to_list()], dtype=np.int32)
    # Geometry is keyed by the same park code the embedding uses, so a park with
    # no geometry row still gets its learned embedding and simply carries
    # has_geometry = 0 alongside it.
    geo_all = geometry_features(df["park_id"].to_list(), geometry_table)

    # pitch_type is a Statcast string code, and it is null on 0.38% of rows.
    # A null is a missing measurement, not a pitch of unknown kind, so it maps to
    # the "other" class and is excluded from the type loss via type_valid.
    codes = df["pitch_type"].to_list()
    ptype = np.array([PITCH_CODE_TO_CLASS.get(c, OTHER_CLASS) if c is not None
                      else OTHER_CLASS for c in codes], dtype=np.int32)
    type_valid = np.array([c is not None for c in codes], dtype=np.float32)

    stuff = np.stack([df[c].to_numpy().astype(np.float32) for c in STUFF_COLS], -1)
    stuff_valid = np.isfinite(stuff).all(-1).astype(np.float32)
    stuff = np.nan_to_num(stuff)
    stuff = (stuff - STUFF_CENTRE) / STUFF_SCALE

    if all(f in df.columns for f in EVENT_FLAGS):
        ev_all = np.stack([df[f].to_numpy().astype(np.float32)
                           for f in EVENT_FLAGS], -1)
    else:
        ev_all = None

    # swing/contact/foul are Booleans with no nulls.
    swing = df["swing"].to_numpy().astype(np.float32)
    contact = df["contact"].to_numpy().astype(np.float32)
    foul = df["foul"].to_numpy().astype(np.float32)

    # Segment boundaries without a Python-level group_by over millions of rows.
    gp = df["game_pk"].to_numpy()
    hf = df["half"].to_numpy()
    newseg = np.empty(len(gp), dtype=bool)
    newseg[0] = True
    newseg[1:] = (gp[1:] != gp[:-1]) | (hf[1:] != hf[:-1])
    starts = np.flatnonzero(newseg)
    ends = np.append(starts[1:], len(gp))

    # A long half is split into consecutive windows rather than truncated, so no
    # pitches are silently discarded from training.
    spans = []
    for s, e in zip(starts, ends):
        for a in range(s, e, max_len):
            spans.append((a, min(a + max_len, e)))

    n = len(spans)
    out = {
        "pitcher_idx": np.zeros((n, max_len), np.int32),
        "batter_idx": np.zeros((n, max_len), np.int32),
        "park_idx": np.zeros((n, max_len), np.int32),
        "ctx": np.zeros((n, max_len, D_CTX_TOTAL), np.float32),
        "geom": np.zeros((n, max_len, N_GEOMETRY), np.float32),
        "pitch_type": np.zeros((n, max_len), np.int32),
        "type_valid": np.zeros((n, max_len), np.float32),
        "stuff": np.zeros((n, max_len, len(STUFF_COLS)), np.float32),
        "stuff_valid": np.zeros((n, max_len), np.float32),
        "swing": np.zeros((n, max_len), np.float32),
        "contact": np.zeros((n, max_len), np.float32),
        "foul": np.zeros((n, max_len), np.float32),
        "valid": np.zeros((n, max_len), np.float32),
    }
    if ev_all is not None:
        out["events"] = np.zeros((n, max_len, len(EVENT_FLAGS)), np.float32)
    for i, (a, b) in enumerate(spans):
        L = b - a
        out["pitcher_idx"][i, :L] = pit[a:b]
        out["batter_idx"][i, :L] = bat[a:b]
        out["park_idx"][i, :L] = prk[a:b]
        out["ctx"][i, :L] = ctx_all[a:b]
        out["geom"][i, :L] = geo_all[a:b]
        out["pitch_type"][i, :L] = ptype[a:b]
        out["type_valid"][i, :L] = type_valid[a:b]
        out["stuff"][i, :L] = stuff[a:b]
        out["stuff_valid"][i, :L] = stuff_valid[a:b]
        out["swing"][i, :L] = swing[a:b]
        out["contact"][i, :L] = contact[a:b]
        out["foul"][i, :L] = foul[a:b]
        out["valid"][i, :L] = 1.0
        if ev_all is not None:
            out["events"][i, :L] = ev_all[a:b]
    return out


def load_seasons(seasons, path="data/processed/pitches_{}.parquet"):
    return [pl.read_parquet(path.format(s)) for s in seasons]
