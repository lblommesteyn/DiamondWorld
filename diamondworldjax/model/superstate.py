"""The shared super-state every pitch-level head conditions on.

The v0 pitch-level line (model/hurdle.py, model/transition.py, model/joint.py) already
implements most of the A/B/C stack: pitch type, location and velocity, then the
swing/called-strike/contact/foul hurdle, then the transition events. What it lacks is a
single well-defined state object those heads share, and two of the pieces named in the
design are genuinely absent from it.

WHAT joint._make_game_state_8 ALREADY CARRIES

    balls, strikes, outs, base_state, score_diff, tto, inning, half

`half` is the home/away indicator, so "external game state" in the sense of which side
is batting is present, though only implicitly. Park enters elsewhere as a LEARNED
embedding (PARK_DIM = 8 in pa_model, D_PARK_EMB = 16 in batted_ball), never as physical
geometry.

WHAT THIS MODULE ADDS

  * an explicit home/away flag and the derived batting-team perspective, so a head does
    not have to rediscover them from `half`;
  * late-and-close context (inning, score margin, outs, base state combined into the
    leverage-ish signals that drive manager behaviour);
  * a slot for RULES-BASED ARENA GEOMETRY, kept separate from the learned park
    embedding, because they answer different questions: the embedding memorises "runs
    play up in this park", geometry says "the right-field wall is 314 feet away and 37
    feet high", which is what a batted-ball head actually needs to resolve a fly ball.

ON THE GEOMETRY TABLE

The geometry is deliberately loaded from `data/parks/geometry.csv` rather than baked in.
Park dimensions are not in the processed Statcast schema and writing thirty stadiums'
dimensions from memory is exactly the kind of plausible-looking fabrication this
codebase has been bitten by before. `geometry_features` returns a zero vector plus a
`has_geometry = 0` flag for any park with no row, so a partially populated table
degrades to the current behaviour instead of silently inventing a wall.

The has_geometry flag matters: without it a missing park is indistinguishable from a
park whose every dimension happens to be zero, and the head would learn from the
sentinel.
"""
from __future__ import annotations

import csv
import os
from typing import Optional

import numpy as np

# Columns expected in data/parks/geometry.csv, in this order after `park_id`.
GEOMETRY_COLS = (
    "lf_line_ft", "lcf_ft", "cf_ft", "rcf_ft", "rf_line_ft",
    "lf_wall_ft", "cf_wall_ft", "rf_wall_ft", "elevation_ft",
)
N_GEOMETRY = len(GEOMETRY_COLS) + 1  # + has_geometry flag

# Normalisers, so a 400-foot fence and a 37-foot wall enter on comparable scales.
# Centres are rough league-typical values; the exact constants only set the origin.
_CENTRE = np.array([330.0, 375.0, 405.0, 375.0, 330.0, 10.0, 10.0, 10.0, 500.0])
_SCALE = np.array([25.0, 25.0, 20.0, 25.0, 25.0, 8.0, 8.0, 8.0, 900.0])


# The canonical table ships INSIDE the package. data/ is a symlink to the Windows
# side of the machine, so anything under it is beyond a symbolic link as far as git
# is concerned and cannot be tracked; a reference table that is not version
# controlled is one that silently differs between checkouts.
_PACKAGED = os.path.join(os.path.dirname(__file__), "..", "data", "parks",
                         "geometry.csv")


def load_geometry(path: str | None = None) -> dict[str, np.ndarray]:
    """park_id -> raw geometry vector. Missing file yields an empty table.

    Looks for an explicit path first, then data/parks/geometry.csv for a local
    override, then the packaged table.
    """
    if path is None:
        for cand in ("data/parks/geometry.csv", os.path.normpath(_PACKAGED)):
            if os.path.exists(cand):
                path = cand
                break
        else:
            return {}
    if not os.path.exists(path):
        return {}
    table: dict[int, np.ndarray] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            pid = row.get("park_id")
            if pid is None or str(pid).strip() == "":
                continue
            pid = str(pid).strip()
            # A BLANK single field reads as league-average rather than
            # disqualifying the whole park. Requiring all nine meant one missing
            # column (elevation, which has no source as trustworthy as the rest)
            # threw away eight verified measurements per park. A blank becomes the
            # centre value, so it normalises to 0 and contributes nothing, which is
            # the right default for a linear head. has_geometry still means "this
            # park has real measurements", because a row with nothing usable is
            # skipped below.
            vals, n_present = [], 0
            for i, c in enumerate(GEOMETRY_COLS):
                v = row.get(c, "")
                if v is None or str(v).strip() == "":
                    vals.append(_CENTRE[i])
                else:
                    try:
                        vals.append(float(v))
                        n_present += 1
                    except ValueError:
                        vals.append(_CENTRE[i])
            if n_present:
                table[pid] = np.asarray(vals, dtype=np.float32)
    return table


def geometry_features(park_ids: np.ndarray,
                      table: Optional[dict[int, np.ndarray]] = None) -> np.ndarray:
    """(N,) park ids -> (N, N_GEOMETRY) normalised geometry + has_geometry flag.

    Parks absent from the table get zeros and has_geometry = 0, which is the same
    information the model has today, rather than a fabricated stadium.
    """
    if table is None:
        table = load_geometry()
    # park_id in the processed data is a STRING team code ("LAA"), not an int.
    # Coercing to int here raised on every row, so the lookup is done on the key
    # as given and the table is expected to be keyed the same way.
    park_ids = list(park_ids) if not isinstance(park_ids, np.ndarray) else         park_ids.reshape(-1).tolist()
    out = np.zeros((len(park_ids), N_GEOMETRY), dtype=np.float32)
    for i, pid in enumerate(park_ids):
        vec = table.get(pid)
        if vec is None:
            continue
        out[i, :len(GEOMETRY_COLS)] = (vec - _CENTRE) / _SCALE
        out[i, -1] = 1.0
    return out


def context_features(balls, strikes, outs, base_state, score_diff, inning, half,
                     tto) -> np.ndarray:
    """Derived game context beyond the raw 8 fields joint.py already stacks.

    `half` is 1 for the bottom of the inning, so the batting team is the home team
    exactly when half == 1. That identity is spelled out here because the extractor
    once had the analogous lineup/staff mapping crossed.

    Returns (N, 7): is_home_batting, base-state occupancy bits (3), a scoring-position
    flag, late_and_close, and a normalised inning.
    """
    half = np.asarray(half).reshape(-1)
    base_state = np.asarray(base_state).reshape(-1).astype(np.int64)
    outs = np.asarray(outs).reshape(-1)
    inning = np.asarray(inning).reshape(-1)
    score_diff = np.asarray(score_diff).reshape(-1)

    is_home_batting = (half == 1).astype(np.float32)
    on1 = (base_state & 1 > 0).astype(np.float32)
    on2 = (base_state & 2 > 0).astype(np.float32)
    on3 = (base_state & 4 > 0).astype(np.float32)
    in_scoring_pos = np.maximum(on2, on3)
    # "Late and close" in the conventional sense: 7th inning or later with the game
    # within one run. This is the regime where manager behaviour, which Transformer C
    # would have to model, stops resembling the rest of the game.
    late_and_close = ((inning >= 7) & (np.abs(score_diff) <= 1)).astype(np.float32)
    inning_norm = (inning - 5.0) / 4.0

    return np.stack([is_home_batting, on1, on2, on3, in_scoring_pos,
                     late_and_close, inning_norm], axis=-1).astype(np.float32)


def super_state(batch: dict, geometry_table: Optional[dict] = None) -> np.ndarray:
    """Assemble the full per-pitch super-state from a batch dict.

    Concatenates the derived context (7) with the arena geometry (N_GEOMETRY). The
    player embeddings and the raw 8-field game state are NOT duplicated here; they are
    already assembled inside joint.py and this is meant to extend that vector, not
    replace it.
    """
    ctx = context_features(
        batch["balls"], batch["strikes"], batch["outs"], batch["base_state"],
        batch["score_diff"], batch["inning"], batch["half"], batch.get("tto"),
    )
    park = batch.get("park_id")
    if park is None:
        geo = np.zeros((len(ctx), N_GEOMETRY), dtype=np.float32)
    else:
        geo = geometry_features(park, geometry_table)
    return np.concatenate([ctx, geo], axis=-1)


SUPER_STATE_DIM = 7 + N_GEOMETRY
