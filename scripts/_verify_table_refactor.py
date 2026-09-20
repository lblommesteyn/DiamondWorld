"""Regression check: the shrink_toward_league refactor must reproduce the player
table stored inside an existing checkpoint bit-for-bit.

If this drifts, every v20/v21/v22/v27 comparison silently changes meaning.
"""
import pickle
import sys

import numpy as np

from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.paths import processed_root
from diamondworldjax.scripts.train_pa import _build_player_table

path = sys.argv[1]
ck = pickle.load(open(path, "rb"))
print("checkpoint keys:", sorted(ck.keys()))

meta = ck.get("pa_metadata")
if meta is None:
    print("no pa_metadata in this checkpoint; cannot compare a stored table")
    raise SystemExit(0)

cfg = meta["config"]
print("stored config:", {k: cfg.get(k) for k in
                         ("train_end", "recency_halflife", "contact_quality",
                          "per_stat_shrink", "shrink_contact_quality")})
stored = np.asarray(meta["player_table"]["stats"])
trp = load_seasons(list(meta["train_seasons"]), data_root=processed_root())
reb = np.asarray(_build_player_table(
    trp,
    recency_halflife=cfg.get("recency_halflife"),
    contact_quality=cfg.get("contact_quality", False),
    per_stat_shrink=cfg.get("per_stat_shrink", False),
    shrink_contact_quality=cfg.get("shrink_contact_quality", False),
)["stats"])

print("shapes:", stored.shape, reb.shape)
if stored.shape != reb.shape:
    raise SystemExit("SHAPE MISMATCH")
print("max abs diff:", float(np.max(np.abs(stored - reb))))
print("IDENTICAL" if np.array_equal(stored, reb) else "DIFFERS")
