"""Regression check: rebuild the player table with the PRE-refactor code and with
the current code, and require bit-identical output for every flag combination
that existing checkpoints used.

The pre-refactor source comes from git (`git show HEAD:...`), so this compares
the real previous implementation rather than a reimplementation of it.
"""
import importlib.util
import sys

import numpy as np

from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.paths import processed_root
from diamondworldjax.scripts.train_pa import _build_player_table as new_build

spec = importlib.util.spec_from_file_location("train_pa_orig", "/tmp/train_pa_orig.py")
orig = importlib.util.module_from_spec(spec)
sys.modules["train_pa_orig"] = orig
spec.loader.exec_module(orig)
old_build = orig._build_player_table

# 2015-2023 is the training span for v15/v16 and everything from v20 onward.
pitches = load_seasons(list(range(2015, 2024)), data_root=processed_root())

combos = [
    dict(recency_halflife=2.0, contact_quality=False, per_stat_shrink=False),  # v13-era
    dict(recency_halflife=2.0, contact_quality=True, per_stat_shrink=False),   # v16
    dict(recency_halflife=2.0, contact_quality=True, per_stat_shrink=True),    # v20/v21/v22/v27
    dict(recency_halflife=None, contact_quality=False, per_stat_shrink=False),  # v6-v11 pooled
]

ok = True
for kw in combos:
    a = np.asarray(old_build(pitches, **kw)["stats"])
    b = np.asarray(new_build(pitches, **kw)["stats"])
    same = np.array_equal(a, b)
    ok &= same
    print(("IDENTICAL " if same else "DIFFERS   ") + str(kw)
          + "   max|diff| " + format(float(np.max(np.abs(a - b))), ".3e"))

# And the new flag must actually change something, in the right direction.
base = np.asarray(new_build(pitches, recency_halflife=2.0, contact_quality=True,
                            per_stat_shrink=True)["stats"])
shr = np.asarray(new_build(pitches, recency_halflife=2.0, contact_quality=True,
                           per_stat_shrink=True, shrink_contact_quality=True)["stats"])
print("\nflag off vs on:")
print("  cols 0:5 unchanged:", np.array_equal(base[:, :5], shr[:, :5]))
print("  cols 5:7 changed:  ", not np.array_equal(base[:, 5:7], shr[:, 5:7]))
seen = base[:, 4] > 0
sd_before = float(base[seen, 5].std())
sd_after = float(shr[seen, 5].std())
print("  expected-hit spread over seen players: "
      + format(sd_before, ".4f") + " -> " + format(sd_after, ".4f")
      + "  (shrinkage must reduce it)")
ok &= np.array_equal(base[:, :5], shr[:, :5])
ok &= not np.array_equal(base[:, 5:7], shr[:, 5:7])
ok &= sd_after < sd_before
print("\n" + ("ALL CHECKS PASS" if ok else "CHECKS FAILED"))
raise SystemExit(0 if ok else 1)
