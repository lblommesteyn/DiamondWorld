"""Tune the contact-quality shrinkage constants on a HOLDOUT season.

The 2024 sweep in _pretest_cq_shrink.py picked its optimum on the test season,
which is exactly the tuning leak projection_levers avoids by tuning on 2023. This
does it honestly: build the player table from 2015-2022, score columns 5 and 6
against 2023, pick the constant there, and only then report what that choice buys
on 2024.
"""
import numpy as np

from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.paths import processed_root
from diamondworldjax.scripts.train_pa import _build_player_table, shrink_toward_league
from diamondworldjax.scripts.projection_levers import _load, counts, actuals

GRID = [0, 100, 200, 400, 700, 1000, 1200, 1600, 2200, 3000, 4000, 6000]
KW = dict(recency_halflife=2.0, contact_quality=True, per_stat_shrink=True)


def cohort(table, act, key):
    rows = [(b, i) for b, i in table["id_to_idx"].items() if i != 0 and b in act[key]]
    ix = np.array([i for _, i in rows])
    y = np.array([act[key][b] for b, _ in rows])
    return ix, y


def sweep(table, ix, y, col):
    out = {}
    for reg in GRID:
        s = shrink_toward_league(table["stats"][:, col:col + 1],
                                 table["stats"][:, 4:5], np.array([float(reg)]))
        out[reg] = float(np.corrcoef(s[ix, 0].astype(float), y)[0, 1])
    return out


print("=== TUNE on 2023 (table built from 2015-2022, 2024 never touched) ===")
tab22 = _build_player_table(load_seasons(list(range(2015, 2023)),
                                         data_root=processed_root()), **KW)
act23 = actuals(counts(_load([2023])[2023]))

chosen = {}
for name, col, key in (("hit", 5, "hit"), ("hr", 6, "hr")):
    ix, y = cohort(tab22, act23, key)
    sc = sweep(tab22, ix, y, col)
    best_reg = max(sc, key=sc.get)
    chosen[name] = best_reg
    print("\ncol " + str(col) + " (" + name + "), " + str(len(y)) + " batters in 2023")
    for reg in GRID:
        mark = "  <-- best" if reg == best_reg else ""
        print("  reg " + str(reg).rjust(5) + "  corr " + format(sc[reg], ".4f") + mark)
    print("  raw (reg 0) " + format(sc[0], ".4f") + " -> chosen " + format(sc[best_reg], ".4f")
          + "   delta " + format(sc[best_reg] - sc[0], "+.4f"))

print("\nCHOSEN ON HOLDOUT: hit=" + str(chosen["hit"]) + ", hr=" + str(chosen["hr"]))

print("\n=== APPLY to 2024 (table built from 2015-2023) ===")
tab23 = _build_player_table(load_seasons(list(range(2015, 2024)),
                                         data_root=processed_root()), **KW)
act24 = actuals(counts(_load([2024])[2024]))
for name, col, key in (("hit", 5, "hit"), ("hr", 6, "hr")):
    ix, y = cohort(tab23, act24, key)
    raw = float(np.corrcoef(tab23["stats"][ix, col].astype(float), y)[0, 1])
    s = shrink_toward_league(tab23["stats"][:, col:col + 1], tab23["stats"][:, 4:5],
                             np.array([float(chosen[name])]))
    got = float(np.corrcoef(s[ix, 0].astype(float), y)[0, 1])
    sc24 = sweep(tab23, ix, y, col)
    oracle_reg = max(sc24, key=sc24.get)
    print("col " + str(col) + " (" + name + "), " + str(len(y)) + " batters in 2024:")
    print("  raw                      " + format(raw, ".4f"))
    print("  holdout-chosen reg " + str(chosen[name]).rjust(4) + "  " + format(got, ".4f")
          + "   delta " + format(got - raw, "+.4f"))
    print("  (2024 oracle reg " + str(oracle_reg).rjust(4) + "  " + format(sc24[oracle_reg], ".4f")
          + ", so the holdout choice costs "
          + format(sc24[oracle_reg] - got, ".4f") + ")")
