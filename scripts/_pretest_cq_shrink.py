"""Cheap pre-test for item 5's fix, before spending GPU on a retrain.

Shrinking columns 5:7 is NOT a monotone transform of the feature, because the
shrinkage factor n/(n+reg) varies across players. So it can change the feature's
own correlation with the realised rate. If it does not improve the feature, there
is no reason to expect it to improve the model, and the retrain is not worth
running.

Scored on the same cohort the headline metric uses: batters with >= 150 PA in the
2024 test season.
"""
import numpy as np

from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.paths import processed_root
from diamondworldjax.scripts.train_pa import _build_player_table, shrink_toward_league
from diamondworldjax.scripts.projection_levers import MIN_PA, _load, counts, actuals

pitches = load_seasons(list(range(2015, 2024)), data_root=processed_root())
kw = dict(recency_halflife=2.0, contact_quality=True, per_stat_shrink=True)
raw = _build_player_table(pitches, **kw)
shr = _build_player_table(pitches, **kw, shrink_contact_quality=True)
id_to_idx = raw["id_to_idx"]

act = actuals(counts(_load([2024])[2024]))

# The scored cohort: real batters with >= 150 PA in 2024.
rows = [(b, i) for b, i in id_to_idx.items() if i != 0 and b in act["hit"]]
ids = [b for b, _ in rows]
ix = np.array([i for _, i in rows])
y_hit = np.array([act["hit"][b] for b in ids])
y_hr = np.array([act["hr"][b] for b in ids])
n_w = raw["stats"][ix, 4].astype(float)

print("cohort: " + str(len(ids)) + " batters with >= " + str(MIN_PA) + " PA in 2024")
print("weighted-PA on this cohort: median " + format(float(np.median(n_w)), ".0f")
      + "  p10 " + format(float(np.percentile(n_w, 10)), ".0f")
      + "  p90 " + format(float(np.percentile(n_w, 90)), ".0f"))
print()

for name, col, y in (("expected hit (col 5)", 5, y_hit), ("expected HR  (col 6)", 6, y_hr)):
    a = raw["stats"][ix, col].astype(float)
    b = shr["stats"][ix, col].astype(float)
    ca = float(np.corrcoef(a, y)[0, 1])
    cb = float(np.corrcoef(b, y)[0, 1])
    print(name + ":")
    print("  raw      corr " + format(ca, ".4f") + "   sd " + format(a.std(), ".4f"))
    print("  shrunk   corr " + format(cb, ".4f") + "   sd " + format(b.std(), ".4f"))
    print("  delta         " + format(cb - ca, "+.4f"))
    # Sweep the constant: is 2200 even the right one for this column?
    best = None
    for reg in (0, 100, 200, 400, 700, 1000, 1200, 1600, 2200, 3000, 4000, 6000):
        s = shrink_toward_league(raw["stats"][:, col:col + 1], raw["stats"][:, 4:5],
                                 np.array([float(reg)]))
        c = float(np.corrcoef(s[ix, 0].astype(float), y)[0, 1])
        if best is None or c > best[0]:
            best = (c, reg)
        print("    reg " + str(reg).rjust(5) + "  corr " + format(c, ".4f"))
    print("  best constant on this cohort: " + str(best[1])
          + " (corr " + format(best[0], ".4f") + ")")
    print()
