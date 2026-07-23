"""Direct head-to-head: DiamondWorld v15 vs Marcel vs real Steamer projections.

SSAC.md previously had to bound v15 against Steamer / ZiPS / THE BAT indirectly,
via the published "professional systems beat Marcel by a few points" folklore,
because their preseason numbers were not reachable. fetch_projections.py sources
Steamer's actual preseason-2024 hitter projections, so that step can be measured
instead of assumed.

One honest wrinkle. Marcel and Steamer are scored here on the batters all three
systems cover; v15's correlations come from prod_playercorr, which scored its own
(larger) batter set. Re-scoring v15 on this exact subset means re-running model
inference on a GPU. So this script reports Marcel on BOTH sets: if restricting to
the common set barely moves Marcel, the v15 numbers transfer and the Steamer gap
can be applied to them. If it moves Marcel a lot, the transfer is not safe and it
says so.

  python -m diamondworldjax.scripts.projection_headtohead
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.paths import processed_root
from diamondworldjax.scripts.marcel_compare import HIT, REG, rates

STATS = ["k", "bb", "hit", "hr"]
MIN_PA = 150            # same qualification as prod_playercorr
PROJ = Path("data/projections_2024.csv")

# v15 from prod_playercorr (2024 test, skill_mode=mean, b_heur recal). Scored on
# its own batter set, hence the transfer check above.
V15 = {"k": 0.741, "bb": 0.645, "hit": 0.411, "hr": 0.580}


def corr(pred: dict, act: dict, keys: list[int]) -> dict:
    out = {}
    for s in STATS:
        a = np.array([pred[s][b] for b in keys])
        b_ = np.array([act[s][b] for b in keys])
        out[s] = float(np.corrcoef(a, b_)[0, 1])
    out["avg"] = float(np.mean([out[s] for s in STATS]))
    return out


def main():
    seasons = {y: load_seasons([y], data_root=processed_root())
               .filter(pl.col("pa_terminal") & pl.col("pa_outcome").is_not_null())
               for y in (2021, 2022, 2023, 2024)}
    n23 = len(seasons[2023])
    lg = dict(hit=seasons[2023]["pa_outcome"].is_in(HIT).sum() / n23,
              bb=seasons[2023]["pa_outcome"].is_in(["BB", "HBP"]).sum() / n23,
              k=(seasons[2023]["pa_outcome"] == "K").sum() / n23,
              hr=(seasons[2023]["pa_outcome"] == "HR").sum() / n23)

    r = {y: rates(seasons[y], "batter_id") for y in (2021, 2022, 2023, 2024)}
    acc: dict[int, dict] = {}
    for y, w in {2023: 5.0, 2022: 4.0, 2021: 3.0}.items():
        for row in r[y].iter_rows(named=True):
            a = acc.setdefault(int(row["batter_id"]), dict(hit=0.0, bb=0.0, k=0.0, hr=0.0, pa=0.0))
            for kk in ("hit", "bb", "k", "hr", "pa"):
                a[kk] += w * row[kk]

    actual = {s: {} for s in STATS}
    for row in r[2024].iter_rows(named=True):
        if row["pa"] < MIN_PA:
            continue
        for s in STATS:
            actual[s][int(row["batter_id"])] = row[s] / row["pa"]

    marcel = {s: {} for s in STATS}
    for b in actual["k"]:
        wa = acc.get(b)
        if wa is None or wa["pa"] < 100:
            continue
        for s in STATS:
            marcel[s][b] = (wa[s] + REG * lg[s]) / (wa["pa"] + REG)

    if not PROJ.exists():
        raise SystemExit(f"{PROJ} missing; run fetch_projections.py first")
    proj = pl.read_csv(PROJ).filter((pl.col("group") == "bat") & (pl.col("system") == "steamer"))
    steamer = {s: {} for s in STATS}
    for row in proj.iter_rows(named=True):
        for s in STATS:
            steamer[s][int(row["mlbam_id"])] = row[f"{s}_rate"]

    marcel_set = sorted(marcel["k"])
    common = sorted(set(marcel_set) & set(steamer["k"]))
    m_full, m_com, s_com = corr(marcel, actual, marcel_set), corr(marcel, actual, common), corr(steamer, actual, common)

    L = []
    L.append("HEAD-TO-HEAD: v15 vs Marcel vs Steamer (2024 actuals, cross-player rate corr)")
    L.append(f"  qualified batters (>=%d PA in 2024): %d" % (MIN_PA, len(actual['k'])))
    L.append(f"  scored by Marcel: {len(marcel_set)}   also projected by Steamer: {len(common)}")
    L.append("")
    L.append(f"  {'stat':5s} {'Marcel(all)':>12s} {'Marcel(comm)':>13s} {'Steamer(comm)':>14s} {'v15(own set)':>13s}")
    for s in STATS:
        L.append(f"  {s.upper():5s} {m_full[s]:12.3f} {m_com[s]:13.3f} {s_com[s]:14.3f} {V15[s]:13.3f}")
    v15_avg = float(np.mean([V15[s] for s in STATS]))
    L.append(f"  {'AVG':5s} {m_full['avg']:12.3f} {m_com['avg']:13.3f} {s_com['avg']:14.3f} {v15_avg:13.3f}")
    L.append("")

    shift = m_com["avg"] - m_full["avg"]
    gap = s_com["avg"] - m_com["avg"]
    L.append(f"  Steamer - Marcel on the common set: {gap:+.3f} AVG corr")
    L.append(f"  set-restriction effect on Marcel:   {shift:+.3f} AVG corr")
    if abs(shift) < 0.02:
        L.append("  -> the common set is not a materially different population, so v15's")
        L.append(f"     numbers transfer: v15 {v15_avg:.3f} vs Marcel {m_full['avg']:.3f} "
                 f"({v15_avg - m_full['avg']:+.3f}), and Steamer sits {gap:+.3f} above Marcel,")
        L.append(f"     putting v15 about {v15_avg - (m_full['avg'] + gap):+.3f} against Steamer.")
    else:
        L.append("  -> restricting the set moves Marcel enough that the v15 transfer is NOT")
        L.append("     safe; v15 must be re-scored on the common set (GPU) before claiming a gap.")
    L.append("")
    L.append("Source: Steamer preseason 2024, Wayback capture of Razzball's public mirror.")
    L.append("ZiPS and THE BAT have no usable preseason-2024 capture and are not estimated.")

    rep = "\n".join(L)
    print(rep)
    Path("data/eval2").mkdir(parents=True, exist_ok=True)
    Path("data/eval2/projection_headtohead.txt").write_text(rep + "\n")


if __name__ == "__main__":
    main()
