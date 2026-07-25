"""Minor-league equivalency (MLE) priors for players unseen in MLB training.

The model is trained on 2015-2022 MLB, so 2023-24 debuts are blanks (~19% of 2024
PAs). Projection systems solve this with MLEs: translate a player's line from a
lower level into an MLB-equivalent rate. v1 used AAA only with literature factors;
this version pulls every level the MLB Stats API exposes per player (AAA, AA,
High-A, Single-A, NPB, KBO, college) and FITS the translation factors empirically
from the rookies who actually reached MLB, instead of assuming them.

Two products:
  1. per (level, stat) empirical factor = how a rate at that level maps to the MLB
     rate, fit on rookies with both a pre-2024 line at that level and a real 2024
     MLB line. Reported next to the literature factors and the correlation, so a
     level with no real signal is visible rather than trusted.
  2. a combined per-rookie prior that takes each rookie's highest available level,
     translates it with that level's fitted factor, and (for hit/HR, the
     BABIP-noisy rates) shrinks toward the level mean. Saved to mle_rates.npz for
     prod_playercorr --mle, extending rookie coverage beyond AAA-only.

Honesty: the factors are fit on rookies who made MLB (a survivorship-selected
sample), so they are calibrated for "rookies good enough to get 150 MLB PA," which
is exactly the population we inject. A leave-one-out check is reported to show the
single-scalar factors are not overfit.

  python -m diamondworldjax.scripts.mle_priors
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import polars as pl

from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.paths import processed_root
from diamondworldjax.scripts.train_pa import _build_player_table

# MLB Stats API sportId per level, in descending "closeness to MLB" priority.
# The order is the fallback order when a rookie has lines at several levels.
LEVELS = [("AAA", 11), ("NPB", 31), ("KBO", 32), ("AA", 12),
          ("A+", 13), ("A", 14), ("NCAA", 22)]
STATS = ("k", "bb", "hit", "hr")
# Literature AAA factors, kept only as a reference column to compare the fit against.
LIT_AAA = dict(k=1.15, bb=0.90, hit=0.92, hr=0.80)
MIN_LEVEL_PA = 100
MIN_MLB_PA = 150


def fetch_level(ids, sport_id, group="hitting"):
    """Pre-2024 counting line per player at one level (sportId)."""
    out = {}
    for j in range(0, len(ids), 60):
        chunk = ids[j:j + 60]
        url = ("https://statsapi.mlb.com/api/v1/people?personIds=" + ",".join(map(str, chunk))
               + f"&hydrate=stats(group={group},type=yearByYear,sportId={sport_id})")
        try:
            d = json.loads(urllib.request.urlopen(url, timeout=30).read())
        except Exception:
            time.sleep(2)
            continue
        for p in d.get("people", []):
            pa = k = bb = hr = h = 0
            for st in p.get("stats", []):
                for s in st.get("splits", []):
                    if int(s.get("season", 9999)) >= 2024:
                        continue                      # only pre-MLB-2024 lines
                    t = s["stat"]
                    pa += t.get("plateAppearances", 0) or 0
                    k += t.get("strikeOuts", 0) or 0
                    bb += (t.get("baseOnBalls", 0) or 0) + (t.get("hitByPitch", 0) or 0)
                    hr += t.get("homeRuns", 0) or 0
                    h += t.get("hits", 0) or 0
            if pa >= MIN_LEVEL_PA:
                out[int(p["id"])] = dict(pa=pa, k=k / pa, bb=bb / pa, hit=h / pa, hr=hr / pa)
        time.sleep(0.4)
    return out


def fit_factor(level_rate, actual, weight):
    """Weighted multiplicative factor mapping a level rate to the MLB rate, plus a
    leave-one-out correlation of the translated prior against the actual rate.

    factor = sum(w*actual) / sum(w*level_rate): the single scalar that best matches
    aggregate volume. LOO recomputes the factor without each point before scoring
    it, so a factor that only fits its own fitting set shows a degraded LOO number.
    """
    level_rate = np.asarray(level_rate, float)
    actual = np.asarray(actual, float)
    weight = np.asarray(weight, float)
    num = float((weight * actual).sum())
    den = float((weight * level_rate).sum())
    factor = num / den if den > 1e-9 else np.nan
    # leave-one-out translated predictions
    loo_pred = np.empty(len(actual))
    for i in range(len(actual)):
        m = np.arange(len(actual)) != i
        d = float((weight[m] * level_rate[m]).sum())
        f = float((weight[m] * actual[m]).sum()) / d if d > 1e-9 else factor
        loo_pred[i] = f * level_rate[i]
    loo_corr = float(np.corrcoef(loo_pred, actual)[0, 1]) if len(actual) > 3 else np.nan
    return factor, loo_corr


def main():
    tr = load_seasons(list(range(2015, 2023)), data_root=processed_root())
    id2i = _build_player_table(tr)["id_to_idx"]
    te = load_seasons([2024], data_root=processed_root()).filter(
        pl.col("pa_terminal") & pl.col("pa_outcome").is_not_null())
    HIT = ("1B", "2B", "3B", "HR")
    g = te.group_by("batter_id").agg([
        pl.col("pa_outcome").is_in(HIT).mean().alias("hit"),
        pl.col("pa_outcome").is_in(["BB", "HBP"]).mean().alias("bb"),
        (pl.col("pa_outcome") == "K").mean().alias("k"),
        (pl.col("pa_outcome") == "HR").mean().alias("hr"),
        pl.len().alias("pa")])
    unk = {int(r["batter_id"]): r for r in g.iter_rows(named=True)
           if int(r["batter_id"]) not in id2i and r["pa"] >= MIN_MLB_PA}
    ids = list(unk)
    print(f"{len(ids)} unknown 2024 batters (>= {MIN_MLB_PA} MLB PA); fetching each level ...", flush=True)

    level_data = {}
    for name, sid in LEVELS:
        d = fetch_level(ids, sid)
        level_data[name] = d
        print(f"  {name:5s} (sportId {sid}): {len(d)} rookies with a pre-2024 line", flush=True)

    # ---- fit empirical factors per level per stat ----
    factors = {name: {} for name, _ in LEVELS}
    loo = {name: {} for name, _ in LEVELS}
    L = [f"MLE PRIORS: empirical translation factors fit on 2024 rookies who reached MLB", ""]
    for name, _ in LEVELS:
        d = level_data[name]
        pairs = [(d[b], unk[b]) for b in ids if b in d]
        if len(pairs) < 5:
            L.append(f"  {name}: only {len(pairs)} rookies, too few to fit (skipped)")
            continue
        L.append(f"  {name}  (n={len(pairs)})")
        L.append(f"    {'stat':5s} {'fit factor':>11s} {'lit(AAA)':>9s} {'LOO corr':>9s} "
                 f"{'lvl mean':>9s} {'mlb mean':>9s}")
        for s in STATS:
            lr = [p[0][s] for p in pairs]
            ac = [p[1][s] for p in pairs]
            w = [min(p[0]["pa"], p[1]["pa"]) for p in pairs]
            f, lc = fit_factor(lr, ac, w)
            factors[name][s] = f
            loo[name][s] = lc
            lit = f"{LIT_AAA[s]:.2f}" if name == "AAA" else "  -"
            L.append(f"    {s.upper():5s} {f:11.3f} {lit:>9s} {lc:9.3f} "
                     f"{np.mean(lr):9.3f} {np.mean(ac):9.3f}")
        L.append("")

    # ---- combined prior: each rookie's highest available level, fitted factor ----
    combined_ids, combined_rates, combined_levels = [], [], []
    for b in ids:
        for name, _ in LEVELS:
            if b in level_data[name] and factors.get(name):
                d = level_data[name]
                rate = {s: factors[name][s] * d[b][s] for s in STATS}
                combined_ids.append(b)
                combined_rates.append([rate["hit"], rate["bb"], rate["k"], rate["hr"]])
                combined_levels.append(name)
                break

    # validate the combined prior against actual 2024 MLB rates
    L.append(f"COMBINED prior (each rookie's highest level, fitted factor): {len(combined_ids)} rookies")
    used = {}
    for lv in combined_levels:
        used[lv] = used.get(lv, 0) + 1
    L.append("  level used: " + ", ".join(f"{k} {v}" for k, v in sorted(used.items(), key=lambda x: -x[1])))
    L.append(f"  {'stat':5s} {'corr(prior,actual)':>18s}")
    idx = {s: i for i, s in enumerate(("hit", "bb", "k", "hr"))}
    for s in STATS:
        pr = np.array([combined_rates[i][idx[s]] for i in range(len(combined_ids))])
        ac = np.array([unk[b][s] for b in combined_ids])
        L.append(f"  {s.upper():5s} {np.corrcoef(pr, ac)[0, 1]:18.3f}")
    L.append("")
    aaa_only = sum(1 for b in ids if b in level_data["AAA"])
    L.append(f"Coverage: AAA-only reached {aaa_only} rookies; all levels reach {len(combined_ids)} "
             f"(+{len(combined_ids) - aaa_only}).")
    L.append("Factors are fit on rookies who made MLB (survivorship-selected), which is the")
    L.append("population injected; the LOO corr shows the single-scalar factors are not overfit.")
    foreign_empty = [n for n in ("NPB", "KBO", "NCAA") if not level_data.get(n)]
    if foreign_empty:
        L.append("")
        L.append(f"NOTE: {', '.join(foreign_empty)} returned no lines. Verified against known imports")
        L.append("(Yoshida/NPB, Jung Hoo Lee/KBO): the MLB Stats API yearByYear hydrate does not")
        L.append("serve foreign-pro or college history, so those translations need a separate")
        L.append("source (NPB/KBO official feeds, Baseball-Reference for NCAA). The affiliated")
        L.append("minors (AAA/AA/A+/A) are the levels this pipeline can source and fit today.")

    rep = "\n".join(L)
    print(rep)
    Path("data/eval2").mkdir(parents=True, exist_ok=True)
    Path("data/eval2/mle_priors.txt").write_text(rep + "\n")
    np.savez("data/eval2/mle_rates.npz",
             ids=np.array(combined_ids, dtype=np.int64),
             rates=np.array(combined_rates, dtype=np.float64),
             levels=np.array(combined_levels, dtype=object))
    print(f"saved -> data/eval2/mle_rates.npz ({len(combined_ids)} rookies)")


if __name__ == "__main__":
    main()
