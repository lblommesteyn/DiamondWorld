"""Minor-league equivalency (MLE) priors for players unseen in MLB training.

The model is trained on 2015-2022 MLB, so 2023-24 debuts are blanks (~19% of 2024
PAs). Projection systems solve this with MLEs: translate a player's AAA/AA/foreign
line into an MLB-equivalent rate. This builds AAA-based priors for the unknown 2024
batters (from the MLB Stats API, sportId 11), translates them with standard
difficulty factors, and validates whether the translated prior predicts each
rookie's ACTUAL 2024 MLB rate -- i.e. whether feeding these as the rate features
would de-blank rookies. v1 uses literature factors; can be refined empirically.
"""
from __future__ import annotations
import json, time, urllib.request
import numpy as np, polars as pl
from diamondworldjax.paths import processed_root
from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.scripts.train_pa import _build_player_table

# AAA -> MLB translation (standard sabermetric MLEs; multiplicative on the rate)
FACT = dict(k=1.15, bb=0.90, hit=0.92, hr=0.80)


def fetch_aaa(ids, group="hitting"):
    """AAA (sportId 11) career-through-2023 counting stats per player id."""
    out = {}
    for j in range(0, len(ids), 60):
        chunk = ids[j:j+60]
        url = ("https://statsapi.mlb.com/api/v1/people?personIds=" + ",".join(map(str, chunk))
               + f"&hydrate=stats(group={group},type=yearByYear,sportId=11)")
        try:
            d = json.loads(urllib.request.urlopen(url, timeout=30).read())
        except Exception:
            time.sleep(2); continue
        for p in d.get("people", []):
            pa = k = bb = hr = h = 0
            for st in p.get("stats", []):
                for s in st.get("splits", []):
                    if int(s.get("season", 9999)) >= 2024:
                        continue   # only pre-MLB-2024 AAA
                    t = s["stat"]
                    pa += t.get("plateAppearances", 0) or 0; k += t.get("strikeOuts", 0) or 0
                    bb += (t.get("baseOnBalls", 0) or 0) + (t.get("hitByPitch", 0) or 0)
                    hr += t.get("homeRuns", 0) or 0; h += t.get("hits", 0) or 0
            if pa >= 100:
                out[int(p["id"])] = dict(pa=pa, k=k, bb=bb, hr=hr, hit=h)
        time.sleep(0.5)
    return out


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
        (pl.col("pa_outcome") == "HR").mean().alias("hr"), pl.len().alias("pa")])
    unk = [(int(r["batter_id"]), r) for r in g.iter_rows(named=True)
           if int(r["batter_id"]) not in id2i and r["pa"] >= 150]
    ids = [b for b, _ in unk]
    print(f"fetching AAA for {len(ids)} unknown 2024 batters (>=150 MLB PA)...", flush=True)
    aaa = fetch_aaa(ids)
    print(f"  got AAA history for {len(aaa)}/{len(ids)}", flush=True)

    rows = []
    for bid, r in unk:
        a = aaa.get(bid)
        if not a:
            continue
        # AAA rate -> translated MLB-equivalent
        aaa_k, aaa_bb = a["k"]/a["pa"], a["bb"]/a["pa"]
        aaa_hit, aaa_hr = a["hit"]/a["pa"], a["hr"]/a["pa"]
        pred = dict(k=aaa_k*FACT["k"], bb=aaa_bb*FACT["bb"], hit=aaa_hit*FACT["hit"], hr=aaa_hr*FACT["hr"])
        rows.append((pred, dict(k=r["k"], bb=r["bb"], hit=r["hit"], hr=r["hr"])))

    out = [f"MLE PRIORS for unknown rookies: translated AAA vs actual 2024 MLB rate",
           f"  rookies with AAA data + >=150 MLB PA: {len(rows)}", ""]
    out.append(f"  {'stat':5s} {'corr(MLE,actual)':>18s} {'MLE mean':>10s} {'actual mean':>12s}")
    for s in ("k", "bb", "hit", "hr"):
        p = np.array([r[0][s] for r in rows]); a = np.array([r[1][s] for r in rows])
        out.append(f"  {s.upper():5s} {np.corrcoef(p,a)[0,1]:18.3f} {p.mean():10.3f} {a.mean():12.3f}")
    out.append("")
    out.append(f"Read: a positive correlation means AAA lines carry real signal about rookie")
    out.append(f"MLB rates, so feeding translated-AAA rate features would de-blank the ~19% of")
    out.append(f"2024 PAs from unseen players (196 batters). v1 uses literature factors;")
    out.append(f"empirical factors + AA/NPB/KBO/college + retraining are the next refinements.")
    rep = "\n".join(out)
    print(rep)
    open("data/eval2/mle_priors.txt", "w").write(rep + "\n")
    # save the translated priors for injection into the player table
    np.savez("data/eval2/mle_rates.npz",
             ids=np.array([b for b, _ in unk if aaa.get(b)]),
             rates=np.array([[r[0]["hit"], r[0]["bb"], r[0]["k"], r[0]["hr"]] for r in rows]))


if __name__ == "__main__":
    main()
