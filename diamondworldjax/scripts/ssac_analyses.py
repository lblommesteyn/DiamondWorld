"""Applied analyses on the v13 simulator: what only a full generative game model
can do that per-player projection systems cannot.

  1. Counterfactual / what-if: swap a starter or a hitter, measure the causal
     change in win probability and run distribution.
  2. Lineup construction by simulation: search batting orders for the one that
     maximizes expected runs / win probability.
  3. Correlated tail-risk: the full per-game outcome distribution (blowouts,
     shutouts, extras) with within-game correlation that independent per-player
     models cannot produce.

All specs are batched into a single simulate() call. Saves data/eval2/ssac_*.
"""
from __future__ import annotations
import json, itertools
from pathlib import Path
import numpy as np
import polars as pl
import urllib.request

from diamondworldjax.paths import processed_root
from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.scripts.scenario_sim import Sim
from diamondworldjax.scripts.seq_models import pitcher_rates

R = 250
_NAMECACHE = Path("/tmp/mlb_names.json")


def names(ids):
    ids = [int(i) for i in ids]
    cache = json.loads(_NAMECACHE.read_text()) if _NAMECACHE.exists() else {}
    miss = [i for i in ids if str(i) not in cache]
    for j in range(0, len(miss), 100):
        chunk = miss[j:j+100]
        url = "https://statsapi.mlb.com/api/v1/people?personIds=" + ",".join(map(str, chunk))
        try:
            d = json.loads(urllib.request.urlopen(url, timeout=20).read())
            for p in d.get("people", []):
                cache[str(p["id"])] = p["fullName"]
        except Exception:
            pass
    _NAMECACHE.write_text(json.dumps(cache))
    return {i: cache.get(str(i), f"#{i}") for i in ids}


def wp(home, away):  # per-spec win prob for the HOME team
    return (home > away).mean(axis=1)


def main():
    s = Sim()
    idx2id = {v: k for k, v in s.id2i.items()}
    # pitcher quality (K-rate allowed) from training; ace = high K, replacement = low
    train = load_seasons([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
                         data_root=processed_root()).filter(pl.col("pa_terminal"))
    pit = pitcher_rates(train, s.id2i, len(s.ptab["hand"]))   # (P,4) [hit,bb,k,hr] allowed
    pcol = "pitcher_id" if "pitcher_id" in train.columns else "pitcher_idx"
    n_bf = train.group_by(pcol).len()
    bf_map = {s.id2i[int(r[pcol])]: r["len"] for r in n_bf.iter_rows(named=True) if int(r[pcol]) in s.id2i}
    starters = np.array([i for i, n in bf_map.items() if n >= 2000])   # high-volume = real starters
    ace_idx = starters[np.argmax(pit[starters, 2])]                    # dominant high-K starter
    repl_idx = starters[np.argmin(pit[starters, 2] - pit[starters, 0])]  # low K, high hits allowed
    # batter quality (wOBA-ish), qualified hitters only (>=300 PA in the weighted table)
    bstat = s.stats
    bq = np.where(bstat[:, 4] >= 300,
                  bstat[:, 0] + 1.8 * bstat[:, 3] + 0.7 * bstat[:, 1] - 0.3 * bstat[:, 2], -np.inf)

    games = s.real_games(2024, limit=500)
    def known(g):   # all key players seen in training (index != 0 = not the unknown slot)
        return (g["park"] != 0 and g["home_staff"] and g["away_staff"]
                and g["home_staff"][0] != 0 and g["away_staff"][0] != 0
                and sum(1 for x in g["home_lineup"] + g["away_lineup"] if x != 0) >= 17)
    clean = [g for g in games if known(g)]
    print(f"clean games (all key players known): {len(clean)}/{len(games)}", flush=True)
    game = clean[len(clean) // 2]                        # a normal game with known players
    home = list(game["home_lineup"]); away = list(game["away_lineup"])
    hsp, asp = game["home_staff"][0], game["away_staff"][0]

    def spec(**kw):
        base = dict(away_lineup=list(away), home_lineup=list(home),
                    away_staff=list(game["away_staff"]), home_staff=list(game["home_staff"]),
                    park=game["park"])
        base.update(kw); return base

    specs, labels = [], []
    specs.append(spec()); labels.append("baseline")
    # counterfactual 1: replace home starter with the league ace
    specs.append(spec(home_staff=[int(ace_idx)] + list(game["home_staff"][1:])))
    labels.append("home SP -> ace")
    # counterfactual 2: replace home starter with a replacement-level arm
    specs.append(spec(home_staff=[int(repl_idx)] + list(game["home_staff"][1:])))
    labels.append("home SP -> replacement")
    # counterfactual 3: upgrade the weakest home hitter to a star bat
    weak_slot = int(np.argmin([bq[i] for i in home]))
    star = int(np.argsort(bq)[-1])
    up = list(home); up[weak_slot] = star
    specs.append(spec(home_lineup=up)); labels.append("weakest hitter -> star bat")

    # lineup optimization: search orders of the home 9 (actual + heuristics + random)
    rng = np.random.default_rng(0)
    orders = [list(home)]
    orders.append(sorted(home, key=lambda i: -bq[i]))                 # best-to-worst
    orders.append(sorted(home, key=lambda i: -(bstat[i,0]+0.7*bstat[i,1])))  # OBP-ish leadoff
    for _ in range(37):
        o = list(home); rng.shuffle(o); orders.append(o)
    lo_start = len(specs)
    for o in orders:
        specs.append(spec(home_lineup=o))
    n_lo = len(orders)

    print(f"simulating {len(specs)} specs x R={R} ...", flush=True)
    H, A = s.run(specs, R=R, seed=7)

    # ---- report ----
    out = []
    hp = [int(idx2id[i]) for i in home]; ap = [int(idx2id[i]) for i in away]
    nm = names(hp + ap + [int(idx2id[ace_idx]), int(idx2id[repl_idx]), int(idx2id[star]),
                          int(idx2id[hsp]), int(idx2id[asp])])
    hometeam_ids = hp
    base_wp = wp(H[:1], A[:1])[0]; base_runs = H[0].mean()
    out.append(f"GAME: home SP {nm[int(idx2id[hsp])]} vs away SP {nm[int(idx2id[asp])]}, park {game['park']}")
    out.append(f"  baseline home win prob = {base_wp:.3f},  expected home runs = {base_runs:.2f}")
    out.append("")
    out.append("=== 1. COUNTERFACTUAL / WHAT-IF (change in home win prob) ===")
    for k in range(1, 4):
        w = wp(H[k:k+1], A[k:k+1])[0]; r = H[k].mean()
        detail = {1: f"(ace = {nm[int(idx2id[ace_idx])]})",
                  2: f"(repl = {nm[int(idx2id[repl_idx])]})",
                  3: f"(star = {nm[int(idx2id[star])]} into slot {weak_slot+1})"}[k]
        out.append(f"  {labels[k]:26s} {detail:34s}  WP {w:.3f}  (delta {w-base_wp:+.3f})  runs {r:.2f} ({r-base_runs:+.2f})")
    out.append("")
    out.append("=== 2. LINEUP CONSTRUCTION BY SIMULATION (home 9, expected runs) ===")
    lo_H = H[lo_start:lo_start+n_lo]
    exp_runs = lo_H.mean(axis=1); lo_wp = wp(lo_H, A[lo_start:lo_start+n_lo])
    actual_runs = exp_runs[0]; best = int(np.argmax(exp_runs))
    out.append(f"  actual order:    {actual_runs:.2f} runs/game,  WP {lo_wp[0]:.3f}")
    out.append(f"  best-found order:{exp_runs[best]:.2f} runs/game,  WP {lo_wp[best]:.3f}   (+{exp_runs[best]-actual_runs:.2f} runs, {lo_wp[best]-lo_wp[0]:+.3f} WP)")
    out.append(f"  worst order:     {exp_runs.min():.2f} runs/game   (spread across {n_lo} orders = {exp_runs.max()-exp_runs.min():.2f} runs)")
    out.append(f"  optimal batting order: " + " -> ".join(nm[int(idx2id[i])].split()[-1] for i in orders[best]))
    out.append("")
    out.append("=== 3. CORRELATED TAIL-RISK (baseline game, full distribution) ===")
    tot = H[0] + A[0]; margin = H[0] - A[0]
    out.append(f"  mean total {tot.mean():.2f}, std {tot.std():.2f}")
    out.append(f"  P(total >= 10) = {(tot>=10).mean():.3f}   P(<=5) = {(tot<=5).mean():.3f}")
    out.append(f"  P(shutout by either) = {((H[0]==0)|(A[0]==0)).mean():.3f}   P(extras, tie after reg proxy |margin|-blur) ~ P(margin==0 pre) = {(margin==0).mean():.3f}")
    out.append(f"  P(blowout >=5) = {(np.abs(margin)>=5).mean():.3f}   P(home wins by >=5) = {(margin>=5).mean():.3f}")
    # independent baseline: if each team's runs were independent Poisson(mean), the
    # total-variance would be mean(total); the sim's is larger => within-game correlation.
    indep_var = tot.mean()
    out.append(f"  total variance: sim {tot.var():.2f} vs independent-Poisson {indep_var:.2f} "
               f"(ratio {tot.var()/indep_var:.2f}x => correlated overdispersion)")

    rep = "\n".join(out)
    print(rep)
    Path("data/eval2/ssac_analyses.txt").write_text(rep)
    np.savez("data/eval2/ssac_dists.npz", base_home=H[0], base_away=A[0],
             cf_home=H[:4], cf_away=A[:4], lo_runs=exp_runs, labels=np.array(labels[:4], dtype=object))
    print("saved -> data/eval2/ssac_analyses.txt (+ ssac_dists.npz)")


if __name__ == "__main__":
    main()
