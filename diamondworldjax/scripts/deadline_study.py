"""Trade-deadline decision study: what did the simulator say a starter was worth at the deadline,
and did the market price his post-trade starts that way?

For each 2024 starter who changed teams mid-season with enough starts on both sides:

  at the deadline   his simulator value per start, from PRE-trade starts only (within-series
                    credit relative to his old rotation, market-calibrated). This is the number a
                    front office could have had when deciding.
  counterfactual    every post-trade start re-simulated with him and, separately, with each of his
                    new team's other regular starters in his place (common random numbers, so the
                    swap is the only difference). Summed over his starts, that is the wins he added
                    over the alternatives the acquiring team actually had.
  market check      the market's per-start credit for his POST-trade starts, which the
                    at-deadline number never saw.

Magnitudes are put in market units with the within-series calibration slope of the same simulator
(Table 3), so they are not refit here. With nine or so pitchers this is a worked example with an
illustrative correlation, not a powered validation; the powered starter test is starter_value.py.

  python -m diamondworldjax.scripts.deadline_study --ckpt <ckpt> --arrays <r2000 arrays>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from diamondworldjax.scripts.deadline_starters import game_dates, starts_by_team, team_names
from diamondworldjax.scripts.scenario_sim import Sim
from diamondworldjax.scripts.starter_value import starter_credits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arrays", required=True,
                    help="leak-free season arrays of the SAME checkpoint, for the per-start credits")
    ap.add_argument("--r", type=int, default=2000)
    ap.add_argument("--cal", type=float, default=0.412,
                    help="within-series market calibration slope of this simulator (Table 3)")
    ap.add_argument("--min-before", type=int, default=8)
    ap.add_argument("--min-after", type=int, default=5)
    ap.add_argument("--min-rot", type=int, default=3,
                    help="post-deadline starts needed to count as one of the new team's regulars")
    ap.add_argument("--ratings-through", default=None,
                    help="ISO date: rebuild player ratings from training seasons plus every 2024 game "
                         "BEFORE this date (e.g. 2024-07-30, the trade deadline), as run_inseason_sim "
                         "does, and run the counterfactuals with them. Default: frozen training ratings.")
    ap.add_argument("--tag", default="v22L")
    args = ap.parse_args()

    names = {int(k): v for k, v in json.loads(
        Path("data/cache/projections/mlb_names.json").read_text()).items()}
    dates, tnames = game_dates(), team_names()
    sbt = starts_by_team()
    credit, _ = starter_credits(args.arrays)
    cmap = {pid: {pk: (s, m) for pk, s, m in c} for pid, c in credit.items()}

    movers = []
    for pid, starts in sbt.items():
        teams = [t for _, t, _ in starts]
        if len(set(teams)) < 2:
            continue
        k = next(i for i, t in enumerate(teams) if t != teams[0])
        new = teams[k]
        pre = starts[:k]
        post = [s for s in starts[k:] if s[1] == new]
        if len(pre) >= args.min_before and len(post) >= args.min_after:
            movers.append((pid, teams[0], new, dates[post[0][0]], pre, post))

    s = Sim(ckpt=args.ckpt, hook_model=True)
    if args.ratings_through:
        # Decision-time ratings: what a front office knew on the cutoff date, nothing later.
        import pickle
        import polars as pl
        from diamondworldjax.data.pipeline import load_seasons
        from diamondworldjax.paths import processed_root
        from diamondworldjax.scripts.run_inseason_sim import extend_table, install
        from diamondworldjax.scripts.train_pa import _build_player_table
        meta = pickle.load(open(args.ckpt, "rb"))["pa_metadata"]
        cfg = meta["config"]
        train = load_seasons(list(meta["train_seasons"]), data_root=processed_root())
        te = load_seasons([2024], data_root=processed_root())
        before_pks = [p for p, d in dates.items() if d < args.ratings_through]
        before = te.filter(pl.col("game_pk").is_in(before_pks))
        fresh = _build_player_table(
            pl.concat([train, before], how="diagonal_relaxed"),
            recency_halflife=cfg.get("recency_halflife"),
            contact_quality=cfg.get("contact_quality", False),
            per_stat_shrink=cfg.get("per_stat_shrink", False),
            shrink_contact_quality=cfg.get("shrink_contact_quality", False),
            pitcher_rates=cfg.get("pitcher_rates", False))
        table, n_extra = extend_table(s.ptab, fresh)
        install(s, table, dict(s.params), n_extra)
        print(f"ratings through {args.ratings_through} (exclusive): {len(before):,} 2024 pitches, "
              f"{n_extra} players added", flush=True)
    games = {int(g["game_pk"]): g for g in s.real_games(2024, limit=10000, pregame_staff=True)}
    id2i = s.id2i

    rows = []
    for pid, old, new, sw, pre, post in movers:
        # regulars of the new team after the switch, excluding the acquired pitcher
        cnt = {}
        for q, st in sbt.items():
            if q == pid:
                continue
            c = sum(1 for pk, t, _ in st if t == new and dates.get(pk, "") >= sw)
            if c >= args.min_rot and q in id2i:
                cnt[q] = c
        reps = sorted(cnt, key=lambda q: -cnt[q])
        specs, meta = [], []
        for pk, _, home in post:
            g = games.get(pk)
            if g is None or not reps:
                continue
            side = "home_staff" if home else "away_staff"
            for who in [pid] + reps:
                staff = list(g[side])
                wi = id2i[who]
                staff = [wi] + [x for x in staff[1:] if x != wi]
                spec = dict(away_lineup=list(g["away_lineup"]), home_lineup=list(g["home_lineup"]),
                            away_staff=list(g["away_staff"]), home_staff=list(g["home_staff"]),
                            park=g["park"])
                spec[side] = staff
                specs.append(spec)
                meta.append((pk, home, who == pid))
        if not specs:
            continue
        H, A = s.run(specs, R=args.r, seed=7, crn=True)
        wp = (H > A).mean(1)
        # own-team win probability
        own = np.array([w if home else 1 - w for w, (_, home, _) in zip(wp, meta)])
        deltas = []
        i = 0
        n_alt = len(reps)
        while i < len(own):
            deltas.append(own[i] - own[i + 1:i + 1 + n_alt].mean())
            i += 1 + n_alt
        deltas = np.array(deltas)
        pre_sim = [cmap[pid][pk][0] for pk, _, _ in pre if pk in cmap.get(pid, {})]
        post_mkt = [cmap[pid][pk][1] for pk, _, _ in post if pk in cmap.get(pid, {})]
        rows.append(dict(
            name=names.get(pid, str(pid)), old=tnames.get(old, old), new=tnames.get(new, new),
            switch=sw, n_pre=len(pre), n_post=len(deltas), n_rep=n_alt,
            pre_sim=args.cal * np.mean(pre_sim) * 100 if pre_sim else np.nan,
            post_mkt=np.mean(post_mkt) * 100 if post_mkt else np.nan,
            cf_per_start=args.cal * deltas.mean() * 100,
            cf_wins=args.cal * deltas.sum()))

    L = ["TRADE-DEADLINE DECISION STUDY (2024), win probability in market-calibrated points",
         f"  ratings: {'through ' + args.ratings_through + ' (exclusive)' if args.ratings_through else 'frozen at end of training'}; "
         f"per-start credits from {args.arrays}",
         f"  simulator: {args.ckpt}, R={args.r} per counterfactual, calibration slope {args.cal}",
         "  at deadline = sim value per start from PRE-trade starts only;",
         "  counterfactual = post-trade starts re-simulated with him vs each new-team regular;",
         "  market = market credit per start on his POST-trade starts (unseen by the at-deadline number)",
         "",
         f"  {'pitcher':20s} {'trade':38s} {'pre':>3s} {'post':>4s} {'at deadline':>11s} "
         f"{'counterfact':>11s} {'wins added':>10s} {'market':>7s}"]
    for r in sorted(rows, key=lambda r: -r["cf_wins"]):
        L.append(f"  {r['name'][:20]:20s} {(r['old'] + ' -> ' + r['new'])[:38]:38s} {r['n_pre']:3d} "
                 f"{r['n_post']:4d} {r['pre_sim']:+11.2f} {r['cf_per_start']:+11.2f} "
                 f"{r['cf_wins']:+10.2f} {r['post_mkt']:+7.2f}")
    pre = np.array([r["pre_sim"] for r in rows]); cf = np.array([r["cf_per_start"] for r in rows])
    mk = np.array([r["post_mkt"] for r in rows])
    ok = np.isfinite(pre) & np.isfinite(mk)
    L.append("")
    L.append(f"  {len(rows)} pitchers. corr(at-deadline sim value, post-trade market value) = "
             f"{np.corrcoef(pre[ok], mk[ok])[0, 1]:+.2f}; corr(counterfactual, market) = "
             f"{np.corrcoef(cf[ok], mk[ok])[0, 1]:+.2f}  (illustrative at this n)")
    L.append(f"  total counterfactual wins added across all acquisitions: {sum(r['cf_wins'] for r in rows):+.2f}")
    rep = "\n".join(L)
    print(rep)
    Path(f"data/eval2/deadline_study_{args.tag}.txt").write_text(rep + "\n")


if __name__ == "__main__":
    main()
