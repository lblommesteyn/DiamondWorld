"""Decision backtest: does simulator-optimized lineup construction beat the actual order,
with the winner's-curse corrected (addresses the "demonstrate the application" point).

For each 2024 game we take the home nine and generate candidate batting orders (the
actual order, two rate heuristics, and random legal permutations). We SELECT the order
that maximizes expected runs on one set of replica seeds, then EVALUATE the selected
order, the actual order, and a random order on an INDEPENDENT set of seeds. The
selection/evaluation seed split removes the winner's curse: the best of many candidates
looks inflated on its own selection seeds, so we never score a choice on the seeds used
to pick it.

Honesty: a counterfactual lineup's real-world outcome is unobservable (the game was
played once, with the actual order), so the reported gain is the simulator's own
unbiased estimate of the lineup lever, not a realized-outcome backtest. It is
complemented by the market validation of the counterfactual machinery itself.

  python -m diamondworldjax.scripts.lineup_backtest --games 90 --cands 24
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from diamondworldjax.scripts.scenario_sim import Sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=90)
    ap.add_argument("--cands", type=int, default=24)
    ap.add_argument("--r-sel", type=int, default=150)
    ap.add_argument("--r-eval", type=int, default=300)
    ap.add_argument("--chunk-games", type=int, default=8)
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint .pkl; default is Sim's legacy V15. New checkpoints restore "
                         "their own feature table and config.")
    ap.add_argument("--pregame-staff", action="store_true",
                    help="bullpen from prior games only (leak-free), as in run_pregame_sim")
    ap.add_argument("--tag", default=None, help="report suffix; default keeps lineup_backtest.txt")
    args = ap.parse_args()

    s = Sim(ckpt=args.ckpt, hook_model=True) if args.ckpt else Sim()
    unknown = s.unknown_idx
    bstat = s.stats
    bq = bstat[:, 0] + 1.8 * bstat[:, 3] + 0.7 * bstat[:, 1] - 0.3 * bstat[:, 2]     # wOBA-ish
    obp = bstat[:, 0] + 0.7 * bstat[:, 1]
    games = [g for g in s.real_games(2024, limit=1500, pregame_staff=args.pregame_staff)
             if g["park"] != 0 and g["home_staff"] and g["home_staff"][0] != unknown
             and g["away_staff"] and g["away_staff"][0] != unknown
             and sum(1 for x in g["home_lineup"] if x != unknown) == 9][:args.games]
    print(f"{len(games)} games; {args.cands} candidate orders each", flush=True)

    rng = np.random.default_rng(0)
    sel_specs, order_of = [], []                    # candidate specs for selection
    for gi, g in enumerate(games):
        home = list(g["home_lineup"])
        orders = [list(home),
                  sorted(home, key=lambda i: -bq[i]),
                  sorted(home, key=lambda i: -obp[i])]
        while len(orders) < args.cands:
            o = list(home); rng.shuffle(o); orders.append(o)
        for o in orders:
            sel_specs.append(dict(away_lineup=list(g["away_lineup"]), home_lineup=o,
                                  away_staff=list(g["away_staff"]), home_staff=list(g["home_staff"]),
                                  park=g["park"]))
            order_of.append((gi, o))

    print(f"selection sim: {len(sel_specs)} specs x R={args.r_sel} (chunked) ...", flush=True)
    chunk = args.chunk_games * args.cands
    outs = []
    for i in range(0, len(sel_specs), chunk):
        H, _ = s.run(sel_specs[i:i + chunk], R=args.r_sel, seed=11, crn=False)
        outs.append(H.mean(1))
        print(f"  sel {min(i + chunk, len(sel_specs))}/{len(sel_specs)}", flush=True)
    sel_runs = np.concatenate(outs)

    # pick best order per game on selection seeds
    best_order = {}; actual_order = {}
    c = args.cands
    for gi in range(len(games)):
        block = sel_runs[gi * c:(gi + 1) * c]
        best_order[gi] = order_of[gi * c + int(np.argmax(block))][1]
        actual_order[gi] = list(games[gi]["home_lineup"])

    # evaluate selected, actual, and a random order on INDEPENDENT seeds
    ev_specs, tag = [], []
    for gi, g in enumerate(games):
        rnd = list(g["home_lineup"]); rng.shuffle(rnd)
        for o, t in ((best_order[gi], "sel"), (actual_order[gi], "act"), (rnd, "rnd")):
            ev_specs.append(dict(away_lineup=list(g["away_lineup"]), home_lineup=o,
                                 away_staff=list(g["away_staff"]), home_staff=list(g["home_staff"]),
                                 park=g["park"]))
            tag.append(t)
    print(f"evaluation sim (independent seeds): {len(ev_specs)} specs x R={args.r_eval} (chunked) ...", flush=True)
    echunk = args.chunk_games * 3
    eouts = []
    for i in range(0, len(ev_specs), echunk):
        H, _ = s.run(ev_specs[i:i + echunk], R=args.r_eval, seed=99, crn=False)
        eouts.append(H.mean(1))
    ev_runs = np.concatenate(eouts)
    ev = {t: ev_runs[i::3] for i, t in enumerate(["sel", "act", "rnd"])}

    gain = ev["sel"] - ev["act"]                     # honest gain (independent-seed eval)
    naive = np.array([sel_runs[gi * c + int(np.argmax(sel_runs[gi * c:(gi + 1) * c]))]
                      for gi in range(len(games))]) - np.array([
        sel_runs[gi * c:(gi + 1) * c][
            next(k for k, (g2, o) in enumerate(order_of[gi * c:(gi + 1) * c]) if o == actual_order[gi])]
        for gi in range(len(games))])
    boot = np.array([gain[rng.integers(0, len(gain), len(gain))].mean() for _ in range(3000)])
    ci = (np.percentile(boot, 2.5), np.percentile(boot, 97.5))

    L = ["LINEUP DECISION BACKTEST (winner's-curse corrected, independent selection/eval seeds)", ""]
    L.append(f"  {len(games)} games, {args.cands} candidate orders, R_sel={args.r_sel}, R_eval={args.r_eval}")
    L.append("")
    L.append(f"  naive gain (biased, selection seeds):     {naive.mean():+.3f} runs/game   <- winner's curse")
    L.append(f"  honest gain (independent-seed eval):      {gain.mean():+.3f} runs/game   95% CI "
             f"[{ci[0]:+.3f}, {ci[1]:+.3f}]")
    L.append(f"  fraction of games improved:               {(gain > 0).mean():.2f}")
    L.append(f"  selected vs random order (independent):   {(ev['sel']-ev['rnd']).mean():+.3f} runs/game")
    L.append(f"  actual  vs random order (independent):    {(ev['act']-ev['rnd']).mean():+.3f} runs/game")
    L.append("")
    L.append("  Read: the honest gain is the winner's-curse-corrected estimate; it is far below the")
    L.append("  naive selection-seed gain, and it is the simulator's own (not realized-outcome)")
    L.append("  estimate of the lineup lever. Over a 162-game season it is roughly "
             f"{gain.mean()*162:+.1f} runs.")
    rep = "\n".join(L)
    print(rep)
    Path("data/eval2").mkdir(parents=True, exist_ok=True)
    out = f"data/eval2/lineup_backtest_{args.tag}.txt" if args.tag else "data/eval2/lineup_backtest.txt"
    Path(out).write_text(rep + "\n")


if __name__ == "__main__":
    main()
