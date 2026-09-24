"""Run the simulator pre-game (hook model = realistic, look-ahead-free bullpen) over
all 2024 games and save the full per-game replica distributions, so
simulator_benchmarks.py can score it against Log5, the market, and independent-Poisson.

This is the current calibrated model, the fitted starter-pull hazard rather than the
actual hook times, and enough replicas (R=100) that per-game win probability is not
dominated by sampling noise.

KNOWN LEAKAGE, do not describe this run as pre-game without the caveat. The hazard
controls WHEN the starter is pulled, but game_extract.extract_games derives each
team's staff from the completed game's PA data, in actual appearance order, and the
lineup likewise. So the simulator is told which relievers appeared and in what
sequence, which a genuine pre-game forecast could not know and which correlates with
how the game went. Removing it needs a staff-selection model that draws from a team's
roster using only information available at first pitch. Until then the game-level
numbers this feeds (win probability, run-distribution coverage, market comparisons)
carry an unquantified optimistic bias.

Saves data/eval2/calib_<tag>_arrays.npz with the same schema the benchmark reads.

  python -m diamondworldjax.scripts.run_pregame_sim --tag v15-pregame-hook --r 100
"""
from __future__ import annotations

import argparse
import gc
import json
import os

import numpy as np
import polars as pl

from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.paths import processed_root
from diamondworldjax.scripts.scenario_sim import Sim


def real_runs(season=2024):
    te = (load_seasons([season], data_root=processed_root()).filter(pl.col("pa_terminal"))
          .group_by(["game_pk", "half_bin"]).agg(pl.col("runs_scored").sum().alias("r")))
    home = {int(r["game_pk"]): r["r"] for r in te.filter(pl.col("half_bin") == 1).iter_rows(named=True)}
    away = {int(r["game_pk"]): r["r"] for r in te.filter(pl.col("half_bin") == 0).iter_rows(named=True)}
    return {g: (home[g], away[g]) for g in home if g in away}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v15-pregame-hook")
    ap.add_argument("--r", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--season", type=int, default=2024,
                    help="season to simulate; must be after the checkpoint's train_end")
    # Which model to simulate with. Without these the script silently loaded v15
    # regardless of what the --tag claimed, which is how a sweep labelled v16 came
    # to have been run on v15. Now the checkpoint is explicit.
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint .pkl; default is Sim's V15 (say so in --tag)")
    ap.add_argument("--recency-halflife", type=float, default=2.0,
                    help="Player-table recency half-life for legacy checkpoints; modern checkpoints restore it.")
    ap.add_argument("--contact-quality", action="store_true",
                    help="must match how the checkpoint was trained (v16+, v22+)")
    ap.add_argument("--per-stat-shrink", action="store_true",
                    help="must match how the checkpoint's player table was built")
    ap.add_argument("--skill-prior", choices=["iso", "learned", "lkj", "walk"],
                    default="iso", help="must match how the checkpoint was trained")
    ap.add_argument("--pitchformer", action="store_true",
                    help="must match: checkpoint trained with --pitchformer")
    ap.add_argument("--pa-arch", type=str, default="transformer",
                    choices=["transformer", "gru", "gru_skip"],
                    help="PA sequence model architecture (requires --pitchformer).")
    ap.add_argument("--train-end", type=int, default=2023)
    ap.add_argument("--recal-file", default="data/eval2/v13_cal_params.npz",
                    help="Calibration .npz containing the selected vector (default: v13 heuristic file).")
    ap.add_argument("--recal-key", default="b_heur",
                    help="Array key in --recal-file (default: b_heur).")
    ap.add_argument("--recal-scale", type=float, default=0.18,
                    help="Multiplier for the calibration vector (default: 0.18).")
    ap.add_argument("--no-recal", action="store_true",
                    help="Sample raw outcome logits without a calibration vector.")
    ap.add_argument("--pregame-staff", action="store_true",
                    help="select relievers from prior games only, removing the "
                         "realized-bullpen leak described above")
    args = ap.parse_args()

    sim_kw = dict(hook_model=True, train_end=args.train_end,
                  recency_hl=args.recency_halflife,
                  contact_quality=args.contact_quality,
                  per_stat_shrink=args.per_stat_shrink,
                  skill_prior=args.skill_prior,
                  pitchformer=args.pitchformer,
                  pa_arch=args.pa_arch,
                  recal=args.recal_file,
                  recal_key=args.recal_key,
                  scale=args.recal_scale,
                  apply_recal=not args.no_recal)
    if args.ckpt:
        sim_kw["ckpt"] = args.ckpt
    print(f"model: {sim_kw.get('ckpt', 'V15 default')}  contact_quality={args.contact_quality}"
          f"  per_stat_shrink={args.per_stat_shrink}  skill_prior={args.skill_prior}"
          f"  pitchformer={args.pitchformer}  recal="
          f"{'off' if args.no_recal else f'{args.recal_file}[{args.recal_key}] x {args.recal_scale:g}'}",
          flush=True)
    s = Sim(**sim_kw)
    print("checkpoint config resolved: "
          f"train_end={s.train_end} recency_halflife={s.recency_hl} "
          f"contact_quality={s.contact_quality} per_stat_shrink={s.per_stat_shrink} "
          f"skill_prior={s.skill_prior} pitchformer={s.pitchformer}", flush=True)
    if args.season <= s.train_end:
        raise SystemExit(f"--season {args.season} is inside the checkpoint's training range "
                         f"(train_end={s.train_end}); the test season must be held out")
    outcomes = real_runs(args.season)
    games = [g for g in s.real_games(args.season, limit=10000, pregame_staff=args.pregame_staff)
             if g["park"] != 0 and int(g["game_pk"]) in outcomes]
    if args.limit:
        games = games[:args.limit]
    pk = np.array([int(g["game_pk"]) for g in games])
    rh = np.array([outcomes[int(g["game_pk"])][0] for g in games], float)
    ra = np.array([outcomes[int(g["game_pk"])][1] for g in games], float)
    print(f"pre-game sim over {len(games)} games x R={args.r} (hook model, crn off, "
          f"staff={'pregame' if args.pregame_staff else 'REALIZED (leaky)'})", flush=True)

    # Each chunk is checkpointed to its own file before the next one starts.
    #
    # Three runs of this script have now died mid-sweep with no traceback and no
    # non-zero exit: job 337 at step 5,000 of 50,000, job 393 at game 250 of 2,429,
    # job 395 at game 1,000 of 2,429. Detaching the process group (scripts/run_detached.sh)
    # ruled out the parent-tree hangup, and the deaths still happen, at a DIFFERENT
    # point each time. A silent SIGKILL that moves around under a box whose 35 GB of
    # RAM is shared with other jobs is the signature of the kernel OOM killer, not of
    # a bug in the simulation, so the chunk index is not the thing to debug.
    #
    # Rather than keep guessing at the cause, make a death cost one chunk instead of
    # the whole sweep: write each chunk out, and skip on restart what is already on
    # disk. A six-hour sweep that has to be perfect to finish is worse than one that
    # can be resumed six times.
    ckdir = f"data/chunks/{args.tag}"
    os.makedirs(ckdir, exist_ok=True)
    # A tag identifies a nominal run, but callers often sweep calibration scale
    # or replica count under that tag.  Never splice those incompatible draws
    # together merely because they cover the same game ids.
    cache_config = json.dumps({
        "ckpt": args.ckpt,
        "season": args.season,
        "r": args.r,
        "train_end": args.train_end,
        "contact_quality": args.contact_quality,
        "per_stat_shrink": args.per_stat_shrink,
        "skill_prior": args.skill_prior,
        "pitchformer": args.pitchformer,
        "recal_file": None if args.no_recal else args.recal_file,
        "recal_key": None if args.no_recal else args.recal_key,
        "recal_scale": None if args.no_recal else args.recal_scale,
    }, sort_keys=True)
    starts = list(range(0, len(games), args.chunk))
    for i in starts:
        ck = f"{ckdir}/chunk_{i:06d}.npz"
        # Resuming is only sound if the chunk on disk covers the SAME games this run
        # is about to simulate. The game list is rebuilt from scratch each run, so a
        # change in filtering or ordering would otherwise silently glue together
        # results for different games. Store the slice's game_pk and check it.
        want = pk[i:i + args.chunk]
        if os.path.exists(ck):
            with np.load(ck) as z:
                cached_config = str(z["config"].item()) if "config" in z else None
                if ("pk" in z and np.array_equal(z["pk"], want)
                        and cached_config == cache_config):
                    print(f"  {min(i + args.chunk, len(games))}/{len(games)} (cached)",
                          flush=True)
                    continue
            print(f"  chunk {i} has a different game/configuration, recomputing", flush=True)
        H, A = s.run(games[i:i + args.chunk], R=args.r, seed=0, skill_mode="mean", crn=False)
        # Write to a temporary name and rename, so a death DURING the write cannot
        # leave a truncated chunk that a later resume would trust.
        np.savez(ck + ".tmp.npz", H=H, A=A, pk=want, config=cache_config)
        os.replace(ck + ".tmp.npz", ck)
        del H, A
        gc.collect()
        print(f"  {min(i + args.chunk, len(games))}/{len(games)}", flush=True)

    Hs, As = [], []
    for i in starts:
        with np.load(f"{ckdir}/chunk_{i:06d}.npz") as z:
            if not np.array_equal(z["pk"], pk[i:i + args.chunk]):
                raise SystemExit(f"chunk {i} game_pk mismatch at assembly")
            if "config" not in z or str(z["config"].item()) != cache_config:
                raise SystemExit(f"chunk {i} configuration mismatch at assembly")
            Hs.append(z["H"]); As.append(z["A"])
    sh = np.concatenate(Hs, 0); sa = np.concatenate(As, 0)
    if len(sh) != len(games):
        raise SystemExit(f"assembled {len(sh)} games, expected {len(games)}")

    out = f"data/eval2/calib_{args.tag}_arrays.npz"
    np.savez(out, sim_home=sh, sim_away=sa, sim_total=sh + sa,
             real_home=rh, real_away=ra, real_total=rh + ra, game_pk=pk)
    print(f"saved -> {out}  (sim mean total {(sh+sa).mean():.2f}, real {(rh+ra).mean():.2f})")


if __name__ == "__main__":
    main()
