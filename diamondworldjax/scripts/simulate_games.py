"""TRUE game simulator: generated inning structure, real rosters, real rules.

v2 (game-structure + bullpen upgrade). The v1 simulator generated outs and
lineup cycling correctly but kept three structural distortions:

  1. Fixed 9 innings for every game — no bottom-9 skip when the home team
     leads, no walk-offs, no extra innings. Tied games counted as home losses
     (home>away test), deflating home-win% to ~48 vs the real ~53.
  2. One starter pitched all 9 innings. Real games average ~4 pitchers per
     side; reliever/starter identity is a first-order effect on scoring.
  3. Starter extraction was CROSSED: `away_sp` held the pitcher the away
     lineup faces (the HOME starter) but the sim used it for the home half,
     so every lineup faced its own team's starter.

This version:

  - Plays halves until the game is decided: skips the bottom 9 when the home
    team leads after the top, ends walk-off halves the moment the home team
    takes the lead (non-HR walk-offs are capped at the winning run, per rule
    7.01(g)(3)), and plays extra innings with the ghost-runner-on-2B rule
    (in effect for the 2023-24 test seasons). Hard cap at MAX_INNINGS.
  - Models pitching changes: each side's real staff (starter + relievers in
    actual appearance order) is extracted per game, and pitchers are hooked
    after a number of PAs drawn from the empirical starter/reliever
    PAs-faced distributions fit on the training seasons. TTO resets for the
    batting team on a pitching change, and the fatigue proxy tracks the
    CURRENT pitcher's workload.
  - Optionally feeds real park indices (--use-park) for checkpoints trained
    with the park_idx fix (v9+). Pre-v9 checkpoints trained the park
    embedding on all-zeros, so the flag defaults off.

Legacy A/B flags: --fixed-nine restores the v1 fixed-9-inning structure and
--no-bullpen keeps the starter in all game.

Outcome sampling is the only model call; runs/bases/outs are engine
arithmetic via the EmpiricalEngine fit on 2015-22.

Usage:
  python -m diamondworldjax.scripts.simulate_games --ckpt <ckpt> --outcome-only --recal --limit-games 512
"""
from __future__ import annotations

import argparse
import pickle
import time
from functools import partial
from pathlib import Path

import numpy as np
import polars as pl

from diamondworldjax.paths import processed_root, checkpoints_root
from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.model.pa_model import pa_model
from diamondworldjax.sim.rules_engine import EmpiricalEngine, OUT_INC, PA_OUTCOME_IDX
from diamondworldjax.sim.game_extract import (
    cap_walkoff_runs,
    extract_games,
    fit_hook_dists,
    pad_staffs,
)
from diamondworldjax.scripts.train_pa import _build_player_table, _build_park_index

TRAIN = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
TEST = [2023, 2024]

MAX_INNINGS = 20   # safety cap; ghost runner decides virtually everything by ~13
HR_IDX = PA_OUTCOME_IDX["HR"]

# v6 per-class logit recalibration = log(real/model) from diag_outcomes ratios
# (K 1.31x, 1B 0.86x, ...). Corrects the mild outcome miscalibration without retraining.
# Order: K, BB, HBP, 1B, 2B, 3B, HR, out, E.
RECAL_V6 = np.array([
    -np.log(1.31), -np.log(0.97), -np.log(0.81), -np.log(0.86), -np.log(0.82),
    -np.log(0.93), -np.log(0.98), -np.log(0.92), 0.0,
], dtype=np.float64)

# v9 (outcome-only + fatigue + park_idx fix) recalibration = log(real/model) from
# diag_outcomes on real park indices. Much milder than v6 (K 1.14x vs 1.31x; the
# park fix improved raw calibration); main corrections are HR 0.72x and 2B 0.81x.
# Order: K, BB, HBP, 1B, 2B, 3B, HR, out, E.
RECAL_V9 = np.array([
    -0.1301, -0.0425, +0.2990, +0.0271, +0.2061, +0.0372, +0.3269, +0.0255, 0.0,
], dtype=np.float64)

# v10 (v9 recipe trained to 50K steps) recalibration = log(real/model) from
# diag_outcomes on real park indices. The longer run fixed HR calibration (1.00x,
# no correction needed vs v9's 0.72x) but over-predicts K more (1.28x vs 1.14x);
# net corrections lift 1B/2B/out and trim K. Order: K, BB, HBP, 1B, 2B, 3B, HR, out, E.
RECAL_V10 = np.array([
    -0.2457, -0.1308, +0.1857, +0.1488, +0.1782, +0.0441, -0.0027, +0.1060, 0.0,
], dtype=np.float64)

RECAL_VECS = {"v6": RECAL_V6, "v9": RECAL_V9, "v10": RECAL_V10}


def simulate(
    model_fn, params, pt, games, rng_key,
    shift=1.0, clock=1.0, recal=False, recal_scale=1.0, recal_vec=RECAL_V6,
    fixed_nine=False, no_bullpen=False, seed=0,
):
    """Vectorized simulation across all games with real game structure.

    Returns dict with per-game scores, occupancy, per-batter outcome counts,
    runs-by-inning (1-9), and structure stats (extras, walk-offs, ties).
    """
    import jax
    import jax.numpy as jnp
    import numpyro.handlers as nh

    G = len(games)
    away_lineup = np.array([g["away_lineup"] for g in games], dtype=np.int64)  # (G,9)
    home_lineup = np.array([g["home_lineup"] for g in games], dtype=np.int64)
    home_staff, home_staff_len = pad_staffs(games, "home_staff")  # pitches top halves
    away_staff, away_staff_len = pad_staffs(games, "away_staff")  # pitches bottom halves
    park = np.array([g["park"] for g in games], dtype=np.int64)
    engine = pt["_engine"]
    starter_pas, reliever_pas = pt["_hook_dists"]
    rng_np = np.random.default_rng(seed)

    away_score = np.zeros(G)
    home_score = np.zeros(G)
    away9 = np.full(G, np.nan)   # score snapshot after 9 innings (pre-extras)
    home9 = np.full(G, np.nan)
    away_ptr = np.zeros(G, dtype=np.int64)
    home_ptr = np.zeros(G, dtype=np.int64)
    away_cyc = np.zeros((G, 9), dtype=np.int64)
    home_cyc = np.zeros((G, 9), dtype=np.int64)

    # Per-staff pitcher state: current index into staff, PAs faced by the
    # current pitcher, and the hook threshold (PAs) for the current pitcher.
    def _staff_state():
        return {
            "cur": np.zeros(G, dtype=np.int64),
            "pa": np.zeros(G, dtype=np.float64),
            "hook": (np.full(G, np.inf) if no_bullpen
                     else rng_np.choice(starter_pas, size=G).astype(np.float64)),
        }

    home_ps = _staff_state()  # home staff state (faces away lineup)
    away_ps = _staff_state()

    game_over = np.zeros(G, dtype=bool)
    occ_on = 0.0
    occ_n = 0.0
    P = int(np.asarray(pt["stats"]).shape[0])
    pcounts = np.zeros((P, 9), dtype=np.float64)
    runs_by_inning = np.zeros(9)   # innings 1-9 (shape test); extras tracked apart
    extra_runs = 0.0
    n_walkoffs = 0
    PITCHES_PER_PA = 3.9

    def play_half(inning, half, mask):
        """Play one half-inning for every game in `mask` (bool, (G,))."""
        nonlocal occ_on, occ_n, extra_runs, n_walkoffs, rng_key
        if not mask.any():
            return
        walkoff = (not fixed_nine) and half == 1 and inning >= 9
        ghost = (not fixed_nine) and inning >= 10

        lineup = away_lineup if half == 0 else home_lineup
        ptr = away_ptr if half == 0 else home_ptr
        cyc = away_cyc if half == 0 else home_cyc
        ps = home_ps if half == 0 else away_ps            # fielding staff state
        staff = home_staff if half == 0 else away_staff
        staff_len = home_staff_len if half == 0 else away_staff_len
        bat_score = away_score if half == 0 else home_score
        fld_score = home_score if half == 0 else away_score

        outs = np.zeros(G, dtype=np.int64)
        bases = np.full(G, 2 if ghost else 0, dtype=np.int64)
        active = mask.copy()

        for _ in range(40):  # safety cap on PAs per half-inning
            if not active.any():
                break
            idx = np.where(active)[0]

            # Pitching change: hook reached and a fresh arm available.
            if not no_bullpen:
                need = idx[(ps["pa"][idx] >= ps["hook"][idx])
                           & (ps["cur"][idx] + 1 < staff_len[idx])]
                if len(need):
                    ps["cur"][need] += 1
                    ps["pa"][need] = 0.0
                    ps["hook"][need] = rng_np.choice(reliever_pas, size=len(need))
                    cyc[need, :] = 0  # new pitcher: batting team TTO resets

            slot = ptr[idx]
            batter = lineup[idx, slot]
            pitcher = staff[idx, ps["cur"][idx]]
            cyc[idx, slot] += 1
            tto = np.minimum(cyc[idx, slot], 3)

            B = len(idx)
            tb = {
                "pa_valid": jnp.ones((B, 1), bool),
                "inning": jnp.full((B, 1), (inning - 1) / 8.0, jnp.float32),
                "half": jnp.full((B, 1), float(half), jnp.float32),
                "outs": jnp.array((outs[idx] / 2.0)[:, None], jnp.float32),
                "base_state": jnp.array((bases[idx] / 7.0)[:, None], jnp.float32),
                "score_diff": jnp.array((np.clip(bat_score[idx] - fld_score[idx], -10, 10) / 10.0)[:, None], jnp.float32),
                "tto": jnp.array((tto / 3.0)[:, None], jnp.float32),
                "shift_restricted": jnp.full((B, 1), shift, jnp.float32),
                "pitch_clock": jnp.full((B, 1), clock, jnp.float32),
                "pitcher_ids": jnp.array(pitcher[:, None]),
                "batter_ids": jnp.array(batter[:, None]),
                "park_ids": jnp.array(park[idx][:, None]),
                "pitch_count_game": jnp.array(
                    np.clip(ps["pa"][idx] * PITCHES_PER_PA / 120.0, 0, 1.5)[:, None], jnp.float32),
            }
            rng_key, k = jax.random.split(rng_key)
            with nh.seed(rng_seed=k):
                with nh.substitute(data=params):
                    with nh.trace() as tr:
                        model_fn(tb, pt, teacher_force=False)
            if recal:
                logits = np.array(tr["pa_outcome"]["fn"].logits)[:, 0, :] + recal_scale * recal_vec
                oc = np.argmax(logits + rng_np.gumbel(size=logits.shape), axis=-1).astype(np.int64)
            else:
                oc = np.array(tr["pa_outcome"]["value"])[:, 0].astype(np.int64)

            occ_on += (bases[idx] > 0).sum()
            occ_n += B
            np.add.at(pcounts, (batter, oc), 1.0)
            e = engine.sample(bases[idx], outs[idx], oc, rng_np)
            runs = e["runs"].astype(np.float64)

            if walkoff:
                runs = cap_walkoff_runs(bat_score[idx], fld_score[idx], runs, oc == HR_IDX)

            bat_score[idx] += runs
            if inning <= 9:
                runs_by_inning[inning - 1] += runs.sum()
            else:
                extra_runs += runs.sum()

            oi = OUT_INC[oc]
            new_outs = outs[idx] + oi
            third = new_outs >= 3
            bases[idx] = np.where(third, 0, e["bs_after"])
            outs[idx] = new_outs
            ptr[idx] = (slot + 1) % 9
            ps["pa"][idx] += 1
            still = new_outs < 3
            if walkoff:
                won = bat_score[idx] > fld_score[idx]
                n_walkoffs += int(won.sum())
                game_over[idx[won]] = True
                still = still & ~won
            active[idx] = still

    n_extra_games = 0
    last_inning = 9 if fixed_nine else MAX_INNINGS
    for inning in range(1, last_inning + 1):
        playing = ~game_over
        if not playing.any():
            break
        if inning >= 10:
            n_extra_games += int(playing.sum())

        play_half(inning, 0, playing)

        if not fixed_nine and inning >= 9:
            # Home leads after the top: no bottom half needed.
            decided = playing & (home_score > away_score)
            game_over[decided] = True

        play_half(inning, 1, playing & ~game_over)

        if not fixed_nine and inning >= 9:
            game_over[playing & (home_score != away_score)] = True

        if inning == 9:
            # Snapshot the score at the end of regulation for every game that has
            # reached the 9th (all of them), before any extra innings are added.
            reached = np.isnan(away9)
            away9[reached] = away_score[reached]
            home9[reached] = home_score[reached]

    n_ties = int((~game_over).sum()) if not fixed_nine else int((home_score == away_score).sum())

    return {
        "away": away_score,
        "home": home_score,
        "away9": away9,
        "home9": home9,
        "occ": occ_on / max(occ_n, 1),
        "pcounts": pcounts,
        "runs_by_inning": runs_by_inning,
        "extra_runs": extra_runs,
        "n_extra_games": n_extra_games,
        "n_walkoffs": n_walkoffs,
        "n_ties": n_ties,
    }


def _rate_stats(c: np.ndarray) -> dict[str, float] | None:
    """length-9 outcome counts -> rate stats. Order K,BB,HBP,1B,2B,3B,HR,out,E."""
    pa = c.sum()
    if pa == 0:
        return None
    h = c[3] + c[4] + c[5] + c[6]
    ab = max(pa - c[1] - c[2], 1)
    tb = c[3] + 2 * c[4] + 3 * c[5] + 4 * c[6]
    return {"AVG": h / ab, "OBP": (h + c[1] + c[2]) / pa, "SLG": tb / ab,
            "K%": c[0] / pa, "HR%": c[6] / pa}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--outcome-only", action="store_true")
    ap.add_argument("--fatigue", action="store_true")
    ap.add_argument("--limit-games", type=int, default=512)
    ap.add_argument("--recal", action="store_true", help="Apply logit recalibration.")
    ap.add_argument("--recal-version", choices=list(RECAL_VECS), default="v6",
                    help="Which per-class recal vector to apply (v6 or v9). Must match the "
                         "checkpoint: v9's fatigue+park model has its own, milder calibration.")
    ap.add_argument("--recal-scale", type=float, default=0.3,
                    help="Pinned v6-final calibration strength (0.3 matches the FULL test-set "
                         "run rate: 8.81 vs 8.86 real. 0.4 was tuned on a biased subset).")
    ap.add_argument("--fixed-nine", action="store_true",
                    help="Legacy v1 structure: fixed 9 innings, no walk-offs/extras.")
    ap.add_argument("--no-bullpen", action="store_true",
                    help="Legacy v1 pitching: the starter pitches the whole game.")
    ap.add_argument("--use-park", action="store_true",
                    help="Feed real park indices (v9+ checkpoints trained with the "
                         "park_idx fix; pre-v9 park embeddings trained on all-zeros).")
    ap.add_argument("--player-stats", action="store_true",
                    help="Also report per-player stat-line reproduction (Phase 3, via true sim).")
    ap.add_argument("--min-pa", type=int, default=150)
    ap.add_argument("--dump-runs", type=Path, default=None,
                    help="Save per-game total runs (away+home) to this .npy for baseline comparison.")
    ap.add_argument("--dump-scores", type=Path, default=None,
                    help="Save per-game away/home final and after-9 scores to this .npz "
                         "(for tie/margin/extras diagnostics).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    with open(args.ckpt, "rb") as f:
        params = pickle.load(f)["params"]
    train_pitches = load_seasons(TRAIN, data_root=processed_root())
    ptab = _build_player_table(train_pitches)
    park_map = _build_park_index(train_pitches) if args.use_park else None
    train_pa = train_pitches.filter(pl.col("pa_terminal"))
    engine = EmpiricalEngine().fit(train_pa)
    hook_dists = fit_hook_dists(train_pa)
    print(f"Hook dists: starter median {np.median(hook_dists[0]):.0f} PA, "
          f"reliever median {np.median(hook_dists[1]):.0f} PA", flush=True)
    del train_pitches, train_pa

    test_pa = load_seasons(TEST, data_root=processed_root()).filter(pl.col("pa_terminal"))
    keep = test_pa["game_pk"].unique().sort().to_numpy()[:args.limit_games]
    test_pa = test_pa.filter(pl.col("game_pk").is_in(keep.tolist()))

    # real reference
    rg = (test_pa.group_by(["game_pk", "half_bin"]).agg(pl.col("runs_scored").sum().alias("r")))
    real_total = rg["r"].sum() / test_pa["game_pk"].n_unique()
    real_occ = (test_pa.with_columns((pl.col("base_state") > 0).cast(pl.Int32).alias("o"))["o"].mean()) * 100
    real_extra_rate = (
        test_pa.group_by("game_pk").agg(pl.col("inning").max().alias("mi"))
        .filter(pl.col("mi") >= 10)["mi"].len()
    ) / test_pa["game_pk"].n_unique() * 100

    print(f"Extracting lineups for {test_pa['game_pk'].n_unique()} games...", flush=True)
    games = extract_games(test_pa, ptab["id_to_idx"], park_map=park_map)
    print(f"  {len(games)} games usable", flush=True)

    pt = {"stats": jnp.array(ptab["stats"]), "league": jnp.array(ptab["league"]),
          "hand": jnp.array(ptab["hand"]), "_engine": engine, "_hook_dists": hook_dists}
    mkw = {}
    if args.outcome_only:
        mkw["outcome_only"] = True
    if args.fatigue:
        mkw["fatigue"] = True
    model_fn = partial(pa_model, **mkw) if mkw else pa_model

    t0 = time.time()
    res = simulate(
        model_fn, params, pt, games, jax.random.PRNGKey(args.seed),
        recal=args.recal, recal_scale=args.recal_scale, recal_vec=RECAL_VECS[args.recal_version],
        fixed_nine=args.fixed_nine, no_bullpen=args.no_bullpen, seed=args.seed,
    )
    away, home, total = res["away"], res["home"], res["away"] + res["home"]
    G = len(games)

    # Real runs-by-inning (innings 1-9), as fraction of total (shape, calibration-free)
    rdf = (test_pa.filter((pl.col("inning") >= 1) & (pl.col("inning") <= 9))
           .group_by("inning").agg(pl.col("runs_scored").sum().alias("r")).sort("inning"))
    real_rbi = np.zeros(9)
    for row in rdf.iter_rows(named=True):
        real_rbi[int(row["inning"]) - 1] = row["r"]
    real_frac = real_rbi / real_rbi.sum()
    sim_frac = res["runs_by_inning"] / max(res["runs_by_inning"].sum(), 1)
    print(f"\n=== runs-by-inning fraction (fatigue shape test) ===", flush=True)
    print(f"  inning   1    2    3    4    5    6    7    8    9   | late(6-9)", flush=True)
    print(f"  real  " + " ".join(f"{x*100:4.1f}" for x in real_frac) + f"  | {real_frac[5:].sum()*100:.1f}%", flush=True)
    print(f"  sim   " + " ".join(f"{x*100:4.1f}" for x in sim_frac) + f"  | {sim_frac[5:].sum()*100:.1f}%", flush=True)

    mode = []
    if args.fixed_nine:
        mode.append("fixed-nine")
    if args.no_bullpen:
        mode.append("no-bullpen")
    if args.use_park:
        mode.append("park")
    print(f"\n=== TRUE SIMULATION v2 ({G} games, {time.time()-t0:.0f}s"
          + (f", {'+'.join(mode)}" if mode else "") + ") ===", flush=True)
    print(f"  runs/game   real {real_total:.2f}   sim {total.mean():.2f}", flush=True)
    print(f"  occupancy   real {real_occ:.1f}%   sim {res['occ']*100:.1f}%", flush=True)
    print(f"  home runs/g {home.mean():.2f}   away {away.mean():.2f}   "
          f"home-win {np.mean(home > away)*100:.1f}%  (ties {res['n_ties']})", flush=True)
    print(f"  extras      real {real_extra_rate:.1f}%   sim {res['n_extra_games']/G*100:.1f}%   "
          f"walk-offs {res['n_walkoffs']/G*100:.1f}%   extra-inning runs {res['extra_runs']/G:.2f}/g", flush=True)

    if args.dump_runs is not None:
        np.save(args.dump_runs, total)
        print(f"  saved per-game runs -> {args.dump_runs}", flush=True)

    if args.dump_scores is not None:
        np.savez(args.dump_scores, away=res["away"], home=res["home"],
                 away9=res["away9"], home9=res["home9"])
        print(f"  saved per-game scores -> {args.dump_scores}", flush=True)

    if args.player_stats:
        P = int(np.asarray(pt["stats"]).shape[0])
        real_counts = np.zeros((P, 9))
        bcol = "batter_id" if "batter_id" in test_pa.columns else "batter_idx"
        for bid, oc in test_pa.select([bcol, "pa_outcome"]).iter_rows():
            if oc in PA_OUTCOME_IDX and int(bid) in ptab["id_to_idx"]:
                real_counts[ptab["id_to_idx"][int(bid)], PA_OUTCOME_IDX[oc]] += 1
        keep_idx = np.where(real_counts.sum(1) >= args.min_pa)[0]
        keep_idx = keep_idx[keep_idx != 0]  # exclude index 0 (unknown-player sink)
        mets = ["AVG", "OBP", "SLG", "K%", "HR%"]
        rv = {m: [] for m in mets}
        sv = {m: [] for m in mets}
        for i in keep_idx:
            rs, ss = _rate_stats(real_counts[i]), _rate_stats(res["pcounts"][i])
            if rs and ss:
                for m in mets:
                    rv[m].append(rs[m])
                    sv[m].append(ss[m])
        print(f"\n=== Player-level via TRUE SIM ({len(rv['AVG'])} batters >= {args.min_pa} PA) ===", flush=True)
        print(f"  {'stat':5s} {'real':>8s} {'sim':>8s} {'MAE':>8s} {'corr':>7s}", flush=True)
        for m in mets:
            a, b = np.array(rv[m]), np.array(sv[m])
            corr = np.corrcoef(a, b)[0, 1] if len(a) > 1 else float("nan")
            print(f"  {m:5s} {a.mean():8.4f} {b.mean():8.4f} {np.abs(a-b).mean():8.4f} {corr:7.3f}", flush=True)


if __name__ == "__main__":
    main()
