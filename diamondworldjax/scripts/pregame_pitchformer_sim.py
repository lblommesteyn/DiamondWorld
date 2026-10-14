"""Pre-game full-game simulation with the A--D pitch-level transformer stack.

eval_pitchformer_games.py rolls the pitch model out on each held-out game's OBSERVED schedule: the
batters who actually came up in each half and the pitchers who actually threw are read from the
finished game. That is fine for checking pitch-level realism but leaks the outcome into any
game-level forecast (a starter knocked out early shows up as an early bullpen in the schedule).
This driver uses only pre-game information, exactly as run_pregame_sim does for the PA model:

  lineups     the posted starting nine and batting order (extract_games), cycled with a pointer
              that the simulated game advances, so who leads off each inning is generated;
  staff       starter + relievers from prior games only (sim.pregame_staff, optionally date-ordered
              and fatigue-aware); the starter is pulled by the fitted state hazard, relievers by a
              sampled PAs-faced threshold, the same hook model the PA simulator uses;
  context     handedness (switch hitters bat opposite the pitcher), park, park geometry, weather.

Each (game, replica) is one row. Rows run inning by inning through rollout_batch, which generates
every pitch, count, base/out state and run; the transformer K/V caches are carried across halves
for the whole game as in training (history_reset=game). Limitation: pitching changes happen only
between half-innings (the rollout holds one pitcher per half); within-inning hooks move to the
next inning boundary.

Output has the run_pregame_sim array format, so validation_stats and the paired tests read it.

  python -m diamondworldjax.scripts.pregame_pitchformer_sim --tag abcdid_te2024_s42 --season 2024 \
      --r 500 --staff-by-date --out-tag abcdid-2024-pregame-r500
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import jax
import numpy as np
import polars as pl

from diamondworldjax.data.pipeline import load_seasons as load_pa_seasons
from diamondworldjax.data.pitch_seq import D_CTX_TOTAL, _env, load_seasons
from diamondworldjax.model.pitchformer import TransformerA, TransformerB
from diamondworldjax.model.pitchformer_checkpoint import add_skill_season, head_kwargs, restore_metadata
from diamondworldjax.model.superstate import geometry_features, load_geometry
from diamondworldjax.model.transformer_c import TransformerC
from diamondworldjax.model.transformer_d import TransformerD
from diamondworldjax.paths import processed_root
from diamondworldjax.sim.c_transition_engine import CTransitionEngine
from diamondworldjax.sim.game_extract import extract_games, fit_hook_dists, fit_hook_model, starter_pull_prob
from diamondworldjax.sim.pregame_staff import pregame_staffs
from diamondworldjax.sim.rules_engine import EmpiricalEngine
from diamondworldjax.simulate.pitchformer_rollout import PitchformerHeads, _cache_template, rollout_batch

PAS_PER_HALF = 12          # schedule slots; the rollout cycles them if a half needs more
PITCHES_PER_PA = 6         # token spacing of the slots (decode horizon = 12 * 6 * 1.5)
MAX_INNINGS = 20
MAX_CONTINUATIONS = 4


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def handedness(seasons):
    """MLB id -> batter stand (1 R, 0 L, -1 switch) and pitcher throws (1 R, 0 L)."""
    df = pl.concat([pl.read_parquet(f"data/processed/pitches_{y}.parquet",
                                    columns=["batter_id", "pitcher_id", "stand", "p_throws"])
                    for y in seasons])
    b = (df.group_by("batter_id").agg((pl.col("stand") == "R").mean().alias("r")))
    stand = {int(i): (1.0 if r > 0.9 else 0.0 if r < 0.1 else -1.0) for i, r in b.iter_rows()}
    p = df.group_by("pitcher_id").agg((pl.col("p_throws") == "R").mean().alias("r"))
    throws = {int(i): float(r > 0.5) for i, r in p.iter_rows()}
    return stand, throws


class Block:
    """State for a block of (game, replica) rows that are simulated together."""

    def __init__(self, specs, n):
        self.n = n
        self.specs = specs                      # per row: game dict (ABCD indices etc.)
        z = np.zeros(n, np.int32)
        self.home, self.away = z.copy(), z.copy()
        self.done = np.zeros(n, bool)
        self.ptr = np.zeros((n, 2), np.int32)   # lineup pointer per batting side (0 away, 1 home)
        self.cur = np.zeros((n, 2), np.int32)   # staff index per FIELDING side (0 away, 1 home)
        self.pa = np.zeros((n, 2), np.float64)  # PAs faced by the current pitcher
        self.ra = np.zeros((n, 2), np.float64)  # runs allowed by the current pitcher
        self.np_ = np.zeros((n, 2), np.int32)   # pitches thrown by the current pitcher
        self.hook = np.full((n, 2), np.inf)     # reliever PAs threshold


def build_batch(blk, rows_bat, half, inning, args, metadata):
    """Synthetic half-inning schedule for every row (ended rows included, so shapes are fixed)."""
    n = blk.n
    T = PAS_PER_HALF * PITCHES_PER_PA
    bat_side, fld_side = half, 1 - half
    out = {
        "pa_outcome": np.full((n, T), -1, np.int32), "pa_terminal": np.zeros((n, T), bool),
        "season": np.full((n, T), args.season, np.int32),
        "pitcher_idx": np.zeros((n, T), np.int32), "batter_idx": np.zeros((n, T), np.int32),
        "park_idx": np.zeros((n, T), np.int32), "ctx": np.zeros((n, T, D_CTX_TOTAL), np.float32),
        "geom": None, "pitch_type": np.zeros((n, T), np.int32), "type_valid": np.zeros((n, T), np.float32),
        "stuff": np.zeros((n, T, 5), np.float32), "stuff_valid": np.zeros((n, T), np.float32),
        "swing": np.zeros((n, T), np.float32), "contact": np.zeros((n, T), np.float32),
        "foul": np.zeros((n, T), np.float32), "hbp": np.zeros((n, T), np.float32),
        "valid": np.ones((n, T), np.float32), "pa_start": np.zeros((n, T), bool),
        "launch": np.zeros((n, T, 2), np.float32), "launch_valid": np.zeros((n, T), np.float32),
        "batted_out": np.zeros((n, T), np.int32), "batted_valid": np.zeros((n, T), np.float32),
    }
    geoms = []
    for i in range(n):
        g = blk.specs[i]
        staff = g["staff"][fld_side]
        pit = staff[min(blk.cur[i, fld_side], len(staff) - 1)]
        throws = g["throws"].get(pit, 1.0)
        out["pitcher_idx"][i] = g["pmap"].get(pit, 0)
        out["park_idx"][i] = g["park_idx"]
        out["ctx"][i, :, 16:] = g["env"]
        geoms.append(g["geom"])
        lineup = g["lineup"][bat_side]
        for k in range(PAS_PER_HALF):
            s = k * PITCHES_PER_PA
            bid = lineup[(blk.ptr[i, bat_side] + k) % len(lineup)]
            st = g["stand"].get(bid, 1.0)
            if st < 0:                          # switch hitter bats opposite the pitcher
                st = 1.0 - throws
            out["pa_start"][i, s] = True
            out["batter_idx"][i, s:s + PITCHES_PER_PA] = g["bmap"].get(bid, 0)
            out["ctx"][i, s:s + PITCHES_PER_PA, 13] = st
            out["ctx"][i, s:s + PITCHES_PER_PA, 14] = throws
            out["ctx"][i, s:s + PITCHES_PER_PA, 15] = float(st == throws)
    out["geom"] = np.repeat(np.stack(geoms)[:, None, :], T, axis=1).astype(np.float32)
    # dynamic context columns (count, bases, score, inning, half, tto) are written by the rollout
    out["ctx"][:, :, 9] = half
    out["ctx"][:, :, 10] = 1 - half
    out["ctx"][:, :, 8] = (inning - 5.0) / 4.0
    return add_skill_season(out, metadata)


def play_half(blk, half, inning, heads, engine, c_engine, cache, args, metadata, hook_model,
              reliever_pas, rng, seed):
    bat, fld = half, 1 - half
    # bottom of the 9th or later is not played when the home team already leads
    if half == 1 and inning >= 9:
        blk.done |= blk.home > blk.away
    live = ~blk.done
    if not live.any():
        return cache
    batch = build_batch(blk, None, half, inning, args, metadata)
    base0 = 2 if inning >= 10 else 0           # automatic runner on second in extras
    state = {
        "balls": np.zeros(blk.n, np.int32), "strikes": np.zeros(blk.n, np.int32),
        "outs": np.zeros(blk.n, np.int32), "base": np.full(blk.n, base0, np.int32),
        "home_score": blk.home.copy(), "away_score": blk.away.copy(),
        "inning": np.full(blk.n, inning, np.int32), "half": np.full(blk.n, half, np.int32),
        "tto": np.minimum(blk.pa[:, fld] // 9 + 1, 3).astype(np.int32),
        "pitch_count": blk.np_[:, fld].copy(), "pa_slot": np.zeros(blk.n, np.int32),
        "ended": blk.done.copy(),
    }
    n_pa = np.zeros(blk.n, np.int32)
    start_runs = np.where(half == 0, blk.away, blk.home).copy()
    for cont in range(MAX_CONTINUATIONS):
        rolled = rollout_batch(heads, batch, seed=seed + cont, engine=engine, c_engine=c_engine,
                               initial_state=state, initial_cache=cache, stop_when_decided=True,
                               decode_len=PAS_PER_HALF * PITCHES_PER_PA)
        cache = rolled.get("final_cache", cache)
        state = {k: np.asarray(v).copy() for k, v in rolled["final_state"].items()}
        k_pa = (rolled["pa_terminal"] & live[:, None]).sum(1).astype(np.int32)
        n_pa += k_pa
        blk.ptr[:, bat] += k_pa
        unfinished = live & ~state["ended"] & (state["inning"] == inning) & (state["half"] == half)
        if not unfinished.any():
            break
        # a very long half used up the decode horizon: continue it with the next batters due up,
        # parking the rows that are already finished
        state["pa_slot"][:] = 0
        state["ended"] = state["ended"] | ~unfinished
        batch = build_batch(blk, None, half, inning, args, metadata)
    else:
        blk.trunc += int(unfinished.sum())
    blk.np_[:, fld] = np.where(live, state["pitch_count"], blk.np_[:, fld])
    blk.home = np.where(live, state["home_score"], blk.home).astype(np.int32)
    blk.away = np.where(live, state["away_score"], blk.away).astype(np.int32)
    runs = np.where(half == 0, blk.away, blk.home) - start_runs
    # hooks for the fielding side, applied at the half boundary
    before = blk.pa[:, fld].copy()
    blk.pa[:, fld] += np.where(live, n_pa, 0)
    blk.ra[:, fld] += np.where(live, runs, 0)
    staff_len = np.array([len(s["staff"][fld]) for s in blk.specs])
    can = live & (blk.cur[:, fld] + 1 < staff_len)
    starter = blk.cur[:, fld] == 0
    fire = np.zeros(blk.n, bool)
    if hook_model is not None:
        # survive every PA of this half under the starter-pull hazard
        maxk = int(n_pa.max()) if len(n_pa) else 0
        for k in range(1, maxk + 1):
            pas = before + k
            p = starter_pull_prob(pas, np.full(blk.n, inning), np.minimum(pas // 9 + 1, 3),
                                  blk.ra[:, fld], hook_model)
            fire |= starter & (k <= n_pa) & (rng.random(blk.n) < p)
    fire |= (~starter) & (blk.pa[:, fld] >= blk.hook[:, fld])
    fire &= can
    blk.cur[fire, fld] += 1
    blk.pa[fire, fld] = 0
    blk.ra[fire, fld] = 0
    blk.np_[fire, fld] = 0
    blk.hook[fire, fld] = rng.choice(reliever_pas, size=int(fire.sum()))
    # game over after the bottom of the 9th or later with a decision
    if half == 1 and inning >= 9:
        blk.done |= blk.home != blk.away
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="ABCD checkpoint tag in --params-dir")
    ap.add_argument("--params-dir", default="checkpoints/pitchformer")
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--r", type=int, default=100)
    ap.add_argument("--rows", type=int, default=1024, help="(game, replica) rows per rollout block")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--staff-by-date", action="store_true")
    ap.add_argument("--bullpen-fatigue", action="store_true")
    ap.add_argument("--game-context", default="data/processed/game_context.parquet")
    ap.add_argument("--events", default="data/processed/events.parquet")
    ap.add_argument("--out-tag", required=True)
    # architecture defaults; restore_metadata overwrites them from the checkpoint
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--no-env", action="store_true")
    ap.add_argument("--no-geom", action="store_true")
    args = ap.parse_args()

    metadata = restore_metadata(args, args.tag)
    if not metadata:
        raise SystemExit(f"no metadata for {args.tag} in {args.params_dir}")
    maps = metadata["maps"]
    kw = head_kwargs(args, maps, metadata)
    root = Path(args.params_dir)
    heads = PitchformerHeads(
        TransformerA(**kw), TransformerB(**kw), TransformerC(**kw), TransformerD(**kw),
        _load(root / f"A_{args.tag}_params.pkl"), _load(root / f"B_{args.tag}_params.pkl"),
        _load(root / f"C_{args.tag}_params.pkl"), _load(root / f"D_{args.tag}_params.pkl"))
    carry = (metadata["config"].get("history_reset") == "game" and heads.a.window_size > 0)
    years = list(metadata["train_years"])
    print(f"ABCD {args.tag}: trained {years[0]}-{years[-1]}, history carry across halves: {carry}",
          flush=True)

    train = load_seasons(years)
    engine = EmpiricalEngine().fit(pl.concat(train).filter(pl.col("pa_terminal")))
    c_engine = CTransitionEngine(event_mode=getattr(heads.c, "c_event_mode", "legacy")).fit(
        pl.concat(train), pl.read_parquet(args.events))
    del train
    train_pa = load_pa_seasons(years, data_root=processed_root()).filter(pl.col("pa_terminal"))
    hook_model = fit_hook_model(train_pa)
    reliever_pas = fit_hook_dists(train_pa)[1]
    del train_pa

    # pre-game game specs, in MLB ids (identity id map), exactly as the PA simulator builds them
    te = load_pa_seasons([args.season], data_root=processed_root()).filter(pl.col("pa_terminal"))
    ids = set(te["batter_id"].unique().to_list()) | set(te["pitcher_id"].unique().to_list())
    ident = {int(i): int(i) for i in ids}
    games = extract_games(te, ident, unknown_idx=-1)
    dates = None
    if args.staff_by_date or args.bullpen_fatigue:
        from diamondworldjax.scripts.run_inseason_sim import game_dates
        dates = game_dates(args.season)
    staffs = pregame_staffs(te, ident, -1, dates=dates, fatigue=args.bullpen_fatigue)
    raw = pl.read_parquet(f"data/processed/pitches_{args.season}.parquet", columns=["game_pk", "park_id"])
    park_of = dict(raw.unique("game_pk").iter_rows())
    gctx = pl.read_parquet(args.game_context) if Path(args.game_context).exists() else None
    geo_table = load_geometry()
    stand, throws = handedness(years + [args.season])   # handedness is fixed, known pre-game

    from diamondworldjax.scripts.run_pregame_sim import real_runs
    outcomes = real_runs(args.season)
    specs = []
    for g in games:
        gp = int(g["game_pk"])
        if gp not in outcomes or gp not in park_of:
            continue
        st = {0: staffs.get((gp, 1)), 1: staffs.get((gp, 0))}   # away fields the bottom half
        if not st[0] or not st[1] or st[0][0] == -1 or st[1][0] == -1:
            continue
        park = park_of[gp]
        env = np.zeros(D_CTX_TOTAL - 16, np.float32)
        if gctx is not None and not args.no_env:
            row = gctx.filter(pl.col("game_pk") == gp)
            if row.height:
                env = _env(row)[0]
        geom = geometry_features([park], geo_table)[0]
        if args.no_geom:
            geom = np.zeros_like(geom)
        specs.append(dict(game_pk=gp, lineup={0: list(g["away_lineup"]), 1: list(g["home_lineup"])},
                          staff=st, park_idx=maps["park"].get(park, 0), env=env, geom=geom,
                          bmap=maps["batter"], pmap=maps["pitcher"], stand=stand, throws=throws))
    if args.limit:
        specs = specs[:args.limit]
    G, R = len(specs), args.r
    print(f"{G} games x R={R} = {G * R:,} rows, blocks of {args.rows}", flush=True)

    rows = [(gi, r) for gi in range(G) for r in range(R)]
    sim_h = np.zeros((G, R), np.int32)
    sim_a = np.zeros((G, R), np.int32)
    trunc = unresolved = 0
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    for b0 in range(0, len(rows), args.rows):
        chunk = rows[b0:b0 + args.rows]
        pad = args.rows - len(chunk)                     # fixed shapes: pad the last block
        chunk_specs = [specs[gi] for gi, _ in chunk] + [specs[chunk[0][0]]] * pad
        blk = Block(chunk_specs, args.rows)
        blk.trunc = 0
        if pad:
            blk.done[-pad:] = True
        cache = None
        if carry:
            tmpl = build_batch(blk, None, 0, 1, args, metadata)
            cache = tuple(_cache_template(m, p, tmpl) for m, p in
                          [(heads.a, heads.a_params), (heads.b, heads.b_params),
                           (heads.c, heads.c_params), (heads.d, heads.d_params)])
        for inning in range(1, MAX_INNINGS + 1):
            for half in (0, 1):
                seed = args.seed * 1_000_003 + b0 * 97 + inning * 1000 + half * 100
                new_cache = play_half(blk, half, inning, heads, engine, c_engine, cache, args,
                                      metadata, hook_model, reliever_pas, rng, seed)
                cache = new_cache if carry else None
            if inning >= 9:
                blk.done |= blk.home != blk.away
            if blk.done.all():
                break
        n = len(chunk)
        unresolved += int((~blk.done[:n]).sum())
        trunc += blk.trunc
        gi = np.array([c[0] for c in chunk]); ri = np.array([c[1] for c in chunk])
        sim_h[gi, ri] = blk.home[:n]
        sim_a[gi, ri] = blk.away[:n]
        done_rows = b0 + n
        el = time.time() - t0
        print(f"  {done_rows:,}/{len(rows):,} rows  {el / 60:.1f} min  "
              f"(eta {el / done_rows * (len(rows) - done_rows) / 60:.0f} min)  "
              f"mean runs so far {(sim_h.sum() + sim_a.sum()) / done_rows:.2f}", flush=True)

    pk = np.array([s["game_pk"] for s in specs])
    rh = np.array([outcomes[p][0] for p in pk], float)
    ra = np.array([outcomes[p][1] for p in pk], float)
    out = f"data/eval2/calib_{args.out_tag}_arrays.npz"
    np.savez(out, sim_home=sim_h, sim_away=sim_a, sim_total=sim_h + sim_a,
             real_home=rh, real_away=ra, real_total=rh + ra, game_pk=pk)
    print(f"saved -> {out}  ({G} games, sim mean total {(sim_h + sim_a).mean():.2f}, real "
          f"{(rh + ra).mean():.2f}; home win {(sim_h > sim_a).mean():.3f}; truncated halves {trunc}; "
          f"unresolved ties {unresolved})")


if __name__ == "__main__":
    main()
