"""Leak-free season simulation with player ratings updated during the season.

run_pregame_sim freezes every player's rating at the end of training, so a pitcher who is
obviously elite by May (or a rookie with no major-league history at all) is simulated in
September as he looked the previous winter. The market does not work that way, and neither does
a front office. This rebuilds the player table before each month of the test season from the
training seasons PLUS every test-season game played before that month started, then simulates that
month's games with it. Nothing from a game on or after the month's first day is used, so every
input is still known before first pitch.

What changes and what does not:
  * the rate features of every player are recomputed with the same recipe as training (the
    checkpoint's stored config: recency half-life, contact quality, shrinkage, pitcher rates);
  * players with no training history get their OWN row instead of the shared unknown slot, with
    a zero skill vector (the prior mean), so their embedding comes from their stats to date;
  * the learned model parameters, the hook model, the rules engine and the leak-free staff
    selection are unchanged.

With a checkpoint trained without --pitcher-rates only hitters' ratings move, because a pitcher's
row carries his own batting rates; with --pitcher-rates both do.

Output arrays have the same format as run_pregame_sim, so every downstream script applies.

  python -m diamondworldjax.scripts.run_inseason_sim --ckpt <ckpt> --season 2024 --r 2000 --tag X
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.paths import processed_root
from diamondworldjax.scripts.run_pregame_sim import real_runs
from diamondworldjax.scripts.scenario_sim import Sim
from diamondworldjax.scripts.train_pa import _build_player_table

TABLE_KEYS = ("stats", "league", "hand", "bat_hand", "pit_hand")


def game_dates(season):
    for p in (Path(f"data/cache/sched_{season}.json"), Path(f"data/cache/market/sched_{season}.json")):
        if p.exists():
            sched = json.loads(p.read_text())
            return {int(g["gamePk"]): d["date"] for d in sched["dates"] for g in d["games"]}
    raise SystemExit(f"no cached schedule for {season}")


def extend_table(base, fresh):
    """Reorder `fresh` into `base`'s player order and append players `base` has never seen.

    Base players keep their indices, so the checkpoint's per-player skill vectors still line up;
    new players go after them and the unknown slot moves to the new end.
    """
    fid = fresh["id_to_idx"]
    base_ids = [int(p) for p in base["all_ids"]]
    missing = [p for p in base_ids if p not in fid]
    if missing:
        raise ValueError(f"{len(missing)} training players absent from the rebuilt table")
    extra = [int(p) for p in fresh["all_ids"] if int(p) not in base["id_to_idx"]]
    order = [fid[p] for p in base_ids] + [fid[p] for p in extra]
    out = {k: np.asarray(fresh[k])[order] for k in TABLE_KEYS if k in fresh}
    all_ids = np.array(base_ids + extra, dtype=np.int64)
    out.update(all_ids=all_ids, id_to_idx={int(p): i for i, p in enumerate(all_ids)},
               unknown_index=len(all_ids))
    return out, len(extra)


def install(sim, table, base_params, n_extra):
    """Point a Sim at a new player table, padding the skill posterior for new players."""
    import jax.numpy as jnp
    params = dict(base_params)
    for k in ("player_mu", "player_sigma"):
        if k in params:
            v = np.asarray(params[k])
            fill = 0.0 if k == "player_mu" else 1.0      # prior mean / prior scale
            pad = np.full((n_extra,) + v.shape[1:], fill, dtype=v.dtype)
            params[k] = jnp.asarray(np.concatenate([v, pad], 0))
    sim.params = params
    sim.ptab = table
    sim.id2i = table["id_to_idx"]
    sim.unknown_idx = table["unknown_index"]
    sim.stats = table["stats"]
    sim.pt.update(stats=jnp.array(table["stats"]), league=jnp.array(table["league"]),
                  hand=jnp.array(table["hand"]), unknown_index=table["unknown_index"],
                  bat_hand=np.asarray(table["bat_hand"]), pit_hand=np.asarray(table["pit_hand"]))
    # The fast inference adapter captures the table and parameters when first built; without
    # this reset every month after the first would silently reuse the first month's ratings.
    sim._mean_inference = None
    sim._mean_seq_inference = None
    sim._mean_adapter_built = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--r", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--limit", type=int, default=None, help="games per month, for smoke tests")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    ck = pickle.load(open(args.ckpt, "rb"))
    meta = ck.get("pa_metadata")
    if not meta:
        raise SystemExit("in-season updating needs a checkpoint with stored pa_metadata")
    cfg = meta["config"]
    s = Sim(ckpt=args.ckpt, hook_model=True)
    if args.season <= s.train_end:
        raise SystemExit(f"--season {args.season} is inside training (train_end={s.train_end})")
    base_table, base_params = s.ptab, dict(s.params)
    train = load_seasons(list(meta["train_seasons"]), data_root=processed_root())
    test = load_seasons([args.season], data_root=processed_root())
    dates = game_dates(args.season)
    test = test.with_columns(pl.col("game_pk").map_elements(
        lambda p: dates.get(int(p), "9999-99-99"), return_dtype=pl.Utf8).alias("_date"))
    outcomes = real_runs(args.season)

    months = defaultdict(list)
    for pk, d in dates.items():
        if pk in outcomes:
            months[d[:7]].append(pk)
    order = sorted(months)
    # March has a handful of openers; fold it into April so the first block is not tiny
    if order and order[0].endswith("-03") and len(order) > 1:
        months[order[1]] += months.pop(order[0])
        order = order[1:]

    Hs, As, pks = [], [], []
    for m in order:
        start = min(dates[p] for p in months[m])
        before = test.filter(pl.col("_date") < start).drop("_date")
        fresh = _build_player_table(
            pl.concat([train, before], how="diagonal_relaxed") if len(before) else train,
            recency_halflife=cfg.get("recency_halflife"),
            contact_quality=cfg.get("contact_quality", False),
            per_stat_shrink=cfg.get("per_stat_shrink", False),
            shrink_contact_quality=cfg.get("shrink_contact_quality", False),
            pitcher_rates=cfg.get("pitcher_rates", False))
        table, n_extra = extend_table(base_table, fresh)
        install(s, table, base_params, n_extra)
        want = set(months[m])
        games = [g for g in s.real_games(args.season, limit=100000, pregame_staff=True)
                 if int(g["game_pk"]) in want and g["park"] != 0]
        if args.limit:
            games = games[:args.limit]
        n_unk = sum(1 for g in games for i in list(g["home_lineup"]) + list(g["away_lineup"])
                    + [g["home_staff"][0], g["away_staff"][0]] if i == s.unknown_idx)
        print(f"{m}: {len(games)} games, rates through {start} (exclusive), {len(before):,} "
              f"in-season pitches, {n_extra} new players, {n_unk} unknown lineup/starter slots",
              flush=True)
        for i in range(0, len(games), args.chunk):
            H, A = s.run(games[i:i + args.chunk], R=args.r, seed=0, skill_mode="mean", crn=False)
            Hs.append(H); As.append(A)
        pks += [int(g["game_pk"]) for g in games]

    sh, sa = np.concatenate(Hs, 0), np.concatenate(As, 0)
    pk = np.array(pks)
    rh = np.array([outcomes[p][0] for p in pk], float)
    ra = np.array([outcomes[p][1] for p in pk], float)
    out = f"data/eval2/calib_{args.tag}_arrays.npz"
    np.savez(out, sim_home=sh, sim_away=sa, sim_total=sh + sa,
             real_home=rh, real_away=ra, real_total=rh + ra, game_pk=pk)
    print(f"saved -> {out}  ({len(pk)} games, sim mean total {(sh + sa).mean():.2f}, "
          f"real {(rh + ra).mean():.2f})")


if __name__ == "__main__":
    main()
