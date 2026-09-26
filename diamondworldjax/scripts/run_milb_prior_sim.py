"""Season simulation where players new to MLB get a translated prior instead of the blank placeholder.

The model maps every player absent from its training table to one shared placeholder (a zero
embedding), so a rookie or an import with a strong record elsewhere is simulated exactly like one with
a weak record. This appends a row for each such player who appears in a test-season game, filled from
his prior-season lines in other leagues, translated to MLB scale with MLE-style factors and shrunk
toward the league with the model's per-stat constants.

  hitters   AAA/AA (always) and NPB (--npb) batting lines -> batting columns 0..4
  pitchers  AAA/AA and NPB pitching lines -> pitcher allowed-rate columns 7..11 (--pitchers; needs a
            checkpoint trained with --pitcher-rates, since the locked model has no pitcher columns)

Factors are fit on players seen in the source league in season t and in MLB in t+1, with t+1 before
the test season, so the test season never informs its own factors. Prior lines come from the two
seasons before the test season (the later one at full weight, the earlier at half). New rows get a
zero skill vector (the prior mean); the model itself is unchanged, and nothing from the test season is
used except each player's handedness, which is known before the game.

  python -m diamondworldjax.scripts.run_milb_prior_sim --ckpt <ckpt> --season 2024 --r 2000 --tag X \
      [--pitchers] [--npb]
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np
import polars as pl

from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.paths import processed_root
from diamondworldjax.scripts.milb_translation import SHRINK, STATS, milb_rates, mlb_rates
from diamondworldjax.scripts.run_inseason_sim import install
from diamondworldjax.scripts.run_pregame_sim import real_runs
from diamondworldjax.scripts.scenario_sim import Sim
from diamondworldjax.scripts.train_pa import PITCHER_BF_COL, PITCHER_RATE_COLS, PITCHER_SHRINK_REG

RATE = {"hit": "hit", "bb": "bb_rate", "k": "k", "hr": "hr_rate"}
COUNT = {"hit": "n_hit", "bb": "n_bb", "k": "n_k", "hr": "n_hr"}
PSHRINK = dict(zip(STATS, PITCHER_SHRINK_REG))


def with_rates(df):
    return df.with_columns([(pl.col("h") / pl.col("pa")).alias("hit"),
                            ((pl.col("bb") + pl.col("hbp")) / pl.col("pa")).alias("bb_rate"),
                            (pl.col("so") / pl.col("pa")).alias("k"),
                            (pl.col("hr") / pl.col("pa")).alias("hr_rate")])


def milb_pitching(path="data/cache/milb/milb_pitching.csv"):
    m = pl.read_csv(path).group_by(["player_id", "season", "level"]).agg(
        [pl.col(c).sum() for c in ("pa", "h", "bb", "hbp", "so", "hr")])
    return with_rates(m)


def mlb_pitching(seasons):
    t = (load_seasons(seasons, data_root=processed_root())
         .filter(pl.col("pa_terminal") & pl.col("pa_outcome").is_not_null())
         .select(["pitcher_id", "season", "pa_outcome"]))
    o = pl.col("pa_outcome")
    return (t.group_by(["pitcher_id", "season"]).agg([
        pl.len().alias("pa"),
        o.is_in(["1B", "2B", "3B", "HR"]).sum().alias("n_hit"),
        o.is_in(["BB", "HBP"]).sum().alias("n_bb"),
        (o == "K").sum().alias("n_k"),
        (o == "HR").sum().alias("n_hr")]).rename({"pitcher_id": "player_id"}))


# ---------------- NPB name matching ----------------
def norm(name):
    """Order-free, accent-free, romanization-tolerant key: 'Otani, Shohei' == 'Shohei Ohtani'."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ,]", "", s)
    toks = [t for t in re.split(r"[ ,]+", s) if t]
    fix = []
    for t in toks:
        for a, b in (("ou", "o"), ("oo", "o"), ("uu", "u"), ("oh", "o")):
            t = t.replace(a, b)
        fix.append(t)
    return " ".join(sorted(fix))


def mlb_people(seasons):
    """MLBAM id -> full name for every MLB player in the given seasons (cached)."""
    cache = Path("data/cache/npb/mlb_people.json")
    known = json.loads(cache.read_text()) if cache.exists() else {}
    for s in seasons:
        if str(s) in known.get("_seasons", []):
            continue
        d = json.load(urllib.request.urlopen(
            f"https://statsapi.mlb.com/api/v1/sports/1/players?season={s}", timeout=60))
        for p in d.get("people", []):
            known[str(p["id"])] = p["fullName"]
        known.setdefault("_seasons", []).append(str(s))
    cache.write_text(json.dumps(known))
    return {int(k): v for k, v in known.items() if k != "_seasons"}


def npb_with_ids(group, people):
    """NPB lines with an MLB id, where the romanized name matches exactly one MLB player."""
    path = Path(f"data/cache/npb/npb_{group}.csv")
    if not path.exists():
        return None
    by_key = {}
    for pid, nm in people.items():
        by_key.setdefault(norm(nm), []).append(pid)
    npb = pl.read_csv(path).with_columns(
        pl.col("name").map_elements(lambda n: (lambda ids: ids[0] if len(ids) == 1 else None)(
            by_key.get(norm(n), [])), return_dtype=pl.Int64).alias("player_id"))
    npb = npb.filter(pl.col("player_id").is_not_null())
    npb = npb.group_by(["player_id", "season"]).agg([pl.col(c).sum() for c in ("pa", "h", "bb", "hbp", "so", "hr")])
    return with_rates(npb.with_columns(pl.lit("NPB").alias("level")))


def fit_factors(src, mlb, T, levels, min_src=150, min_mlb=100):
    pairs = (src.filter(pl.col("pa") >= min_src).with_columns((pl.col("season") + 1).alias("next"))
             .join(mlb.filter(pl.col("pa") >= min_mlb), left_on=["player_id", "next"],
                   right_on=["player_id", "season"], suffix="_mlb")
             .filter(pl.col("next") < T))
    out = {}
    for lvl in levels:
        p = pairs.filter(pl.col("level") == lvl)
        out[lvl] = {s: float(p[COUNT[s]].sum() / max((p["pa_mlb"] * p[RATE[s]]).sum(), 1e-9)) for s in STATS}
        out[lvl]["n"] = len(p)
    return out


def translate(lines, factors, league, shrink, T):
    pa_eff, acc = 0.0, dict.fromkeys(STATS, 0.0)
    for ln in lines.iter_rows(named=True):
        w = 1.0 if ln["season"] == T - 1 else 0.5
        f = factors[ln["level"]]
        for k in STATS:
            acc[k] += w * ln["pa"] * ln[RATE[k]] * f[k]
        pa_eff += w * ln["pa"]
    return [(acc[k] + shrink[k] * league[k]) / (pa_eff + shrink[k]) for k in STATS], pa_eff


def prepare_priors(T, npb=False, pitchers=False):
    """Source lines, translation factors and league rates for test season T (all pre-T)."""
    log = []
    people = mlb_people(range(2015, T + 1)) if npb else {}
    bat_src = milb_rates().select(["player_id", "season", "level", "pa", "hit", "bb_rate", "k", "hr_rate"])
    bat_levels = ["AAA", "AA"]
    if npb:
        nb = npb_with_ids("batting", people)
        if nb is not None:
            bat_src = pl.concat([bat_src, nb.select(bat_src.columns)])
            bat_levels.append("NPB")
    mlb_b = mlb_rates(list(range(2015, T + 1)))
    fb = fit_factors(bat_src, mlb_b, T, bat_levels)
    prev = mlb_b.filter(pl.col("season") < T)
    ctx = dict(T=T, people=people, bat_src=bat_src.filter(pl.col("season").is_in([T - 1, T - 2])),
               fb=fb, lg_b={k: float(prev[COUNT[k]].sum() / prev["pa"].sum()) for k in STATS}, log=log)
    log.append(f"batting factors {json.dumps(fb)}")
    if pitchers:
        pit_src = milb_pitching().select(bat_src.columns)
        pit_levels = ["AAA", "AA"]
        if npb:
            npit = npb_with_ids("pitching", people)
            if npit is not None:
                pit_src = pl.concat([pit_src, npit.select(pit_src.columns)])
                pit_levels.append("NPB")
        mlb_p = mlb_pitching(list(range(2015, T + 1)))
        fp = fit_factors(pit_src, mlb_p, T, pit_levels)
        prevp = mlb_p.filter(pl.col("season") < T)
        ctx.update(pit_src=pit_src.filter(pl.col("season").is_in([T - 1, T - 2])), fp=fp,
                   lg_p={k: float(prevp[COUNT[k]].sum() / prevp["pa"].sum()) for k in STATS})
        log.append(f"pitching factors {json.dumps(fp)}")
    return ctx


def append_priors(base, te, ctx, cfg, hitters=True, pitchers=False):
    """Append a translated-prior row for every player in `te` who is absent from `base`.

    Returns (table, n_new, names). `base` keeps its rows and order, so the checkpoint's skill
    vectors still line up; install() pads the skill posterior for the n_new appended rows.
    """
    T = ctx["T"]
    stats = np.asarray(base["stats"])
    known = base["id_to_idx"]
    # template rows: what a table pitcher looks like in the batting columns, and vice versa
    bf_col = stats[:, PITCHER_BF_COL] if cfg.get("pitcher_rates") else np.zeros(len(stats))
    pitcher_like = (stats[:, 4] == 0) & (bf_col > 0)
    hitter_like = (stats[:, 4] > 0) & (bf_col == 0)
    tmpl_p = stats[pitcher_like].mean(0) if pitcher_like.any() else stats.mean(0)
    tmpl_h = stats[hitter_like].mean(0) if hitter_like.any() else stats.mean(0)

    new_ids, rows, bat_hand, pit_hand, names = [], [], [], [], []
    if hitters:
        hs = (te.filter(~pl.col("batter_id").is_in(list(known)))
                .group_by("batter_id").agg((pl.col("batter_hand") == "R").mean().alias("r")))
        for r in hs.iter_rows(named=True):
            lines = ctx["bat_src"].filter(pl.col("player_id") == r["batter_id"])
            if len(lines) == 0:
                continue
            rates, pa_eff = translate(lines, ctx["fb"], ctx["lg_b"], SHRINK, T)
            row = tmpl_h.copy()
            row[:4] = rates
            row[4] = pa_eff
            new_ids.append(int(r["batter_id"])); rows.append(row)
            bat_hand.append(1.0 if (r["r"] or 0) >= 0.5 else 0.0); pit_hand.append(0.5)
            names.append(("H", int(r["batter_id"]), sorted(set(lines["level"].to_list()))))
    if pitchers:
        taken = set(new_ids)
        ps = (te.filter(~pl.col("pitcher_id").is_in(list(known)))
                .group_by("pitcher_id").agg((pl.col("pitcher_hand") == "R").mean().alias("r")))
        bf_log = cfg.get("pitcher_bf_log", False)
        for r in ps.iter_rows(named=True):
            if int(r["pitcher_id"]) in taken:
                continue
            lines = ctx["pit_src"].filter(pl.col("player_id") == r["pitcher_id"])
            if len(lines) == 0:
                continue
            rates, bf = translate(lines, ctx["fp"], ctx["lg_p"], PSHRINK, T)
            row = tmpl_p.copy()
            row[list(PITCHER_RATE_COLS)] = rates
            row[PITCHER_BF_COL] = np.log1p(bf) / np.log1p(2500.0) if bf_log else bf
            new_ids.append(int(r["pitcher_id"])); rows.append(row)
            bat_hand.append(0.5); pit_hand.append(1.0 if (r["r"] or 0) >= 0.5 else 0.0)
            names.append(("P", int(r["pitcher_id"]), sorted(set(lines["level"].to_list()))))

    n = len(new_ids)
    table = {k: np.asarray(base[k]) for k in ("stats", "league", "hand", "bat_hand", "pit_hand")}
    if n == 0:
        table.update(all_ids=np.asarray(base["all_ids"]), id_to_idx=dict(known),
                     unknown_index=base["unknown_index"])
        return table, 0, names
    table["stats"] = np.concatenate([table["stats"], np.array(rows, dtype=stats.dtype).reshape(n, -1)], 0)
    table["league"] = np.concatenate([table["league"], np.zeros(n, table["league"].dtype)])
    bh, ph = np.array(bat_hand, np.float32), np.array(pit_hand, np.float32)
    table["bat_hand"] = np.concatenate([table["bat_hand"], bh])
    table["pit_hand"] = np.concatenate([table["pit_hand"], ph])
    hand = np.where(ph != 0.5, ph, bh)
    table["hand"] = np.concatenate([table["hand"], (hand >= 0.5).astype(table["hand"].dtype)])
    all_ids = np.concatenate([np.asarray(base["all_ids"], dtype=np.int64), np.array(new_ids, np.int64)])
    table.update(all_ids=all_ids, id_to_idx={int(p): i for i, p in enumerate(all_ids)},
                 unknown_index=len(all_ids))
    return table, n, names


def describe(names, ctx):
    out = [f"{len(names)} new players given priors: {sum(1 for t, _, _ in names if t == 'H')} hitters, "
           f"{sum(1 for t, _, _ in names if t == 'P')} pitchers, "
           f"{sum(1 for _, _, lv in names if 'NPB' in lv)} with NPB lines"]
    if ctx["people"]:
        out.append("NPB-sourced: " + ", ".join(ctx["people"].get(i, str(i))
                                               for _, i, lv in names if "NPB" in lv))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--r", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pitchers", action="store_true", help="also give new pitchers priors (needs --pitcher-rates checkpoint)")
    ap.add_argument("--npb", action="store_true", help="also use NPB lines (imports)")
    ap.add_argument("--no-hitters", action="store_true")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    T = args.season

    cfg = pickle.load(open(args.ckpt, "rb"))["pa_metadata"]["config"]
    if args.pitchers and not cfg.get("pitcher_rates"):
        raise SystemExit("--pitchers needs a checkpoint trained with --pitcher-rates")
    s = Sim(ckpt=args.ckpt, hook_model=True)
    if T <= s.train_end:
        raise SystemExit(f"--season {T} is inside training (train_end={s.train_end})")
    te = load_seasons([T], data_root=processed_root()).filter(pl.col("pa_terminal"))
    ctx = prepare_priors(T, npb=args.npb, pitchers=args.pitchers)
    table, n, names = append_priors(s.ptab, te, ctx, cfg, hitters=not args.no_hitters,
                                    pitchers=args.pitchers)
    install(s, table, dict(s.params), n)
    log = list(ctx["log"]) + describe(names, ctx)
    for line in log:
        print(line, flush=True)

    outcomes = real_runs(T)
    games = [g for g in s.real_games(T, limit=100000, pregame_staff=True)
             if g["park"] != 0 and int(g["game_pk"]) in outcomes]
    if args.limit:
        games = games[:args.limit]
    Hs, As = [], []
    for i in range(0, len(games), args.chunk):
        H, A = s.run(games[i:i + args.chunk], R=args.r, seed=0, skill_mode="mean", crn=False)
        Hs.append(H); As.append(A)
    sh, sa = np.concatenate(Hs, 0), np.concatenate(As, 0)
    pk = np.array([int(g["game_pk"]) for g in games])
    rh = np.array([outcomes[p][0] for p in pk], float)
    ra = np.array([outcomes[p][1] for p in pk], float)
    out = f"data/eval2/calib_{args.tag}_arrays.npz"
    np.savez(out, sim_home=sh, sim_away=sa, sim_total=sh + sa,
             real_home=rh, real_away=ra, real_total=rh + ra, game_pk=pk)
    Path(f"data/eval2/priors_{args.tag}.txt").write_text("\n".join(log) + "\n")
    print(f"saved -> {out}  ({len(pk)} games, sim mean total {(sh + sa).mean():.2f}, real {(rh + ra).mean():.2f})")


if __name__ == "__main__":
    main()
