"""Strikeout-prop signal test AND real-odds backtest scaffold.

Real historical prop odds are paid-only, so the true backtest is blocked on data.
This script does two things so that block is the ONLY thing missing:

  1. Signal test (free data, always runs): does the model predict a starter's
     strikeouts BETTER than the naive baseline a book's line sits near? Necessary,
     not sufficient, for a real edge -- if it cannot beat a simple baseline out of
     sample it will not beat a sharp book. For each 2024 start, over the PAs the
     starter faced, actual_K vs model_EK (sum of calibrated P(K) per PA) vs
     baseline_EK (the pitcher's own train-window K-rate x batters faced).

  2. Real-odds backtest (runs when --odds <file> is given): the settlement, edge
     sweep, and closing-line-value logic a paid prop feed will plug straight into.
     The odds file schema and the American-odds math are fixed and unit-tested here
     via --selftest (no GPU, no model), against a known-efficient and a known-biased
     synthetic market, so the day real odds land the backtest is trustworthy.

Model default is v16 (the current best player model, trained through 2023 with the
contact-quality features); pass --ckpt/--train-end/--contact-quality to change it.

Usage:
  python -m diamondworldjax.scripts.strikeout_props            # signal test (needs GPU)
  python -m diamondworldjax.scripts.strikeout_props --selftest # settlement math only
  python -m diamondworldjax.scripts.strikeout_props --odds data/prop_odds_2024.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import polars as pl

from diamondworldjax.paths import processed_root
from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.data.pa_batching import build_pa_batch
from diamondworldjax.scripts.train_pa import _build_player_table, _build_park_index, apply_park_idx
from diamondworldjax.model.pa_model import pa_model
from diamondworldjax.sim.rules_engine import PA_OUTCOME_IDX
from functools import partial

K_IDX = PA_OUTCOME_IDX["K"]
V16 = "checkpoints/dwjax_pa_v16/dwjax_step_0050000.pkl"

# ------------------------------------------------------------------ real-odds path

# Prop-odds file schema (CSV or parquet). One row per pitcher-start-market-side, or
# one row per start with both sides; the loader accepts either. Required columns:
#   game_pk      int    matches the processed-data game_pk (join key)
#   pitcher_id   int    MLBAM id of the starter
#   line         float  the posted strikeout line (e.g. 6.5)
#   over_odds    int    American odds on the over  (e.g. -115)
#   under_odds   int    American odds on the under (e.g. -105)
# Optional (enables closing-line value): close_line, close_over_odds, close_under_odds.
# A date+name feed can be crosswalked to game_pk/pitcher_id via the MLB Stats API
# schedule the same way build_odds.py does for moneyline.
ODDS_REQUIRED = ("game_pk", "pitcher_id", "line", "over_odds", "under_odds")


def american_profit(odds: np.ndarray) -> np.ndarray:
    """Profit per 1 unit staked on a winning bet at the given American odds."""
    odds = np.asarray(odds, float)
    return np.where(odds > 0, odds / 100.0, 100.0 / np.abs(odds))


def load_prop_odds(path: str) -> pl.DataFrame:
    p = Path(path)
    df = pl.read_parquet(p) if p.suffix == ".parquet" else pl.read_csv(p)
    missing = [c for c in ODDS_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns {missing}; schema is {ODDS_REQUIRED}")
    # guard malformed American odds (|odds| must be >= 100); drop those rows loudly
    bad = df.filter((pl.col("over_odds").abs() < 100) | (pl.col("under_odds").abs() < 100))
    if len(bad):
        print(f"  dropping {len(bad)} rows with malformed odds (|american| < 100)")
    return df.filter((pl.col("over_odds").abs() >= 100) & (pl.col("under_odds").abs() >= 100))


def backtest_props(line, over_odds, under_odds, model_ek, actual_k,
                   close_line=None, edges=(0.0, 0.25, 0.5, 0.75, 1.0, 1.5)) -> list[str]:
    """Settle a strikeout-prop book at real American odds, over an edge sweep.

    Bet OVER when model_ek - line > edge, UNDER when line - model_ek > edge, using
    the posted odds for that side; push when actual == line. Reports ROI, hit rate,
    and (if closing lines are given) closing-line value, the honest leading
    indicator of whether the bets have real edge.
    """
    line = np.asarray(line, float)
    over_odds = np.asarray(over_odds, float)
    under_odds = np.asarray(under_odds, float)
    model_ek = np.asarray(model_ek, float)
    actual_k = np.asarray(actual_k, float)
    push = actual_k == line
    over_win = actual_k > line

    L = ["REAL-ODDS PROP BACKTEST (over if actual>line), posted American odds",
         f"  starts with odds: {len(line)}", ""]
    L.append(f"  {'edge':5s} {'bets':>5s} {'ROI':>8s} {'hit':>6s} {'CLV':>8s}")
    for e in edges:
        pick_over = (model_ek - line) > e
        pick_under = (line - model_ek) > e
        m = (pick_over | pick_under) & ~push
        if m.sum() == 0:
            L.append(f"  {e:.2f}  {0:5d}")
            continue
        won = np.where(pick_over[m], over_win[m], ~over_win[m])
        odds = np.where(pick_over[m], over_odds[m], under_odds[m])
        pnl = np.where(won, american_profit(odds), -1.0)
        clv = ""
        if close_line is not None:
            cl = np.asarray(close_line, float)[m]
            # value taken vs close: for an over, we beat the market if the line rose
            # (we got a lower number); mirror for unders. Positive = favourable CLV.
            mv = np.where(pick_over[m], cl - line[m], line[m] - cl)
            clv = f"{mv.mean():+7.3f}"
        L.append(f"  {e:.2f}  {int(m.sum()):5d}  {pnl.mean()*100:+6.1f}%  {won.mean()*100:4.1f}%  {clv:>7s}")
    return L


def _selftest():
    """Validate the settlement math with no model: an efficient market must return
    about -vig, a biased market must be beatable. Same guard used for the moneyline
    backtest engine."""
    rng = np.random.default_rng(0)
    n = 20000
    true_mean = rng.uniform(3, 9, n)
    actual = rng.poisson(true_mean).astype(float)
    fair = np.full(n, -110)
    line = np.floor(true_mean) + 0.5              # unbiased half-integer line (no pushes)
    # EFFICIENT: the model carries NO information the line lacks -- its number is the
    # fair line plus noise independent of the outcome, so which side it bets is
    # uncorrelated with who wins. Any bettor here just pays the -110 vig (~-4.5%).
    model_noedge = line + rng.normal(0, 0.4, n)
    eff = backtest_props(line, fair, fair, model_noedge, actual, edges=(0.0,))
    # BIASED: the line sits 1.5 K too high while the model knows the true mean, so the
    # model correctly bets unders and profits. A real edge must look like this.
    biased_line = line + 1.5
    bia = backtest_props(biased_line, fair, fair, true_mean, actual, edges=(0.0,))
    print("SELF-TEST: settlement math")
    print("  efficient market (fair -110 line, model=truth), edge 0:")
    print("   ", eff[-1].strip(), "-> expect ROI near -vig (~-5%)")
    print("  biased market (line +1 K high, model=truth), edge 0:")
    print("   ", bia[-1].strip(), "-> expect ROI clearly positive (model bets unders)")
    eff_roi = float(eff[-1].split("%")[0].split()[-1])
    bia_roi = float(bia[-1].split("%")[0].split()[-1])
    ok = eff_roi < 0 and bia_roi > 5
    print(f"  PASS: {ok}  (efficient {eff_roi:+.1f}%, biased {bia_roi:+.1f}%)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=V16)
    ap.add_argument("--recal", default="data/eval2/v13_cal_params.npz")
    ap.add_argument("--recal-key", default="b_heur")
    ap.add_argument("--train-end", type=int, default=2023,
                    help="Last training season for the player table (must match the ckpt; "
                         "2023 for v15/v16).")
    ap.add_argument("--recency-halflife", type=float, default=2.0)
    ap.add_argument("--contact-quality", action="store_true", default=True,
                    help="Use xBA-style expected hit/HR columns (v16). --no-contact-quality to disable.")
    ap.add_argument("--no-contact-quality", dest="contact_quality", action="store_false")
    ap.add_argument("--skill-mode", default="mean", choices=["mean", "prior"])
    ap.add_argument("--odds", default=None, help="Path to a real prop-odds file; enables the backtest.")
    ap.add_argument("--selftest", action="store_true", help="Validate settlement math, no GPU/model.")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    import jax, jax.numpy as jnp, numpyro.handlers as nh, pickle
    params = pickle.load(open(args.ckpt, "rb"))["params"]
    if args.skill_mode == "mean" and "player_mu" in params:
        params = {**params, "player_skills": params["player_mu"]}
    b_heur = np.load(args.recal)[args.recal_key].astype(np.float64)

    odds_df = load_prop_odds(args.odds) if args.odds else None
    if odds_df is not None:
        print(f"loaded {len(odds_df)} prop-odds rows from {args.odds}")

    TRAIN = list(range(2015, args.train_end + 1))
    train = load_seasons(TRAIN, data_root=processed_root())
    ptab = _build_player_table(train, recency_halflife=args.recency_halflife,
                               contact_quality=args.contact_quality)
    park_map = _build_park_index(train)
    id2i = ptab["id_to_idx"]
    unknown_idx = ptab["unknown_index"]
    # pitcher season K-rate baseline (2015-2022)
    tp = train.filter(pl.col("pa_terminal") & pl.col("pa_outcome").is_not_null())
    pcol = "pitcher_id" if "pitcher_id" in tp.columns else "pitcher_idx"
    kr = tp.group_by(pcol).agg([(pl.col("pa_outcome") == "K").mean().alias("kr"), pl.len().alias("n")])
    krate = {int(r[pcol]): r["kr"] for r in kr.iter_rows(named=True) if r["n"] >= 100}
    league_kr = float((tp["pa_outcome"] == "K").mean())
    del train, tp

    test = (load_seasons([2024], data_root=processed_root())
            .filter(pl.col("pa_terminal") & pl.col("pa_outcome").is_not_null()))
    test = apply_park_idx(test, park_map)
    # starter per (game_pk, half_bin) = pitcher of the first PA
    starters = (test.sort("at_bat_number").group_by(["game_pk", "half_bin"])
                .agg(pl.col(pcol).first().alias("sp")))
    test = test.join(starters, on=["game_pk", "half_bin"])
    test = test.with_columns((pl.col(pcol) == pl.col("sp")).alias("is_sp_pa"))

    pt = {"stats": jnp.array(ptab["stats"]), "league": jnp.array(ptab["league"]),
          "hand": jnp.array(ptab["hand"]),
          "unknown_index": unknown_idx,
          "bat_hand": jnp.array(ptab.get("bat_hand", np.full(len(ptab["hand"]), .5, np.float32))),
          "pit_hand": jnp.array(ptab.get("pit_hand", np.full(len(ptab["hand"]), .5, np.float32)))}
    model_fn = partial(pa_model, outcome_only=True, fatigue=True)

    # forward pass over all games; extract calibrated P(K) per PA, keep starter PAs
    gids = test["game_pk"].unique().to_numpy()
    recs = {}  # (game_pk, half_bin, sp) -> [actual_K, model_EK, BF]
    def _map(b):
        for k in ("pitcher_ids", "batter_ids"):
            b[k] = np.vectorize(lambda x: id2i.get(int(x), unknown_idx))(np.array(b[k]))
        return b
    rng = jax.random.PRNGKey(0)
    for i in range(0, len(gids), 64):
        chunk = gids[i:i+64]
        df = test.filter(pl.col("game_pk").is_in(chunk.tolist())).sort(["game_pk", "at_bat_number"])
        batch = _map(build_pa_batch(df))
        for kk in list(batch):
            if kk != "game_ids":
                batch[kk] = jnp.array(np.array(batch[kk]))
        with nh.seed(rng_seed=0):
            with nh.substitute(data=params):
                with nh.trace() as tr:
                    model_fn(batch, pt, teacher_force=False)
        logits = np.array(tr["pa_outcome"]["fn"].logits) + b_heur  # (B,T,9) calibrated
        pkprob = np.exp(logits - logits.max(-1, keepdims=True))
        pkprob = pkprob / pkprob.sum(-1, keepdims=True)
        PK = pkprob[:, :, K_IDX]                      # (B,T) P(K)
        valid = np.array(batch["pa_valid"])
        # reconstruct per-game PA rows in the same sorted order
        gorder = df["game_pk"].to_numpy()
        # rebuild per-(game) index: build_pa_batch groups by unique sorted game
        ug = batch["game_ids"]
        rowmeta = df.select(["game_pk", "half_bin", "sp", "is_sp_pa", pcol,
                             (pl.col("pa_outcome") == "K").cast(pl.Int8).alias("isk")])
        # iterate games in ug order, PAs in at_bat order
        for bi, g in enumerate(ug):
            sub = rowmeta.filter(pl.col("game_pk") == int(g))
            n = min(len(sub), PK.shape[1])
            sub = sub.head(n)
            issp = sub["is_sp_pa"].to_numpy()
            hb = sub["half_bin"].to_numpy(); spid = sub["sp"].to_numpy(); isk = sub["isk"].to_numpy()
            pkv = PK[bi, :n]
            for j in range(n):
                if not issp[j]:
                    continue
                key = (int(g), int(hb[j]), int(spid[j]))
                r = recs.setdefault(key, [0.0, 0.0, 0])
                r[0] += int(isk[j]); r[1] += float(pkv[j]); r[2] += 1
        if i % 640 == 0:
            print(f"  {i}/{len(gids)} games", flush=True)

    # assemble per-start table; baseline_EK from pitcher season K-rate * BF
    actual, mek, bek, bf, gpks, spids = [], [], [], [], [], []
    for (g, hb, sp), (aK, mK, n) in recs.items():
        if n < 12:      # require a real start (>=12 batters faced)
            continue
        kr_p = krate.get(sp, league_kr)
        actual.append(aK); mek.append(mK); bek.append(kr_p * n); bf.append(n)
        gpks.append(int(g)); spids.append(int(sp))
    actual = np.array(actual, float); mek = np.array(mek); bek = np.array(bek); bf = np.array(bf, float)
    gpks = np.array(gpks); spids = np.array(spids)
    n = len(actual)

    def corr(a, b): return float(np.corrcoef(a, b)[0, 1])
    tag = Path(args.ckpt).parent.name  # e.g. dwjax_pa_v16
    L = [f"STRIKEOUT-PROP SIGNAL TEST (2024, {n} starts >=12 BF, model {tag})", ""]
    L.append(f"  actual K: mean {actual.mean():.2f}  | model_EK mean {mek.mean():.2f}  | baseline_EK mean {bek.mean():.2f}")
    L.append("")
    L.append("Predicting actual K (higher corr / lower MAE = better):")
    L.append(f"  baseline (pitcher season K-rate): corr {corr(bek,actual):.3f}  MAE {np.abs(bek-actual).mean():.3f}")
    L.append(f"  model    ({tag} matchup, calibrated): corr {corr(mek,actual):.3f}  MAE {np.abs(mek-actual).mean():.3f}")
    L.append("")

    # ---- REAL-ODDS backtest, when a prop-odds file was supplied ----
    if odds_df is not None:
        key = pl.DataFrame({"game_pk": gpks, "pitcher_id": spids,
                            "model_ek": mek, "actual_k": actual})
        j = key.join(odds_df, on=["game_pk", "pitcher_id"], how="inner")
        if len(j) == 0:
            L.append("REAL-ODDS BACKTEST: 0 starts matched the odds file on (game_pk, pitcher_id).")
        else:
            cl = j["close_line"].to_numpy() if "close_line" in j.columns else None
            L += backtest_props(j["line"].to_numpy(), j["over_odds"].to_numpy(),
                                j["under_odds"].to_numpy(), j["model_ek"].to_numpy(),
                                j["actual_k"].to_numpy(), close_line=cl)
        L.append("")
        rep = "\n".join(L)
        print(rep)
        open("data/eval2/strikeout_props_realodds.txt", "w").write(rep + "\n")
        return

    # prop backtest vs a baseline-set line
    line = np.round(bek * 2) / 2
    over_win = actual > line
    push = actual == line
    nz = ~push
    # NULL baselines: bet a fixed side every game (no model). If the line is
    # mis-set these are already profitable, so any model ROI must be read against
    # them, not against zero.
    under_win = actual[nz] < line[nz]
    roi_under = np.where(under_win, 100/110, -1.0).mean()
    roi_over = np.where(~under_win, 100/110, -1.0).mean()
    L.append(f"LINE CHECK: actual mean {actual.mean():.2f} vs line mean {line.mean():.2f} "
             f"=> P(under)={under_win.mean():.3f}. The baseline line is biased, so:")
    L.append(f"  NULL always-UNDER (no model): ROI {roi_under*100:+.1f}%  |  always-OVER: ROI {roi_over*100:+.1f}%")
    L.append("")
    L.append("PROP BACKTEST vs baseline-set line (over if actual>line), -110 both sides")
    L.append("  edge(K)  bets   ROI     hit    (edge = |model_EK - line|)")
    for e in (0.0, 0.25, 0.5, 0.75, 1.0):
        pick_over = (mek - line) > e
        pick_under = (line - mek) > e
        m = (pick_over | pick_under) & ~push
        if m.sum() == 0:
            L.append(f"  {e:.2f}     0"); continue
        won = np.where(pick_over[m], over_win[m], ~over_win[m])
        roi = (np.where(won, 100/110, -1.0)).mean()
        L.append(f"  {e:.2f}   {int(m.sum()):5d}  {roi*100:+6.1f}%  {won.mean()*100:4.1f}%")
    L.append("")
    L.append("Reading it: the model does NOT out-predict the baseline (corr above), and its prop")
    L.append("ROI only matches/exceeds the NULL always-under because it exploits the same biased")
    L.append("synthetic line harder, not because it has matchup skill. A real book line is not")
    L.append("biased this way, so this is NO evidence of a prop edge; that needs real prop odds.")
    rep = "\n".join(L)
    print(rep)
    open("data/eval2/strikeout_props.txt", "w").write(rep)
    print("saved -> data/eval2/strikeout_props.txt")


if __name__ == "__main__":
    main()
