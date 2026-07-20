"""Strikeout-prop signal test (free data): does the model beat a baseline line?

Real historical prop odds are paid-only, so a true prop backtest is out of reach.
But the question that decides whether props are worth paying for is answerable with
free data: does the model predict a starting pitcher's strikeouts BETTER than the
naive baseline a book's line sits near? If it cannot beat a simple baseline out of
sample, it will not beat a sharp book.

For each 2024 start, over the PAs the starter actually faced:
  actual_K    = real strikeouts
  model_EK    = sum of the model's calibrated P(K) per PA (v12, conditioned + recal)
  baseline_EK = sum of the pitcher's 2015-2022 K-rate (matchup-blind)
The "line" is baseline_EK rounded to the nearest 0.5 (books set K-lines near the
pitcher's expectation). We bet over/under by whether model_EK beats the line, settle
vs actual_K at -110, and also compare how well model vs baseline rank actual K.

Caveat, stated plainly: real book lines are opponent-adjusted and sharper than this
baseline, and using the pitcher's actual batters-faced leaks game length equally into
both predictors. Beating this baseline is NECESSARY, not sufficient, for a real edge.

Usage: python -m diamondworldjax.scripts.strikeout_props
"""
from __future__ import annotations
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
TRAIN = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]


def main():
    import jax, jax.numpy as jnp, numpyro.handlers as nh, pickle
    V12 = "checkpoints/dwjax_pa_v12/dwjax_step_0050000.pkl"
    params = pickle.load(open(V12, "rb"))["params"]
    b_heur = np.load("data/eval2/v12_cal_params.npz")["b_heur"].astype(np.float64)

    train = load_seasons(TRAIN, data_root=processed_root())
    ptab = _build_player_table(train, recency_halflife=2.0)
    park_map = _build_park_index(train)
    id2i = ptab["id_to_idx"]
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
          "bat_hand": jnp.array(ptab.get("bat_hand", np.full(len(ptab["hand"]), .5, np.float32))),
          "pit_hand": jnp.array(ptab.get("pit_hand", np.full(len(ptab["hand"]), .5, np.float32)))}
    model_fn = partial(pa_model, outcome_only=True, fatigue=True)

    # forward pass over all games; extract calibrated P(K) per PA, keep starter PAs
    gids = test["game_pk"].unique().to_numpy()
    recs = {}  # (game_pk, half_bin, sp) -> [actual_K, model_EK, BF]
    def _map(b):
        for k in ("pitcher_ids", "batter_ids"):
            b[k] = np.vectorize(lambda x: id2i.get(int(x), 0))(np.array(b[k]))
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
    actual, mek, bek, bf = [], [], [], []
    for (g, hb, sp), (aK, mK, n) in recs.items():
        if n < 12:      # require a real start (>=12 batters faced)
            continue
        kr_p = krate.get(sp, league_kr)
        actual.append(aK); mek.append(mK); bek.append(kr_p * n); bf.append(n)
    actual = np.array(actual, float); mek = np.array(mek); bek = np.array(bek); bf = np.array(bf, float)
    n = len(actual)

    def corr(a, b): return float(np.corrcoef(a, b)[0, 1])
    L = [f"STRIKEOUT-PROP SIGNAL TEST (2024, {n} starts >=12 BF)", ""]
    L.append(f"  actual K: mean {actual.mean():.2f}  | model_EK mean {mek.mean():.2f}  | baseline_EK mean {bek.mean():.2f}")
    L.append("")
    L.append("Predicting actual K (higher corr / lower MAE = better):")
    L.append(f"  baseline (pitcher season K-rate): corr {corr(bek,actual):.3f}  MAE {np.abs(bek-actual).mean():.3f}")
    L.append(f"  model    (v12 matchup, calibrated): corr {corr(mek,actual):.3f}  MAE {np.abs(mek-actual).mean():.3f}")
    L.append("")

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
