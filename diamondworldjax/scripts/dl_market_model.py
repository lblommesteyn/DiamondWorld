"""A discriminative deep-learning model vs the closing line, incl. a CLV loss.

The generative simulator (v10-v12) loses to the moneyline market (backtest.py):
its edge is overconfidence, not alpha. This asks the question a different way with
a direct discriminative model on game-level features, and operationalizes the
"train against the closing line" idea two ways:

  - Model A (plain): P(home) = sigmoid(MLP(features)). Learns to predict the
    outcome from lineup/starter/park quality, ignoring the market.
  - Model B (market-anchored / CLV): logit(home) = market_logit + MLP(features).
    The net can only move the prediction OFF the closing line, so it is
    structurally trained to predict where the market is WRONG (the residual). If
    the market is efficient the residual is unlearnable out of sample and the net
    collapses to ~0; if there is a real inefficiency the net finds it.

Strict temporal split (train 2023, test 2024), evaluated by out-of-sample
log-loss vs the market and by an actual closing-line backtest. Features come from
2015-2022 player rates only (no leakage). Player rates are the same stale-stat
inputs the generative model uses, so this is a like-for-like discriminative test.

Usage:
  python -m diamondworldjax.scripts.dl_market_model
"""
from __future__ import annotations
import numpy as np
import polars as pl

from diamondworldjax.paths import processed_root
from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.sim.game_extract import extract_games
from diamondworldjax.scripts.train_pa import _build_player_table, _build_park_index
from diamondworldjax.scripts.simulate_games import TRAIN, TEST

HIT = ("1B", "2B", "3B", "HR")


def pitcher_allowed(train_pa, id_to_idx, P):
    """Per-pitcher allowed rates [hit,bb,k,hr] over training PAs -> (P,4)."""
    pcol = "pitcher_id" if "pitcher_id" in train_pa.columns else "pitcher_idx"
    df = train_pa.filter(pl.col("pa_outcome").is_not_null())
    g = df.group_by(pcol).agg([
        pl.col("pa_outcome").is_in(HIT).mean().alias("hit"),
        pl.col("pa_outcome").is_in(["BB", "HBP"]).mean().alias("bb"),
        (pl.col("pa_outcome") == "K").mean().alias("k"),
        (pl.col("pa_outcome") == "HR").mean().alias("hr"),
        pl.len().alias("n"),
    ])
    out = np.full((P, 4), np.nan)
    for r in g.iter_rows(named=True):
        i = id_to_idx.get(int(r[pcol]))
        if i is not None and r["n"] >= 50:
            out[i] = [r["hit"], r["bb"], r["k"], r["hr"]]
    # fill unknowns with league mean
    mean = np.nanmean(out, axis=0)
    out[np.isnan(out).any(1)] = mean
    return out


def build_features():
    train_pitches = load_seasons(TRAIN, data_root=processed_root())
    ptab = _build_player_table(train_pitches)
    park_map = _build_park_index(train_pitches)
    id_to_idx = ptab["id_to_idx"]
    P = len(ptab["hand"])
    bat = ptab["stats"][:, :4]            # [hit,bb,k,hr] per batter (as batter)
    pit = pitcher_allowed(train_pitches.filter(pl.col("pa_terminal")), id_to_idx, P)

    test_pa = load_seasons(TEST, data_root=processed_root()).filter(pl.col("pa_terminal"))
    # park run factor: avg runs/game per park (training), as one scalar feature
    games = extract_games(test_pa, id_to_idx, park_map=park_map)

    rows, pks = [], []
    for g in games:
        hl, al = np.array(g["home_lineup"]), np.array(g["away_lineup"])
        hsp = g["home_staff"][0] if g["home_staff"] else 0   # home starter (faces away)
        asp = g["away_staff"][0] if g["away_staff"] else 0    # away starter (faces home)
        feat = np.concatenate([
            bat[hl].mean(0), bat[al].mean(0),      # home & away lineup batting
            pit[asp], pit[hsp],                    # pitcher home-bats-vs, away-bats-vs
        ])
        rows.append(feat)
        pks.append(g["game_pk"])
    return np.array(rows), np.array(pks)


def devig(mlh, mla):
    def a2p(a):
        a = np.where(np.abs(a) < 100, np.nan, a)
        return np.where(a < 0, -a / (-a + 100), 100 / (a + 100))
    ph, pa = a2p(mlh), a2p(mla)
    return ph / (ph + pa)


def a2dec(a):
    a = np.where(np.abs(a) < 100, np.nan, a)
    return np.where(a < 0, 1 + 100 / (-a), 1 + a / 100)


def train_mlp(X, y, Xval, yval, market_logit=None, ml_val=None, linear=False,
              hidden=16, steps=4000, lr=1e-3, l2=3e-2, seed=0):
    """Train with heavy L2 + early stopping on a held-out val set (2023 tail), so
    the market test is fair rather than an overfit blowup. linear=True -> logistic
    regression (no hidden layers)."""
    import jax, jax.numpy as jnp
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    def init(m, n): return rng.standard_normal((m, n)) * np.sqrt(2 / m)
    if linear:
        P = {"W3": init(d, 1)[:, :] * 0.0, "b3": np.zeros(1)}
    else:
        P = {"W1": init(d, hidden), "b1": np.zeros(hidden),
             "W3": init(hidden, 1) * 0.01, "b3": np.zeros(1)}
    P = {k: jnp.array(v) for k, v in P.items()}
    Xj, yj = jnp.array(X), jnp.array(y)
    mlj = jnp.array(market_logit) if market_logit is not None else None

    def fwd(P, X):
        if linear:
            return (X @ P["W3"] + P["b3"])[:, 0]
        h = jnp.tanh(X @ P["W1"] + P["b1"])
        return (h @ P["W3"] + P["b3"])[:, 0]

    def loss(P, X, y, ml):
        logit = fwd(P, X)
        if ml is not None:
            logit = ml + logit          # market-anchored: net predicts the residual
        p = jax.nn.sigmoid(logit)
        bce = -jnp.mean(y * jnp.log(p + 1e-7) + (1 - y) * jnp.log(1 - p + 1e-7))
        reg = l2 * sum(jnp.sum(w ** 2) for k, w in P.items() if k.startswith("W"))
        return bce + reg

    def predict_p(P, Xn, mln):
        logit = np.array(fwd(P, jnp.array(Xn)))
        if mln is not None:
            logit = mln + logit
        return 1 / (1 + np.exp(-logit))

    def vll(P):
        p = np.clip(predict_p(P, Xval, ml_val), 1e-6, 1 - 1e-6)
        return float(-np.mean(yval * np.log(p) + (1 - yval) * np.log(1 - p)))

    import optax
    opt = optax.adam(lr); st = opt.init(P)
    gl = jax.jit(jax.grad(loss))
    best, bestP, patience, wait = 1e9, P, 400, 0
    for i in range(steps):
        g = gl(P, Xj, yj, mlj)
        upd, st = opt.update(g, st); P = optax.apply_updates(P, upd)
        if i % 50 == 0:
            v = vll(P)
            if v < best - 1e-5:
                best, bestP, wait = v, P, 0
            else:
                wait += 50
                if wait >= patience:
                    break
    return lambda Xn, mln=None: predict_p(bestP, Xn, mln)


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def backtest(p_model, p_mkt, dec_home, dec_away, y, edges=(0.0, 0.02, 0.04, 0.06, 0.10)):
    out = []
    for e in edges:
        # bet home if model prob exceeds market by e, else away symmetrically
        bh = (p_model - p_mkt) > e
        ba = ((1 - p_model) - (1 - p_mkt)) > e
        prof = 0.0; n = 0; wins = 0
        prof += np.where(y[bh] == 1, dec_home[bh] - 1, -1).sum(); n += bh.sum(); wins += y[bh].sum()
        prof += np.where(y[ba] == 0, dec_away[ba] - 1, -1).sum(); n += ba.sum(); wins += (1 - y[ba]).sum()
        roi = prof / n if n else 0.0
        out.append((e, int(n), roi * 100, wins / n * 100 if n else 0))
    return out


def main():
    print("Building features...", flush=True)
    X, pks = build_features()
    # outcomes from the saved sim arrays (ground-truth real scores)
    arr = np.load("data/eval2/calib_v12_bt_arrays.npz")
    y_map = {int(pk): int(h > a) for pk, h, a in
             zip(arr["game_pk"], arr["real_home"], arr["real_away"]) if h != a}
    # odds
    import csv
    odds = {int(r["game_pk"]): r for r in csv.DictReader(open("data/eval2/odds_2023_2024.csv"))}
    # year map from schedules
    import json
    yr = {}
    for s in (2023, 2024):
        d = json.load(open(f"/tmp/sched_{s}.json"))
        for day in d["dates"]:
            for g in day["games"]:
                yr[g["gamePk"]] = s

    keep = [i for i, pk in enumerate(pks)
            if int(pk) in y_map and int(pk) in odds and int(pk) in yr]
    X = X[keep]; pk = pks[keep]
    y = np.array([y_map[int(p)] for p in pk], dtype=float)
    mlh = np.array([float(odds[int(p)]["ml_home"]) for p in pk])
    mla = np.array([float(odds[int(p)]["ml_away"]) for p in pk])
    year = np.array([yr[int(p)] for p in pk])
    p_mkt = devig(mlh, mla)
    dec_h, dec_a = a2dec(mlh), a2dec(mla)
    ok = ~np.isnan(p_mkt) & ~np.isnan(dec_h) & ~np.isnan(dec_a)
    X, y, p_mkt, dec_h, dec_a, year, mlh, mla = [v[ok] for v in (X, y, p_mkt, dec_h, dec_a, year, mlh, mla)]

    # standardize features on train; split 2023 into fit (first 80%) + val (last 20%)
    tr, te = year == 2023, year == 2024
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xs = (X - mu) / sd
    mkt_logit = np.log(np.clip(p_mkt, 1e-4, 1 - 1e-4) / np.clip(1 - p_mkt, 1e-4, 1 - 1e-4))
    tr_idx = np.where(tr)[0]
    cut = int(len(tr_idx) * 0.8)
    fit_i, val_i = tr_idx[:cut], tr_idx[cut:]

    L = [f"DL MARKET MODEL  fit=2023[:80%] (n={len(fit_i)})  val=2023[80%:] (n={len(val_i)})  "
         f"test=2024 (n={te.sum()})", ""]
    L.append(f"  test base rate home-win={y[te].mean():.3f}  market mean P(home)={p_mkt[te].mean():.3f}")
    L.append("  models: heavy L2 + early stopping on val (fair test, not an overfit blowup)")
    L.append("")
    L.append("OUT-OF-SAMPLE LOG-LOSS (2024, lower=better)")
    L.append(f"  market (closing line)      {logloss(p_mkt[te], y[te]):.4f}")

    preds = {}
    print("Training logistic (features)...", flush=True)
    fL = train_mlp(Xs[fit_i], y[fit_i], Xs[val_i], y[val_i], linear=True)
    preds["Logistic"] = fL(Xs[te])
    L.append(f"  Logistic(features)         {logloss(preds['Logistic'], y[te]):.4f}")

    print("Training Model A (MLP features)...", flush=True)
    fA = train_mlp(Xs[fit_i], y[fit_i], Xs[val_i], y[val_i])
    preds["Model A (MLP)"] = fA(Xs[te])
    L.append(f"  Model A: MLP(features)     {logloss(preds['Model A (MLP)'], y[te]):.4f}")

    print("Training Model B (market-anchored / CLV)...", flush=True)
    fB = train_mlp(Xs[fit_i], y[fit_i], Xs[val_i], y[val_i],
                   market_logit=mkt_logit[fit_i], ml_val=mkt_logit[val_i])
    preds["Model B (CLV)"] = fB(Xs[te], mkt_logit[te])
    L.append(f"  Model B: market + MLP      {logloss(preds['Model B (CLV)'], y[te]):.4f}")
    L.append(f"    Model B residual: mean|pB-market|={np.mean(np.abs(preds['Model B (CLV)']-p_mkt[te])):.4f} "
             f"(~0 => market efficient, nothing learnable)")
    L.append("")

    for name, p in preds.items():
        L.append(f"CLOSING-LINE BACKTEST 2024 - {name}")
        L.append("  edge   bets   ROI     hit")
        for e, n, roi, hit in backtest(p, p_mkt[te], dec_h[te], dec_a[te], y[te]):
            L.append(f"  {e:.02f}  {n:5d}  {roi:+6.1f}%  {hit:4.1f}%")
        L.append("")

    rep = "\n".join(L)
    print(rep)
    open("data/eval2/dl_market_model.txt", "w").write(rep)
    print("saved -> data/eval2/dl_market_model.txt")


if __name__ == "__main__":
    main()
