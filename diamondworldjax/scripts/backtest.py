"""Backtest harness: does the model's edge survive a real market?

Matching real run distributions (RESULTS.md) is necessary but NOT sufficient for
a betting edge: the market already prices lineups, weather, and starters. The
only real test is ROI against actual closing lines. This harness:

  1. Turns the per-matchup predictive ensemble (from calib_audit's *_arrays.npz)
     into model probabilities for each market: moneyline P(home) and totals
     over/under at the posted line.
  2. Reads an odds CSV of CLOSING prices, converts American odds to implied
     probabilities, and removes the vig (two-way overround normalization) to get
     the market's fair probability p_mkt.
  3. Bets when the model edge (p_model - p_mkt) clears a threshold, stakes flat
     or fractional-Kelly, settles against the real outcome, and reports ROI,
     hit rate, and per-market breakdown.

Betting the CLOSING line is the strict test: sharp closing lines are the
efficient benchmark; beating them is the gold standard.

Odds CSV schema (one row per game; missing markets left blank):
  game_pk,ml_home,ml_away,total_line,over_odds,under_odds
  American odds (e.g. -145, +130). total_line e.g. 8.5.

Validation: with no real odds, --synthetic {efficient,noisy} runs a self-contained
unit test on a KNOWN generative process (set true probs, sample outcomes, price
the market at true prob + vig). An honest engine returns ROI ~ -vig against an
efficient market and turns profitable only when the bettor is genuinely sharper
than a noisy market. This checks the accounting, not the real model.

Usage:
  # real backtest (needs odds.csv of closing lines)
  python -m diamondworldjax.scripts.backtest --arrays data/eval2/calib_v10_arrays.npz \
    --odds data/odds_2023_2024.csv --market both --edge 0.03 --kelly 0.25
  # engine self-test, no external data
  python -m diamondworldjax.scripts.backtest --synthetic efficient
  python -m diamondworldjax.scripts.backtest --synthetic noisy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# ---------- odds helpers ----------
def american_to_prob(a: np.ndarray) -> np.ndarray:
    """American odds -> implied probability (with vig)."""
    a = np.asarray(a, dtype=float)
    p = np.where(a < 0, (-a) / ((-a) + 100.0), 100.0 / (a + 100.0))
    return p


def american_to_decimal(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return np.where(a < 0, 1.0 + 100.0 / (-a), 1.0 + a / 100.0)


def devig_two_way(p1: np.ndarray, p2: np.ndarray):
    """Normalize a two-way market to sum to 1 (proportional / multiplicative)."""
    s = p1 + p2
    return p1 / s, p2 / s


def kelly_fraction(p: np.ndarray, dec_odds: np.ndarray) -> np.ndarray:
    """Full-Kelly stake fraction for prob p at decimal odds. b = dec-1."""
    b = dec_odds - 1.0
    f = (p * b - (1 - p)) / b
    return np.clip(f, 0.0, 1.0)


def settle(stake, dec_odds, won, unit=1.0):
    """Return profit in units for stakes (fraction of unit) at dec_odds given win mask."""
    profit = np.where(won, stake * (dec_odds - 1.0), -stake)
    return profit * unit


# ---------- reporting ----------
def summarize(name, p_model, p_mkt, dec, won, mask, edge_thresh, kelly, out_lines):
    bet = mask & (p_model - p_mkt > edge_thresh)
    n = int(bet.sum())
    out_lines.append(f"--- {name} ---")
    if n == 0:
        out_lines.append(f"  no bets clear edge>{edge_thresh:.3f}")
        return 0.0, 0
    if kelly > 0:
        stake = kelly * kelly_fraction(p_model[bet], dec[bet])
    else:
        stake = np.full(n, 1.0)  # flat 1 unit
    profit = settle(stake, dec[bet], won[bet])
    staked = stake.sum()
    roi = profit.sum() / staked if staked > 0 else 0.0
    hit = won[bet].mean()
    avg_edge = (p_model[bet] - p_mkt[bet]).mean()
    out_lines.append(f"  bets={n}  staked={staked:.1f}u  profit={profit.sum():+.1f}u  "
                     f"ROI={roi*100:+.1f}%  hit={hit*100:.1f}%  avg_edge={avg_edge*100:+.1f}%")
    return profit.sum(), n


# ---------- real backtest ----------
def run_real(args):
    arr = np.load(args.arrays, allow_pickle=True)
    sim_total = arr["sim_total"]; sim_home = arr["sim_home"]; sim_away = arr["sim_away"]
    real_total = arr["real_total"]; real_home = arr["real_home"]; real_away = arr["real_away"]
    game_pk = arr["game_pk"]
    pk_to_i = {int(pk): i for i, pk in enumerate(game_pk)}

    import csv
    odds = {}
    with open(args.odds) as f:
        for row in csv.DictReader(f):
            try:
                pk = int(row["game_pk"])
            except (KeyError, ValueError):
                continue
            if pk in pk_to_i:
                odds[pk] = row
    L = [f"REAL BACKTEST  arrays={Path(args.arrays).name}  odds={Path(args.odds).name}",
         f"  games matched with odds: {len(odds)} / {len(game_pk)}",
         f"  edge threshold={args.edge}  staking={'Kelly x'+str(args.kelly) if args.kelly>0 else 'flat 1u'}", ""]
    idx = np.array([pk_to_i[pk] for pk in odds], dtype=int)
    rows = [odds[pk] for pk in odds]

    def col(key):
        out = np.full(len(rows), np.nan)
        for j, r in enumerate(rows):
            v = r.get(key, "")
            if v not in ("", None):
                out[j] = float(v)
        return out

    total_profit = 0.0
    if args.market in ("moneyline", "both"):
        mlh, mla = col("ml_home"), col("ml_away")
        have = ~np.isnan(mlh) & ~np.isnan(mla)
        ph_v, pa_v = american_to_prob(mlh), american_to_prob(mla)
        p_mkt_home, _ = devig_two_way(ph_v, pa_v)
        dec_home = american_to_decimal(mlh)
        dec_away = american_to_decimal(mla)
        p_model_home = (sim_home[idx] > sim_away[idx]).mean(axis=1)
        decided = real_home[idx] != real_away[idx]
        won_home = (real_home[idx] > real_away[idx])
        m = have & decided
        tp, _ = summarize("MONEYLINE home", p_model_home, p_mkt_home, dec_home, won_home, m, args.edge, args.kelly, L)
        total_profit += tp
        tp, _ = summarize("MONEYLINE away", 1 - p_model_home, 1 - p_mkt_home, dec_away, ~won_home, m, args.edge, args.kelly, L)
        total_profit += tp

    if args.market in ("totals", "both"):
        tl, oo, uo = col("total_line"), col("over_odds"), col("under_odds")
        have = ~np.isnan(tl) & ~np.isnan(oo) & ~np.isnan(uo)
        po_v, pu_v = american_to_prob(oo), american_to_prob(uo)
        p_mkt_over, _ = devig_two_way(po_v, pu_v)
        dec_over, dec_under = american_to_decimal(oo), american_to_decimal(uo)
        # model P(over) at each game's posted line; push handling: strict >
        p_model_over = np.array([(sim_total[i] > tl[j]).mean() if have[j] else np.nan
                                 for j, i in enumerate(idx)])
        push = real_total[idx] == tl
        won_over = real_total[idx] > tl
        m = have & ~push
        tp, _ = summarize("TOTALS over", p_model_over, p_mkt_over, dec_over, won_over, m, args.edge, args.kelly, L)
        total_profit += tp
        tp, _ = summarize("TOTALS under", 1 - p_model_over, 1 - p_mkt_over, dec_under, ~won_over, m, args.edge, args.kelly, L)
        total_profit += tp

    L.append("")
    L.append(f"TOTAL profit across selected markets: {total_profit:+.1f}u")
    report = "\n".join(L)
    print(report)
    if args.out:
        Path(args.out).write_text(report)
        print(f"saved -> {args.out}")


# ---------- synthetic engine self-test ----------
def run_synthetic(mode: str, n: int = 20000, vig: float = 0.045, seed: int = 0):
    """Validate the accounting. Generate true probs, sample outcomes, price the
    market at true prob (+vig). 'efficient': model == truth (should net ~ -vig on
    flat bets over all games; ~0 when only +EV bets are taken, since none exist).
    'noisy': market is truth + noise while model == truth => model is sharper and
    should net positive."""
    rng = np.random.default_rng(seed)
    p_true = rng.uniform(0.30, 0.70, n)
    y = (rng.random(n) < p_true).astype(float)

    if mode == "efficient":
        p_model = p_true.copy()
        p_mkt_fair = p_true.copy()
    elif mode == "noisy":
        p_model = p_true.copy()
        p_mkt_fair = np.clip(p_true + rng.normal(0, 0.05, n), 0.02, 0.98)
    else:
        raise SystemExit(f"unknown synthetic mode {mode}")

    # market posts vigged odds: two-way implied probs sum to 1+vig (full overround)
    p_yes = p_mkt_fair * (1 + vig)
    p_no = (1 - p_mkt_fair) * (1 + vig)
    dec_yes = 1.0 / p_yes
    # de-vig back out (what the bettor estimates as fair)
    p_mkt_devig, _ = devig_two_way(p_yes, p_no)

    L = [f"SYNTHETIC ENGINE SELF-TEST  mode={mode}  n={n}  vig={vig*100:.1f}%", ""]
    # baseline: bet every game (no edge filter) -> pays the vig on a fair market
    profit_all = settle(np.ones(n), dec_yes, y.astype(bool))
    L.append(f"  bet-ALL   : bets={n:5d}  ROI={profit_all.sum()/n*100:+.2f}%  "
             f"hit={y.mean()*100:.1f}%   (baseline: fair market pays ~ -vig)")
    for edge in (1e-6, 0.02, 0.04):
        bet = (p_model - p_mkt_devig) > edge
        tag = "edge>0" if edge < 1e-3 else f"edge>{edge:.02f}"
        if bet.sum() == 0:
            L.append(f"  {tag:10s}: no bets clear filter"); continue
        stake = np.ones(bet.sum())
        profit = settle(stake, dec_yes[bet], y[bet].astype(bool))
        roi = profit.sum() / stake.sum()
        L.append(f"  {tag:10s}: bets={int(bet.sum()):5d}  ROI={roi*100:+.2f}%  "
                 f"hit={y[bet].mean()*100:.1f}%")
    if mode == "efficient":
        L.append("  EXPECT: bet-ALL ~ -vig; edge-filtered ~0 bets (no real +EV exists).")
    else:
        L.append("  EXPECT: positive ROI, growing with the edge filter (sharper than market).")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", type=Path, help="calib_audit *_arrays.npz")
    ap.add_argument("--odds", type=Path, help="closing-odds CSV")
    ap.add_argument("--market", choices=["moneyline", "totals", "both"], default="both")
    ap.add_argument("--edge", type=float, default=0.03)
    ap.add_argument("--kelly", type=float, default=0.0, help="0=flat 1u; else Kelly multiplier")
    ap.add_argument("--synthetic", choices=["efficient", "noisy"], default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.synthetic:
        run_synthetic(args.synthetic)
    elif args.arrays and args.odds:
        run_real(args)
    else:
        ap.error("provide --synthetic MODE, or both --arrays and --odds")


if __name__ == "__main__":
    main()
