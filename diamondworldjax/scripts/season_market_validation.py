"""Out-of-sample, cross-market validation of the causal signal: does the model's
game-specific win-probability signal still track an independent market on a season the
model never saw, and on a prediction market (Kalshi), not just a sportsbook?

The model's player rates are frozen at <=2023 (v16). We test its game-specific signal
against the 2026 season (genuinely out-of-sample) priced by Kalshi's per-game markets
(KXMLBGAME), using the same within-series design and the same pitching/hitting channel
decomposition as the 2024 sportsbook validation -- no GPU simulation, so it is purely
the model's inputs vs the market. Kalshi only lists MLB game markets from ~mid-2026, so
2025 is not available on Kalshi; a sportsbook path is used for 2025 if odds exist.

Kalshi pre-game implied probability: each game has two markets (one per team, ...-TEAM);
the closing candlestick price just before first pitch (encoded in the ticker) is the
market P(that team wins). We map to home/away via the MLB schedule and read P(home).

  python -m diamondworldjax.scripts.season_market_validation --season 2026
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.paths import processed_root
from diamondworldjax.scripts.train_pa import _build_player_table
from diamondworldjax.scripts.seq_models import pitcher_rates
from diamondworldjax.scripts.whatif_channels import woba_bat, allowed_idx

KAL = "https://api.elections.kalshi.com/trade-api/v2"
CACHE = Path("data/cache/market"); CACHE.mkdir(parents=True, exist_ok=True)
MON = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
       "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
# Kalshi -> StatsAPI abbreviation normalization
NORM = {"CHW": "CWS", "WAS": "WSH", "AZ": "ARI", "SD": "SDP", "TB": "TBR", "KC": "KCR",
        "SF": "SFG", "SFO": "SFG"}


def _get(url, cache_name=None, retries=3):
    if cache_name:
        p = CACHE / cache_name
        if p.exists() and p.stat().st_size > 2:
            return json.loads(p.read_text())
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            r = json.load(urllib.request.urlopen(req, timeout=30))
            if cache_name:
                (CACHE / cache_name).write_text(json.dumps(r))
            return r
        except Exception:
            time.sleep(1.5)
    return None


def norm_team(t):
    return NORM.get(t, t)


def kalshi_games(season):
    """Return {(date, frozenset{teamA,teamB})): {team: pre-game P(win)}} from Kalshi."""
    mkts, cursor = [], ""
    for _ in range(30):
        d = _get(f"{KAL}/markets?series_ticker=KXMLBGAME&status=settled&limit=1000"
                 + (f"&cursor={cursor}" if cursor else ""))
        if not d or not d.get("markets"):
            break
        mkts += d["markets"]; cursor = d.get("cursor", "")
        if not cursor:
            break
    # Group the two markets of each game by (date, time, teams_str); the two -SIDE
    # codes are the two teams, so no ambiguous concatenation splitting is needed.
    by_game = defaultdict(dict)
    for m in mkts:
        tk = m["ticker"]                                  # KXMLBGAME-26JUL261920NYYPHI-PHI
        parts = tk.split("-")
        if len(parts) < 3:
            continue
        body, side = parts[1], parts[2]
        try:
            yy, mon, dd = int(body[:2]), MON[body[2:5]], int(body[5:7])
        except Exception:
            continue
        if 2000 + yy != season:
            continue
        hhmm, teams_str = body[7:11], body[11:]
        gdate = f"{2000+yy:04d}-{mon:02d}-{dd:02d}"
        start = dt.datetime(2000 + yy, mon, dd, int(hhmm[:2]), int(hhmm[2:])) + dt.timedelta(hours=4)
        sts = int(calendar.timegm(start.timetuple()))
        cs = _get(f"{KAL}/series/KXMLBGAME/markets/{tk}/candlesticks"
                  f"?period_interval=60&start_ts={sts-64800}&end_ts={sts}", cache_name=f"cs_{tk}.json")
        px = None
        if cs and cs.get("candlesticks"):
            vals = [c["price"].get("close_dollars") for c in cs["candlesticks"] if c["price"].get("close_dollars")]
            px = float(vals[-1]) if vals else None
        if px is None:
            continue
        by_game[(gdate, hhmm, teams_str)][norm_team(side)] = px
    out = {}
    for (gdate, hhmm, teams_str), sides in by_game.items():
        if len(sides) < 2:                                # need both teams' prices
            continue
        out[(gdate, frozenset(sides.keys()))] = sides
    return out


def sched_boxscores(season, want_keys):
    """MLB schedule + boxscores for games matching want_keys (date, {abbrs})."""
    sched = _get(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={season}&gameType=R",
                 cache_name=f"sched_{season}.json")
    # team id -> abbrev
    teams = _get("https://statsapi.mlb.com/api/v1/teams?sportId=1", cache_name="teams.json")
    ab = {t["id"]: t["abbreviation"] for t in teams["teams"]}
    games = []
    for d in sched.get("dates", []):
        gdate = d["date"]
        for g in d.get("games", []):
            h = g["teams"]["home"]; a = g["teams"]["away"]
            if "score" not in h or "score" not in a:
                continue
            ha, aa = ab.get(h["team"]["id"]), ab.get(a["team"]["id"])
            key = (gdate, frozenset({ha, aa}))
            if key in want_keys:
                games.append((int(g["gamePk"]), gdate, ha, aa, h["score"], a["score"]))
    return games, ab


def boxscore_lineup(game_pk, id2i):
    bs = _get(f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore", cache_name=f"box_{game_pk}.json")
    if not bs:
        return None
    out = {}
    for side in ("home", "away"):
        t = bs["teams"][side]
        order = [int(str(pid).replace("ID", "")) for pid in t.get("battingOrder", [])][:9]
        bat = [id2i.get(pid, 0) for pid in order]
        pitchers = [int(str(p).replace("ID", "")) for p in t.get("pitchers", [])]
        sp = id2i.get(pitchers[0], 0) if pitchers else 0
        out[side] = (bat, sp)
    return out


def american_implied(ml):
    ml = float(ml)
    return 100.0 / (ml + 100.0) if ml > 0 else -ml / (-ml + 100.0)


def _uget(u):
    import urllib.request
    r = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(r, timeout=40))


def polymarket_games(season):
    """{(date, frozenset{abbrevA,abbrevB}): {abbrev: pre-game P(win)}} from Polymarket.

    Full-game moneyline = the market whose question equals the "Team A vs. Team B"
    event title (outcomes are the two team names). Pre-game price = the last CLOB
    price point before first pitch (from the MLB schedule), for outcome[0]'s token.
    """
    teams = _get("https://statsapi.mlb.com/api/v1/teams?sportId=1", cache_name="teams.json")
    name2ab = {t["name"]: t["abbreviation"] for t in teams["teams"]}
    sched = _get(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={season}&gameType=R",
                 cache_name=f"sched_{season}.json")
    # frozenset{full names} -> list of (game official-date, first-pitch ts). A team pair
    # plays several times a season, so we keep them all and match Polymarket to the
    # nearest game date (Polymarket's startDate is the listing date, not first pitch).
    start_of = defaultdict(list)
    for d in sched.get("dates", []):
        for g in d.get("games", []):
            names = frozenset({g["teams"]["home"]["team"]["name"], g["teams"]["away"]["team"]["name"]})
            try:
                ts = int(calendar.timegm(dt.datetime.strptime(g["gameDate"][:19], "%Y-%m-%dT%H:%M:%S").timetuple()))
                start_of[names].append((d["date"], ts))
            except Exception:
                pass
    evs = []
    for off in range(0, 2000, 100):
        d = _uget(f"https://gamma-api.polymarket.com/events?limit=100&offset={off}"
                  f"&closed=true&order=startDate&ascending=false&tag_slug=mlb")
        d = d if isinstance(d, list) else d.get("data", [])
        evs += d
        if len(d) < 100:
            break
    out = {}
    n_match = n_clob_ok = n_pre = 0
    for e in evs:
        title = e.get("title", "")
        if " vs" not in title.lower() or any(s in title for s in ("First 5", "Props", "Run Line", "Total", "Strikeout", "Home Runs", ": ")):
            continue
        sd = str(e.get("startDate", ""))[:10]
        if not sd.startswith(str(season)):
            continue
        ml = [m for m in e.get("markets", []) if m.get("question", "").strip() == title.strip()]
        if not ml:
            continue
        m = ml[0]
        oc = m.get("outcomes"); oc = json.loads(oc) if isinstance(oc, str) else oc
        tk = m.get("clobTokenIds"); tk = json.loads(tk) if isinstance(tk, str) else tk
        if not oc or not tk or len(oc) < 2:
            continue
        ab0, ab1 = name2ab.get(oc[0]), name2ab.get(oc[1])
        if not ab0 or not ab1:
            continue
        cands = start_of.get(frozenset({oc[0], oc[1]}), [])
        if not cands:
            continue
        try:
            sd_ts = int(calendar.timegm(dt.datetime.strptime(sd, "%Y-%m-%d").timetuple()))
        except Exception:
            continue
        gdate, gts = min(cands, key=lambda gt: abs(gt[1] - sd_ts))
        if abs(gts - sd_ts) > 3 * 86400:
            continue
        n_match += 1
        cache = CACHE / f"pm_{tk[0][:24]}.json"
        if not (cache.exists() and cache.stat().st_size > 2):
            time.sleep(0.4)                            # throttle: the CLOB rate-limits bulk fetches
        cs = _get(f"https://clob.polymarket.com/prices-history?market={tk[0]}&interval=max&fidelity=60",
                  cache_name=f"pm_{tk[0][:24]}.json")
        if not cs or not cs.get("history"):
            continue
        n_clob_ok += 1
        pre = [p["p"] for p in cs["history"] if p.get("t", 0) <= gts]
        if not pre:
            continue
        n_pre += 1
        p0 = float(pre[-1])
        if not (0.02 < p0 < 0.98):
            continue
        out[(gdate, frozenset({ab0, ab1}))] = {ab0: p0, ab1: 1 - p0}
    print(f"  [polymarket diag] schedule-matched {n_match}, CLOB ok {n_clob_ok}, "
          f"had pre-game pts {n_pre}, usable {len(out)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--train-end", type=int, default=2023,
                    help="Last season in the rate table. Use 2024 for a 1-year-out 2025 test.")
    ap.add_argument("--odds", default=None,
                    help="Sportsbook odds CSV (game_pk,ml_home,ml_away).")
    ap.add_argument("--market", choices=["kalshi", "polymarket"], default="kalshi",
                    help="Prediction market to use when --odds is not given.")
    ap.add_argument("--min-known", type=int, default=6, help="Min known batters of 9 per lineup.")
    args = ap.parse_args()

    print(f"building player rate tables (<= {args.train_end}) ...", flush=True)
    tr = load_seasons(list(range(2015, args.train_end + 1)), data_root=processed_root())
    ptab = _build_player_table(tr, recency_halflife=2.0, contact_quality=True)
    pit = pitcher_rates(tr.filter(pl.col("pa_terminal")), ptab["id_to_idx"], len(ptab["hand"]))
    id2i, stats = ptab["id_to_idx"], ptab["stats"]
    lg_allowed = float(np.mean([allowed_idx(pit[i]) for i in range(1, len(pit)) if pit[i].sum() > 0]))
    del tr

    if args.odds:                                   # sportsbook path (P(home) by game_pk)
        od = pl.read_csv(args.odds)
        om = {int(r["game_pk"]): american_implied(r["ml_home"]) /
              (american_implied(r["ml_home"]) + american_implied(r["ml_away"]))
              for r in od.iter_rows(named=True) if r["ml_home"] is not None and r["ml_away"] is not None}
        print(f"loaded {len(om)} sportsbook game prices from {args.odds}", flush=True)
        # need schedule to get lineups; want-keys = all games (match by game_pk directly)
        sched = _get(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={args.season}&gameType=R",
                     cache_name=f"sched_{args.season}.json")
        teams = _get("https://statsapi.mlb.com/api/v1/teams?sportId=1", cache_name="teams.json")
        abn = {t["id"]: t["abbreviation"] for t in teams["teams"]}
        games = []
        for d in sched.get("dates", []):
            for g in d.get("games", []):
                h, a = g["teams"]["home"], g["teams"]["away"]
                if "score" in h and "score" in a and int(g["gamePk"]) in om:
                    games.append((int(g["gamePk"]), d["date"], abn.get(h["team"]["id"]),
                                  abn.get(a["team"]["id"]), h["score"], a["score"]))
        price_of = lambda gpk, ha: om.get(gpk)      # already P(home)
    else:                                           # prediction-market path (Kalshi or Polymarket)
        print(f"fetching {args.market} {args.season} game markets ...", flush=True)
        kg = polymarket_games(args.season) if args.market == "polymarket" else kalshi_games(args.season)
        print(f"  {len(kg)} {args.market} games with pre-game prices", flush=True)
        games, _ = sched_boxscores(args.season, set(kg.keys()))
    print(f"  {len(games)} games matched to the schedule", flush=True)

    def channel_idx(bat_known, sp):
        Ho = np.mean([woba_bat(stats[i]) for i in bat_known]) if bat_known else 0.0
        Hp = allowed_idx(pit[sp]) if sp != 0 else lg_allowed   # impute league-avg for unknown starter
        return Ho, Hp

    rows = []
    for gpk, gdate, ha, aa, hs, as_ in games:
        if args.odds:
            p_home = om.get(gpk)
        else:
            prices = kg.get((gdate, frozenset({ha, aa})), {})
            p_home = prices.get(ha)
        if p_home is None or not (0.02 < p_home < 0.98):
            continue
        lu = boxscore_lineup(gpk, id2i)
        if not lu:
            continue
        hbat = [i for i in lu["home"][0] if i != 0]; abat = [i for i in lu["away"][0] if i != 0]
        if len(hbat) < args.min_known or len(abat) < args.min_known:
            continue
        Ho, Hp = channel_idx(hbat, lu["home"][1]); Ao, Ap = channel_idx(abat, lu["away"][1])
        rows.append((ha, aa, p_home, Hp - Ap, Ho - Ao))
    n = len(rows)
    print(f"  {n} games with market price + known lineups/starters", flush=True)

    H = np.array([r[0] for r in rows]); A = np.array([r[1] for r in rows])
    mk = np.array([r[2] for r in rows]); pitch = np.array([r[3] for r in rows]); hit = np.array([r[4] for r in rows])
    grp = defaultdict(list)
    for r in range(n):
        grp[(H[r], A[r])].append(r)
    dm, dp, dh = [], [], []
    nser = 0
    for idx in grp.values():
        if len(idx) < 2:
            continue
        nser += 1; idx = np.array(idx)
        dm.extend(mk[idx] - mk[idx].mean()); dp.extend(pitch[idx] - pitch[idx].mean()); dh.extend(hit[idx] - hit[idx].mean())
    dm, dp, dh = np.array(dm), np.array(dp), np.array(dh)
    mkt_name = "sportsbook" if args.odds else f"{args.market} prediction market"
    L = [f"SEASON x MARKET VALIDATION -- {args.season} vs {mkt_name} "
         f"(out-of-sample; rates frozen <= {args.train_end})", ""]
    L.append(f"  {n} games priced+known, {nser} series, {len(dm)} within-series deviations")
    if len(dm) > 20:
        dpz, dhz = dp / (dp.std() + 1e-9), dh / (dh.std() + 1e-9)
        X = np.column_stack([np.ones(len(dm)), dpz, dhz]); beta, *_ = np.linalg.lstsq(X, dm, rcond=None)
        res = dm - X @ beta; s2 = res @ res / max(len(dm) - 3, 1); se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
        L.append(f"  within-series regression of {mkt_name} P(home) move on model channels:")
        L.append(f"    pitching channel t = {beta[1]/se[1]:+.2f}   hitting channel t = {beta[2]/se[2]:+.2f}")
        # combined model index vs market, within series
        comb = -pitch + hit    # home-favorable: better home starter is lower allowed (so -pitch), better home hitting +hit
        dc = []
        for idx in grp.values():
            if len(idx) < 2: continue
            idx = np.array(idx); dc.extend(comb[idx] - comb[idx].mean())
        dc = np.array(dc)
        L.append(f"  corr(model game-specific index, {mkt_name} move) within series = {np.corrcoef(dc, dm)[0,1]:.3f}")
    rep = "\n".join(L)
    print(rep)
    Path("data/eval2").mkdir(parents=True, exist_ok=True)
    Path(f"data/eval2/season_market_validation_{args.season}.txt").write_text(rep + "\n")


if __name__ == "__main__":
    main()
