"""Pure-python helpers for the true game simulator (no JAX imports).

Kept separate from scripts/simulate_games.py so the extraction and
hook-distribution logic is unit-testable without the JAX/NumPyro stack.
"""
from __future__ import annotations

import numpy as np
import polars as pl

MAX_STAFF = 12  # max pitchers per side per game


def fit_hook_dists(train_pa: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Empirical PAs-faced distributions for starters and relievers.

    Per (game, half) the pitcher with the earliest at-bat is the starter;
    everyone else is a reliever. Returns (starter_pas, reliever_pas) arrays
    to sample hook thresholds from.
    """
    g = (
        train_pa.group_by(["game_pk", "half_bin", "pitcher_id"])
        .agg(pl.len().alias("n"), pl.col("at_bat_number").min().alias("fab"))
        .with_columns(
            pl.col("fab").rank("ordinal").over(["game_pk", "half_bin"]).alias("rk")
        )
    )
    starters = g.filter(pl.col("rk") == 1)["n"].to_numpy().astype(np.int64)
    relievers = g.filter(pl.col("rk") > 1)["n"].to_numpy().astype(np.int64)
    return starters, relievers


# PAs-faced, times-through-order, and runs-allowed. Inning is deliberately excluded:
# it is near-collinear with PAs faced (~9 batters per turn), and keeping both drives
# the runs-allowed coefficient to an uninterpretable ~0. Dropping it lets each stay
# interpretable (pas +, ra +) at no cost to held-out accuracy.
HOOK_FEATS = ("pas", "tto", "ra")


def _fit_logistic(X, y, iters=30, l2=1.0):
    """Regularized IRLS logistic fit (no sklearn). X includes the intercept col."""
    beta = np.zeros(X.shape[1])
    ridge = l2 * np.eye(X.shape[1])
    ridge[0, 0] = 0.0  # don't regularize the intercept
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30, 30)))
        w = np.clip(p * (1 - p), 1e-6, None)
        XtWX = (X.T * w) @ X + ridge
        grad = X.T @ (y - p) - ridge @ beta
        try:
            beta = beta + np.linalg.solve(XtWX, grad)
        except np.linalg.LinAlgError:
            break
    return beta


def fit_hook_model(train_pa: pl.DataFrame) -> dict:
    """State-dependent starter-pull hazard: P(pulled after this PA | state).

    The marginal hook (fit_hook_dists) draws a fixed PAs-faced threshold up front,
    so a shelled starter is as likely to stay as a cruising one. This fits the real
    managerial decision instead: a discrete-time hazard over the starter's PAs, with
    state = (PAs faced so far, inning, times-through-order, runs allowed so far).
    Each starter PA is one row, labelled 1 only on the PA after which the starter was
    actually replaced (game/half with >1 pitcher), else 0. Applied in the simulator,
    it makes bullpen usage endogenous to the simulated game and uses only pre-game
    information (the fitted hazard + the unfolding state), not the actual bullpen.

    Returns {beta, mean, std, feats} for a standardized logistic model, plus the
    reliever PAs marginal (relievers are ~1 inning; their variance is small and not
    worth a state model here).
    """
    df = train_pa.select(["game_pk", "half_bin", "pitcher_id", "at_bat_number",
                          "inning", "tto", "runs_scored"])
    # starter per (game, half) = earliest at-bat; and pitcher count per half
    firsts = (df.group_by(["game_pk", "half_bin", "pitcher_id"])
              .agg(pl.col("at_bat_number").min().alias("fab"))
              .with_columns(pl.col("fab").rank("ordinal")
                            .over(["game_pk", "half_bin"]).alias("rk")))
    npitch = (firsts.group_by(["game_pk", "half_bin"])
              .agg(pl.len().alias("npitch")))
    starters = firsts.filter(pl.col("rk") == 1).select(["game_pk", "half_bin", "pitcher_id"])
    s = (df.join(starters, on=["game_pk", "half_bin", "pitcher_id"], how="inner")
         .join(npitch, on=["game_pk", "half_bin"], how="left")
         .sort(["game_pk", "half_bin", "at_bat_number"]))
    grp = ["game_pk", "half_bin"]
    s = s.with_columns([
        pl.col("at_bat_number").cum_count().over(grp).alias("pas"),      # 1..N
        pl.col("runs_scored").cum_sum().over(grp).alias("ra"),
        pl.col("at_bat_number").max().over(grp).alias("last_ab"),
    ])
    # target: pulled after this PA = this is the last starter PA and a reliever followed
    s = s.with_columns(
        ((pl.col("at_bat_number") == pl.col("last_ab")) & (pl.col("npitch") > 1))
        .cast(pl.Float64).alias("y"))
    feat = np.column_stack([s[c].to_numpy().astype(np.float64) for c in HOOK_FEATS])
    y = s["y"].to_numpy().astype(np.float64)
    mean = feat.mean(0)
    std = feat.std(0) + 1e-9
    Xs = np.column_stack([np.ones(len(feat)), (feat - mean) / std])
    beta = _fit_logistic(Xs, y)
    _, reliever_pas = fit_hook_dists(train_pa)
    return {"beta": beta, "mean": mean, "std": std, "feats": HOOK_FEATS,
            "reliever_pas": reliever_pas, "base_rate": float(y.mean())}


def starter_pull_prob(pas, inning, tto, ra, model: dict) -> np.ndarray:
    """Vectorized P(pull) for the fitted hazard, given per-game state arrays.

    Accepts all four candidate state arrays and selects the columns the model was
    actually fit on (model["feats"]), so the feature set can change without touching
    the simulator's call site.
    """
    avail = {"pas": np.asarray(pas, float), "inning": np.asarray(inning, float),
             "tto": np.asarray(tto, float), "ra": np.asarray(ra, float)}
    feat = np.column_stack([avail[f] for f in model["feats"]])
    Xs = np.column_stack([np.ones(len(feat)), (feat - model["mean"]) / model["std"]])
    return 1.0 / (1.0 + np.exp(-np.clip(Xs @ model["beta"], -30, 30)))


def extract_games(
    test_pa: pl.DataFrame,
    id_to_idx: dict,
    park_map: dict | None = None,
    unknown_idx: int | None = None,
) -> list[dict]:
    """Per game: lineups (9 batter idx each), pitching staffs in appearance
    order, park idx.

    LEAKAGE WARNING. Both the lineup and the staff are read off `test_pa`, the
    COMPLETED game. The staff is the actual relievers in the actual order they were
    used, and the lineup is the realized batting order including any early
    substitution. Callers that describe themselves as pre-game (run_pregame_sim.py)
    are leak-free only in hook TIMING, which the fitted hazard supplies; the identity
    and ordering of the relievers still come from the finished game. A true pre-game
    extractor would need to select a staff from the roster using information
    available at first pitch.

    The staff a lineup FACES belongs to the fielding team: half_bin 0 (top,
    away batting) is pitched by the HOME staff and vice versa. (The v1
    extractor tagged these crossed, so every lineup faced its own starter.)

    Unknown players map to a dedicated one-past-the-table sentinel.  The model
    converts that sentinel to a neutral embedding, so an unseen player no longer
    impersonates whichever real player happens to occupy index 0.  The sentinel
    keeps full game coverage without changing checkpoint parameter shapes.
    """
    if unknown_idx is None:
        unknown_idx = max(id_to_idx.values(), default=-1) + 1

    bcol = "batter_id" if "batter_id" in test_pa.columns else "batter_idx"
    pcol = "pitcher_id" if "pitcher_id" in test_pa.columns else "pitcher_idx"
    games = []
    df = test_pa.sort(["game_pk", "at_bat_number"])
    for gid, gdf in df.group_by("game_pk", maintain_order=True):
        rec = {"game_pk": int(gid[0] if isinstance(gid, tuple) else gid)}
        ok = True
        for half, batting, fielding in [(0, "away", "home"), (1, "home", "away")]:
            h = gdf.filter(pl.col("half_bin") == half)
            if len(h) == 0:
                ok = False
                break
            bats = [id_to_idx.get(int(b), unknown_idx) for b in h[bcol].to_list()]
            seen, lineup = set(), []
            for b in bats:
                if b not in seen:
                    seen.add(b)
                    lineup.append(b)
                if len(lineup) == 9:
                    break
            while len(lineup) < 9:
                lineup.append(lineup[0] if lineup else unknown_idx)
            rec[f"{batting}_lineup"] = lineup

            staff, pseen = [], set()
            for p in h[pcol].to_list():
                p = int(p)
                if p not in pseen:
                    pseen.add(p)
                    staff.append(id_to_idx.get(p, unknown_idx))
                if len(staff) == MAX_STAFF:
                    break
            rec[f"{fielding}_staff"] = staff
        if not ok:
            continue
        park = 0
        if park_map is not None and "park_id" in gdf.columns:
            pid = gdf["park_id"].to_list()[0]
            if pid is not None:
                park = park_map.get(pid, 0)
        rec["park"] = park
        games.append(rec)
    return games


def cap_walkoff_runs(
    bat_score: np.ndarray,
    fld_score: np.ndarray,
    runs: np.ndarray,
    is_hr: np.ndarray,
) -> np.ndarray:
    """Cap sampled runs on walk-off plays.

    The game ends the moment the winning run scores; on non-HR plays only the
    winning run counts (MLB rule 7.01(g)(3)). Home runs count in full.
    """
    would = bat_score + runs
    nonhr_walkoff = (would > fld_score) & ~is_hr
    return np.where(nonhr_walkoff, fld_score + 1 - bat_score, runs)


def pad_staffs(
    games: list[dict], key: str, unknown_idx: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """(G, MAX_STAFF) staff idx padded with the last pitcher + (G,) lengths."""
    G = len(games)
    out = np.zeros((G, MAX_STAFF), dtype=np.int64)
    lens = np.zeros(G, dtype=np.int64)
    for i, g in enumerate(games):
        s = g[key] or [unknown_idx]
        lens[i] = len(s)
        out[i, : len(s)] = s
        out[i, len(s):] = s[-1]
    return out, lens
