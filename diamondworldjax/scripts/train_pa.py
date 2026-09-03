"""DiamondWorldJAX PA-level model training entry point.

Usage
-----
    python -m diamondworldjax.scripts.train_pa [--steps 50000] [--batch 64]

Trains the PA-level model via SVI and saves checkpoints to
checkpoints/dwjax_pa/.
"""
from __future__ import annotations

import argparse
import itertools
import time
from functools import partial
from pathlib import Path

import numpy as np

from diamondworldjax.paths import processed_root, checkpoints_root, results_root

_DATA_ROOT = processed_root()
_CKPT_DIR  = checkpoints_root() / "dwjax_pa_v5"
_LOG_PATH  = results_root() / "dwjax_pa_v5_elbo.json"

TRAIN_SEASONS = list(range(2015, 2023))
DEFAULT_F_PLAYER = 16


EV_BIN, LA_BIN = 2.0, 3.0        # exit-velo (mph) and launch-angle (deg) bin widths
MIN_BIN_N = 25                   # bins thinner than this are not trusted


def _contact_quality(terminal, batter_col, id_to_idx, P, weights):
    """Expected hit / HR rate per batter from contact quality (the xBA idea).

    Columns 0..3 of the stat table are outcome rates, so a batter's hit rate
    carries a season of BABIP luck: where the ball landed and who was standing
    there. This scores the batter on the contact they MADE instead. Bin every
    batted ball by (exit velocity, launch angle), take the league hit/HR frequency
    in that bin, and average over the batter's PAs. Strikeouts and walks contribute
    zero expected hits, so the result is directly comparable to the observed rate.

    projection_levers.py measured this construction lifting hit-rate correlation
    0.418 -> 0.497 and HR 0.608 -> 0.641 on a Marcel-style projection, which is
    why it is worth a feature slot here.

    The bin table is built from the TRAINING seasons only (`terminal` is already
    filtered), so no test-season information leaks in.
    """
    import polars as pl

    d = terminal.filter(pl.col("pa_outcome").is_not_null()).with_columns([
        pl.col("pa_outcome").is_in(["1B", "2B", "3B", "HR"]).cast(pl.Float64).alias("f_hit"),
        (pl.col("pa_outcome") == "HR").cast(pl.Float64).alias("f_hr"),
    ])
    hit_bb = d.filter(pl.col("launch_speed").is_not_null() & pl.col("launch_angle").is_not_null())
    if len(hit_bb) < 1000:
        return None
    binned = hit_bb.with_columns([
        (pl.col("launch_speed") / EV_BIN).floor().alias("ev_b"),
        (pl.col("launch_angle") / LA_BIN).floor().alias("la_b"),
    ])
    tbl = (binned.group_by(["ev_b", "la_b"])
           .agg([pl.col("f_hit").mean().alias("p_hit"),
                 pl.col("f_hr").mean().alias("p_hr"),
                 pl.len().alias("n")])
           .filter(pl.col("n") >= MIN_BIN_N)
           .select(["ev_b", "la_b", "p_hit", "p_hr"]))

    d = (d.with_columns([
            (pl.col("launch_speed") / EV_BIN).floor().alias("ev_b"),
            (pl.col("launch_angle") / LA_BIN).floor().alias("la_b")])
         .join(tbl, on=["ev_b", "la_b"], how="left"))
    # No launch data, or a bin too thin to trust: fall back to what actually
    # happened, so those PAs are neither credited nor penalised.
    d = d.with_columns([
        pl.col("p_hit").fill_null(pl.col("f_hit")).alias("x_hit"),
        pl.col("p_hr").fill_null(pl.col("f_hr")).alias("x_hr"),
        pl.Series("w", weights),
    ])
    agg = (d.group_by(batter_col).agg([
        (pl.col("x_hit") * pl.col("w")).sum().alias("xh"),
        (pl.col("x_hr") * pl.col("w")).sum().alias("xhr"),
        pl.col("w").sum().alias("wpa")]))

    x = np.zeros((P, 2), dtype=np.float32)
    for row in agg.iter_rows(named=True):
        i = id_to_idx.get(int(row[batter_col]))
        if i is not None and row["wpa"] > 0:
            x[i, 0] = row["xh"] / row["wpa"]
            x[i, 1] = row["xhr"] / row["wpa"]
    return x


def _build_player_table(pitches, recency_halflife: float | None = None,
                        contact_quality: bool = False,
                        per_stat_shrink: bool = False) -> dict:
    """Build the per-player stat/handedness table.

    recency_halflife (seasons): if set, each PA's contribution to a player's rate
    stats is weighted 0.5 ** ((max_season - season) / halflife), so recent form
    dominates. A leakage-free "current-season" prior: the most recent TRAINING
    season (2022) is weighted highest as the best proxy for 2023-24 talent. None =
    uniform (pooled 2015-2022), the v6..v11 behavior.

    contact_quality: fill stat columns 5 and 6 with expected hit / HR rates built
    from launch speed and angle rather than from what fell in (see
    _contact_quality). Off by default: it changes the model's input distribution,
    so a checkpoint trained without it must be evaluated without it.
    """
    import polars as pl

    terminal = pitches.filter(pl.col("pa_terminal"))
    pitcher_col = "pitcher_id" if "pitcher_id" in pitches.columns else "pitcher_idx"
    batter_col  = "batter_id"  if "batter_id"  in pitches.columns else "batter_idx"
    max_season = int(terminal["season"].max()) if "season" in terminal.columns else 0

    all_ids = np.unique(np.concatenate([
        pitches[pitcher_col].to_numpy(),
        pitches[batter_col].to_numpy(),
    ])).astype(np.int32)
    P = len(all_ids)
    id_to_idx = {int(pid): i for i, pid in enumerate(all_ids)}

    stats  = np.zeros((P, DEFAULT_F_PLAYER), dtype=np.float32)
    league = np.zeros(P, dtype=np.int32)
    hand   = np.zeros(P, dtype=np.int32)

    if "pa_outcome" in terminal.columns:
        for row in terminal.filter(pl.col("pa_outcome").is_not_null()).iter_rows(named=True):
            bid  = row.get(batter_col, 0)
            bidx = id_to_idx.get(int(bid), 0)
            outcome = row.get("pa_outcome", "")
            if recency_halflife:
                w = 0.5 ** ((max_season - int(row.get("season", max_season))) / recency_halflife)
            else:
                w = 1.0
            if outcome in ("1B", "2B", "3B", "HR"):
                stats[bidx, 0] += w
            if outcome in ("BB", "HBP"):
                stats[bidx, 1] += w
            if outcome == "K":
                stats[bidx, 2] += w
            if outcome == "HR":
                stats[bidx, 3] += w
            stats[bidx, 4] += w

    pa_count = np.maximum(stats[:, 4:5], 1)
    stats[:, :4] /= pa_count

    if per_stat_shrink:
        # Per-stat empirical-Bayes shrinkage toward the league rate.
        #
        # Columns 0..3 are RAW observed rates, so a 150-PA batter's hit rate is
        # mostly noise while his strikeout rate is already informative. Feeding
        # both in raw asks the model to work out how much to trust each one from
        # the PA count in column 4. In principle it can; the evidence from the
        # v17/v18/v19 series is that this model does not learn what it could
        # learn, and that FEATURES move the metric where architecture and
        # objective do not.
        #
        # The constants are measured, not guessed: projection_levers.py found
        # each rate stabilises at a different sample size (K ~200 PA, BB ~400,
        # hit and HR ~2200), and flagged the single shipped REG=1200 as "badly
        # wrong at both ends". This applies each stat's own constant.
        #
        # Order of columns 0..3 is (hit, bb, k, hr); mismatching this would
        # silently shrink strikeouts with the home-run constant, so it is
        # asserted against PA_OUTCOME order in the tests.
        reg = np.array([2200.0, 400.0, 200.0, 2200.0], dtype=np.float64)
        n = stats[:, 4:5].astype(np.float64)
        seen = (n > 0).ravel()
        # League baseline from PA-weighted observed rates, so it is not dragged
        # by the many near-zero-PA rows in the table.
        #
        # NOTE the name: `league` is already taken by the league_id int array a
        # few lines above, which feeds LeagueEmbedding. Shadowing it here silently
        # replaced those ints with float rates and blew up inside nn.Embed with
        # "Input type must be an integer" from three frames away.
        league_rate = ((stats[seen, :4].astype(np.float64) * n[seen]).sum(0)
                       / max(n[seen].sum(), 1.0))
        stats[:, :4] = ((stats[:, :4].astype(np.float64) * n + reg * league_rate)
                        / (n + reg)).astype(np.float32)

    if contact_quality and {"launch_speed", "launch_angle"}.issubset(terminal.columns):
        tnn = terminal.filter(pl.col("pa_outcome").is_not_null())
        if recency_halflife and "season" in tnn.columns:
            w = 0.5 ** ((max_season - tnn["season"].to_numpy()) / recency_halflife)
        else:
            w = np.ones(len(tnn))
        xq = _contact_quality(terminal, batter_col, id_to_idx, P, w.astype(np.float64))
        if xq is not None:
            stats[:, 5:7] = xq

    # Per-player modal handedness for the simulator + the hand embedding.
    # bat_hand: modal batting side (stand); pit_hand: modal throw hand (p_throws).
    # R=1, L=0, unknown=0.5 (a batter who never appears keeps 0.5).
    bat_hand = np.full(P, 0.5, dtype=np.float32)
    pit_hand = np.full(P, 0.5, dtype=np.float32)
    bat_hand_col = "batter_hand" if "batter_hand" in terminal.columns else "stand"
    pit_hand_col = "pitcher_hand" if "pitcher_hand" in terminal.columns else "p_throws"
    if bat_hand_col in terminal.columns:
        m = (terminal.group_by(batter_col)
             .agg((pl.col(bat_hand_col) == "R").mean().alias("r")))
        for row in m.iter_rows(named=True):
            i = id_to_idx.get(int(row[batter_col]), None)
            if i is not None and row["r"] is not None:
                bat_hand[i] = 1.0 if row["r"] >= 0.5 else 0.0
    if pit_hand_col in terminal.columns:
        m = (terminal.group_by(pitcher_col)
             .agg((pl.col(pit_hand_col) == "R").mean().alias("r")))
        for row in m.iter_rows(named=True):
            i = id_to_idx.get(int(row[pitcher_col]), None)
            if i is not None and row["r"] is not None:
                pit_hand[i] = 1.0 if row["r"] >= 0.5 else 0.0
    # hand embedding input (int 0/1): pitchers use throw hand, else batting side.
    hand = np.where(pit_hand != 0.5, pit_hand, bat_hand)
    hand = (hand >= 0.5).astype(np.int32)

    return {"stats": stats, "league": league, "hand": hand,
            "bat_hand": bat_hand, "pit_hand": pit_hand,
            "id_to_idx": id_to_idx, "all_ids": all_ids,
            # P is deliberately outside the parameter table. PlayerRegistry
            # maps it to a neutral embedding without shifting existing indices.
            "unknown_index": P}


def _build_park_index(pitches) -> dict:
    """Dense park_id -> park_idx mapping fit on the training seasons.

    Index 0 is reserved for unknown parks (test-era venues never seen in
    training), matching the unknown-player convention. Deterministic given the
    training seasons (sorted unique park_id), so eval/sim scripts can rebuild
    the identical mapping instead of persisting it in checkpoints.

    NOTE: before v9 this mapping did not exist anywhere in the pipeline, so
    `pa_batching._col("park_idx", 0.0)` silently filled 0 for every PA and the
    park embedding trained as a constant. Apply with `apply_park_idx` before
    building batches.
    """
    import polars as pl

    # park_id is a string venue code (e.g. "HOU") in the processed parquets.
    ids = sorted(pitches["park_id"].drop_nulls().unique().to_list())
    return {p: i + 1 for i, p in enumerate(ids)}


def apply_park_idx(df, park_map: dict):
    """Materialise the park_idx column pa_batching expects."""
    import polars as pl

    return df.with_columns(
        pl.col("park_id")
        .replace_strict(park_map, default=0, return_dtype=pl.Int32)
        .alias("park_idx")
    )


def _map_player_ids(batch: dict, id_to_idx: dict, unknown_index: int | None = None) -> dict:
    """Map raw ids to table rows; unseen and padding ids use a neutral sentinel."""
    import jax.numpy as jnp

    if unknown_index is None:
        unknown_index = max(id_to_idx.values(), default=-1) + 1

    def remap(arr):
        arr_np = np.array(arr)
        out = np.vectorize(lambda x: id_to_idx.get(int(x), unknown_index))(arr_np)
        return jnp.array(out.astype(np.int32))

    batch["pitcher_ids"] = remap(batch["pitcher_ids"])
    batch["batter_ids"]  = remap(batch["batter_ids"])
    return batch


def _make_pa_batch(pa_df, game_id_chunk, id_to_idx, player_table_np):
    import polars as pl
    import jax.numpy as jnp
    from diamondworldjax.data.pa_batching import build_pa_batch

    chunk_df = pa_df.filter(pl.col("game_pk").is_in(game_id_chunk.tolist()))
    if len(chunk_df) == 0:
        return None, None

    batch = build_pa_batch(chunk_df)
    batch = _map_player_ids(batch, id_to_idx)

    pt = {
        "stats":  jnp.array(player_table_np["stats"]),
        "league": jnp.array(player_table_np["league"]),
        "hand":   jnp.array(player_table_np["hand"]),
    }
    return batch, pt


def _infinite_batch_iter(pa_df, chunks, id_to_idx, player_table_np):
    for chunk in itertools.cycle(chunks):
        result = _make_pa_batch(pa_df, chunk, id_to_idx, player_table_np)
        if result[0] is not None:
            yield result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",  type=int,   default=50_000)
    parser.add_argument("--lr",     type=float, default=3e-4)
    parser.add_argument("--seed",   type=int,   default=0)
    parser.add_argument("--batch",  type=int,   default=64,
                        help="Games per mini-batch (PA model is lightweight — 64 fits easily)")
    parser.add_argument("--resume",    type=str,   default=None)
    parser.add_argument("--ss-rate",   type=float, default=0.25,
                        help="Max fraction of complete games with generated state histories (0=disabled)")
    parser.add_argument("--ss-warmup", type=int,   default=20_000,
                        help="Steps to ramp ss_rate from 0 to ss_max_rate")
    parser.add_argument("--cosine-alpha", type=float, default=0.0,
                        help="Cosine LR floor as fraction of init LR (0=decay to 0, 0.1=decay to 10%%)")
    parser.add_argument("--contact-quality", action="store_true",
                        help="Fill player stat columns 5-6 with expected hit/HR rates from "
                             "launch speed and angle (xBA-style) instead of relying only on "
                             "outcome rates, which carry BABIP luck. Changes the input "
                             "distribution: eval must pass the same flag. Fresh train.")
    parser.add_argument("--outcome-only", action="store_true",
                        help="Train the outcome-only model (v6): single pa_outcome head, "
                             "runs + base_state handled by the rules engine at eval.")
    parser.add_argument("--engine-ss", action="store_true",
                        help="Deprecated compatibility flag; scheduled sampling now always rolls "
                             "complete game states through the rules engine.")
    parser.add_argument("--fatigue", action="store_true",
                        help="Phase-4: add pitcher cumulative game pitch count to the model "
                             "context (STATE_DIM 8 -> 9). Fresh train (changes context dim).")
    parser.add_argument("--platoon", action="store_true",
                        help="Add batter side + pitcher throw hand (real per-PA stand/p_throws) "
                             "to the context (+2 dims). Fresh train (changes context dim).")
    parser.add_argument("--recency-halflife", type=float, default=None,
                        help="Recency-weight player rate stats by season (half-life in seasons). "
                             "Leakage-free current-form prior. Eval scripts must pass the same.")
    parser.add_argument("--no-kl-scale", action="store_true",
                        help="Disable the minibatch player_skills KL scaling (reproduce the "
                             "pre-fix v6..v12 latent-collapse behavior).")
    parser.add_argument("--bilinear-rank", type=int, default=0,
                        help="Rank of the bilinear batter x pitcher interaction added to the "
                             "outcome logits (0 = off, the v6..v16 behavior). Generalises the "
                             "hand-coded --platoon lever to all interactions. Eval scripts must "
                             "pass the same value.")
    parser.add_argument("--nested", action="store_true",
                        help="Two-stage outcome head: {K,BB,HBP,in-play} then in-play -> "
                             "{1B,2B,3B,HR,out,E}, instead of a flat 9-way softmax. Eval "
                             "scripts must pass the same flag.")
    parser.add_argument("--skill-prior", choices=["iso", "learned", "lkj", "walk"], default="iso",
                        help="Prior on the per-player skill latent. iso = N(0,I) (v6..v16); "
                             "learned = per-dimension learned scale (adaptive shrinkage); "
                             "lkj = full learned correlation. Eval scripts must pass the same.")
    parser.add_argument("--player-agg-weight", type=float, default=0.0,
                        help="Weight on the per-batter aggregation loss (0 = off). Reweights "
                             "the objective from per-PA (where ~99.7%% of the signal is the "
                             "league marginal) toward per-BATTER, the axis the eval metric "
                             "measures. Training-only; adds no parameters, so eval scripts do "
                             "NOT need a matching flag.")
    parser.add_argument("--player-agg-shrink", type=float, default=20.0,
                        help="Shrinkage constant k in n/(n+k) for the aggregation loss; "
                             "downweights batters with few PAs in the minibatch.")
    parser.add_argument("--per-stat-shrink", action="store_true",
                        help="Shrink each rate feature toward the league rate using its OWN "
                             "measured stabilisation constant (K 200 PA, BB 400, hit/HR 2200) "
                             "instead of feeding raw rates. Changes the input distribution, so "
                             "eval scripts must pass the same flag.")
    parser.add_argument("--pitchformer", action="store_true",
                        help="Replace the flat context with a causal PA-level transformer "
                             "adapted from the pitchformer architecture. Each PA attends to "
                             "all previous PAs in the game. Fresh train (changes context dim). "
                             "Eval/sim scripts must pass the same flag.")
    parser.add_argument("--tag", type=str, default=None,
                        help="Checkpoint/log dir tag override (e.g. v6).")
    parser.add_argument("--train-end", type=int, default=2022,
                        help="Last season included in training (inclusive). Default 2022 "
                             "(v6..v13). Set 2023 to fold in the previous season: the "
                             "recency-weighted rate features then weight 2023 highest, the "
                             "single most valuable prior for 2024 (see prev_season_ablation). "
                             "Eval must then test 2024 only (2023 becomes in-sample).")
    args = parser.parse_args()

    global _CKPT_DIR, _LOG_PATH, TRAIN_SEASONS
    TRAIN_SEASONS = list(range(2015, args.train_end + 1))
    if args.tag:
        _CKPT_DIR = checkpoints_root() / f"dwjax_pa_{args.tag}"
        _LOG_PATH = results_root() / f"dwjax_pa_{args.tag}_elbo.json"
    elif args.outcome_only:
        _CKPT_DIR = checkpoints_root() / "dwjax_pa_v6"
        _LOG_PATH = results_root() / "dwjax_pa_v6_elbo.json"

    print("Importing JAX + NumPyro...", flush=True)
    import jax
    print(f"  JAX devices: {jax.devices()}", flush=True)

    import polars as pl
    from diamondworldjax.data.pipeline import load_seasons
    from diamondworldjax.model.pa_model import pa_model
    from diamondworldjax.train.svi import train

    print(f"Loading training seasons {TRAIN_SEASONS}...", flush=True)
    pitches = load_seasons(TRAIN_SEASONS, data_root=_DATA_ROOT)
    print(f"  {len(pitches):,} pitches loaded.", flush=True)

    print("Building player table...", flush=True)
    player_table_np = _build_player_table(pitches, recency_halflife=args.recency_halflife,
                                          contact_quality=args.contact_quality,
                                          per_stat_shrink=args.per_stat_shrink)
    print(f"  {len(player_table_np['all_ids']):,} unique players.", flush=True)

    print("Filtering to PA-terminal rows...", flush=True)
    pa_df = pitches.filter(pl.col("pa_terminal"))
    park_map = _build_park_index(pitches)
    pa_df = apply_park_idx(pa_df, park_map)
    print(f"  {len(pa_df):,} plate appearances, {len(park_map)} parks.", flush=True)

    game_ids = pa_df["game_pk"].unique().to_numpy()
    np.random.shuffle(game_ids)
    chunks = [game_ids[i:i + args.batch] for i in range(0, len(game_ids), args.batch)]
    batch_iter = _infinite_batch_iter(pa_df, chunks, player_table_np["id_to_idx"], player_table_np)

    # Minibatch-SVI correction: the global player_skills KL must be scaled to the
    # minibatch fraction (batch games / total games), else it is ~total/batch times
    # over-weighted and the latent collapses to the prior. --no-kl-scale disables
    # it (reproduces the pre-fix v6..v12 behavior).
    kl_scale = 1.0 if args.no_kl_scale else (args.batch / max(len(game_ids), 1))

    print(f"Starting SVI: {args.steps} steps, lr={args.lr}"
          f"{'  [outcome-only v6]' if args.outcome_only else ''}", flush=True)
    t0 = time.time()

    _mkw = {}
    if args.outcome_only:
        _mkw["outcome_only"] = True
    if args.fatigue:
        _mkw["fatigue"] = True
    if args.platoon:
        _mkw["platoon"] = True
    if args.bilinear_rank > 0:
        _mkw["bilinear_rank"] = args.bilinear_rank
    if args.nested:
        _mkw["nested"] = True
    if args.skill_prior != "iso":
        _mkw["skill_prior"] = args.skill_prior
    if args.skill_prior == "walk":
        # Derive the season axis from the actual training range rather than a
        # default, so --train-end changes the latent shape correctly.
        _mkw["season_base"] = TRAIN_SEASONS[0]
        _mkw["n_seasons"] = len(TRAIN_SEASONS)
    if args.player_agg_weight > 0:
        _mkw["player_agg_weight"] = args.player_agg_weight
        _mkw["player_agg_shrink"] = args.player_agg_shrink
    if args.pitchformer:
        _mkw["pitchformer"] = True
    _mkw["kl_scale"] = kl_scale
    model_fn = partial(pa_model, **_mkw)

    svi_state, guide, losses = train(
        model            = model_fn,
        batch_iter       = batch_iter,
        n_steps          = args.steps,
        lr               = args.lr,
        seed             = args.seed,
        ckpt_dir         = _CKPT_DIR,
        log_path         = _LOG_PATH,
        resume_path      = args.resume,
        cosine_decay     = True,
        cosine_alpha     = args.cosine_alpha,
        ss_max_rate      = args.ss_rate,
        ss_warmup_steps  = args.ss_warmup,
        ss_start_step    = 0 if args.resume else 5_000,
        engine_ss        = args.engine_ss,
        kl_scale         = kl_scale,
        skill_prior      = args.skill_prior,
        n_seasons        = len(TRAIN_SEASONS) if args.skill_prior == "walk" else 1,
    )

    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed/3600:.2f}h.", flush=True)
    print(f"Final ELBO = {-losses[-1]:.2f}", flush=True)
    print(f"Checkpoint dir: {_CKPT_DIR}", flush=True)


if __name__ == "__main__":
    main()
