"""TRUE game simulator: generated inning structure, real rosters, real rules.

v2 (game-structure + bullpen upgrade). The v1 simulator generated outs and
lineup cycling correctly but kept three structural distortions:

  1. Fixed 9 innings for every game — no bottom-9 skip when the home team
     leads, no walk-offs, no extra innings. Tied games counted as home losses
     (home>away test), deflating home-win% to ~48 vs the real ~53.
  2. One starter pitched all 9 innings. Real games average ~4 pitchers per
     side; reliever/starter identity is a first-order effect on scoring.
  3. Starter extraction was CROSSED: `away_sp` held the pitcher the away
     lineup faces (the HOME starter) but the sim used it for the home half,
     so every lineup faced its own team's starter.

This version:

  - Plays halves until the game is decided: skips the bottom 9 when the home
    team leads after the top, ends walk-off halves the moment the home team
    takes the lead (non-HR walk-offs are capped at the winning run, per rule
    7.01(g)(3)), and plays extra innings with the ghost-runner-on-2B rule
    (in effect for the 2023-24 test seasons). Hard cap at MAX_INNINGS.
  - Models pitching changes: each side's real staff (starter + relievers in
    actual appearance order) is extracted per game, and pitchers are hooked
    after a number of PAs drawn from the empirical starter/reliever
    PAs-faced distributions fit on the training seasons. TTO resets for the
    batting team on a pitching change, and the fatigue proxy tracks the
    CURRENT pitcher's workload.
  - Optionally feeds real park indices (--use-park) for checkpoints trained
    with the park_idx fix (v9+). Pre-v9 checkpoints trained the park
    embedding on all-zeros, so the flag defaults off.

Legacy A/B flags: --fixed-nine restores the v1 fixed-9-inning structure and
--no-bullpen keeps the starter in all game.

Outcome sampling is the only model call; runs/bases/outs are engine
arithmetic via the EmpiricalEngine fit on 2015-22.

Usage:
  python -m diamondworldjax.scripts.simulate_games --ckpt <ckpt> --outcome-only --recal --limit-games 512
"""
from __future__ import annotations

import argparse
import pickle
import time
from functools import partial
from pathlib import Path

import numpy as np
import polars as pl

from diamondworldjax.paths import processed_root, checkpoints_root
from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.model.pa_model import pa_model
from diamondworldjax.sim.rules_engine import EmpiricalEngine, PA_OUTCOME_IDX
from diamondworldjax.sim.game_extract import (
    cap_walkoff_runs,
    extract_games,
    fit_hook_dists,
    pad_staffs,
    starter_pull_prob,
)
from diamondworldjax.scripts.train_pa import _build_player_table, _build_park_index

TRAIN = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
TEST = [2023, 2024]

MAX_INNINGS = 20   # safety cap; ghost runner decides virtually everything by ~13
HR_IDX = PA_OUTCOME_IDX["HR"]

# v6 per-class logit recalibration = log(real/model) from diag_outcomes ratios
# (K 1.31x, 1B 0.86x, ...). Corrects the mild outcome miscalibration without retraining.
# Order: K, BB, HBP, 1B, 2B, 3B, HR, out, E.
RECAL_V6 = np.array([
    -np.log(1.31), -np.log(0.97), -np.log(0.81), -np.log(0.86), -np.log(0.82),
    -np.log(0.93), -np.log(0.98), -np.log(0.92), 0.0,
], dtype=np.float64)

# v9 (outcome-only + fatigue + park_idx fix) recalibration = log(real/model) from
# diag_outcomes on real park indices. Much milder than v6 (K 1.14x vs 1.31x; the
# park fix improved raw calibration); main corrections are HR 0.72x and 2B 0.81x.
# Order: K, BB, HBP, 1B, 2B, 3B, HR, out, E.
RECAL_V9 = np.array([
    -0.1301, -0.0425, +0.2990, +0.0271, +0.2061, +0.0372, +0.3269, +0.0255, 0.0,
], dtype=np.float64)

# v10 (v9 recipe trained to 50K steps) recalibration = log(real/model) from
# diag_outcomes on real park indices. The longer run fixed HR calibration (1.00x,
# no correction needed vs v9's 0.72x) but over-predicts K more (1.28x vs 1.14x);
# net corrections lift 1B/2B/out and trim K. Order: K, BB, HBP, 1B, 2B, 3B, HR, out, E.
RECAL_V10 = np.array([
    -0.2457, -0.1308, +0.1857, +0.1488, +0.1782, +0.0441, -0.0027, +0.1060, 0.0,
], dtype=np.float64)

RECAL_VECS = {"v6": RECAL_V6, "v9": RECAL_V9, "v10": RECAL_V10}


def simulate(
    model_fn, params, pt, games, rng_key,
    shift=1.0, clock=1.0, recal=False, recal_scale=1.0, recal_vec=RECAL_V6,
    fixed_nine=False, no_bullpen=False, seed=0, platoon=False, recal_temp=1.0,
    skill_mode="prior", crn_keys=None, hook_model=None, max_pa_per_half=40,
    pitchformer=False, skill_prior="iso", simulation_season=2024,
):
    """Vectorized simulation across all games with real game structure.

    Returns dict with per-game scores, occupancy, per-batter outcome counts,
    runs-by-inning (1-9), and structure stats (extras, walk-offs, ties).

    crn_keys: optional per-game integer stream id (length G) enabling common
    random numbers. Two games with the same key draw the same outcome-gumbel and
    base-advancement sequence (until their play diverges), so shared game
    randomness cancels when you difference scenarios. The intended use is to pass
    the replica index, shared across scenarios, so scenario A replica r and
    scenario B replica r are paired. Hooks use a separate per-game stream, so a
    pitching-change timing difference does not scramble the outcome stream. When
    None, a single global generator is used (the original behaviour).

    max_pa_per_half: safety cap for a generated half-inning. ``None`` disables
    truncation. When the cap is reached, the result records affected games in
    ``truncated`` rather than silently presenting them as complete.
    """
    import jax
    import jax.numpy as jnp
    import numpyro.handlers as nh
    from diamondworldjax.model.pa_inference import bucket_size, build_pa_inference

    if max_pa_per_half == 0:
        max_pa_per_half = None
    elif max_pa_per_half is not None and max_pa_per_half < 0:
        raise ValueError("max_pa_per_half must be positive or None")

    # Skill handling: the SVI player-skill latent collapsed to the prior
    # (player_mu ~ 0, player_sigma ~ 1), so by default the model samples it from
    # N(0,1) at eval (aleatoric noise). "mean" substitutes player_mu (~0),
    # removing that noise; "sample" draws once from the posterior. Player signal
    # otherwise lives entirely in the deterministic rate-stat encoder.
    params = dict(params)
    if skill_mode in ("mean", "sample") and "player_mu" in params:
        mu = np.asarray(params["player_mu"])
        if skill_mode == "sample":
            sig = np.asarray(params.get("player_sigma", np.ones_like(mu)))
            mu = mu + sig * np.random.default_rng(seed).standard_normal(mu.shape)
        if skill_prior == "walk":
            params["player_skill_eps"] = jnp.asarray(mu)
            if "skill_walk_sigma_loc" in params:
                params["skill_walk_sigma"] = params["skill_walk_sigma_loc"]
        else:
            params["player_skills"] = jnp.asarray(mu)

    # The NumPyro model samples a global player-skill latent whenever it is not
    # substituted.  In a PA-by-PA simulator that used to redraw the *global*
    # quantity for every PA, which is both slow and incoherent.  Materialize one
    # world-level draw here instead, allowing the cached pure-inference path below
    # to reuse its player embeddings for the entire simulation.
    if skill_mode == "prior" and "player_mu" in params:
        skill_rng = np.random.default_rng(seed)
        if skill_prior == "walk" and "player_skill_eps" not in params:
            shape = np.asarray(params["player_mu"]).shape
            params["player_skill_eps"] = jnp.asarray(skill_rng.standard_normal(shape))
            if "skill_walk_sigma_loc" in params:
                params["skill_walk_sigma"] = params["skill_walk_sigma_loc"]
        elif skill_prior == "iso" and "player_skills" not in params:
            shape = np.asarray(params["player_mu"]).shape
            params["player_skills"] = jnp.asarray(skill_rng.standard_normal(shape))

    G = len(games)
    away_lineup = np.array([g["away_lineup"] for g in games], dtype=np.int64)  # (G,9)
    home_lineup = np.array([g["home_lineup"] for g in games], dtype=np.int64)
    P = int(np.asarray(pt["stats"]).shape[0])
    unknown_idx = int(pt.get("unknown_index", P))
    home_staff, home_staff_len = pad_staffs(games, "home_staff", unknown_idx)  # pitches top halves
    away_staff, away_staff_len = pad_staffs(games, "away_staff", unknown_idx)  # pitches bottom halves
    park = np.array([g["park"] for g in games], dtype=np.int64)
    engine = pt["_engine"]
    starter_pas, reliever_pas = pt["_hook_dists"]
    rng_np = np.random.default_rng(seed)

    # Standard PA models have no generated-history dependency.  Their logits can
    # therefore be evaluated without entering NumPyro on every PA.  Build the
    # immutable player/park tables once.  Custom model functions and unsupported
    # prior families retain the legacy path below.
    inference = None
    _fn = model_fn.func if isinstance(model_fn, partial) else model_fn
    _kw = dict(model_fn.keywords or {}) if isinstance(model_fn, partial) else {}
    if _fn is pa_model and not _kw.get("pitchformer", False):
        try:
            inference = build_pa_inference(
                params, pt,
                outcome_only=_kw.get("outcome_only", False),
                fatigue=_kw.get("fatigue", False),
                platoon=_kw.get("platoon", False),
                nested=_kw.get("nested", False),
                bilinear_rank=_kw.get("bilinear_rank", 0),
                skill_prior=_kw.get("skill_prior", skill_prior),
                season_base=_kw.get("season_base", 2015),
                n_seasons=_kw.get("n_seasons", 9),
                simulation_season=simulation_season,
            )
        except (KeyError, ValueError):
            # Checkpoints from experimental prior families can still be simulated
            # through the original, fully general NumPyro route.
            inference = None

    # Common-random-numbers state: one generator per game for the outcome stream
    # (gumbel + base-advancement uniform) and one for hooks. Games sharing a
    # crn_key are seeded identically, so they produce identical sequences and the
    # noise cancels in a scenario difference wherever the play matches.
    crn = crn_keys is not None
    if crn:
        crn_keys = np.asarray(crn_keys, dtype=np.int64)
        assert len(crn_keys) == G, "crn_keys must have one entry per game"
        out_rng = [np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=(int(k), 0)))
                   for k in crn_keys]
        hook_rng = [np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=(int(k), 1)))
                    for k in crn_keys]

    away_score = np.zeros(G)
    home_score = np.zeros(G)
    away_hits = np.zeros(G)
    home_hits = np.zeros(G)
    away9 = np.full(G, np.nan)   # score snapshot after 9 innings (pre-extras)
    home9 = np.full(G, np.nan)
    away_ptr = np.zeros(G, dtype=np.int64)
    home_ptr = np.zeros(G, dtype=np.int64)
    away_cyc = np.zeros((G, 9), dtype=np.int64)
    home_cyc = np.zeros((G, 9), dtype=np.int64)

    # Per-staff pitcher state: current index into staff, PAs faced by the
    # current pitcher, and the hook threshold (PAs) for the current pitcher.
    # With a fitted hazard model the starter is governed by P(pull|state), so its
    # threshold is unused (set to inf); relievers still use a sampled PAs threshold.
    def _staff_state():
        if no_bullpen:
            hook0 = np.full(G, np.inf)
        elif hook_model is not None:
            hook0 = np.full(G, np.inf)  # starter: hazard-driven, not threshold
        elif crn:
            hook0 = np.array([hook_rng[g].choice(starter_pas) for g in range(G)], np.float64)
        else:
            hook0 = rng_np.choice(starter_pas, size=G).astype(np.float64)
        return {
            "cur": np.zeros(G, dtype=np.int64),
            "pa": np.zeros(G, dtype=np.float64),
            "ra": np.zeros(G, dtype=np.float64),   # runs allowed by the current pitcher
            "hook": hook0,
        }

    home_ps = _staff_state()  # home staff state (faces away lineup)
    away_ps = _staff_state()

    # --- Pitchformer: PA history buffer for causal attention ---------------
    # When pitchformer=True, the model needs to see all previous PAs in the
    # game. We allocate a fixed-size buffer per game and fill it as PAs are
    # played. Each model call sends (B, t_max) where t_max is the furthest
    # any active game has progressed, and pa_valid masks unused positions.
    MAX_PAS = 120   # generous upper bound (~70 real PAs per game)
    if pitchformer:
        pa_hist = {
            "inning":           np.zeros((G, MAX_PAS), np.float32),
            "half":             np.zeros((G, MAX_PAS), np.float32),
            "outs":             np.zeros((G, MAX_PAS), np.float32),
            "base_state":       np.zeros((G, MAX_PAS), np.float32),
            "score_diff":       np.zeros((G, MAX_PAS), np.float32),
            "tto":              np.zeros((G, MAX_PAS), np.float32),
            "shift_restricted": np.zeros((G, MAX_PAS), np.float32),
            "pitch_clock":      np.zeros((G, MAX_PAS), np.float32),
            "pitcher_ids":      np.zeros((G, MAX_PAS), np.int64),
            "batter_ids":       np.zeros((G, MAX_PAS), np.int64),
            "park_ids":         np.zeros((G, MAX_PAS), np.int64),
            "pitch_count_game": np.zeros((G, MAX_PAS), np.float32),
            "pa_valid":         np.zeros((G, MAX_PAS), bool),
        }
        if skill_prior == "walk":
            pa_hist["season"] = np.full((G, MAX_PAS), simulation_season, np.int32)
        if platoon:
            pa_hist["bat_side"] = np.full((G, MAX_PAS), 0.5, np.float32)
            pa_hist["pit_hand"] = np.full((G, MAX_PAS), 0.5, np.float32)
        pa_step = np.zeros(G, dtype=np.int64)  # next write position per game

    game_over = np.zeros(G, dtype=bool)
    occ_on = 0.0
    occ_n = 0.0
    pcounts = np.zeros((P, 9), dtype=np.float64)
    runs_by_inning = np.zeros(9)   # innings 1-9 (shape test); extras tracked apart
    extra_runs = 0.0
    n_walkoffs = 0
    truncated = np.zeros(G, dtype=bool)
    truncated_half_innings = 0
    PITCHES_PER_PA = 3.9

    def play_half(inning, half, mask):
        """Play one half-inning for every game in `mask` (bool, (G,))."""
        nonlocal occ_on, occ_n, extra_runs, n_walkoffs, rng_key, truncated_half_innings
        if not mask.any():
            return
        walkoff = (not fixed_nine) and half == 1 and inning >= 9
        ghost = (not fixed_nine) and inning >= 10

        lineup = away_lineup if half == 0 else home_lineup
        ptr = away_ptr if half == 0 else home_ptr
        cyc = away_cyc if half == 0 else home_cyc
        ps = home_ps if half == 0 else away_ps            # fielding staff state
        staff = home_staff if half == 0 else away_staff
        staff_len = home_staff_len if half == 0 else away_staff_len
        bat_score = away_score if half == 0 else home_score
        fld_score = home_score if half == 0 else away_score

        outs = np.zeros(G, dtype=np.int64)
        bases = np.full(G, 2 if ghost else 0, dtype=np.int64)
        active = mask.copy()

        pa_number = 0
        while active.any() and (max_pa_per_half is None or pa_number < max_pa_per_half):
            pa_number += 1
            if not active.any():
                break
            idx = np.where(active)[0]

            # Pitching change: decide who is pulled, then bring in a fresh arm.
            if not no_bullpen:
                a = idx
                can = ps["cur"][a] + 1 < staff_len[a]
                if hook_model is not None:
                    # Starter governed by the fitted state hazard; relievers keep the
                    # sampled PAs threshold. TTO proxied from PAs faced (~9 per turn).
                    is_starter = ps["cur"][a] == 0
                    pas_a = ps["pa"][a]
                    tto_a = np.minimum(pas_a // 9 + 1, 3)
                    p_pull = starter_pull_prob(pas_a, np.full(len(a), inning), tto_a,
                                               ps["ra"][a], hook_model)
                    u_pull = (np.array([hook_rng[g].random() for g in a]) if crn
                              else rng_np.random(len(a)))
                    starter_fire = is_starter & (pas_a >= 1) & (u_pull < p_pull)
                    reliever_fire = (~is_starter) & (ps["pa"][a] >= ps["hook"][a])
                    fire = (starter_fire | reliever_fire) & can
                else:
                    fire = (ps["pa"][a] >= ps["hook"][a]) & can
                need = a[fire]
                if len(need):
                    ps["cur"][need] += 1
                    ps["pa"][need] = 0.0
                    ps["ra"][need] = 0.0
                    ps["hook"][need] = (np.array([hook_rng[g].choice(reliever_pas) for g in need])
                                        if crn else rng_np.choice(reliever_pas, size=len(need)))
                    cyc[need, :] = 0  # new pitcher: batting team TTO resets

            slot = ptr[idx]
            batter = lineup[idx, slot]
            pitcher = staff[idx, ps["cur"][idx]]
            cyc[idx, slot] += 1
            tto = np.minimum(cyc[idx, slot], 3)

            B = len(idx)

            # --- Compute per-PA feature values (shared by both paths) ------
            _inn_val = (inning - 1) / 8.0
            _outs_val = outs[idx] / 2.0
            _bs_val = bases[idx] / 7.0
            _sd_val = np.clip(bat_score[idx] - fld_score[idx], -10, 10) / 10.0
            _tto_val = tto / 3.0
            _pc_val = np.clip(ps["pa"][idx] * PITCHES_PER_PA / 120.0, 0, 1.5).astype(np.float32)

            if pitchformer:
                # Write current PA features into the history buffer.
                for i_loc, g_idx in enumerate(idx):
                    s = pa_step[g_idx]
                    pa_hist["inning"][g_idx, s] = _inn_val
                    pa_hist["half"][g_idx, s] = float(half)
                    pa_hist["outs"][g_idx, s] = _outs_val[i_loc]
                    pa_hist["base_state"][g_idx, s] = _bs_val[i_loc]
                    pa_hist["score_diff"][g_idx, s] = _sd_val[i_loc]
                    pa_hist["tto"][g_idx, s] = _tto_val[i_loc]
                    pa_hist["shift_restricted"][g_idx, s] = shift
                    pa_hist["pitch_clock"][g_idx, s] = clock
                    pa_hist["pitcher_ids"][g_idx, s] = pitcher[i_loc]
                    pa_hist["batter_ids"][g_idx, s] = batter[i_loc]
                    pa_hist["park_ids"][g_idx, s] = park[g_idx]
                    pa_hist["pitch_count_game"][g_idx, s] = _pc_val[i_loc]
                    pa_hist["pa_valid"][g_idx, s] = True
                    if platoon:
                        P_ = P  # avoid shadowing
                        b_known = (batter[i_loc] >= 0) & (batter[i_loc] < P_)
                        p_known = (pitcher[i_loc] >= 0) & (pitcher[i_loc] < P_)
                        pa_hist["bat_side"][g_idx, s] = (
                            np.asarray(pt["bat_hand"])[batter[i_loc]] if b_known else 0.5)
                        pa_hist["pit_hand"][g_idx, s] = (
                            np.asarray(pt["pit_hand"])[pitcher[i_loc]] if p_known else 0.5)

                # Build tb from the full history, padded to MAX_PAS so JAX
                # sees one fixed shape and compiles once (not per-T).
                steps = pa_step[idx] + 1       # per-game sequence length
                pos = (steps - 1).astype(np.int64)  # index of the current PA
                cols = np.arange(MAX_PAS)
                tb = {
                    "pa_valid":         jnp.array(pa_hist["pa_valid"][np.ix_(idx, cols)]),
                    "inning":           jnp.array(pa_hist["inning"][np.ix_(idx, cols)]),
                    "half":             jnp.array(pa_hist["half"][np.ix_(idx, cols)]),
                    "outs":             jnp.array(pa_hist["outs"][np.ix_(idx, cols)]),
                    "base_state":       jnp.array(pa_hist["base_state"][np.ix_(idx, cols)]),
                    "score_diff":       jnp.array(pa_hist["score_diff"][np.ix_(idx, cols)]),
                    "tto":              jnp.array(pa_hist["tto"][np.ix_(idx, cols)]),
                    "shift_restricted": jnp.array(pa_hist["shift_restricted"][np.ix_(idx, cols)]),
                    "pitch_clock":      jnp.array(pa_hist["pitch_clock"][np.ix_(idx, cols)]),
                    "pitcher_ids":      jnp.array(pa_hist["pitcher_ids"][np.ix_(idx, cols)]),
                    "batter_ids":       jnp.array(pa_hist["batter_ids"][np.ix_(idx, cols)]),
                    "park_ids":         jnp.array(pa_hist["park_ids"][np.ix_(idx, cols)]),
                    "pitch_count_game": jnp.array(pa_hist["pitch_count_game"][np.ix_(idx, cols)]),
                }
                if skill_prior == "walk":
                    tb["season"] = jnp.array(pa_hist["season"][np.ix_(idx, cols)])
                if platoon:
                    tb["bat_side"] = jnp.array(pa_hist["bat_side"][np.ix_(idx, cols)])
                    tb["pit_hand"] = jnp.array(pa_hist["pit_hand"][np.ix_(idx, cols)])

                rng_key, k = jax.random.split(rng_key)
                with nh.seed(rng_seed=k):
                    with nh.substitute(data=params):
                        with nh.trace() as tr:
                            model_fn(tb, pt, teacher_force=False)

                # Extract logits/value at each game's CURRENT position.
                if recal:
                    full_logits = np.array(tr["pa_outcome"]["fn"].logits)  # (B, MAX_PAS, 9)
                    logits_cur = full_logits[np.arange(B), pos, :]         # (B, 9)
                    logits_cur = (logits_cur + recal_scale * recal_vec) / recal_temp
                    gum = (np.stack([out_rng[g].gumbel(size=logits_cur.shape[1]) for g in idx])
                           if crn else rng_np.gumbel(size=logits_cur.shape))
                    oc = np.argmax(logits_cur + gum, axis=-1).astype(np.int64)
                else:
                    # Use NumPyro's traced sample at the current position so
                    # the JAX RNG stream is respected (preserves CRN pairing).
                    oc = np.array(tr["pa_outcome"]["value"])[np.arange(B), pos].astype(np.int64)

                # Advance the step counter AFTER extracting logits.
                pa_step[idx] += 1

            else:
                # Ordinary PA path.  The fast adapter accepts a single PA per
                # row and uses power-of-two buckets, preventing the shrinking
                # active set from creating a new JIT shape at nearly every PA.
                if platoon:
                    bat_known = (batter >= 0) & (batter < P)
                    pit_known = (pitcher >= 0) & (pitcher < P)
                    bat_side = np.full(B, 0.5, dtype=np.float32)
                    pit_side = np.full(B, 0.5, dtype=np.float32)
                    bat_side[bat_known] = np.asarray(pt["bat_hand"])[batter[bat_known]]
                    pit_side[pit_known] = np.asarray(pt["pit_hand"])[pitcher[pit_known]]
                else:
                    bat_side = pit_side = np.zeros(B, dtype=np.float32)

                if inference is not None:
                    M = bucket_size(B)

                    def _pad(values, dtype):
                        out = np.zeros(M, dtype=dtype)
                        out[:B] = values
                        return out

                    logits = np.asarray(inference.logits(
                        inning=np.full(M, _inn_val, np.float32),
                        half=np.full(M, float(half), np.float32),
                        outs=_pad(_outs_val, np.float32),
                        base_state=_pad(_bs_val, np.float32),
                        score_diff=_pad(_sd_val, np.float32),
                        tto=_pad(_tto_val, np.float32),
                        shift_restricted=np.full(M, shift, np.float32),
                        pitch_clock=np.full(M, clock, np.float32),
                        pitch_count_game=_pad(_pc_val, np.float32),
                        pitcher_ids=_pad(pitcher, np.int32),
                        batter_ids=_pad(batter, np.int32),
                        park_ids=_pad(park[idx], np.int32),
                        bat_side=_pad(bat_side, np.float32),
                        pit_hand=_pad(pit_side, np.float32),
                    ))[:B]
                    if recal:
                        logits = (logits + recal_scale * recal_vec) / recal_temp
                        gum = (np.stack([out_rng[g].gumbel(size=logits.shape[1]) for g in idx])
                               if crn else rng_np.gumbel(size=logits.shape))
                        oc = np.argmax(logits + gum, axis=-1).astype(np.int64)
                    else:
                        rng_key, k = jax.random.split(rng_key)
                        oc = np.asarray(jax.random.categorical(k, jnp.asarray(logits), axis=-1)).astype(np.int64)
                else:
                    # Legacy fallback for custom models and unsupported checkpoint
                    # variants.  Keeping it intact makes the optimization
                    # transparent to external callers that supply their own model.
                    tb = {
                        "pa_valid": jnp.ones((B, 1), bool),
                        "inning": jnp.full((B, 1), _inn_val, jnp.float32),
                        "half": jnp.full((B, 1), float(half), jnp.float32),
                        "outs": jnp.array(_outs_val[:, None], jnp.float32),
                        "base_state": jnp.array(_bs_val[:, None], jnp.float32),
                        "score_diff": jnp.array(_sd_val[:, None], jnp.float32),
                        "tto": jnp.array(_tto_val[:, None], jnp.float32),
                        "shift_restricted": jnp.full((B, 1), shift, jnp.float32),
                        "pitch_clock": jnp.full((B, 1), clock, jnp.float32),
                        "pitcher_ids": jnp.array(pitcher[:, None]),
                        "batter_ids": jnp.array(batter[:, None]),
                        "park_ids": jnp.array(park[idx][:, None]),
                        "pitch_count_game": jnp.array(_pc_val[:, None], jnp.float32),
                    }
                    if skill_prior == "walk":
                        tb["season"] = jnp.full((B, 1), simulation_season, jnp.int32)
                    if platoon:
                        tb["bat_side"] = jnp.array(bat_side[:, None])
                        tb["pit_hand"] = jnp.array(pit_side[:, None])
                    rng_key, k = jax.random.split(rng_key)
                    with nh.seed(rng_seed=k):
                        with nh.substitute(data=params):
                            with nh.trace() as tr:
                                model_fn(tb, pt, teacher_force=False)
                    if recal:
                        logits = (np.array(tr["pa_outcome"]["fn"].logits)[:, 0, :] + recal_scale * recal_vec) / recal_temp
                        gum = (np.stack([out_rng[g].gumbel(size=logits.shape[1]) for g in idx])
                               if crn else rng_np.gumbel(size=logits.shape))
                        oc = np.argmax(logits + gum, axis=-1).astype(np.int64)
                    else:
                        oc = np.array(tr["pa_outcome"]["value"])[:, 0].astype(np.int64)

            occ_on += (bases[idx] > 0).sum()
            occ_n += B
            known_batters = (batter >= 0) & (batter < P)
            np.add.at(pcounts, (batter[known_batters], oc[known_batters]), 1.0)
            # Base advancement: one per-game uniform from the same stream (drawn
            # after the gumbel, so the per-game order is fixed) feeds the engine's
            # CRN path; else the shared generator.
            u = np.array([out_rng[g].random() for g in idx]) if crn else None
            e = engine.sample(bases[idx], outs[idx], oc, rng_np, u=u)
            runs = e["runs"].astype(np.float64)

            if walkoff:
                runs = cap_walkoff_runs(bat_score[idx], fld_score[idx], runs, oc == HR_IDX)

            bat_score[idx] += runs
            hits = np.isin(oc, (PA_OUTCOME_IDX["1B"], PA_OUTCOME_IDX["2B"],
                                 PA_OUTCOME_IDX["3B"], PA_OUTCOME_IDX["HR"])).astype(float)
            if half == 0:
                away_hits[idx] += hits
            else:
                home_hits[idx] += hits
            if inning <= 9:
                runs_by_inning[inning - 1] += runs.sum()
            else:
                extra_runs += runs.sum()

            oi = e["out_inc"]
            new_outs = outs[idx] + oi
            third = new_outs >= 3
            bases[idx] = np.where(third, 0, e["bs_after"])
            outs[idx] = new_outs
            ptr[idx] = (slot + 1) % 9
            ps["pa"][idx] += 1
            ps["ra"][idx] += runs   # runs allowed by the current fielding pitcher
            still = new_outs < 3
            if walkoff:
                won = bat_score[idx] > fld_score[idx]
                n_walkoffs += int(won.sum())
                game_over[idx[won]] = True
                still = still & ~won
            active[idx] = still

        if active.any():
            truncated[active] = True
            truncated_half_innings += int(active.sum())

    n_extra_games = 0
    last_inning = 9 if fixed_nine else MAX_INNINGS
    for inning in range(1, last_inning + 1):
        playing = ~game_over
        if not playing.any():
            break
        if inning >= 10:
            n_extra_games += int(playing.sum())

        play_half(inning, 0, playing)

        if not fixed_nine and inning >= 9:
            # Home leads after the top: no bottom half needed.
            decided = playing & (home_score > away_score)
            game_over[decided] = True

        play_half(inning, 1, playing & ~game_over)

        if not fixed_nine and inning >= 9:
            game_over[playing & (home_score != away_score)] = True

        if inning == 9:
            # Snapshot the score at the end of regulation for every game that has
            # reached the 9th (all of them), before any extra innings are added.
            reached = np.isnan(away9)
            away9[reached] = away_score[reached]
            home9[reached] = home_score[reached]

    n_ties = int((~game_over).sum()) if not fixed_nine else int((home_score == away_score).sum())

    return {
        "away": away_score,
        "home": home_score,
        "away_hits": away_hits,
        "home_hits": home_hits,
        "away9": away9,
        "home9": home9,
        "occ": occ_on / max(occ_n, 1),
        "pcounts": pcounts,
        "runs_by_inning": runs_by_inning,
        "extra_runs": extra_runs,
        "n_extra_games": n_extra_games,
        "n_walkoffs": n_walkoffs,
        "n_ties": n_ties,
        "truncated": truncated,
        "n_truncated": int(truncated.sum()),
        "truncated_half_innings": truncated_half_innings,
    }


def _rate_stats(c: np.ndarray) -> dict[str, float] | None:
    """length-9 outcome counts -> rate stats. Order K,BB,HBP,1B,2B,3B,HR,out,E."""
    pa = c.sum()
    if pa == 0:
        return None
    h = c[3] + c[4] + c[5] + c[6]
    ab = max(pa - c[1] - c[2], 1)
    tb = c[3] + 2 * c[4] + 3 * c[5] + 4 * c[6]
    return {"AVG": h / ab, "OBP": (h + c[1] + c[2]) / pa, "SLG": tb / ab,
            "K%": c[0] / pa, "HR%": c[6] / pa}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--outcome-only", action="store_true")
    ap.add_argument("--fatigue", action="store_true")
    ap.add_argument("--limit-games", type=int, default=512)
    ap.add_argument("--recal", action="store_true", help="Apply logit recalibration.")
    ap.add_argument("--recal-version", choices=list(RECAL_VECS), default="v6",
                    help="Which per-class recal vector to apply (v6 or v9). Must match the "
                         "checkpoint: v9's fatigue+park model has its own, milder calibration.")
    ap.add_argument("--recal-scale", type=float, default=0.3,
                    help="Pinned v6-final calibration strength (0.3 matches the FULL test-set "
                         "run rate: 8.81 vs 8.86 real. 0.4 was tuned on a biased subset).")
    ap.add_argument("--recal-file", type=Path, default=None,
                    help="Load a fitted per-class recal vector from npz (overrides --recal-version). "
                         "Key set by --recal-key (default 'b'). From fit_calibration.py.")
    ap.add_argument("--recal-key", type=str, default="b",
                    help="Array key in --recal-file to use as the recal vector (e.g. b, b_heur).")
    ap.add_argument("--recal-temp", type=float, default=1.0,
                    help="Temperature for learned calibration: logits=(logits+scale*vec)/T.")
    ap.add_argument("--skill-mode", choices=["prior", "mean", "sample"], default="prior",
                    help="Latent skill at eval: prior=sample N(0,1) (default, legacy); "
                         "mean=use posterior mean player_mu; sample=one posterior draw.")
    ap.add_argument("--recency-halflife", type=float, default=None,
                    help="Match a recency-trained model (v12+): same half-life as training.")
    ap.add_argument("--fixed-nine", action="store_true",
                    help="Legacy v1 structure: fixed 9 innings, no walk-offs/extras.")
    ap.add_argument("--no-bullpen", action="store_true",
                    help="Legacy v1 pitching: the starter pitches the whole game.")
    ap.add_argument("--platoon", action="store_true",
                    help="Feed batter side + pitcher throw hand (v11+ platoon models).")
    ap.add_argument("--pitchformer", action="store_true",
                    help="Use the causal PA-level transformer for context (must match "
                         "the checkpoint's training flag).")
    ap.add_argument("--pitchformer-dim", type=int, default=128,
                    help="PA transformer hidden width; must match the checkpoint.")
    ap.add_argument("--pitchformer-layers", type=int, default=2)
    ap.add_argument("--pitchformer-heads", type=int, default=4)
    ap.add_argument("--pitchformer-dropout", type=float, default=0.0)
    ap.add_argument("--use-park", action="store_true",
                    help="Feed real park indices (v9+ checkpoints trained with the "
                         "park_idx fix; pre-v9 park embeddings trained on all-zeros).")
    ap.add_argument("--player-stats", action="store_true",
                    help="Also report per-player stat-line reproduction (Phase 3, via true sim).")
    ap.add_argument("--min-pa", type=int, default=150)
    ap.add_argument("--dump-runs", type=Path, default=None,
                    help="Save per-game total runs (away+home) to this .npy for baseline comparison.")
    ap.add_argument("--dump-scores", type=Path, default=None,
                    help="Save per-game away/home final and after-9 scores to this .npz "
                         "(for tie/margin/extras diagnostics).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-pa-per-half", type=int, default=40,
                    help="Safety cap for generated half-innings; 0 disables truncation.")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    with open(args.ckpt, "rb") as f:
        params = pickle.load(f)["params"]
    train_pitches = load_seasons(TRAIN, data_root=processed_root())
    ptab = _build_player_table(train_pitches, recency_halflife=args.recency_halflife)
    park_map = _build_park_index(train_pitches) if args.use_park else None
    train_pa = train_pitches.filter(pl.col("pa_terminal"))
    engine = EmpiricalEngine().fit(train_pa)
    hook_dists = fit_hook_dists(train_pa)
    print(f"Hook dists: starter median {np.median(hook_dists[0]):.0f} PA, "
          f"reliever median {np.median(hook_dists[1]):.0f} PA", flush=True)
    del train_pitches, train_pa

    test_pa = load_seasons(TEST, data_root=processed_root()).filter(pl.col("pa_terminal"))
    keep = test_pa["game_pk"].unique().sort().to_numpy()[:args.limit_games]
    test_pa = test_pa.filter(pl.col("game_pk").is_in(keep.tolist()))

    # real reference
    rg = (test_pa.group_by(["game_pk", "half_bin"]).agg(pl.col("runs_scored").sum().alias("r")))
    real_total = rg["r"].sum() / test_pa["game_pk"].n_unique()
    real_occ = (test_pa.with_columns((pl.col("base_state") > 0).cast(pl.Int32).alias("o"))["o"].mean()) * 100
    real_extra_rate = (
        test_pa.group_by("game_pk").agg(pl.col("inning").max().alias("mi"))
        .filter(pl.col("mi") >= 10)["mi"].len()
    ) / test_pa["game_pk"].n_unique() * 100

    print(f"Extracting lineups for {test_pa['game_pk'].n_unique()} games...", flush=True)
    games = extract_games(
        test_pa, ptab["id_to_idx"], park_map=park_map,
        unknown_idx=ptab["unknown_index"],
    )
    print(f"  {len(games)} games usable", flush=True)

    pt = {"stats": jnp.array(ptab["stats"]), "league": jnp.array(ptab["league"]),
          "hand": jnp.array(ptab["hand"]),
          "unknown_index": ptab["unknown_index"],
          "bat_hand": np.asarray(ptab["bat_hand"]), "pit_hand": np.asarray(ptab["pit_hand"]),
          "_engine": engine, "_hook_dists": hook_dists}
    mkw = {}
    if args.outcome_only:
        mkw["outcome_only"] = True
    if args.fatigue:
        mkw["fatigue"] = True
    if args.platoon:
        mkw["platoon"] = True
    if args.pitchformer:
        mkw.update(
            pitchformer=True,
            pitchformer_dim=args.pitchformer_dim,
            pitchformer_layers=args.pitchformer_layers,
            pitchformer_heads=args.pitchformer_heads,
            pitchformer_dropout=args.pitchformer_dropout,
        )
    model_fn = partial(pa_model, **mkw) if mkw else pa_model

    if args.recal_file is not None:
        _recal_vec = np.load(args.recal_file)[args.recal_key].astype(np.float64)
        print(f"  recal from {args.recal_file} [{args.recal_key}] T={args.recal_temp}", flush=True)
    else:
        _recal_vec = RECAL_VECS[args.recal_version]

    if args.max_pa_per_half < 0:
        ap.error("--max-pa-per-half must be non-negative")
    t0 = time.time()
    res = simulate(
        model_fn, params, pt, games, jax.random.PRNGKey(args.seed),
        recal=args.recal, recal_scale=args.recal_scale, recal_vec=_recal_vec,
        fixed_nine=args.fixed_nine, no_bullpen=args.no_bullpen, seed=args.seed,
        platoon=args.platoon, recal_temp=args.recal_temp, skill_mode=args.skill_mode,
        max_pa_per_half=None if args.max_pa_per_half == 0 else args.max_pa_per_half,
        pitchformer=args.pitchformer,
    )
    away, home, total = res["away"], res["home"], res["away"] + res["home"]
    G = len(games)

    # Real runs-by-inning (innings 1-9), as fraction of total (shape, calibration-free)
    rdf = (test_pa.filter((pl.col("inning") >= 1) & (pl.col("inning") <= 9))
           .group_by("inning").agg(pl.col("runs_scored").sum().alias("r")).sort("inning"))
    real_rbi = np.zeros(9)
    for row in rdf.iter_rows(named=True):
        real_rbi[int(row["inning"]) - 1] = row["r"]
    real_frac = real_rbi / real_rbi.sum()
    sim_frac = res["runs_by_inning"] / max(res["runs_by_inning"].sum(), 1)
    print(f"\n=== runs-by-inning fraction (fatigue shape test) ===", flush=True)
    print(f"  inning   1    2    3    4    5    6    7    8    9   | late(6-9)", flush=True)
    print(f"  real  " + " ".join(f"{x*100:4.1f}" for x in real_frac) + f"  | {real_frac[5:].sum()*100:.1f}%", flush=True)
    print(f"  sim   " + " ".join(f"{x*100:4.1f}" for x in sim_frac) + f"  | {sim_frac[5:].sum()*100:.1f}%", flush=True)

    mode = []
    if args.fixed_nine:
        mode.append("fixed-nine")
    if args.no_bullpen:
        mode.append("no-bullpen")
    if args.use_park:
        mode.append("park")
    print(f"\n=== TRUE SIMULATION v2 ({G} games, {time.time()-t0:.0f}s"
          + (f", {'+'.join(mode)}" if mode else "") + ") ===", flush=True)
    print(f"  runs/game   real {real_total:.2f}   sim {total.mean():.2f}", flush=True)
    print(f"  occupancy   real {real_occ:.1f}%   sim {res['occ']*100:.1f}%", flush=True)
    print(f"  home runs/g {home.mean():.2f}   away {away.mean():.2f}   "
          f"home-win {np.mean(home > away)*100:.1f}%  (ties {res['n_ties']})", flush=True)
    print(f"  extras      real {real_extra_rate:.1f}%   sim {res['n_extra_games']/G*100:.1f}%   "
          f"walk-offs {res['n_walkoffs']/G*100:.1f}%   extra-inning runs {res['extra_runs']/G:.2f}/g", flush=True)
    if res["n_truncated"]:
        print(f"  WARNING: truncated {res['n_truncated']}/{G} games across "
              f"{res['truncated_half_innings']} half-innings; exclude these draws or rerun "
              "with --max-pa-per-half 0.", flush=True)

    if args.dump_runs is not None:
        np.save(args.dump_runs, total)
        print(f"  saved per-game runs -> {args.dump_runs}", flush=True)

    if args.dump_scores is not None:
        np.savez(args.dump_scores, away=res["away"], home=res["home"],
                 away9=res["away9"], home9=res["home9"])
        print(f"  saved per-game scores -> {args.dump_scores}", flush=True)

    if args.player_stats:
        P = int(np.asarray(pt["stats"]).shape[0])
        real_counts = np.zeros((P, 9))
        bcol = "batter_id" if "batter_id" in test_pa.columns else "batter_idx"
        for bid, oc in test_pa.select([bcol, "pa_outcome"]).iter_rows():
            if oc in PA_OUTCOME_IDX and int(bid) in ptab["id_to_idx"]:
                real_counts[ptab["id_to_idx"][int(bid)], PA_OUTCOME_IDX[oc]] += 1
        keep_idx = np.where(real_counts.sum(1) >= args.min_pa)[0]
        mets = ["AVG", "OBP", "SLG", "K%", "HR%"]
        rv = {m: [] for m in mets}
        sv = {m: [] for m in mets}
        for i in keep_idx:
            rs, ss = _rate_stats(real_counts[i]), _rate_stats(res["pcounts"][i])
            if rs and ss:
                for m in mets:
                    rv[m].append(rs[m])
                    sv[m].append(ss[m])
        print(f"\n=== Player-level via TRUE SIM ({len(rv['AVG'])} batters >= {args.min_pa} PA) ===", flush=True)
        print(f"  {'stat':5s} {'real':>8s} {'sim':>8s} {'MAE':>8s} {'corr':>7s}", flush=True)
        for m in mets:
            a, b = np.array(rv[m]), np.array(sv[m])
            corr = np.corrcoef(a, b)[0, 1] if len(a) > 1 else float("nan")
            print(f"  {m:5s} {a.mean():8.4f} {b.mean():8.4f} {np.abs(a-b).mean():8.4f} {corr:7.3f}", flush=True)


if __name__ == "__main__":
    main()
