"""Teacher-forced, posterior-predictive, and stateful rollout utilities.

``free_rollout_samples`` has two supported call forms:

* the original posterior-predictive form used by ``eval_v0.py``;
* a stateful predictor form used for generated, autoregressive pitch rollouts.

Keeping both forms in one public function preserves the older evaluation API while
making the generated-state contract explicit and testable.
"""
from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import Predictive

from diamondworldjax.domain import ArrayGameState, PitchResult, apply_pitch_result, batting_score_diff


class StepDistribution(NamedTuple):
    """Logits emitted by one stateful predictor step.

    Every tensor has leading shape ``(B,)`` except categorical logits, which have
    shape ``(B, n_classes)``.  The outcome and transition fields are consumed by
    :func:`apply_pitch_result`, so counts, bases, inning changes, and score updates
    are derived from the generated state rather than a supplied test-game sequence.
    """

    pitch_type_logits: jnp.ndarray
    swing_logits: jnp.ndarray
    called_strike_logits: jnp.ndarray
    contact_logits: jnp.ndarray
    foul_logits: jnp.ndarray
    runs_logits: jnp.ndarray
    base_state_logits: jnp.ndarray
    outs_added_logits: jnp.ndarray
    pa_outcome_logits: jnp.ndarray


def initial_game_state(batch_size: int) -> ArrayGameState:
    """Return an empty top-of-first game state for ``batch_size`` simulations."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    zeros = jnp.zeros((batch_size,), dtype=jnp.int32)
    return ArrayGameState(
        inning=jnp.ones((batch_size,), dtype=jnp.int32),
        half=zeros,
        balls=zeros,
        strikes=zeros,
        outs=zeros,
        base_state=zeros,
        home_score=zeros,
        away_score=zeros,
        pitch_count_game=zeros,
        pitch_count_inning=zeros,
        pitch_count_pa=zeros,
        tto=jnp.ones((batch_size,), dtype=jnp.int32),
    )


def _state_features(state: ArrayGameState) -> jnp.ndarray:
    """Normalised model features for a generated game state."""
    batting_diff = jnp.where(
        state.half == 0,
        state.away_score - state.home_score,
        state.home_score - state.away_score,
    )
    return jnp.stack(
        [
            (state.inning - 1) / 8.0,
            state.half,
            state.balls / 3.0,
            state.strikes / 2.0,
            state.outs / 2.0,
            state.base_state / 7.0,
            batting_diff / 10.0,
            state.pitch_count_game / 100.0,
            state.pitch_count_inning / 30.0,
            state.pitch_count_pa / 10.0,
            state.tto / 3.0,
        ],
        axis=-1,
    ).astype(jnp.float32)


def _sample_step(
    predictor: Callable,
    params: Any,
    state: ArrayGameState,
    history: jnp.ndarray,
    history_mask: jnp.ndarray,
    exogenous: Any,
    rng_key: jax.Array,
) -> tuple[ArrayGameState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate and apply one pitch, returning state and observable samples."""
    distn = predictor(params, _state_features(state), history, history_mask, exogenous)
    keys = jax.random.split(rng_key, 9)
    pitch_type = jax.random.categorical(keys[0], distn.pitch_type_logits, axis=-1)
    swing = jax.random.bernoulli(keys[1], jax.nn.sigmoid(distn.swing_logits))
    called_strike = jax.random.bernoulli(keys[2], jax.nn.sigmoid(distn.called_strike_logits))
    contact = jax.random.bernoulli(keys[3], jax.nn.sigmoid(distn.contact_logits))
    foul = jax.random.bernoulli(keys[4], jax.nn.sigmoid(distn.foul_logits))
    runs_scored = jax.random.categorical(keys[5], distn.runs_logits, axis=-1)
    base_state_after = jax.random.categorical(keys[6], distn.base_state_logits, axis=-1)
    outs_added = jax.random.categorical(keys[7], distn.outs_added_logits, axis=-1)
    pa_outcome = jax.random.categorical(keys[8], distn.pa_outcome_logits, axis=-1)

    stepped = apply_pitch_result(
        state,
        PitchResult(
            swing=swing,
            called_strike=called_strike,
            contact=contact,
            foul=foul,
            runs_scored=runs_scored,
            base_state_after=base_state_after,
            outs_added=outs_added,
            pa_outcome=pa_outcome,
        ),
    )
    return (
        stepped.state,
        stepped.pa_terminal,
        stepped.outcome,
        stepped.state.base_state,
        pitch_type,
        stepped.game_over,
    )


def _keep_active_state(
    previous: ArrayGameState,
    updated: ArrayGameState,
    active: jnp.ndarray,
) -> ArrayGameState:
    """Apply an update only to games that had not already finished."""
    return ArrayGameState(*[jnp.where(active, new, old) for old, new in zip(previous, updated)])


def _stateful_free_rollout_samples(
    predictor: Callable,
    params: Any,
    state: ArrayGameState,
    rng_key: jax.Array,
    *,
    num_samples: int = 1,
    max_steps: int = 400,
    exogenous: Any = None,
) -> dict:
    """Generate pitch sequences while feeding each generated state into the next step."""
    if num_samples < 1 or max_steps < 1:
        raise ValueError("num_samples and max_steps must be positive")

    batch_size = state.inning.shape[0]
    per_sample: list[dict[str, jnp.ndarray]] = []
    sample_keys = jax.random.split(rng_key, num_samples)
    for sample_key in sample_keys:
        current = state
        # History is intentionally generic: predictors may consume its generated
        # pitch tokens, while simple predictors can ignore it.
        history = jnp.zeros((batch_size, 0, 5), dtype=jnp.float32)
        history_mask = jnp.zeros((batch_size, 0), dtype=bool)
        records: dict[str, list[jnp.ndarray]] = {
            "balls": [], "strikes": [], "outs": [], "base_state": [],
            "runs_scored": [], "pa_terminal": [], "pa_outcome": [],
            "home_score": [], "away_score": [], "inning": [], "half": [], "game_over": [],
        }
        game_over = jnp.zeros((batch_size,), dtype=bool)
        keys = jax.random.split(sample_key, max_steps)
        for step_key in keys:
            previous_home, previous_away = current.home_score, current.away_score
            updated, terminal, outcome, _, pitch_type, ended = _sample_step(
                predictor, params, current, history, history_mask, exogenous, step_key
            )
            active = ~game_over
            current = _keep_active_state(current, updated, active)
            terminal = terminal & active
            outcome = jnp.where(active, outcome, -1)
            game_over = game_over | (ended & active)
            runs = (current.home_score - previous_home) + (current.away_score - previous_away)
            records["balls"].append(current.balls)
            records["strikes"].append(current.strikes)
            records["outs"].append(current.outs)
            records["base_state"].append(current.base_state)
            records["runs_scored"].append(runs)
            records["pa_terminal"].append(terminal)
            records["pa_outcome"].append(outcome)
            records["home_score"].append(current.home_score)
            records["away_score"].append(current.away_score)
            records["inning"].append(current.inning)
            records["half"].append(current.half)
            records["game_over"].append(game_over)

            token = jnp.stack(
                [
                    pitch_type.astype(jnp.float32),
                    current.strikes.astype(jnp.float32),
                    current.outs.astype(jnp.float32),
                    current.base_state.astype(jnp.float32),
                    terminal.astype(jnp.float32),
                ],
                axis=-1,
            )
            history = jnp.concatenate([history, token[:, None, :]], axis=1)
            history_mask = jnp.concatenate(
                [history_mask, active[:, None]], axis=1
            )
        per_sample.append({name: jnp.stack(values, axis=0) for name, values in records.items()})

    return {
        name: jnp.stack([sample[name] for sample in per_sample], axis=0)
        for name in per_sample[0]
    }


# ---------------------------------------------------------------------------
# Teacher-forced rollout
# ---------------------------------------------------------------------------

def teacher_forced_samples(
    model: Callable,
    guide: Any,
    params: dict,
    batch: dict,
    player_table: dict,
    rng_key: jax.Array,
    num_samples: int = 1,
) -> dict:
    """
    Run posterior predictive sampling with observed pitch data as conditioning.

    Useful for calibration: the model sees all real observations (teacher_force=True)
    and we examine what it samples at each site.

    Returns
    -------
    dict mapping site name → array of shape (num_samples, B, T, ...)
    """
    predictive = Predictive(
        model,
        guide      = guide,
        params     = params,
        num_samples= num_samples,
        return_sites = _ALL_OBSERVABLE_SITES,
    )
    return predictive(
        rng_key,
        batch,
        player_table,
        teacher_force = True,
    )


# ---------------------------------------------------------------------------
# Free rollout (game simulation)
# ---------------------------------------------------------------------------

def free_rollout_samples(
    model_or_predictor: Callable,
    guide_or_params: Any,
    params_or_state: Any,
    batch_or_rng_key: Any,
    player_table: dict | None = None,
    rng_key: jax.Array | None = None,
    num_samples: int = 1,
    max_steps: int | None = None,
    exogenous: Any = None,
) -> dict:
    """
    Draw free samples from either supported rollout interface.

    Stateful form::

        free_rollout_samples(predictor, params, initial_state, rng_key,
                             num_samples=..., max_steps=...)

    calls ``predictor`` once per generated pitch and feeds the state produced by
    that pitch into the next call.  It returns arrays shaped ``(S, steps, B)``.

    Posterior-predictive form (kept for existing callers)::

        free_rollout_samples(model, guide, params, batch, player_table, rng_key,
                             num_samples=...)

    This retains its original behaviour and output layout ``(S, B, T, ...)``.

    Returns
    -------
    dict mapping site name → array of shape (num_samples, B, T, ...)
    """
    if isinstance(params_or_state, ArrayGameState):
        if max_steps is None:
            raise ValueError("stateful free_rollout_samples requires max_steps")
        return _stateful_free_rollout_samples(
            model_or_predictor,
            guide_or_params,
            params_or_state,
            batch_or_rng_key,
            num_samples=num_samples,
            max_steps=max_steps,
            exogenous=exogenous,
        )

    if player_table is None or rng_key is None:
        raise TypeError(
            "posterior-predictive free_rollout_samples requires model, guide, params, "
            "batch, player_table, and rng_key"
        )
    predictive = Predictive(
        model_or_predictor,
        guide      = guide_or_params,
        params     = params_or_state,
        num_samples= num_samples,
        return_sites = _ALL_OBSERVABLE_SITES,
    )
    return predictive(
        rng_key,
        batch_or_rng_key,
        player_table,
        teacher_force = False,
    )


def autoregressive_joint_rollout_samples(
    model: Callable,
    guide: Any,
    params: dict,
    batch: dict,
    player_table: dict,
    rng_key: jax.Array,
    num_samples: int = 1,
) -> dict:
    """Generate the joint pitch model without replaying observed pitch history.

    The legacy posterior-predictive route calls the model once on a complete
    recorded game, which makes position ``t`` read the real pitches at
    ``< t``.  This routine instead calls the model once per pitch, writes the
    sampled pitch back into its history buffer, and advances balls, strikes,
    bases, outs, scores, and pitch counts with :func:`apply_pitch_result`.

    Player/park IDs and rule-era settings remain exogenous.  This is therefore a
    generated *pitch and game-state* rollout on an observed roster schedule, not
    yet a lineup/roster generator.
    """
    import numpy as np
    import numpyro.handlers as handlers

    valid = np.asarray(batch["pitch_valid"], dtype=bool)
    B, T = valid.shape
    if num_samples < 1:
        raise ValueError("num_samples must be positive")

    # Only exogenous covariates survive from the supplied recorded batch. Every
    # generated pitch/state field starts neutral and is filled causally below.
    preserve = {"pitch_valid", "pitcher_ids", "batter_ids", "park_ids", "game_ids",
                "shift_restricted", "pitch_clock"}
    template: dict[str, np.ndarray] = {}
    for name, value in batch.items():
        array = np.asarray(value)
        if name in preserve or array.shape != (B, T):
            template[name] = array.copy()
        else:
            template[name] = np.zeros_like(array)

    def _draw_player_skills(key):
        with handlers.seed(rng_seed=key):
            with handlers.substitute(data=params):
                with handlers.trace() as trace:
                    guide(batch, player_table, teacher_force=False)
        try:
            return trace["player_skills"]["value"]
        except KeyError as exc:
            raise ValueError(
                "joint autoregressive rollout requires a guide with a player_skills site"
            ) from exc

    def _write_state(rolling: dict[str, np.ndarray], state: ArrayGameState, t: int):
        rolling["inning"][:, t] = np.asarray((state.inning - 1) / 8.0)
        rolling["half"][:, t] = np.asarray(state.half)
        rolling["balls"][:, t] = np.asarray(state.balls / 3.0)
        rolling["strikes"][:, t] = np.asarray(state.strikes / 2.0)
        rolling["outs"][:, t] = np.asarray(state.outs / 2.0)
        rolling["base_state"][:, t] = np.asarray(state.base_state / 7.0)
        rolling["score_diff"][:, t] = np.asarray(batting_score_diff(state) / 10.0)
        rolling["pitch_count_game"][:, t] = np.asarray(state.pitch_count_game / 120.0)
        rolling["pitch_count_inning"][:, t] = np.asarray(state.pitch_count_inning / 30.0)
        rolling["pitch_count_pa"][:, t] = np.asarray(state.pitch_count_pa / 10.0)
        rolling["tto"][:, t] = np.asarray(state.tto / 3.0)

    samples: list[dict[str, np.ndarray]] = []
    sample_keys = jax.random.split(rng_key, num_samples)
    for sample_key in sample_keys:
        sample_key, guide_key = jax.random.split(sample_key)
        player_skills = _draw_player_skills(guide_key)
        rolling = {name: value.copy() for name, value in template.items()}
        state = initial_game_state(B)
        game_over = jnp.zeros((B,), dtype=bool)
        record_dtypes = {
            "pitch_type": batch["pitch_type"].dtype,
            "plate_x": batch["plate_x"].dtype,
            "plate_z": batch["plate_z"].dtype,
            "release_speed": batch["release_speed"].dtype,
            "swing": batch["obs_swing"].dtype,
            "called_strike": batch["obs_called_strike"].dtype,
            "contact": batch["obs_contact"].dtype,
            "foul": batch["obs_foul"].dtype,
            "launch_speed": batch["launch_speed"].dtype,
            "launch_angle": batch["launch_angle"].dtype,
            "spray_angle": batch["spray_angle"].dtype,
            "hit_distance": batch["hit_distance"].dtype,
            "pa_outcome": batch["pa_outcome"].dtype,
            "runs_scored": batch["runs_scored"].dtype,
            "base_state_after": batch["base_state_after"].dtype,
            "outs_added": batch["outs_added"].dtype,
        }
        records = {name: np.zeros((B, T), dtype=dtype) for name, dtype in record_dtypes.items()}
        records["pa_terminal"] = np.zeros((B, T), dtype=bool)

        for t in range(T):
            active = jnp.asarray(valid[:, t]) & ~game_over
            if not bool(np.any(np.asarray(active))):
                continue
            _write_state(rolling, state, t)
            sample_key, step_key = jax.random.split(sample_key)
            model_batch = {name: jnp.asarray(value) for name, value in rolling.items()}
            with handlers.seed(rng_seed=step_key):
                with handlers.substitute(data={**params, "player_skills": player_skills}):
                    with handlers.trace() as trace:
                        model(model_batch, player_table, teacher_force=False)

            def value(name):
                return jnp.asarray(trace[name]["value"])[:, t]

            pitch_type = value("pitch_type")
            plate_x = value("plate_x")
            plate_z = value("plate_z")
            release_speed = value("release_speed")
            swing, called_strike = value("swing"), value("called_strike")
            contact, foul = value("contact"), value("foul")
            runs_scored = value("runs_scored")
            base_after, outs_added = value("base_state_after"), value("outs_added")
            pa_outcome = value("pa_outcome")
            previous_state = state
            stepped = apply_pitch_result(
                state,
                PitchResult(swing, called_strike, contact, foul, runs_scored,
                            base_after, outs_added, pa_outcome),
            )
            state = _keep_active_state(state, stepped.state, active)
            terminal = stepped.pa_terminal & active
            game_over = game_over | (stepped.game_over & active)

            generated = {
                "pitch_type": pitch_type, "plate_x": plate_x, "plate_z": plate_z,
                "release_speed": release_speed,
            }
            for name, generated_value in generated.items():
                previous = jnp.asarray(rolling[name][:, t])
                rolling[name][:, t] = np.asarray(jnp.where(active, generated_value, previous))
            # The model has no movement generator. Neutral movement is already
            # present in the buffer and avoids carrying observed pfx values.

            for name in records:
                if name == "pa_terminal":
                    records[name][:, t] = np.asarray(terminal)
                elif name == "runs_scored":
                    # Record realised scoreboard movement, not a raw transition
                    # draw that might be irrelevant on a non-terminal pitch.
                    records[name][:, t] = np.asarray(
                        (state.home_score + state.away_score)
                        - (previous_state.home_score + previous_state.away_score)
                    )
                elif name == "pa_outcome":
                    records[name][:, t] = np.asarray(jnp.where(terminal, stepped.outcome, -1))
                else:
                    records[name][:, t] = np.asarray(jnp.where(active, value(name), 0))
        samples.append(records)
    return {name: jnp.asarray(np.stack([sample[name] for sample in samples], axis=0))
            for name in samples[0]}


# ---------------------------------------------------------------------------
# Game-level aggregate: extract run totals from a free rollout
# ---------------------------------------------------------------------------

def extract_game_runs(
    samples: dict,
    terminal_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Sum runs_scored at terminal PAs per game per sample.

    Parameters
    ----------
    samples       : output of free_rollout_samples
    terminal_mask : (B, T) — which positions are PA terminals for the
        posterior-predictive layout. Omit it for stateful rollout samples, which
        carry their generated ``pa_terminal`` values.

    Returns
    -------
    game_runs : (num_samples, B) int — total runs per game per sample
    """
    # runs_scored site: (num_samples, B, T) int
    runs = samples.get("runs_scored")
    if runs is None:
        raise KeyError("'runs_scored' not found in samples dict.")

    if terminal_mask is None:
        generated_terminal = samples.get("pa_terminal")
        if generated_terminal is None:
            raise ValueError("terminal_mask is required for samples without 'pa_terminal'.")
        # Stateful layout is (S, steps, B), unlike NumPyro Predictive's (S, B, T).
        return (runs * generated_terminal).sum(axis=1)

    mask = terminal_mask[None, :, :]          # (1, B, T)
    return (runs * mask).sum(axis=-1)         # (num_samples, B)


def simulate_season_runs(
    model: Callable,
    guide: Any,
    params: dict,
    season_batches: list[dict],
    player_table: dict,
    seed: int = 0,
) -> jnp.ndarray:
    """
    Simulate full-season run totals by iterating over all game batches.

    Returns
    -------
    all_runs : (total_games,) float  — one total run value per game,
               averaged over draws.
    """
    rng = jax.random.PRNGKey(seed)
    all_runs = []

    for batch in season_batches:
        rng, key = jax.random.split(rng)
        samples = free_rollout_samples(
            model, guide, params, batch, player_table, key, num_samples=8
        )
        terminal_mask = batch["terminal_mask"]
        game_runs = extract_game_runs(samples, terminal_mask)  # (8, B)
        # Average over samples: (B,)
        all_runs.append(game_runs.mean(axis=0))

    return jnp.concatenate(all_runs, axis=0)


# ---------------------------------------------------------------------------
# Site names returned by the legacy posterior-predictive diagnostic
# ---------------------------------------------------------------------------

_ALL_OBSERVABLE_SITES = [
    # Hurdle
    "pitch_type", "plate_x", "plate_z", "release_speed",
    "swing", "called_strike", "contact", "foul",
    # Batted ball
    "launch_speed", "launch_angle", "spray_angle", "hit_distance",
    # Transition
    "pa_outcome", "runs_scored", "base_state_after", "outs_added",
    # Manager
    "pitching_change", "steal_attempt", "runner_send", "defensive_alignment",
]
