"""SVI training loop for DiamondWorldJAX.

The model is purely discriminative: all neural-network weights live in NumPyro
`param` sites (via flax_module), and every continuous `sample` site receives an
observed value during teacher-forcing.  A small number of discrete `sample`
sites (runner_send, defensive_alignment, called_strike, etc.) are left
unobserved; they are sampled in the ELBO estimate and contribute their
log-probability but do not affect downstream computations.

Guide: empty (no continuous latent variables to approximate).
Loss:  Trace_ELBO â€” sums log-likelihoods of observed sites.
Optimizer: Adam.
"""
from __future__ import annotations

import math
import json
import pickle
import time
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
import numpyro.handlers as nhandlers
import optax
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import optax_to_numpyro

DEFAULT_LR        = 1e-3
DEFAULT_STEPS     = 50_000
LOG_INTERVAL      = 500
CKPT_INTERVAL     = 5_000
GRAD_CLIP_VALUE   = 1.0   # per-element gradient clip threshold

# Must match SKILL_DIM in embeddings.py
SKILL_DIM = 32


def make_player_skills_guide(P: int, skill_dim: int = SKILL_DIM, kl_scale: float = 1.0,
                             skill_prior: str = "iso", n_seasons: int = 1):
    """Dispatch to the guide matching the model's skill prior.

    CRITICAL, and the source of a real bug. A NumPyro guide that omits a latent
    site does NOT raise. Under Trace_ELBO the missing site is simply drawn from
    its prior at every step, so it is never learned. That silently converts
    "give the model a learned prior hyperparameter" into "inject fresh noise into
    the prior on every step", which is a different experiment with a predictably
    worse outcome.

    This bit hard. The v17c (learned scale) and v17d (LKJ) runs were trained
    against the iso-only guide, so `skill_tau` and `skill_L` were resampled from
    their priors every step and never fitted. Their checkpoints contain no
    parameters for those sites, which is how it was caught. Both results measured
    prior NOISE rather than prior freedom, and both were retracted.

    `scripts/_check_guide_coverage.py` asserts coverage for every prior.
    """
    if skill_prior == "iso":
        return _guide_iso(P, skill_dim, kl_scale)
    if skill_prior in ("learned", "lkj"):
        return _guide_scaled(P, skill_dim, kl_scale, lkj=(skill_prior == "lkj"))
    if skill_prior == "walk":
        return _guide_walk(P, skill_dim, kl_scale, n_seasons)
    raise ValueError(f"unknown skill_prior {skill_prior!r}")


def make_shared_task_skills_guide(
    P: int,
    skill_dim: int = SKILL_DIM,
    kl_scale: float = 1.0,
    skill_prior: str = "iso",
    n_seasons: int = 1,
):
    """Mean-field guide for the shared-plus-task-residual hierarchy."""
    def _normal_site(name: str, scale: float):
        shape = ((P, skill_dim) if skill_prior == "iso"
                 else (P, n_seasons, skill_dim))
        mu = numpyro.param(
            f"{name}_mu", jnp.zeros(shape),
            constraint=constraints.interval(-5.0, 5.0),
        )
        sigma = numpyro.param(
            f"{name}_sigma", jnp.full(shape, scale),
            constraint=constraints.interval(0.05, 2.0),
        )
        numpyro.sample(name, dist.Normal(mu, sigma).to_event(len(shape)))

    def guide(batch, player_table, teacher_force=True):
        del batch, player_table, teacher_force
        if skill_prior not in {"iso", "walk"}:
            raise ValueError("shared task skills support only 'iso' and 'walk'")
        with nhandlers.scale(scale=kl_scale):
            if skill_prior == "walk":
                walk = numpyro.param(
                    "skill_walk_sigma_loc", jnp.asarray(0.3),
                    constraint=constraints.interval(0.01, 1.0),
                )
                numpyro.sample("skill_walk_sigma", dist.Delta(walk))
            _normal_site("shared_player_skills", 0.3)
            _normal_site("pa_skill_residual", 0.15)
            _normal_site("pitch_skill_residual", 0.15)
    return guide


def _guide_scaled(P: int, skill_dim: int, kl_scale: float, lkj: bool):
    """Guide for the learned-scale and LKJ priors.

    The per-player skill keeps its mean-field Normal posterior. The GLOBAL
    hyperparameters (tau, and the LKJ Cholesky factor) get point estimates via
    Delta, the standard treatment for a handful of globals shared across all
    players: there is ample data to pin them, and a Delta keeps the ELBO free of
    an extra KL that would need its own subsample scaling.
    """
    base = _guide_iso(P, skill_dim, kl_scale)

    def guide(batch, player_table, teacher_force=True):
        base(batch, player_table, teacher_force)
        tau = numpyro.param(
            "skill_tau_loc", jnp.ones(skill_dim),
            constraint=constraints.interval(0.05, 3.0),
        )
        with nhandlers.scale(scale=kl_scale):
            numpyro.sample("skill_tau", dist.Delta(tau).to_event(1))
            if lkj:
                L = numpyro.param(
                    "skill_L_loc", jnp.eye(skill_dim),
                    # Singleton instance, not a call: constraints.corr_cholesky is
                    # already the constraint object in this NumPyro version.
                    constraint=constraints.corr_cholesky,
                )
                numpyro.sample("skill_L", dist.Delta(L).to_event(2))
    return guide


def _guide_walk(P: int, skill_dim: int, kl_scale: float, n_seasons: int):
    """Guide for the per-season random-walk skill prior.

    The latent is the innovation tensor `player_skill_eps` with shape
    (P, n_seasons, skill_dim), so the posterior is mean-field over that tensor;
    the walk's step size gets a point estimate.
    """
    def guide(batch, player_table, teacher_force=True):
        mu = numpyro.param(
            "player_mu", jnp.zeros((P, n_seasons, skill_dim)),
            constraint=constraints.interval(-5.0, 5.0),
        )
        sigma = numpyro.param(
            "player_sigma", jnp.full((P, n_seasons, skill_dim), 0.3),
            constraint=constraints.interval(0.05, 2.0),
        )
        walk = numpyro.param(
            "skill_walk_sigma_loc", jnp.asarray(0.3),
            constraint=constraints.interval(0.01, 1.0),
        )
        with nhandlers.scale(scale=kl_scale):
            numpyro.sample("skill_walk_sigma", dist.Delta(walk))
            numpyro.sample("player_skill_eps", dist.Normal(mu, sigma).to_event(3))
    return guide


def _guide_iso(P: int, skill_dim: int = SKILL_DIM, kl_scale: float = 1.0):
    """Return a variational guide for the per-player latent skill vectors.

    Parameterises q(player_skills) = Normal(mu, sigma) with hard constraints:
      mu    âˆˆ [-5, 5]      â€” prevents posterior mean from drifting far off-prior
      sigma âˆˆ [0.05, 2.0]  â€” hard floor/ceiling; gradient flows through a smooth
                             sigmoid transform, so it can never overflow

    This is more stable than the rhoâ†’softplus approach: if rho drifts very
    negative the floor doesn't help because the checkpoint bakes in the bad
    value. A constrained sigma param has no such failure mode.
    """
    def guide(batch, player_table, teacher_force=True):
        mu = numpyro.param(
            "player_mu",
            jnp.zeros((P, skill_dim)),
            constraint=constraints.interval(-5.0, 5.0),
        )
        sigma = numpyro.param(
            "player_sigma",
            jnp.full((P, skill_dim), 0.3),
            constraint=constraints.interval(0.05, 2.0),
        )
        # Must match the model's player_skills scale so the ELBO KL is balanced.
        with nhandlers.scale(scale=kl_scale):
            numpyro.sample("player_skills", dist.Normal(mu, sigma).to_event(2))
    return guide


def make_optimizer(lr: float = DEFAULT_LR, n_steps: int | None = None, cosine_decay: bool = False, cosine_alpha: float = 0.0) -> Any:
    """Adam with per-element grad clipping (numpyro-compatible).

    Per-element clip (optax.clip) instead of global-norm clip:
    clip_by_global_norm computes sqrt(sum(g**2)) which overflows float32
    once any individual gradient exceeds ~1e15. Once that overflow makes
    the norm Inf, all clipped grads become NaN. Per-element clipping never
    has to compute a global sum so it can't overflow.

    NaN handling is also done in the training loop (revert state on NaN
    loss). apply_if_finite and zero_nans both use lax.cond / where inside
    the chain, which JIT-deadlocks with NumPyro's SVI wrapper past
    ~step 1000.  Plain-Python revert avoids that entirely.
    """
    lr_or_schedule = (
        optax.cosine_decay_schedule(init_value=lr, decay_steps=n_steps, alpha=cosine_alpha)
        if cosine_decay and n_steps else lr
    )
    chain = optax.chain(
        optax.clip(GRAD_CLIP_VALUE),
        optax.adam(lr_or_schedule),
    )
    return optax_to_numpyro(chain)


def _eval_metrics(
    model: Callable,
    svi: SVI,
    svi_state,
    batch: dict,
    player_table: dict,
) -> dict[str, float]:
    """No-gradient forward pass to extract per-site NLLs from the trace."""
    params = svi.get_params(svi_state)
    # Multi-task batches nest the two independently padded sequence types. Use
    # the pitch batch for the compact diagnostic below; its sample sites are
    # scoped by multitask_model.
    # ``abcd`` is the current Pitchformer likelihood.  It joins PA in the
    # production joint trainer just as the older ``pitch`` likelihood does in
    # train_shared_skills; diagnostic selection must understand both layouts.
    is_multitask = "pa" in batch and ("pitch" in batch or "abcd" in batch)
    metric_batch = batch["pitch"] if "pitch" in batch else (batch["pa"] if is_multitask else batch)
    valid = jnp.asarray(metric_batch.get("pitch_valid", metric_batch.get("pa_valid")))
    n      = float(jnp.sum(valid))
    if n == 0:
        return {}

    with nhandlers.seed(rng_seed=0):
        with nhandlers.trace() as tr:
            with nhandlers.substitute(data=params):
                model(batch, player_table, teacher_force=True)

    out: dict[str, float] = {}
    site_prefix = "pitch/" if "pitch" in batch else ""
    for site_name in ("pitch_type", "swing", "contact"):
        site = tr.get(site_prefix + site_name)
        if site is None:
            continue
        lp = site.get("log_prob")
        if lp is None:
            continue
        lp = jnp.asarray(lp)
        if lp.shape == valid.shape:
            out[f"{site_name}_nll"] = float(-jnp.sum(lp * valid) / n)
    return out


def _rollout_pa_game_states(outcomes: np.ndarray, batch: dict) -> dict[str, jnp.ndarray]:
    """Build coherent PA-level states from sampled outcomes, one game at a time.

    This is the scheduled-sampling state constructor.  It intentionally rolls
    *complete games*, so base state, outs, inning/half, and batting-team score
    differential remain synchronized.  Pitcher identities and pitch counts stay
    exogenous: the PA model does not yet generate roster or manager decisions.
    """
    import numpy as np
    from diamondworldjax.sim.rules_engine import BS_AFTER, OUT_INC, RUNS

    valid = np.asarray(batch["pa_valid"], dtype=bool)
    B, T = valid.shape
    state = {
        name: np.asarray(batch[name]).copy()
        for name in ("inning", "half", "outs", "base_state", "score_diff")
    }

    for b in range(B):
        rows = np.flatnonzero(valid[b])
        if not len(rows):
            continue
        first = int(rows[0])
        inning = max(1, int(round(state["inning"][b, first] * 8.0)) + 1)
        half = int(round(state["half"][b, first]))
        outs = int(round(state["outs"][b, first] * 2.0))
        bases = int(round(state["base_state"][b, first] * 7.0))
        diff = float(state["score_diff"][b, first] * 10.0)
        if half == 0:
            away, home = max(diff, 0.0), max(-diff, 0.0)
        else:
            home, away = max(diff, 0.0), max(-diff, 0.0)

        for t in rows:
            state["inning"][b, t] = (inning - 1) / 8.0
            state["half"][b, t] = half
            state["outs"][b, t] = outs / 2.0
            state["base_state"][b, t] = bases / 7.0
            state["score_diff"][b, t] = ((away - home) if half == 0 else (home - away)) / 10.0

            outcome = int(np.clip(outcomes[b, t], 0, len(OUT_INC) - 1))
            runs = int(RUNS[bases, outcome])
            if half == 0:
                away += runs
            else:
                home += runs
            outs += int(OUT_INC[outcome])
            if outs >= 3:
                outs, bases = 0, 0
                if half == 1:
                    inning += 1
                half = 1 - half
            else:
                bases = int(BS_AFTER[bases, outcome])

    return {name: jnp.asarray(value) for name, value in state.items()}


def _apply_game_scheduled_sampling(
    model: Callable,
    svi: SVI,
    svi_state,
    batch: dict,
    player_table: dict,
    ss_rate: float,
    rng_key,
) -> dict:
    """Replace complete-game state histories, never independent PA fields.

    A game is selected once for the entire batch sequence.  Its sampled PA
    outcomes are then passed through the deterministic rules engine before the
    supervised update.  The observed next-PA labels are an intentionally
    down-stream robustness target, not counterfactual ground truth; teacher
    forcing remains the dominant objective during the warm-up schedule.
    """
    import numpy as np

    params = svi.get_params(svi_state)
    is_multitask = "pa" in batch and ("pitch" in batch or "abcd" in batch)
    pa_batch = batch["pa"] if is_multitask else batch
    guide_key, model_key, game_key = jax.random.split(rng_key, 3)
    # Variational parameter names are not model sample-site names. Draw the
    # guide once, then replay its complete latent hierarchy into the model.
    guide_trace = nhandlers.trace(nhandlers.seed(
        nhandlers.substitute(svi.guide, data=params), rng_seed=guide_key
    )).get_trace(batch, player_table, teacher_force=False)
    with nhandlers.seed(rng_seed=model_key):
        with nhandlers.substitute(data=params):
            with nhandlers.replay(trace=guide_trace):
                with nhandlers.trace() as tr:
                    model(batch, player_table, teacher_force=False)
    site_name = "pa/pa_outcome" if is_multitask else "pa_outcome"
    if site_name not in tr:
        return batch

    rolled = _rollout_pa_game_states(np.asarray(tr[site_name]["value"]), pa_batch)
    use_generated_game = jax.random.uniform(
        game_key, (pa_batch["pa_valid"].shape[0],)
    ) < ss_rate
    updated_pa = {
        **pa_batch,
        **{
            name: jnp.where(use_generated_game[:, None], value, pa_batch[name])
            for name, value in rolled.items()
        },
    }
    return {**batch, "pa": updated_pa} if is_multitask else updated_pa


def _validate_resume_metadata(checkpoint, current):
    """Reject a known configuration or registry change before reusing weights."""
    import numpy as np
    saved = checkpoint.get("pa_metadata")
    expected = (current or {}).get("pa_metadata")
    if not saved or not expected:
        return  # Legacy checkpoints still receive parameter-tree checks.
    for field in ("config", "train_seasons", "park_map", "player_table"):
        left, left_tree = jax.tree_util.tree_flatten(saved.get(field))
        right, right_tree = jax.tree_util.tree_flatten(expected.get(field))
        if left_tree != right_tree or any(
                not np.array_equal(a, b, equal_nan=True) if np.asarray(a).dtype.kind in "fc"
                else not np.array_equal(a, b) for a, b in zip(left, right)):
            raise ValueError(f"Cannot resume with changed PA {field}; use a compatible configuration")


def _resume_params(svi, state, key, batch, player_table, loaded):
    """Restore constrained checkpoint values through SVI's inverse transforms.

    This is a parameter warm start, not an exact optimizer/RNG continuation.
    New parameter sites keep their initialized values. Existing sites must have
    matching trees/shapes; incompatible checkpoints must not silently remap them.
    """
    import numpy as np
    current = svi.get_params(state)
    unknown = set(loaded) - set(current)
    if unknown:
        raise ValueError(f"Checkpoint contains incompatible parameter sites: {sorted(unknown)}")
    for name, value in loaded.items():
        old, old_tree = jax.tree_util.tree_flatten(value)
        new, new_tree = jax.tree_util.tree_flatten(current[name])
        if old_tree != new_tree or any(np.shape(a) != np.shape(b) for a, b in zip(old, new)):
            raise ValueError(f"Checkpoint parameter shape/tree mismatch: {name}")
        if any(not np.isfinite(x).all() for x in old):
            raise ValueError(f"Non-finite checkpoint parameter: {name}")
    restored = svi.init(key, batch, player_table, teacher_force=True,
                        init_params={**current, **loaded})
    unconstrained = jax.tree_util.tree_leaves(svi.optim.get_params(restored.optim_state))
    if any(not np.isfinite(x).all() for x in unconstrained):
        raise ValueError("Checkpoint parameters are outside the current guide constraints")
    return restored


def train(
    model: Callable,
    batch_iter,                  # iterable yielding (batch, player_table) tuples
    n_steps: int          = DEFAULT_STEPS,
    lr: float             = DEFAULT_LR,
    seed: int             = 0,
    ckpt_dir: Path | None  = None,
    log_path: Path | None  = None,
    rank: int             = 20,      # UNUSED. Left only so train_v0.py's --rank
                                     # keeps working; there is no low-rank guide.
    resume_path: Path | None = None,
    cosine_decay: bool    = False,
    cosine_alpha: float   = 0.0,   # LR floor as fraction of init_lr (0 = decay to 0)
    ss_max_rate: float    = 0.0,   # scheduled sampling: max mixing probability
    ss_warmup_steps: int  = 25_000, # steps to ramp ss_rate from 0 â†’ ss_max_rate
    ss_start_step: int    = 5_000,  # don't apply SS until model has learned basics
    engine_ss: bool       = False,  # retained for CLI compatibility; all SS is game-level
    kl_scale: float       = 1.0,    # scale for the global player_skills KL (= batch/total_games)
    skill_prior: str      = "iso",  # MUST match the model, else its extra latents go
                                    # uncovered by the guide and are silently resampled
                                    # from the prior every step instead of being learned
    n_seasons: int        = 1,      # season axis, only used by skill_prior="walk"
    shared_task_skills: bool = False,
    checkpoint_metadata: dict | None = None,
    update_chunk_size: int = 16,
    prefetch_depth: int = 2,
    start_step: int = 0,
    metrics_fn: Callable[[Callable, SVI, Any, dict, dict], dict[str, float]] | None = None,
) -> tuple[Any, Any, list[float]]:
    """
    Run SVI training.

    Parameters
    ----------
    model       : NumPyro model function (batch, player_table, teacher_force=True)
    batch_iter  : infinite iterator; each call to next() returns (batch, player_table)
    n_steps     : total SVI update steps.  With ``start_step``, training runs
                  from that restored global step through this target.
    lr          : Adam learning rate
    seed        : JAX PRNG seed
    ckpt_dir    : if given, save params checkpoint every CKPT_INTERVAL steps
    log_path    : if given, write ELBO log as JSON
    resume_path : if given, load params from this checkpoint before training
    start_step  : global step represented by ``resume_path``.  This makes a
                  resumed run preserve checkpoint numbering and schedules.
    metrics_fn  : optional model-specific, read-only diagnostics for log lines

    Returns
    -------
    svi_state : final SVI state (contains params)
    guide     : variational player-skill guide used by SVI and posterior replay
    losses    : list of ELBO values (one per step)
    """
    optimizer = make_optimizer(lr, n_steps=n_steps, cosine_decay=cosine_decay, cosine_alpha=cosine_alpha)

    if not 0 <= start_step <= n_steps:
        raise ValueError("start_step must be non-negative and no greater than n_steps")

    from .runtime import prefetch, shape_signature, stack_batches, safe_update
    if update_chunk_size < 1:
        raise ValueError("update_chunk_size must be positive")
    batch_iter = prefetch(batch_iter, prefetch_depth)
    first_batch, first_player_table = next(batch_iter)
    P = first_player_table["stats"].shape[0]
    guide = (
        make_shared_task_skills_guide(
            P, kl_scale=kl_scale, skill_prior=skill_prior, n_seasons=n_seasons
        )
        if shared_task_skills else
        make_player_skills_guide(P, kl_scale=kl_scale,
                                 skill_prior=skill_prior, n_seasons=n_seasons)
    )
    print(f"  player_skills KL scale = {kl_scale:.6g}", flush=True)

    svi = SVI(
        model,
        guide,
        optimizer,
        loss=Trace_ELBO(num_particles=1),
    )

    rng_key = jax.random.PRNGKey(seed)
    rng_key, init_key = jax.random.split(rng_key)
    print("Initialising SVI...", flush=True)
    t0 = time.time()
    svi_state = svi.init(init_key, first_batch, first_player_table, teacher_force=True)
    print(f"  Init done in {time.time()-t0:.1f}s", flush=True)

    if resume_path is not None:
        with open(resume_path, "rb") as f:
            checkpoint = pickle.load(f)
        _validate_resume_metadata(checkpoint, checkpoint_metadata)
        svi_state = _resume_params(svi, svi_state, init_key, first_batch,
                                   first_player_table, checkpoint["params"])
        print(f"  Resumed parameters from {resume_path} (optimizer/schedule restarted)", flush=True)

    losses: list[float] = []
    nan_skips = 0

    if ckpt_dir is not None:
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    @jax.jit
    def update_chunk(state, key, batches, table):
        def advance(carry, batch):
            state, key = carry
            key, _ = jax.random.split(key)
            state, loss = safe_update(svi.update, state, batch, table, teacher_force=True)
            return (state, key), loss
        return jax.lax.scan(advance, (state, key), batches)

    step = start_step
    pending = None
    try:
        while step < n_steps:
            batch, player_table = next(batch_iter) if pending is None else pending
            pending = None
            count = min(update_chunk_size, n_steps - step,
                        LOG_INTERVAL - ((step - 1) % LOG_INTERVAL),
                        CKPT_INTERVAL - ((step - 1) % CKPT_INTERVAL))
            if ss_max_rate > 0.0:
                count = min(count, max(1, ss_start_step - step))
            selected = [batch]
            # Only combine equal shapes and a shared immutable player table.
            for _ in range(count - 1):
                candidate, table = next(batch_iter)
                if table is not player_table or shape_signature(candidate) != shape_signature(batch):
                    pending = candidate, table
                    break
                selected.append(candidate)
            if ss_max_rate > 0.0 and step >= ss_start_step:
                progress = min(1.0, (step - ss_start_step) / max(ss_warmup_steps, 1))
                rng_key, _ = jax.random.split(rng_key)
                rng_key, ss_key = jax.random.split(rng_key)
                batch = _apply_game_scheduled_sampling(
                    model, svi, svi_state, batch, player_table, ss_max_rate * progress, ss_key)
                svi_state, loss = safe_update(svi.update, svi_state, batch, player_table, teacher_force=True)
                chunk_losses = [float(loss)]
            else:
                (svi_state, rng_key), values = update_chunk(
                    svi_state, rng_key, stack_batches(selected), player_table)
                chunk_losses = jax.device_get(values).tolist()
                batch = selected[-1]
            losses.extend(chunk_losses)
            nan_skips += sum(not math.isfinite(value) for value in chunk_losses)
            step += len(selected) - 1
            loss_f = chunk_losses[-1]

            if step % LOG_INTERVAL == 0:
                elapsed = time.time() - t0
                metrics = _eval_metrics(model, svi, svi_state, batch, player_table)
                if metrics_fn is not None:
                    metrics.update(metrics_fn(model, svi, svi_state, batch, player_table))
                m_str = "  ".join(
                    f"{k} = {v:.4f}" for k, v in metrics.items()
                )
                print(
                    f"  step {step:6d}  ELBO = {-loss_f:10.2f}  {m_str}  "
                    f"elapsed = {elapsed:.0f}s  nan_skips = {nan_skips}",
                    flush=True,
                )

            if ckpt_dir is not None and step > 0 and step % CKPT_INTERVAL == 0:
                _save_checkpoint(svi, svi_state, ckpt_dir, step, checkpoint_metadata)

            step += 1
    finally:
        batch_iter.close()

    if ckpt_dir is not None:
        _save_checkpoint(svi, svi_state, ckpt_dir, n_steps, checkpoint_metadata)

    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(losses, indent=2))

    return svi_state, guide, losses


def _save_checkpoint(svi: SVI, state, ckpt_dir: Path, step: int, metadata=None) -> None:
    params = svi.get_params(state)
    path   = ckpt_dir / f"dwjax_step_{step:07d}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"step": step, "params": params, **(metadata or {})}, f)
    print(f"  Checkpoint saved â†’ {path}", flush=True)


def load_checkpoint(path: Path) -> dict:
    """Load params from a checkpoint file."""
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    return ckpt["params"]
