"""SVI training loop for DiamondWorldJAX.

The model is purely discriminative: all neural-network weights live in NumPyro
`param` sites (via flax_module), and every continuous `sample` site receives an
observed value during teacher-forcing.  A small number of discrete `sample`
sites (runner_send, defensive_alignment, called_strike, etc.) are left
unobserved; they are sampled in the ELBO estimate and contribute their
log-probability but do not affect downstream computations.

Guide: empty (no continuous latent variables to approximate).
Loss:  Trace_ELBO — sums log-likelihoods of observed sites.
Optimizer: Adam.
"""
from __future__ import annotations

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
from numpyro.infer.svi import SVIState
from numpyro.optim import optax_to_numpyro

DEFAULT_LR        = 1e-3
DEFAULT_STEPS     = 50_000
LOG_INTERVAL      = 500
CKPT_INTERVAL     = 5_000
GRAD_CLIP_VALUE   = 1.0   # per-element gradient clip threshold

# Must match SKILL_DIM in embeddings.py
SKILL_DIM = 32


def empty_guide(*args, **kwargs) -> None:
    """No continuous latent variables — guide does nothing."""


def make_player_skills_guide(P: int, skill_dim: int = SKILL_DIM, kl_scale: float = 1.0):
    """Return a variational guide for the per-player latent skill vectors.

    Parameterises q(player_skills) = Normal(mu, sigma) with hard constraints:
      mu    ∈ [-5, 5]      — prevents posterior mean from drifting far off-prior
      sigma ∈ [0.05, 2.0]  — hard floor/ceiling; gradient flows through a smooth
                             sigmoid transform, so it can never overflow

    This is more stable than the rho→softplus approach: if rho drifts very
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
    valid  = jnp.asarray(batch.get("pitch_valid", batch.get("pa_valid")))  # (B, T)
    n      = float(jnp.sum(valid))
    if n == 0:
        return {}

    with nhandlers.seed(rng_seed=0):
        with nhandlers.trace() as tr:
            with nhandlers.substitute(data=params):
                model(batch, player_table, teacher_force=True)

    out: dict[str, float] = {}
    for site_name in ("pitch_type", "swing", "contact"):
        site = tr.get(site_name)
        if site is None:
            continue
        lp = site.get("log_prob")
        if lp is None:
            continue
        lp = jnp.asarray(lp)
        if lp.shape == valid.shape:
            out[f"{site_name}_nll"] = float(-jnp.sum(lp * valid) / n)
    return out


def _apply_scheduled_sampling(
    model: Callable,
    svi: SVI,
    svi_state,
    batch: dict,
    player_table: dict,
    ss_rate: float,
    rng_key,
) -> dict:
    """Mix model-predicted base_state_after into batch inputs.

    With probability ss_rate, replaces batch["base_state"][:, t] with the
    model's argmax prediction of base_state_after from the previous PA.
    This is scheduled sampling at the batch level — no lax.scan needed.
    """
    params = svi.get_params(svi_state)
    with nhandlers.seed(rng_seed=rng_key):
        with nhandlers.substitute(data=params):
            with nhandlers.trace() as tr:
                model(batch, player_table, teacher_force=False)

    if "base_state_after" not in tr:
        return batch

    # Model's predicted base state after each PA, normalised to [0, 1]
    pred_bs_after = jnp.array(tr["base_state_after"]["value"]).astype(jnp.float32) / 7.0

    # Shift: prediction at t-1 feeds as base_state at t (t=0 keeps real value)
    pred_bs_shifted = jnp.concatenate(
        [batch["base_state"][:, :1], pred_bs_after[:, :-1]], axis=1
    )

    rng_key, mask_key = jax.random.split(rng_key)
    B, T = pred_bs_shifted.shape
    use_model = jax.random.uniform(mask_key, (B, T)) < ss_rate
    mixed_bs  = jnp.where(use_model, pred_bs_shifted, batch["base_state"])

    return {**batch, "base_state": mixed_bs}


def _apply_engine_scheduled_sampling(
    model: Callable,
    svi: SVI,
    svi_state,
    batch: dict,
    player_table: dict,
    ss_rate: float,
    rng_key,
) -> dict:
    """Phase-2 DAgger: expose the model to its OWN rolled-out base states, but
    derived from the deterministic rules engine (legal by construction).

    The earlier `_apply_scheduled_sampling` detonated (free-rollout KL=12) because
    it fed back the neural `base_state_after` head, which could place runners in
    impossible configurations. Here we instead sample only `pa_outcome` from the
    model and compute the next base state via the engine lookup table, so every
    injected state is a legal baseball state. This lets us anneal in self-generated
    context at a low rate (e.g. 0.05 -> 0.25) without the distribution detonating.

    Only meaningful for the outcome-only model (v6+), whose sole stochastic site
    is `pa_outcome`.
    """
    import numpy as _np
    from diamondworldjax.sim.rules_engine import BS_AFTER

    params = svi.get_params(svi_state)
    with nhandlers.seed(rng_seed=rng_key):
        with nhandlers.substitute(data=params):
            with nhandlers.trace() as tr:
                model(batch, player_table, teacher_force=False)

    if "pa_outcome" not in tr:
        return batch

    outcomes = _np.asarray(tr["pa_outcome"]["value"]).astype(_np.int64)        # (B, T)
    real_bs_int = _np.clip(_np.rint(_np.asarray(batch["base_state"]) * 7.0), 0, 7).astype(_np.int64)
    eng_bs_after = BS_AFTER[real_bs_int, outcomes].astype(_np.float32) / 7.0    # (B, T)

    eng_bs_after = jnp.asarray(eng_bs_after)
    # Shift: engine state after PA t-1 feeds as base_state at t (t=0 keeps real).
    pred_bs_shifted = jnp.concatenate(
        [batch["base_state"][:, :1], eng_bs_after[:, :-1]], axis=1
    )

    rng_key, mask_key = jax.random.split(rng_key)
    B, T = pred_bs_shifted.shape
    use_model = jax.random.uniform(mask_key, (B, T)) < ss_rate
    mixed_bs  = jnp.where(use_model, pred_bs_shifted, batch["base_state"])

    return {**batch, "base_state": mixed_bs}


def train(
    model: Callable,
    batch_iter,                  # iterable yielding (batch, player_table) tuples
    n_steps: int          = DEFAULT_STEPS,
    lr: float             = DEFAULT_LR,
    seed: int             = 0,
    ckpt_dir: Path | None  = None,
    log_path: Path | None  = None,
    rank: int             = 20,
    resume_path: Path | None = None,
    cosine_decay: bool    = False,
    cosine_alpha: float   = 0.0,   # LR floor as fraction of init_lr (0 = decay to 0)
    ss_max_rate: float    = 0.0,   # scheduled sampling: max mixing probability
    ss_warmup_steps: int  = 25_000, # steps to ramp ss_rate from 0 → ss_max_rate
    ss_start_step: int    = 5_000,  # don't apply SS until model has learned basics
    engine_ss: bool       = False,  # use engine-based (legal) DAgger instead of neural bsa
    kl_scale: float       = 1.0,    # scale for the global player_skills KL (= batch/total_games)
) -> tuple[Any, Any, list[float]]:
    """
    Run SVI training.

    Parameters
    ----------
    model       : NumPyro model function (batch, player_table, teacher_force=True)
    batch_iter  : infinite iterator; each call to next() returns (batch, player_table)
    n_steps     : total SVI update steps
    lr          : Adam learning rate
    seed        : JAX PRNG seed
    ckpt_dir    : if given, save params checkpoint every CKPT_INTERVAL steps
    log_path    : if given, write ELBO log as JSON
    resume_path : if given, load params from this checkpoint before training

    Returns
    -------
    svi_state : final SVI state (contains params)
    guide     : empty_guide (returned for API compatibility with Predictive)
    losses    : list of ELBO values (one per step)
    """
    optimizer = make_optimizer(lr, n_steps=n_steps, cosine_decay=cosine_decay, cosine_alpha=cosine_alpha)

    first_batch, first_player_table = next(batch_iter)
    P = first_player_table["stats"].shape[0]
    guide = make_player_skills_guide(P, kl_scale=kl_scale)
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
        loaded_params = load_checkpoint(Path(resume_path))
        # Merge: overlay checkpoint values onto the freshly-initialised param
        # tree so that new params (fusion layer, player_mu, player_sigma) keep
        # their init values and the optimizer state covers the full tree.
        # Reinitialising with only loaded_params (a subset) leaves new params
        # without Adam moment estimates, which produces NaN on the first update.
        current_params = svi.get_params(svi_state)
        merged_params = {**current_params, **loaded_params}
        new_optim_state = svi.optim.init(merged_params)
        svi_state = SVIState(new_optim_state, svi_state.mutable_state, svi_state.rng_key)
        print(f"  Resumed params from {resume_path}", flush=True)
        print(f"  Checkpoint keys loaded: {len(loaded_params)}  "
              f"new keys (fresh init): {len(merged_params) - len(loaded_params)}", flush=True)

    losses: list[float] = []
    nan_skips = 0

    if ckpt_dir is not None:
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    for step in range(n_steps):
        batch, player_table = next(batch_iter)
        rng_key, step_key = jax.random.split(rng_key)

        # Scheduled sampling: replace some real base_states with self-generated ones.
        if ss_max_rate > 0.0 and step >= ss_start_step:
            progress  = min(1.0, (step - ss_start_step) / max(ss_warmup_steps, 1))
            ss_rate   = ss_max_rate * progress
            rng_key, ss_key = jax.random.split(rng_key)
            ss_fn = _apply_engine_scheduled_sampling if engine_ss else _apply_scheduled_sampling
            batch = ss_fn(
                model, svi, svi_state, batch, player_table, ss_rate, ss_key
            )

        # Snapshot state before update so we can revert if loss/state goes NaN.
        # SVIState is a namedtuple, so this just keeps a reference to the
        # immutable arrays — cheap (no copy).
        prev_state = svi_state
        svi_state, loss = svi.update(
            svi_state, batch, player_table, teacher_force=True
        )
        loss_f = float(loss)
        if not jnp.isfinite(loss):
            # Roll back — keeps Adam's momentum buffers clean of any NaN.
            svi_state = prev_state
            nan_skips += 1
            loss_f = float("nan")
        losses.append(loss_f)

        if step % LOG_INTERVAL == 0:
            elapsed = time.time() - t0
            metrics = _eval_metrics(model, svi, svi_state, batch, player_table)
            m_str = "  ".join(
                f"{k} = {v:.4f}" for k, v in metrics.items()
            )
            print(
                f"  step {step:6d}  ELBO = {-loss_f:10.2f}  {m_str}  "
                f"elapsed = {elapsed:.0f}s  nan_skips = {nan_skips}",
                flush=True,
            )

        if ckpt_dir is not None and step > 0 and step % CKPT_INTERVAL == 0:
            _save_checkpoint(svi, svi_state, ckpt_dir, step)

    if ckpt_dir is not None:
        _save_checkpoint(svi, svi_state, ckpt_dir, n_steps)

    if log_path is not None:
        Path(log_path).write_text(json.dumps(losses, indent=2))

    return svi_state, guide, losses


def _save_checkpoint(svi: SVI, state, ckpt_dir: Path, step: int) -> None:
    params = svi.get_params(state)
    path   = ckpt_dir / f"dwjax_step_{step:07d}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"step": step, "params": params}, f)
    print(f"  Checkpoint saved → {path}", flush=True)


def load_checkpoint(path: Path) -> dict:
    """Load params from a checkpoint file."""
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    return ckpt["params"]
