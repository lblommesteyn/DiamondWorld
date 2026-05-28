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
import numpyro.handlers as nhandlers
import optax
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.svi import SVIState
from numpyro.optim import optax_to_numpyro

DEFAULT_LR        = 1e-3
DEFAULT_STEPS     = 50_000
LOG_INTERVAL      = 500
CKPT_INTERVAL     = 5_000
GRAD_CLIP_NORM    = 1.0   # max global grad norm


def empty_guide(*args, **kwargs) -> None:
    """No continuous latent variables — guide does nothing."""


def make_optimizer(lr: float = DEFAULT_LR) -> Any:
    """Adam with global-norm grad clipping (numpyro-compatible).

    NaN handling is done in the training loop (revert state on NaN loss) —
    NOT in the optax chain.  Both apply_if_finite and zero_nans use lax.cond /
    where inside the chain, which JIT-deadlocks with NumPyro's SVI wrapper
    past ~step 1000.  Plain-Python revert avoids that entirely.
    """
    chain = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP_NORM),
        optax.adam(lr),
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
    valid  = jnp.asarray(batch["pitch_valid"])  # (B, T)
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


def train(
    model: Callable,
    batch_iter,                  # iterable yielding (batch, player_table) tuples
    n_steps: int          = DEFAULT_STEPS,
    lr: float             = DEFAULT_LR,
    seed: int             = 0,
    ckpt_dir: Path | None  = None,
    log_path: Path | None  = None,
    # rank kept for API compatibility; unused (empty guide has no latent dim)
    rank: int             = 20,
    resume_path: Path | None = None,
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
    optimizer = make_optimizer(lr)
    svi = SVI(
        model,
        empty_guide,
        optimizer,
        loss=Trace_ELBO(num_particles=1),
    )

    rng_key = jax.random.PRNGKey(seed)

    first_batch, first_player_table = next(batch_iter)
    rng_key, init_key = jax.random.split(rng_key)
    print("Initialising SVI...", flush=True)
    t0 = time.time()
    svi_state = svi.init(init_key, first_batch, first_player_table, teacher_force=True)
    print(f"  Init done in {time.time()-t0:.1f}s", flush=True)

    if resume_path is not None:
        loaded_params = load_checkpoint(Path(resume_path))
        new_optim_state = svi.optim.init(loaded_params)
        svi_state = SVIState(new_optim_state, svi_state.mutable_state, svi_state.rng_key)
        print(f"  Resumed params from {resume_path}", flush=True)

    losses: list[float] = []
    nan_skips = 0

    if ckpt_dir is not None:
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    for step in range(n_steps):
        batch, player_table = next(batch_iter)
        rng_key, step_key = jax.random.split(rng_key)

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

    return svi_state, empty_guide, losses


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
