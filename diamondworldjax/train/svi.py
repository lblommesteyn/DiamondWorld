"""SVI training loop for DiamondWorldJAX.

Guide: AutoLowRankMultivariateNormal (captures correlations among global latents).
Optimizer: Adam with optional cosine-annealing LR schedule.
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
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal
from numpyro.optim import Adam, optax_to_numpyro

# AutoLowRankMultivariateNormal rank — 20 captures key correlations without
# blowing up the guide parameter count.
DEFAULT_RANK = 20
DEFAULT_LR   = 1e-3
DEFAULT_STEPS = 50_000
LOG_INTERVAL  = 500
CKPT_INTERVAL = 5_000


def build_guide(model: Callable, rank: int = DEFAULT_RANK) -> AutoLowRankMultivariateNormal:
    """Construct an AutoLowRankMultivariateNormal guide for `model`."""
    return AutoLowRankMultivariateNormal(model, rank=rank)


def make_optimizer(lr: float = DEFAULT_LR) -> Any:
    """Adam optimizer wrapped for NumPyro."""
    return Adam(lr)


def train(
    model: Callable,
    batch_iter,                  # iterable yielding (batch, player_table) tuples
    n_steps: int         = DEFAULT_STEPS,
    rank: int            = DEFAULT_RANK,
    lr: float            = DEFAULT_LR,
    seed: int            = 0,
    ckpt_dir: Path | None = None,
    log_path: Path | None = None,
) -> tuple[Any, Any, list[float]]:
    """
    Run SVI training.

    Parameters
    ----------
    model       : NumPyro model function (batch, player_table, teacher_force=True)
    batch_iter  : infinite iterator; each call to next() returns (batch, player_table)
    n_steps     : total SVI update steps
    rank        : guide covariance rank
    lr          : Adam learning rate
    seed        : JAX PRNG seed
    ckpt_dir    : if given, save params checkpoint every CKPT_INTERVAL steps
    log_path    : if given, write ELBO log as JSON

    Returns
    -------
    svi_state : final SVI state (contains params)
    guide     : fitted guide (needed for Predictive)
    losses    : list of ELBO values (one per step)
    """
    guide     = build_guide(model, rank=rank)
    optimizer = make_optimizer(lr)
    svi       = SVI(
        model,
        guide,
        optimizer,
        loss = Trace_ELBO(num_particles=4),
    )

    rng_key = jax.random.PRNGKey(seed)

    # Initialise on first batch
    first_batch, first_player_table = next(batch_iter)
    rng_key, init_key = jax.random.split(rng_key)
    print("Initialising SVI...", flush=True)
    t0 = time.time()
    svi_state = svi.init(init_key, first_batch, first_player_table, teacher_force=True)
    print(f"  Init done in {time.time()-t0:.1f}s", flush=True)

    losses: list[float] = []

    if ckpt_dir is not None:
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    for step in range(n_steps):
        batch, player_table = next(batch_iter)
        rng_key, step_key = jax.random.split(rng_key)
        svi_state, loss = svi.update(
            svi_state, batch, player_table, teacher_force=True
        )
        losses.append(float(loss))

        if step % LOG_INTERVAL == 0:
            elapsed = time.time() - t0
            print(
                f"  step {step:6d}  ELBO = {-loss:10.2f}  "
                f"elapsed = {elapsed:.0f}s",
                flush=True,
            )

        if ckpt_dir is not None and step > 0 and step % CKPT_INTERVAL == 0:
            _save_checkpoint(svi, svi_state, guide, ckpt_dir, step)

    # Final checkpoint
    if ckpt_dir is not None:
        _save_checkpoint(svi, svi_state, guide, ckpt_dir, n_steps)

    if log_path is not None:
        Path(log_path).write_text(json.dumps(losses, indent=2))

    return svi_state, guide, losses


def _save_checkpoint(svi: SVI, state, guide, ckpt_dir: Path, step: int) -> None:
    params = svi.get_params(state)
    path   = ckpt_dir / f"dwjax_step_{step:07d}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"step": step, "params": params, "guide": guide}, f)
    print(f"  Checkpoint saved → {path}", flush=True)


def load_checkpoint(path: Path) -> tuple[dict, Any]:
    """Load params and guide from a checkpoint file."""
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    return ckpt["params"], ckpt["guide"]
