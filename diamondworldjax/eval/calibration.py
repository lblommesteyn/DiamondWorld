"""Calibration and evaluation metrics for DiamondWorldJAX.

Pitch-level
-----------
  pitch_nll(samples, batch)  — per-site NLL on held-out pitches

PA-level
--------
  pa_outcome_nll(samples, batch)  — NLL of run-scoring distribution
  pa_calibration(samples, batch)  — calibration curves per hurdle

Game-level
----------
  game_run_metrics(sim_runs, obs_runs)  — KL, Wasserstein, mean/var error
"""
from __future__ import annotations

import numpy as np
from typing import Optional
import jax.numpy as jnp

try:
    from scipy import stats as scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Pitch-level NLL
# ---------------------------------------------------------------------------

def _safe_log(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.log(np.clip(p, eps, 1.0))


def bernoulli_nll(
    probs: np.ndarray,   # (N,) predicted P(=1)
    labels: np.ndarray,  # (N,) observed {0,1}
    mask: Optional[np.ndarray] = None,  # (N,) bool
) -> float:
    """Mean NLL of a Bernoulli prediction."""
    if mask is not None:
        probs  = probs[mask]
        labels = labels[mask]
    n = len(probs)
    if n == 0:
        return float("nan")
    nll = -(labels * _safe_log(probs) + (1 - labels) * _safe_log(1 - probs))
    return float(nll.mean())


def categorical_nll(
    probs: np.ndarray,   # (N, K) predicted per-class probabilities
    labels: np.ndarray,  # (N,) observed class indices
    mask: Optional[np.ndarray] = None,
) -> float:
    """Mean NLL of a Categorical prediction."""
    if mask is not None:
        probs  = probs[mask]
        labels = labels[mask]
    n = len(probs)
    if n == 0:
        return float("nan")
    chosen = probs[np.arange(n), labels.clip(0, probs.shape[1] - 1)]
    return float(-_safe_log(chosen).mean())


def pitch_nll_report(
    samples: dict,
    batch: dict,
) -> dict[str, float]:
    """
    Compute mean NLL per observable site using posterior predictive samples.

    samples : output of teacher_forced_samples() or free_rollout_samples()
              shape (S, B, T, ...) for each site
    batch   : original batch dict

    Returns dict of {site_name: mean_nll}.
    """
    report: dict[str, float] = {}

    S = next(iter(samples.values())).shape[0] if samples else 1

    valid = np.array(batch["pitch_valid"])  # (B, T)

    def _flat(key, default=None):
        arr = batch.get(key, default)
        return None if arr is None else np.array(arr).ravel()

    def _s_flat(key):
        """Mean over samples then flatten."""
        arr = samples.get(key)
        if arr is None:
            return None
        a = np.array(arr)
        # (S, B, T, ...) → mean over S → (B, T, ...)
        return a.mean(axis=0).ravel()

    v = valid.ravel().astype(bool)

    # --- swing ---
    swing_obs  = _flat("obs_swing")
    swing_pred = _s_flat("swing")
    if swing_obs is not None and swing_pred is not None:
        m = v & (swing_obs >= 0)
        report["swing_nll"] = bernoulli_nll(swing_pred, swing_obs, mask=m)

    # --- contact (given swing) ---
    contact_obs  = _flat("obs_contact")
    contact_pred = _s_flat("contact")
    swing_mask_r = _flat("obs_swing")
    if contact_obs is not None and contact_pred is not None:
        m = v & (contact_obs >= 0) & (swing_mask_r == 1 if swing_mask_r is not None else True)
        report["contact_nll"] = bernoulli_nll(contact_pred, contact_obs, mask=m)

    # --- runs_scored (terminal PAs) ---
    runs_obs   = _flat("runs_scored")
    runs_pred  = _s_flat("runs_scored")
    terminal_r = np.array(batch["terminal_mask"]).ravel()
    if runs_obs is not None and runs_pred is not None:
        m = v & terminal_r.astype(bool)
        # Treat as Bernoulli of (>0) for simple NLL
        report["runs_nonzero_nll"] = bernoulli_nll(
            (runs_pred > 0.5).astype(float),
            (runs_obs  > 0).astype(int),
            mask=m,
        )

    return report


# ---------------------------------------------------------------------------
# Game-level run distribution metrics
# ---------------------------------------------------------------------------

def kl_divergence_runs(
    sim_runs: np.ndarray,    # (N_sim,) run totals per game
    obs_runs: np.ndarray,    # (N_obs,) run totals per game
    max_runs: int = 20,
) -> float:
    """KL(obs || sim) on the discrete run distribution."""
    bins = np.arange(max_runs + 2)
    obs_hist, _ = np.histogram(obs_runs, bins=bins, density=True)
    sim_hist, _ = np.histogram(sim_runs, bins=bins, density=True)

    eps = 1e-8
    obs_hist = obs_hist + eps
    sim_hist = sim_hist + eps
    obs_hist /= obs_hist.sum()
    sim_hist /= sim_hist.sum()

    return float(np.sum(obs_hist * np.log(obs_hist / sim_hist)))


def wasserstein_runs(
    sim_runs: np.ndarray,
    obs_runs: np.ndarray,
) -> float:
    """Wasserstein-1 distance between run distributions."""
    if _HAS_SCIPY:
        return float(scipy_stats.wasserstein_distance(obs_runs, sim_runs))
    # Fallback: mean absolute difference
    return float(abs(np.mean(sim_runs) - np.mean(obs_runs)))


def game_run_metrics(
    sim_runs: np.ndarray,
    obs_runs: np.ndarray,
) -> dict[str, float]:
    """
    Full suite of game-level run distribution metrics.

    Matches the keys used in the existing eval_svi_arma.py results table.
    """
    sim_runs = np.asarray(sim_runs, dtype=float)
    obs_runs = np.asarray(obs_runs, dtype=float)

    obs_mean = obs_runs.mean()
    sim_mean = sim_runs.mean()
    obs_var  = obs_runs.var()
    sim_var  = sim_runs.var()

    obs_total = len(obs_runs)
    sim_total = len(sim_runs)

    def _p(runs, threshold):
        return (runs >= threshold).mean()

    return {
        "kl_run_distribution": kl_divergence_runs(sim_runs, obs_runs),
        "wasserstein_runs":    wasserstein_runs(sim_runs, obs_runs),
        "mean_rg_error":       abs(sim_mean - obs_mean),
        "variance_error":      abs(sim_var  - obs_var),
        "p0_error":            abs(_p(sim_runs, 0) - _p(obs_runs, 0)),
        "p5_plus_error":       abs(_p(sim_runs, 5) - _p(obs_runs, 5)),
        "p8_plus_error":       abs(_p(sim_runs, 8) - _p(obs_runs, 8)),
    }


# ---------------------------------------------------------------------------
# Calibration curve per binary site
# ---------------------------------------------------------------------------

def calibration_curve(
    predicted_probs: np.ndarray,  # (N,) predicted P(=1)
    observed:        np.ndarray,  # (N,) observed {0,1}
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute reliability diagram data: (mean predicted prob, mean observed rate)
    per bin.
    """
    bins   = np.linspace(0, 1, n_bins + 1)
    bin_p, bin_o = [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = (predicted_probs >= lo) & (predicted_probs < hi)
        if idx.sum() == 0:
            continue
        bin_p.append(predicted_probs[idx].mean())
        bin_o.append(observed[idx].mean())

    return np.array(bin_p), np.array(bin_o)
