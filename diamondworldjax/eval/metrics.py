"""Canonical DiamondWorldJAX game-level evaluation metrics."""
from __future__ import annotations

import numpy as np


def _run_histogram(values: np.ndarray, max_runs: int = 20) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.int64), 0, max_runs)
    counts = np.bincount(clipped, minlength=max_runs + 1).astype(np.float64)
    return counts / counts.sum() if counts.sum() else counts


def kl_divergence_runs(sim_runs: np.ndarray, obs_runs: np.ndarray, max_runs: int = 20) -> float:
    """KL(observed || simulated), with all overflow in the final bin."""
    observed = _run_histogram(obs_runs, max_runs)
    simulated = _run_histogram(sim_runs, max_runs)
    eps = 1e-8
    observed = (observed + eps) / (observed + eps).sum()
    simulated = (simulated + eps) / (simulated + eps).sum()
    return float(np.sum(observed * np.log(observed / simulated)))


def wasserstein_runs(sim_runs: np.ndarray, obs_runs: np.ndarray) -> float:
    """Exact empirical one-dimensional Wasserstein distance."""
    simulated = np.sort(np.asarray(sim_runs, dtype=np.float64))
    observed = np.sort(np.asarray(obs_runs, dtype=np.float64))
    if not len(simulated) or not len(observed):
        return float("nan")
    grid = np.sort(np.unique(np.concatenate([simulated, observed])))
    sim_cdf = np.searchsorted(simulated, grid, side="right") / len(simulated)
    obs_cdf = np.searchsorted(observed, grid, side="right") / len(observed)
    widths = np.diff(np.concatenate([[grid[0]], grid]))
    return float(np.sum(np.abs(sim_cdf - obs_cdf) * widths))


def game_run_metrics(sim_runs: np.ndarray, obs_runs: np.ndarray) -> dict[str, float]:
    simulated = np.asarray(sim_runs, dtype=float)
    observed = np.asarray(obs_runs, dtype=float)
    if not len(simulated) or not len(observed):
        raise ValueError("game_run_metrics requires non-empty simulated and observed arrays")
    return {
        "kl_run_distribution": kl_divergence_runs(simulated, observed),
        "wasserstein_runs": wasserstein_runs(simulated, observed),
        "mean_rg_error": float(abs(simulated.mean() - observed.mean())),
        "variance_error": float(abs(simulated.var() - observed.var())),
        "p0_error": float(abs(np.mean(simulated == 0) - np.mean(observed == 0))),
        "p5_plus_error": float(abs(np.mean(simulated >= 5) - np.mean(observed >= 5))),
        "p8_plus_error": float(abs(np.mean(simulated >= 8) - np.mean(observed >= 8))),
    }
