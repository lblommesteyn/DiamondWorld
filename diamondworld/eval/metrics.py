from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl


RUN_BINS = list(range(0, 21))
INNING_BINS = [0, 1, 2, 3, 4, 5]


@dataclass(frozen=True)
class MetricReport:
    kl_run_distribution: float
    wasserstein_runs: float
    mean_rg_error: float
    variance_error: float
    p0_error: float
    p5_plus_error: float
    p8_plus_error: float
    crooked_kl: float
    pa_length_kl: float

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def _hist(values: np.ndarray, bins: list[int], *, overflow_last: bool = True) -> np.ndarray:
    counts = np.zeros(len(bins), dtype=np.float64)
    for value in values:
        idx = int(value)
        if overflow_last and idx >= bins[-1]:
            counts[-1] += 1
        elif idx in bins:
            counts[bins.index(idx)] += 1
    total = counts.sum()
    return counts / total if total else counts


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return math.nan
    a = np.sort(a.astype(np.float64))
    b = np.sort(b.astype(np.float64))
    grid = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    widths = np.diff(np.concatenate([[grid[0]], grid]))
    return float(np.sum(np.abs(cdf_a - cdf_b) * widths))


def team_runs_per_game(pitches: pl.DataFrame) -> np.ndarray:
    terminal = pitches.filter(pl.col("pa_terminal"))
    grouped = terminal.group_by(["game_pk", "half"]).agg(pl.col("runs_scored").sum().alias("runs"))
    return grouped.get_column("runs").to_numpy()


def inning_runs(pitches: pl.DataFrame) -> np.ndarray:
    terminal = pitches.filter(pl.col("pa_terminal"))
    grouped = terminal.group_by(["game_pk", "inning", "half"]).agg(
        pl.col("runs_scored").sum().alias("runs")
    )
    return grouped.get_column("runs").to_numpy()


def pa_lengths(pitches: pl.DataFrame) -> np.ndarray:
    grouped = pitches.group_by(["game_pk", "at_bat_number"]).agg(pl.len().alias("n"))
    return grouped.get_column("n").to_numpy()


def evaluate_game_logs(
    empirical: pl.DataFrame,
    simulated: pl.DataFrame,
    *,
    skip_pa_length_kl: bool = False,
) -> MetricReport:
    emp_runs = team_runs_per_game(empirical)
    sim_runs = team_runs_per_game(simulated)
    emp_innings = inning_runs(empirical)
    sim_innings = inning_runs(simulated)

    emp_run_hist = _hist(emp_runs, RUN_BINS)
    sim_run_hist = _hist(sim_runs, RUN_BINS)
    emp_inning_hist = _hist(emp_innings, INNING_BINS)
    sim_inning_hist = _hist(sim_innings, INNING_BINS)

    if skip_pa_length_kl:
        pa_length_kl = float("nan")
    else:
        emp_pa_lengths = pa_lengths(empirical)
        sim_pa_lengths = pa_lengths(simulated)
        emp_pa_hist = _hist(emp_pa_lengths, list(range(1, 16)))
        sim_pa_hist = _hist(sim_pa_lengths, list(range(1, 16)))
        pa_length_kl = kl_divergence(emp_pa_hist, sim_pa_hist)

    return MetricReport(
        kl_run_distribution=kl_divergence(emp_run_hist, sim_run_hist),
        wasserstein_runs=wasserstein_1d(emp_runs, sim_runs),
        mean_rg_error=float(abs(np.mean(sim_runs) - np.mean(emp_runs))),
        variance_error=float(abs(np.var(sim_runs) - np.var(emp_runs))),
        p0_error=float(abs(np.mean(sim_runs == 0) - np.mean(emp_runs == 0))),
        p5_plus_error=float(abs(np.mean(sim_runs >= 5) - np.mean(emp_runs >= 5))),
        p8_plus_error=float(abs(np.mean(sim_runs >= 8) - np.mean(emp_runs >= 8))),
        crooked_kl=kl_divergence(emp_inning_hist, sim_inning_hist),
        pa_length_kl=pa_length_kl,
    )
