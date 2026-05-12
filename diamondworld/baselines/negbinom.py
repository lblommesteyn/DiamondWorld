from __future__ import annotations

from collections import defaultdict

import numpy as np
import polars as pl
from scipy import stats

from diamondworld.baselines.base import BaseSimulator, GameLog


class NegBinomSimulator(BaseSimulator):
    """Baseline B2: Per-inning run distribution fitted as NegBinom (or Poisson).

    For each (inning 1-9, half top/bot) combination we fit a negative binomial
    via method of moments to the observed run distribution, then simulate games
    by sampling runs independently per half-inning.
    """

    def fit(self, pitches: pl.DataFrame) -> None:
        terminal = pitches.filter(pl.col("pa_terminal"))
        # Aggregate runs per (game_pk, inning, half)
        grouped = (
            terminal
            .group_by(["game_pk", "inning", "half"])
            .agg(pl.col("runs_scored").sum().alias("runs"))
        )

        # Collect observations per (inning, half)
        obs: dict[tuple, list[int]] = defaultdict(list)
        for row in grouped.iter_rows(named=True):
            inning = int(row["inning"])
            half = row["half"]
            runs = int(row["runs"]) if row["runs"] is not None else 0
            if 1 <= inning <= 9:
                obs[(inning, half)].append(runs)

        self._params: dict[tuple, tuple[float, float]] = {}
        self._is_poisson: dict[tuple, bool] = {}

        for key, values in obs.items():
            arr = np.array(values, dtype=np.float64)
            mu = float(arr.mean()) if len(arr) > 0 else 0.5
            var = float(arr.var()) if len(arr) > 1 else mu

            if var > mu and mu > 0:
                # Negative binomial via method of moments
                p = mu / var
                n = mu * p / (1.0 - p)
                self._params[key] = (max(n, 1e-3), float(np.clip(p, 1e-6, 1 - 1e-6)))
                self._is_poisson[key] = False
            else:
                # Treat as Poisson: use NegBinom with large n (approx Poisson)
                lam = max(mu, 1e-6)
                # NegBinom with p -> 1 and n -> inf, n*(1-p)/p = lam
                # Use large n trick: n=1e6, p=n/(n+lam)
                n_large = 1e6
                p_approx = n_large / (n_large + lam)
                self._params[key] = (n_large, float(p_approx))
                self._is_poisson[key] = True

        # Compute global fallback (all observations pooled)
        all_obs = [r for v in obs.values() for r in v]
        if all_obs:
            arr = np.array(all_obs, dtype=np.float64)
            mu = float(arr.mean())
            var = float(arr.var()) if len(arr) > 1 else mu
            if var > mu and mu > 0:
                p = mu / var
                n = mu * p / (1.0 - p)
                self._fallback_params = (max(n, 1e-3), float(np.clip(p, 1e-6, 1 - 1e-6)))
            else:
                lam = max(mu, 1e-6)
                n_large = 1e6
                self._fallback_params = (n_large, float(n_large / (n_large + lam)))
        else:
            self._fallback_params = (1.0, 0.5)

    def _sample_runs(self, inning: int, half: str) -> int:
        """Sample runs for one (inning, half) from the fitted distribution."""
        params = self._params.get((inning, half), self._fallback_params)
        n, p = params
        # scipy.stats.nbinom(n, p): mean = n*(1-p)/p
        return int(stats.nbinom.rvs(n, p))

    def simulate_game(self, game_context: dict) -> GameLog:
        game_id = game_context.get("game_id", 0)
        inning_runs_away = []  # top = away bats
        inning_runs_home = []  # bot = home bats

        for inning in range(1, 10):
            inning_runs_away.append(self._sample_runs(inning, "top"))
            inning_runs_home.append(self._sample_runs(inning, "bot"))

        return GameLog(
            game_id=game_id,
            home_runs=sum(inning_runs_home),
            away_runs=sum(inning_runs_away),
            inning_runs_home=inning_runs_home,
            inning_runs_away=inning_runs_away,
        )
