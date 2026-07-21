"""Scenario simulator: run arbitrary game specs through v13 with replicas.

The foundation for the applied analyses (counterfactuals, lineup optimization,
correlated tail-risk, win-probability-with-uncertainty). A "spec" is a dict:
  {away_lineup:[9 idx], home_lineup:[9 idx], away_staff:[idx...],
   home_staff:[idx...], park:int}
Replicating a spec R times and simulating once yields its full outcome
distribution (correlated across the whole game, which is the point).
"""
from __future__ import annotations
import pickle
from functools import partial
import numpy as np
import polars as pl

from diamondworldjax.paths import processed_root
from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.model.pa_model import pa_model
from diamondworldjax.sim.rules_engine import EmpiricalEngine
from diamondworldjax.sim.game_extract import extract_games, fit_hook_dists
from diamondworldjax.scripts.train_pa import _build_player_table, _build_park_index, apply_park_idx
from diamondworldjax.scripts.simulate_games import simulate, TRAIN, TEST

V13 = "checkpoints/dwjax_pa_v13/dwjax_step_0050000.pkl"
RECAL = "data/eval2/v13_cal_params.npz"


class Sim:
    def __init__(self, ckpt=V13, recal=RECAL, recency_hl=2.0, scale=0.18,
                 skill_mode="mean", recal_key="b_heur"):
        import jax, jax.numpy as jnp
        self.jax = jax
        self.params = pickle.load(open(ckpt, "rb"))["params"]
        train = load_seasons(TRAIN, data_root=processed_root())
        self.ptab = _build_player_table(train, recency_halflife=recency_hl)
        self.park_map = _build_park_index(train)
        tp = train.filter(pl.col("pa_terminal"))
        engine = EmpiricalEngine().fit(tp)
        hooks = fit_hook_dists(tp)
        del train
        self.id2i = self.ptab["id_to_idx"]
        P = len(self.ptab["hand"])
        self.pt = {"stats": jnp.array(self.ptab["stats"]), "league": jnp.array(self.ptab["league"]),
                   "hand": jnp.array(self.ptab["hand"]),
                   "bat_hand": np.asarray(self.ptab.get("bat_hand", np.full(P, .5, np.float32))),
                   "pit_hand": np.asarray(self.ptab.get("pit_hand", np.full(P, .5, np.float32))),
                   "_engine": engine, "_hook_dists": hooks}
        self.model_fn = partial(pa_model, outcome_only=True, fatigue=True)
        self.recal_vec = np.load(recal)[recal_key].astype(np.float64)
        self.scale, self.skill_mode = scale, skill_mode
        self.stats = self.ptab["stats"]   # per-player [hit,bb,k,hr] rates

    def run(self, specs, R=200, seed=0, skill_mode=None, no_bullpen=False):
        """Return home (n,R), away (n,R) run totals."""
        sm = skill_mode or self.skill_mode
        games = [dict(s, game_pk=i) for i, s in enumerate(specs) for _ in range(R)]
        res = simulate(self.model_fn, self.params, self.pt, games,
                       self.jax.random.PRNGKey(seed), recal=True, recal_scale=self.scale,
                       recal_vec=self.recal_vec, seed=seed, skill_mode=sm, no_bullpen=no_bullpen)
        n = len(specs)
        return res["home"].reshape(n, R), res["away"].reshape(n, R)

    # ---- real games as specs ----
    def real_games(self, season=2024, limit=400):
        te = load_seasons([season], data_root=processed_root()).filter(pl.col("pa_terminal"))
        keep = te["game_pk"].unique().sort().to_numpy()[:limit]
        te = te.filter(pl.col("game_pk").is_in(keep.tolist()))
        te = apply_park_idx(te, self.park_map)
        games = extract_games(te, self.id2i, park_map=self.park_map)
        return games  # each has game_pk, away/home_lineup, away/home_staff, park

    def player_rate(self, idx):
        return self.stats[idx]  # [hit,bb,k,hr]
