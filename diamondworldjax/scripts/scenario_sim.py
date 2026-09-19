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
from diamondworldjax.sim.game_extract import extract_games, fit_hook_dists, fit_hook_model
from diamondworldjax.scripts.train_pa import _build_player_table, _build_park_index, apply_park_idx
from diamondworldjax.scripts.simulate_games import simulate, TRAIN, TEST

V13 = "checkpoints/dwjax_pa_v13/dwjax_step_0050000.pkl"
V15 = "checkpoints/dwjax_pa_v15/dwjax_step_0035000.pkl"
V16 = "checkpoints/dwjax_pa_v16/dwjax_step_0050000.pkl"
RECAL = "data/eval2/v13_cal_params.npz"


class Sim:
    def __init__(self, ckpt=V15, recal=RECAL, recency_hl=2.0, scale=0.18,
                 skill_mode="mean", recal_key="b_heur", train_end=2023, hook_model=False,
                 contact_quality=False, per_stat_shrink=False, skill_prior="iso",
                 pitchformer=False, pa_arch="transformer", apply_recal=True):
        # v15 (default) is trained through 2023, so its player embeddings are index-locked
        # to a 2015-2023 table; train_end must match the checkpoint (2022 for v13).
        # contact_quality must match the checkpoint too (True for v16, else the model
        # gets zeros in the columns it was trained to read).
        import jax, jax.numpy as jnp
        self.jax = jax
        checkpoint = pickle.load(open(ckpt, "rb"))
        self.params = checkpoint["params"]
        metadata = checkpoint.get("pa_metadata")
        config = metadata["config"] if metadata else {}
        if metadata:
            # Use the stored feature table and architecture rather than trusting
            # caller defaults.  A mismatched player table changes model inputs
            # even though parameter shapes still load successfully.
            train_end = config["train_end"]
            recency_hl = config["recency_halflife"]
            contact_quality = config["contact_quality"]
            per_stat_shrink = config["per_stat_shrink"]
            skill_prior = config["skill_prior"]
            pitchformer = config["pitchformer"]
            pa_arch = config["pa_arch"]
        train_seasons = list(range(2015, train_end + 1))
        train = load_seasons(train_seasons, data_root=processed_root())
        if metadata:
            self.ptab = metadata["player_table"]
            self.park_map = metadata["park_map"]
        else:
            self.ptab = _build_player_table(train, recency_halflife=recency_hl,
                                            contact_quality=contact_quality,
                                            per_stat_shrink=per_stat_shrink)
            self.park_map = _build_park_index(train)
        tp = train.filter(pl.col("pa_terminal"))
        engine = EmpiricalEngine().fit(tp)
        hooks = fit_hook_dists(tp)
        # State-dependent starter-pull hazard (endogenous, pre-game-legit bullpen).
        self.hook_model = fit_hook_model(tp) if hook_model else None
        del train
        self.id2i = self.ptab["id_to_idx"]
        self.unknown_idx = self.ptab["unknown_index"]
        P = len(self.ptab["hand"])
        self.pt = {"stats": jnp.array(self.ptab["stats"]), "league": jnp.array(self.ptab["league"]),
                   "hand": jnp.array(self.ptab["hand"]),
                   "unknown_index": self.ptab["unknown_index"],
                   "bat_hand": np.asarray(self.ptab.get("bat_hand", np.full(P, .5, np.float32))),
                   "pit_hand": np.asarray(self.ptab.get("pit_hand", np.full(P, .5, np.float32))),
                   "_engine": engine, "_hook_dists": hooks}
        _mkw = dict(outcome_only=config.get("outcome_only", True),
                    fatigue=config.get("fatigue", True),
                    platoon=config.get("platoon", False),
                    nested=config.get("nested", False),
                    bilinear_rank=config.get("bilinear_rank", 0),
                    skill_prior=skill_prior)
        if skill_prior == "walk":
            _mkw.update(season_base=train_seasons[0], n_seasons=len(train_seasons))
        if pitchformer:
            _mkw.update(pitchformer=True, pa_arch=pa_arch,
                        pitchformer_dim=config.get("pitchformer_dim", 128),
                        pitchformer_layers=config.get("pitchformer_layers", 2),
                        pitchformer_heads=config.get("pitchformer_heads", 4),
                        pitchformer_dropout=config.get("pitchformer_dropout", 0.0),
                        pitchformer_position=config.get("pitchformer_position", "sinusoidal"))
        self.model_fn = partial(pa_model, **_mkw)
        self.pitchformer = pitchformer
        self.pa_arch = pa_arch
        self.skill_prior = skill_prior
        self.train_end = train_end
        self.recency_hl = recency_hl
        self.contact_quality = contact_quality
        self.per_stat_shrink = per_stat_shrink
        self.apply_recal = apply_recal
        self.recal_vec = (
            np.load(recal)[recal_key].astype(np.float64) if apply_recal else None
        )
        self.scale, self.skill_mode = scale, skill_mode
        self.stats = self.ptab["stats"]   # per-player [hit,bb,k,hr] rates
        self._mean_inference = None
        self._mean_seq_inference = None
        self._mean_adapter_built = False

    def _mean_fast_adapter(self):
        """Construct the mean-skill JAX adapter once and reuse it across chunks."""
        if self._mean_adapter_built:
            return self._mean_inference, self._mean_seq_inference
        self._mean_adapter_built = True

        import jax.numpy as jnp
        from diamondworldjax.model.pa_inference import (
            build_pa_inference, build_pa_sequence_inference,
        )

        # Match simulate(..., skill_mode="mean") exactly. The adapter owns only
        # immutable tables/parameters; game state remains in simulate.
        params = dict(self.params)
        if "player_mu" in params:
            if self.skill_prior == "walk":
                params["player_skill_eps"] = jnp.asarray(params["player_mu"])
                if "skill_walk_sigma_loc" in params:
                    params["skill_walk_sigma"] = params["skill_walk_sigma_loc"]
            else:
                params["player_skills"] = jnp.asarray(params["player_mu"])
        common = dict(
            outcome_only=True, fatigue=True, platoon=False,
            nested=False, bilinear_rank=0, skill_prior=self.skill_prior,
            season_base=2015, n_seasons=self.train_end - 2015 + 1,
            simulation_season=2024,
        )
        try:
            if self.pitchformer:
                self._mean_seq_inference = build_pa_sequence_inference(
                    params, self.pt, pa_arch=self.pa_arch, **common)
            else:
                self._mean_inference = build_pa_inference(params, self.pt, **common)
        except (KeyError, ValueError):
            # Keep the established generic fallback for unsupported checkpoints.
            pass
        return self._mean_inference, self._mean_seq_inference

    def run(self, specs, R=200, seed=0, skill_mode=None, no_bullpen=False, crn=True):
        """Return home (n,R), away (n,R) run totals.

        crn (default True): pair the random stream by replica index across
        scenarios, so replica r of every spec sees the same game randomness. For a
        counterfactual (baseline vs one-change) the shared noise cancels in the
        difference, giving a far tighter estimate of the causal delta at the same
        R. Set False for the legacy independent-noise behaviour.
        """
        sm = skill_mode or self.skill_mode
        cached_inference = cached_seq_inference = None
        if sm == "mean":
            cached_inference, cached_seq_inference = self._mean_fast_adapter()
        games = [dict(s, game_pk=i) for i, s in enumerate(specs) for _ in range(R)]
        # Spec-major flatten: position p is spec (p//R), replica (p%R). Keying the
        # stream on the replica index pairs the same replica across all specs.
        crn_keys = np.tile(np.arange(R), len(specs)) if crn else None
        res = simulate(self.model_fn, self.params, self.pt, games,
                       self.jax.random.PRNGKey(seed), recal=self.apply_recal, recal_scale=self.scale,
                       recal_vec=self.recal_vec, seed=seed, skill_mode=sm, no_bullpen=no_bullpen,
                       crn_keys=crn_keys, hook_model=self.hook_model,
                       pitchformer=self.pitchformer, skill_prior=self.skill_prior,
                       simulation_season=2024, cached_inference=cached_inference,
                       cached_seq_inference=cached_seq_inference)
        n = len(specs)
        return res["home"].reshape(n, R), res["away"].reshape(n, R)

    # ---- real games as specs ----
    def real_games(self, season=2024, limit=400, pregame_staff=False):
        """Game specs for the simulator.

        pregame_staff=True replaces each side's realized staff, which extract_games
        reads off the completed game in actual appearance order, with one selected
        from prior games only (diamondworldjax.sim.pregame_staff). The realized
        version is look-ahead and biases every game-level result built on it; see
        the leakage note in game_extract.extract_games.
        """
        te = load_seasons([season], data_root=processed_root()).filter(pl.col("pa_terminal"))
        keep = te["game_pk"].unique().sort().to_numpy()[:limit]
        te = te.filter(pl.col("game_pk").is_in(keep.tolist()))
        te = apply_park_idx(te, self.park_map)
        games = extract_games(te, self.id2i, park_map=self.park_map,
                              unknown_idx=self.ptab["unknown_index"])
        if pregame_staff:
            from diamondworldjax.sim.pregame_staff import pregame_staffs
            staffs = pregame_staffs(te, self.id2i, self.ptab["unknown_index"])
            # half_bin 0 (top, away batting) is pitched by the HOME staff, and vice
            # versa. extract_games tags them this way and the v1 extractor got it
            # crossed, so the mapping is spelled out rather than inferred.
            for g in games:
                gp = int(g["game_pk"])
                for half, fielding in ((0, "home"), (1, "away")):
                    st = staffs.get((gp, half))
                    if st:
                        g[f"{fielding}_staff"] = st
        return games  # each has game_pk, away/home_lineup, away/home_staff, park

    def player_rate(self, idx):
        return self.stats[idx]  # [hit,bb,k,hr]
