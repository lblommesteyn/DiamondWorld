"""Production joint trainer for the PA model and current ABCD Pitchformer.

The standalone PA and ABCD entry points remain deliberately untouched.  This
entry point builds paired complete-game batches, learns one seasonal player
hierarchy, and exports ordinary PA and A/B/C/D checkpoints after fitting.
"""
from __future__ import annotations

import argparse
import copy
import json
import itertools
import pickle
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.handlers as nhandlers

from diamondworldjax.data.pa_batching import build_pa_batch
from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.data.pitch_seq import build_id_maps, make_sequences
from diamondworldjax.model.embeddings import PlayerSeasonEncoder, SkillFusionLayer
from diamondworldjax.model.joint_pa_abcd import joint_pa_abcd_model
from diamondworldjax.model.multitask import task_checkpoint_params
from diamondworldjax.model.pa_checkpoint import MODEL_FIELDS
from diamondworldjax.model.pitchformer_checkpoint import save_metadata
from diamondworldjax.paths import checkpoints_root, results_root
from diamondworldjax.scripts.train_pa import (
    _build_park_index,
    _build_player_table,
    _map_player_ids,
    apply_park_idx,
)
from diamondworldjax.train.svi import train


DEFAULT_TRAIN_SEASONS = "2015,2016,2017,2018,2019,2020,2021,2022,2023"


def _joint_task_metrics(model, svi, svi_state, batch, player_table) -> dict[str, float]:
    """Read PA and ABCD likelihood diagnostics from one posterior draw.

    This is deliberately outside the differentiated update.  The combined ELBO
    remains the training objective; these values make it possible to spot one
    task degrading while the other improves during an unattended production run.
    """
    params = svi.get_params(svi_state)
    guide = nhandlers.substitute(svi.guide, data=params)
    guide_trace = nhandlers.trace(nhandlers.seed(guide, rng_seed=jax.random.PRNGKey(17))).get_trace(
        batch, player_table, teacher_force=True
    )
    conditioned = nhandlers.replay(nhandlers.substitute(model, data=params), trace=guide_trace)
    model_trace = nhandlers.trace(
        nhandlers.seed(conditioned, rng_seed=jax.random.PRNGKey(23))
    ).get_trace(batch, player_table, teacher_force=True)

    metrics: dict[str, float] = {}
    pa_site = model_trace.get("pa/pa_outcome")
    if pa_site is not None:
        valid = jnp.asarray(batch["pa"]["pa_valid"])
        count = jnp.maximum(jnp.sum(valid), 1)
        log_prob = pa_site.get("log_prob")
        if log_prob is None:
            log_prob = pa_site["fn"].log_prob(pa_site["value"])
        metrics["pa_outcome_nll"] = float(-jnp.sum(log_prob * valid) / count)
    abcd_site = model_trace.get("joint/abcd_nll_per_pitch")
    if abcd_site is not None:
        metrics["abcd_nll_per_pitch"] = float(abcd_site["value"])
    return metrics


def _validate_joint_resume(checkpoint: dict, metadata: dict) -> None:
    """Reject a resume that would silently change ABCD task semantics."""
    saved = checkpoint.get("joint_metadata")
    expected = metadata.get("joint_metadata")
    if not saved:
        raise ValueError("--resume must point to a Model 7 joint checkpoint")
    for name in ("train_seasons", "test_season", "abcd_options", "role_global_indices"):
        left, left_tree = jax.tree_util.tree_flatten(saved.get(name))
        right, right_tree = jax.tree_util.tree_flatten(expected.get(name))
        if left_tree != right_tree or len(left) != len(right) or any(
                not np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
                if np.asarray(a).dtype.kind in "fc" else
                not np.array_equal(np.asarray(a), np.asarray(b))
                for a, b in zip(left, right)):
            raise ValueError(f"Cannot resume with changed joint {name}; use the original Model 7 configuration")


def _walk_mean(params: dict, name: str, skill_prior: str) -> np.ndarray:
    """Materialise a guide mean in the same space consumed by each task."""
    mean = np.asarray(params[f"{name}_mu"])
    if skill_prior != "walk":
        return mean
    scale = float(np.asarray(params["skill_walk_sigma_loc"]))
    steps = np.concatenate([mean[:, :1], mean[:, 1:] * scale], axis=1)
    return np.cumsum(steps, axis=1)


def _export_artifacts(checkpoint_path: Path, out: Path, tag: str, *,
                      player_table: dict, maps: dict,
                      role_global_indices: dict[str, np.ndarray],
                      train_seasons: list[int], pa_metadata: dict,
                      abcd_options: dict, d_residual: int,
                      skill_prior: str, args_dict: dict) -> None:
    """Create standard PA and Pitchformer artifacts from the joint checkpoint."""
    with checkpoint_path.open("rb") as f:
        checkpoint = pickle.load(f)
    params = checkpoint["params"]
    out.mkdir(parents=True, exist_ok=True)

    # The PA export has the ordinary parameter names and posterior expected by
    # existing PA evaluators and simulators.
    pa_checkpoint = {
        "step": checkpoint["step"],
        "params": task_checkpoint_params(params, "pa"),
        "pa_metadata": pa_metadata,
        "joint_source": str(checkpoint_path),
    }
    pa_path = out / f"pa_{tag}.pkl"
    with pa_path.open("wb") as f:
        pickle.dump(jax.device_get(pa_checkpoint), f)

    # ABCD receives the posterior-mean shared + ABCD residual skill table.  Its
    # evaluator expects role-local tables, while joint training uses PA's global
    # registry, so remap only at this explicit export boundary.
    skills = (_walk_mean(params, "shared_player_skills", skill_prior)
              + _walk_mean(params, "pitch_skill_residual", skill_prior))
    if skills.ndim == 2:
        skills = skills[:, None, :]
    stats = jnp.asarray(player_table["stats"])
    league = jnp.asarray(player_table["league"])
    hand = jnp.asarray(player_table["hand"])
    encoded = PlayerSeasonEncoder(f_player=stats.shape[-1]).apply(
        {"params": params["abcd_player_encoder$params"]}, stats, league, hand
    )
    det = jnp.broadcast_to(encoded[:, None, :],
                           (skills.shape[0], skills.shape[1], encoded.shape[-1]))
    fused = SkillFusionLayer().apply(
        {"params": params["abcd_player_encoder_skill_fusion$params"]},
        det.reshape(-1, det.shape[-1]), jnp.asarray(skills).reshape(-1, skills.shape[-1]),
    ).reshape(det.shape)
    fused = np.asarray(fused)

    player_data = {}
    for role in ("pitcher", "batter"):
        table = fused[role_global_indices[role]].copy()
        table[0] = 0.0  # Pitchformer index zero is its neutral unknown sentinel.
        player_data[role] = table

    network = params["abcd_network$params"]
    for head in abcd_options["heads"]:
        head_params = copy.deepcopy(network[f"head_{head}"])
        # The joint network computes SuperState and its residual outside each
        # head.  Reinsert both at the conventional location for native ABCD
        # evaluation/rollout code.
        head_params.setdefault("trunk", {})["super_state"] = copy.deepcopy(network["shared_ss"])
        head_params["trunk"]["super_state"]["head_residual"] = copy.deepcopy(network[f"res_{head}"])
        variables = {
            "params": head_params,
            "player_data": {"trunk": {"super_state": player_data}},
        }
        with (out / f"{head.upper()}_{tag}_params.pkl").open("wb") as f:
            pickle.dump(jax.device_get(variables), f)

    model_options = {
        "player_mode": "pa",
        "skill_seasons": int(skills.shape[1]),
        "pitch_history": abcd_options["pitch_history"],
        "dropout": abcd_options["dropout"],
        "observation_masks": abcd_options["observation_masks"],
        "c_event_mode": abcd_options["c_event_mode"],
        "c_support": abcd_options["c_support"],
        "position_encoding": abcd_options["position_encoding"],
        "window_size": abcd_options["window_size"],
        "residual_dim": d_residual,
    }
    save_metadata(out, tag, {
        "version": 1,
        "joint": True,
        "config": args_dict,
        "maps": maps,
        "train_years": train_seasons,
        "skill_season_base": min(train_seasons),
        "model_options": model_options,
    })
    report = {
        "joint": True,
        "checkpoint": str(checkpoint_path),
        "pa_export": str(pa_path),
        "heads": list(abcd_options["heads"]),
        "train_seasons": train_seasons,
        "note": "Use PA and ABCD evaluators separately; the joint checkpoint is the shared training artifact.",
    }
    (out / f"joint_{tag}_report.json").write_text(json.dumps(report, indent=2))


class _PaAbcdBatches:
    """Infinite, shape-stable matched-game batch iterator for SVI."""

    def __init__(self, pa_rows, abcd_arrays: dict, game_ids: np.ndarray, *,
                 game_batch: int, max_pa: int, id_to_idx: dict,
                 player_table_np: dict, seed: int):
        import polars as pl

        self._pl = pl
        self.pa_rows = pa_rows
        self.arrays = abcd_arrays
        self.game_ids = np.asarray(game_ids)
        self.game_batch = game_batch
        self.max_pa = max_pa
        self.id_to_idx = id_to_idx
        self.player_table = jax.device_put({
            name: player_table_np[name] for name in ("stats", "league", "hand")
        })
        row_games = np.asarray(abcd_arrays["game_pk"])[:, 0]
        lookup = {int(g): np.flatnonzero(row_games == g) for g in self.game_ids}
        absent = [int(g) for g in self.game_ids if not len(lookup[int(g)])]
        if absent:
            raise ValueError(f"No ABCD sequences for {len(absent)} PA training games")
        self.max_sequences = max(len(rows) for rows in lookup.values())
        self.sequence_rows = np.zeros((len(self.game_ids), self.max_sequences), np.int32)
        self.sequence_valid = np.zeros((len(self.game_ids), self.max_sequences), bool)
        for i, gid in enumerate(self.game_ids):
            rows = lookup[int(gid)]
            self.sequence_rows[i, :len(rows)] = rows
            self.sequence_valid[i, :len(rows)] = True
        order = np.arange(len(self.game_ids))
        np.random.default_rng(seed).shuffle(order)
        self.chunks = [order[i:i + game_batch]
                       for i in range(0, len(order), game_batch)
                       if len(order[i:i + game_batch]) == game_batch]
        if not self.chunks:
            raise ValueError("Joint batch exceeds the number of training games")

    def __iter__(self):
        for chunk in itertools.cycle(self.chunks):
            games = self.game_ids[chunk]
            pa_chunk = self.pa_rows.filter(self._pl.col("game_pk").is_in(games.tolist()))
            pa_batch = build_pa_batch(pa_chunk, max_pa=self.max_pa)
            pa_batch = _map_player_ids(pa_batch, self.id_to_idx)

            rows = self.sequence_rows[chunk].reshape(-1)
            active = self.sequence_valid[chunk].reshape(-1)
            abcd_batch = {}
            for name, value in self.arrays.items():
                selected = np.asarray(value)[rows].copy()
                selected *= active.reshape((len(active),) + (1,) * (selected.ndim - 1))
                abcd_batch[name] = selected
            yield {
                "pa": pa_batch,
                "abcd": jax.device_put(abcd_batch),
            }, self.player_table


def _build_role_global_indices(maps: dict, id_to_idx: dict) -> dict[str, np.ndarray]:
    """Map ABCD's role-local IDs onto PA's shared player registry."""
    result = {}
    for role in ("pitcher", "batter"):
        value = np.zeros(maps[f"n_{role}"], np.int32)
        for raw_id, local_index in maps[role].items():
            if raw_id not in id_to_idx:
                raise ValueError(f"Training {role} {raw_id} absent from PA player registry")
            value[local_index] = id_to_idx[raw_id]
        result[role] = value
    return result


def _add_skill_season(arrays: dict, season_base: int, n_seasons: int) -> None:
    arrays["skill_season"] = np.clip(
        arrays["season"] - season_base, 0, n_seasons - 1
    ).astype(np.int32)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-seasons", default=DEFAULT_TRAIN_SEASONS)
    parser.add_argument("--test-season", type=int, default=2024)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch", type=int, default=32, help="Complete games per joint update")
    parser.add_argument("--max-pa", type=int, default=128, help="Fixed PA padding length for joint compilation")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="pa_abcd")
    parser.add_argument("--out", type=Path, default=checkpoints_root() / "pa_abcd")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Model 7 checkpoint to resume; --steps remains the total target update count")
    parser.add_argument("--update-chunk-size", type=int, default=1,
                        help="Keep joint batches distinct; one paired game chunk per compiled update")
    parser.add_argument("--prefetch-depth", type=int, default=2)
    parser.add_argument("--pa-weight", type=float, default=1.0)
    parser.add_argument("--abcd-weight", type=float, default=1.0)
    parser.add_argument("--residual-scale", type=float, default=0.35)
    parser.add_argument("--skill-prior", choices=["iso", "walk"], default="walk")
    parser.add_argument("--pa-ss-rate", type=float, default=0.0)
    parser.add_argument("--pa-ss-warmup", type=int, default=20_000)
    parser.add_argument("--missing-samples", type=int, default=2)

    # PA likelihood configuration. Defaults reproduce the production PA member
    # of the six-model runner rather than its optional sequence variants.
    parser.add_argument("--outcome-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fatigue", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--runs-upweight", type=float, default=0.0)
    parser.add_argument("--player-agg-weight", type=float, default=0.0)
    parser.add_argument("--player-agg-shrink", type=float, default=20.0)
    parser.add_argument("--pa-pitchformer", action="store_true")
    parser.add_argument("--pa-arch", choices=["transformer", "gru"], default="transformer")
    parser.add_argument("--pa-pitchformer-dim", type=int, default=128)
    parser.add_argument("--pa-pitchformer-layers", type=int, default=2)
    parser.add_argument("--pa-pitchformer-heads", type=int, default=4)
    parser.add_argument("--pa-pitchformer-dropout", type=float, default=0.0)
    parser.add_argument("--recency-halflife", type=float, default=None)
    parser.add_argument("--contact-quality", action="store_true")
    parser.add_argument("--per-stat-shrink", action="store_true")

    # ABCD configuration mirrors the existing current-Pitchformer flags.
    parser.add_argument("--stack", default="abcd")
    parser.add_argument("--max-len", type=int, default=160)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--d-residual", type=int, default=16)
    parser.add_argument("--pitch-history", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--c-event-mode", choices=["bundles", "legacy"], default="bundles")
    parser.add_argument("--history-reset", choices=["game", "half_inning", "batting_side"], default="game")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--position-encoding", choices=["learned", "sinusoidal"], default="sinusoidal")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--context-len", type=int, default=None)
    parser.add_argument("--events", default="data/processed/events.parquet")
    parser.add_argument("--game-context", default="data/processed/game_context.parquet")
    parser.add_argument("--no-env", action="store_true")
    parser.add_argument("--no-geom", action="store_true")
    parser.add_argument("--limit-train-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seasons = [int(s) for s in args.train_seasons.split(",")]
    if (args.test_season in seasons or len(set(seasons)) != len(seasons)
            or not seasons or max(seasons) != 2023):
        raise ValueError("Joint production split must be distinct train seasons ending in 2023 and held-out 2024")
    if (args.steps < 2 or args.batch < 1 or args.max_pa < 1
            or not 0 < args.residual_scale < 1
            or args.pa_weight <= 0 or args.abcd_weight <= 0
            or not 0 <= args.pa_ss_rate <= 1
            or args.missing_samples < 0
            or not args.stack or set(args.stack) - set("abcd")
            or len(set(args.stack)) != len(args.stack)):
        raise ValueError("Invalid joint training dimensions, weights, schedule, or ABCD head stack")
    if args.missing_samples and "a" not in args.stack:
        raise ValueError("Missing-pitch marginalization requires head A")
    if not 0 <= args.dropout < 1 or not 0 <= args.pa_pitchformer_dropout < 1:
        raise ValueError("Dropout must be in [0, 1)")
    if args.context_len is None:
        args.context_len = args.window_size
    if not 0 < args.window_size <= args.context_len < args.max_len:
        raise ValueError("Require 0 < --window-size <= --context-len < --max-len")

    resume_step = 0
    resume_checkpoint = None
    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(f"Joint resume checkpoint not found: {args.resume}")
        with args.resume.open("rb") as f:
            resume_checkpoint = pickle.load(f)
        resume_step = resume_checkpoint.get("step")
        if not isinstance(resume_step, int) or not 0 <= resume_step < args.steps:
            raise ValueError("--resume checkpoint step must be non-negative and smaller than --steps")

    import os
    import polars as pl
    from diamondworldjax.sim.c_transition_engine import CTransitionEngine

    print(f"Loading joint train seasons {seasons}; test {args.test_season}", flush=True)
    pitches = load_seasons(seasons)
    if args.limit_train_rows:
        pitches = pitches.head(args.limit_train_rows)

    player_table = _build_player_table(
        pitches, recency_halflife=args.recency_halflife,
        contact_quality=args.contact_quality, per_stat_shrink=args.per_stat_shrink,
    )
    park_map = _build_park_index(pitches)
    pa_rows = apply_park_idx(pitches.filter(pl.col("pa_terminal")), park_map)
    game_ids = pa_rows["game_pk"].unique().sort().to_numpy()
    maps = build_id_maps([pitches])
    role_globals = _build_role_global_indices(maps, player_table["id_to_idx"])

    events = None
    if "c" in args.stack:
        events = pl.read_parquet(args.events)
    game_context = None
    if not args.no_env and os.path.exists(args.game_context):
        game_context = pl.read_parquet(args.game_context)
    elif not args.no_env:
        print("Game context unavailable; environment features are neutral.", flush=True)

    train_arrays = make_sequences(
        pitches, maps, args.max_len, events=events, game_ctx=game_context,
        context_len=args.context_len, history_reset=args.history_reset,
        include_game_pk=True,
    )
    if args.no_geom:
        train_arrays["geom"][:] = 0.0
    skill_seasons = len(seasons) if args.skill_prior == "walk" else 1
    _add_skill_season(train_arrays, min(seasons), skill_seasons)

    c_support = None
    if "c" in args.stack and args.c_event_mode == "bundles":
        c_support = CTransitionEngine(event_mode="bundles").fit(pitches, events).support()
        support = np.asarray(c_support).reshape(256, 24)
        target = (train_arrays["events"].astype(np.int32) * (1 << np.arange(8))).sum(-1)
        base_state = sum((train_arrays["ctx"][..., 3 + i] > .5).astype(np.int32) * (1 << i)
                         for i in range(3))
        outs = np.clip(np.rint(train_arrays["ctx"][..., 2] * 2).astype(np.int32), 0, 2)
        train_arrays["c_eligible"] &= support[target, base_state * 3 + outs]

    abcd_options = {
        "n_pitchers": maps["n_pitcher"], "n_batters": maps["n_batter"],
        "n_parks": maps["n_park"], "d_model": args.d_model,
        "n_layers": args.layers, "n_heads": args.heads,
        "d_residual": args.d_residual, "dropout": args.dropout,
        "heads": args.stack, "skill_seasons": skill_seasons,
        "pitch_history": args.pitch_history,
        "position_encoding": args.position_encoding,
        "window_size": args.window_size, "observation_masks": True,
        "c_event_mode": args.c_event_mode, "c_support": c_support,
    }
    pa_kwargs = {
        "outcome_only": args.outcome_only,
        "fatigue": args.fatigue,
        "runs_upweight": args.runs_upweight,
        "player_agg_weight": args.player_agg_weight,
        "player_agg_shrink": args.player_agg_shrink,
        "season_base": min(seasons), "n_seasons": skill_seasons,
        "pitchformer": args.pa_pitchformer,
    }
    if args.pa_pitchformer:
        pa_kwargs.update(
            pitchformer_dim=args.pa_pitchformer_dim,
            pitchformer_layers=args.pa_pitchformer_layers,
            pitchformer_heads=args.pa_pitchformer_heads,
            pitchformer_dropout=args.pa_pitchformer_dropout,
            pa_arch=args.pa_arch, pitchformer_position="sinusoidal",
        )

    pa_config = {
        "outcome_only": args.outcome_only, "fatigue": args.fatigue,
        "platoon": False, "bilinear_rank": 0, "nested": False,
        "skill_prior": args.skill_prior, "pitchformer": args.pa_pitchformer,
        "pitchformer_dim": args.pa_pitchformer_dim,
        "pitchformer_layers": args.pa_pitchformer_layers,
        "pitchformer_heads": args.pa_pitchformer_heads,
        "pitchformer_dropout": args.pa_pitchformer_dropout,
        "pa_arch": args.pa_arch, "pitchformer_position": "sinusoidal",
        "runs_upweight": args.runs_upweight,
        "train_end": max(seasons), "recency_halflife": args.recency_halflife,
        "contact_quality": args.contact_quality,
        "per_stat_shrink": args.per_stat_shrink,
    }
    assert set(MODEL_FIELDS).issubset(pa_config)
    pa_metadata = {
        "version": 1, "train_seasons": seasons, "config": pa_config,
        "player_table": player_table, "park_map": park_map,
    }
    joint_metadata = {
        "joint_metadata": {
            "version": 1, "train_seasons": seasons, "test_season": args.test_season,
            "config": vars(args), "abcd_options": abcd_options,
            "role_global_indices": role_globals,
        },
        "pa_metadata": pa_metadata,
    }
    if resume_checkpoint is not None:
        _validate_joint_resume(resume_checkpoint, joint_metadata)

    iterator = _PaAbcdBatches(
        pa_rows, train_arrays, game_ids, game_batch=args.batch, max_pa=args.max_pa,
        id_to_idx=player_table["id_to_idx"], player_table_np=player_table,
        seed=args.seed,
    )
    kl_scale = args.batch / max(len(game_ids), 1)
    model = partial(
        joint_pa_abcd_model, abcd_options=abcd_options,
        role_global_indices={k: jnp.asarray(v) for k, v in role_globals.items()},
        pa_model_kwargs=pa_kwargs, pa_weight=args.pa_weight,
        abcd_weight=args.abcd_weight, residual_scale=args.residual_scale,
        kl_scale=kl_scale, skill_prior=args.skill_prior, n_seasons=skill_seasons,
        missing_samples=args.missing_samples,
    )
    checkpoint_dir = args.out / f"joint_{args.tag}"
    log_path = results_root() / f"dwjax_joint_{args.tag}_elbo.json"
    print(
        f"Joint PA+ABCD: {len(game_ids):,} games, batch={args.batch}, "
        f"KL scale={kl_scale:.6g}, PA:ABCD={args.pa_weight}:{args.abcd_weight}",
        flush=True,
    )
    if args.resume is not None:
        print(f"Resuming Model 7 at global step {resume_step:,} toward {args.steps:,}", flush=True)
    train(
        model=model, batch_iter=iter(iterator), n_steps=args.steps, lr=args.lr,
        seed=args.seed, ckpt_dir=checkpoint_dir, log_path=log_path,
        resume_path=args.resume, start_step=resume_step,
        cosine_decay=True, ss_max_rate=args.pa_ss_rate,
        ss_warmup_steps=args.pa_ss_warmup, ss_start_step=5_000,
        kl_scale=kl_scale, skill_prior=args.skill_prior, n_seasons=skill_seasons,
        shared_task_skills=True, checkpoint_metadata=joint_metadata,
        update_chunk_size=args.update_chunk_size, prefetch_depth=args.prefetch_depth,
        metrics_fn=_joint_task_metrics,
    )
    checkpoint_path = checkpoint_dir / f"dwjax_step_{args.steps:07d}.pkl"
    _export_artifacts(
        checkpoint_path, args.out, args.tag, player_table=player_table, maps=maps,
        role_global_indices=role_globals, train_seasons=seasons,
        pa_metadata=pa_metadata, abcd_options=abcd_options,
        d_residual=args.d_residual, skill_prior=args.skill_prior,
        args_dict=vars(args),
    )
    print(f"Joint checkpoint: {checkpoint_path}", flush=True)
    print(f"Exported PA and ABCD heads to {args.out}", flush=True)


if __name__ == "__main__":
    main()
