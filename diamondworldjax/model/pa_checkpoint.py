"""Portable PA checkpoint configuration and posterior materialization."""
from __future__ import annotations

import warnings
import numpy as np

MODEL_FIELDS = (
    "outcome_only", "fatigue", "platoon", "bilinear_rank", "nested", "skill_prior",
    "pitchformer", "pitchformer_dim", "pitchformer_layers", "pitchformer_heads",
    "pitchformer_dropout", "pa_arch", "pitchformer_position", "runs_upweight",
)


def add_eval_arguments(parser):
    """Add missing legacy overrides; new checkpoints restore these automatically."""
    options = {a.dest for a in parser._actions}
    defaults = dict(train_end=2022, recency_halflife=None, contact_quality=False,
                    per_stat_shrink=False, platoon=False, nested=False,
                    bilinear_rank=0, skill_prior="iso", skill_mode="mean")
    for name, default in defaults.items():
        if name in options:
            continue
        flag = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(flag, action="store_true")
        elif name == "skill_mode":
            parser.add_argument(flag, choices=["mean", "sample", "prior"], default=default)
        elif name == "skill_prior":
            parser.add_argument(flag, choices=["iso", "walk", "learned", "lkj"], default=default)
        else:
            parser.add_argument(flag, type=float if name == "recency_halflife" else int,
                                default=default)
    parser.add_argument("--test-seasons", default=None,
                        help="Comma-separated held-out seasons; defaults to seasons after training through 2024.")


def restore_config(checkpoint, args):
    meta = checkpoint.get("pa_metadata")
    if meta:
        for name, value in meta["config"].items():
            setattr(args, name, value)
        args.use_park = True
        train = list(meta["train_seasons"])
    else:
        warnings.warn("Legacy PA checkpoint: using explicit CLI configuration; verify the training split and features.")
        train = list(range(2015, args.train_end + 1))
    test_arg = getattr(args, "test_seasons", None)
    test = ([int(s) for s in test_arg.split(",")] if test_arg
            else list(range(max(train) + 1, 2025)))
    if not test or set(train) & set(test):
        raise ValueError("Evaluation requires nonempty held-out --test-seasons disjoint from training")
    return train, test


def model_kwargs(args):
    result = {name: getattr(args, name) for name in MODEL_FIELDS if hasattr(args, name)}
    result.update(season_base=2015, n_seasons=args.train_end - 2015 + 1)
    return result


def player_table(checkpoint, pitches, args):
    from diamondworldjax.scripts.train_pa import _build_player_table, _build_park_index
    meta = checkpoint.get("pa_metadata")
    if meta:
        return meta["player_table"], meta["park_map"] if args.use_park else None
    return (_build_player_table(pitches, recency_halflife=args.recency_halflife,
                               contact_quality=args.contact_quality,
                               per_stat_shrink=args.per_stat_shrink),
            _build_park_index(pitches) if args.use_park else None)


def posterior_params(params, skill_prior="iso", mode="mean", seed=0):
    """Bind sample-site values, never confuse variational parameters with samples."""
    import jax.numpy as jnp
    result = dict(params)
    if mode == "prior":
        return result
    site = "player_skill_eps" if skill_prior == "walk" else "player_skills"
    if "player_mu" not in result:
        if site not in result:
            raise ValueError("Checkpoint has no learned player posterior; explicitly request prior mode for legacy models")
        return result
    value = np.asarray(result["player_mu"])
    if mode == "sample":
        value = value + np.asarray(result["player_sigma"]) * np.random.default_rng(seed).standard_normal(value.shape)
    result[site] = jnp.asarray(value)
    for sample, param in (("skill_walk_sigma", "skill_walk_sigma_loc"),
                          ("skill_tau", "skill_tau_loc"), ("skill_L", "skill_L_loc")):
        if param in result:
            result[sample] = result[param]
    return result
