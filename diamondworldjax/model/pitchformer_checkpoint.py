"""ABCD checkpoint metadata, PA skill transfer, and shared-head export."""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import flax
import optax


def trainable_optimizer(optimizer):
    """Frozen player_data is neither differentiated nor decayed by AdamW."""
    return optax.masked(optimizer, lambda tree: jax.tree_util.tree_map_with_path(
        lambda path, _: path[0].key == "params", tree))


def transfer_pa_skills(path, maps, mode, train_years):
    """Export the exact PA encoder+fusion and posterior-mean skill representation.

    pa-no-latent holds the same encoder/statistics fixed and zeros latent skills.
    Seasonal priors retain their season axis and clamp future seasons as in PA.
    """
    from .embeddings import PlayerSeasonEncoder, SkillFusionLayer
    with Path(path).open("rb") as f:
        ckpt = pickle.load(f)
    meta = ckpt.get("pa_metadata")
    if not meta:
        raise ValueError("PA skill transfer requires a checkpoint with pa_metadata (saved player table/split)")
    if max(meta["train_seasons"]) > max(train_years):
        raise ValueError("PA skill checkpoint contains seasons beyond ABCD training; this would leak evaluation data")
    params, table = ckpt["params"], meta["player_table"]
    cfg = meta["config"]
    mu = jnp.asarray(params["player_mu"])
    if cfg["skill_prior"] == "walk":
        sigma = params["skill_walk_sigma_loc"]
        mu = jnp.cumsum(jnp.concatenate([mu[:, :1], mu[:, 1:] * sigma], axis=1), axis=1)
    else:
        mu = mu[:, None, :]
    if mode == "pa-no-latent":
        mu = jnp.zeros_like(mu)
    enc = PlayerSeasonEncoder(f_player=table["stats"].shape[-1])
    det = enc.apply({"params": params["player_encoder$params"]},
                    jnp.asarray(table["stats"]), jnp.asarray(table["league"]), jnp.asarray(table["hand"]))
    fused = SkillFusionLayer().apply({"params": params["player_encoder_skill_fusion$params"]},
        jnp.broadcast_to(det[:, None, :], (*mu.shape[:2], det.shape[-1])), mu)
    fused = np.asarray(fused)
    result = {}
    for role in ("pitcher", "batter"):
        value = np.zeros((maps[f"n_{role}"], fused.shape[1], 64), np.float32)
        for pid, target in maps[role].items():
            source = table["id_to_idx"].get(pid)
            if source is not None:
                value[target] = fused[source]
        result[role] = value
    return result, min(meta["train_seasons"])


def install_player_data(variables, tables):
    if tables is None:
        return variables
    result = flax.core.unfreeze(variables)
    def fill(node):
        if "pitcher" in node and "batter" in node:
            node.update({k: jnp.asarray(v) for k, v in tables.items()})
        else:
            for value in node.values():
                if isinstance(value, dict):
                    fill(value)
    fill(result["player_data"])
    return flax.core.freeze(result) if isinstance(variables, flax.core.FrozenDict) else result


def export_shared_head(variables, head):
    """Produce an ordinary head tree with an equivalent SuperState residual."""
    source = flax.core.unfreeze(variables)
    out = {"params": source["params"][f"head_{head}"]}
    ss = dict(source["params"]["shared_ss"])
    ss["head_residual"] = source["params"][f"res_{head}"]
    out["params"]["trunk"]["super_state"] = ss
    if "player_data" in source:
        out["player_data"] = {"trunk": {"super_state": source["player_data"]["shared_ss"]}}
    return out


def save_metadata(directory, tag, metadata):
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / f"{tag}_metadata.pkl").open("wb") as f:
        pickle.dump(metadata, f)


def load_metadata(directory, tag):
    path = Path(directory) / f"{tag}_metadata.pkl"
    if not path.exists():
        warnings.warn("Legacy ABCD checkpoint: supply original architecture and training vocabulary")
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def restore_metadata(args, tag=None):
    meta = load_metadata(args.params_dir, tag or args.tag)
    if meta:
        for name in ("d_model", "layers", "heads", "max_len", "no_env", "no_geom",
                     "position_encoding", "window_size", "history_reset", "context_len", "dropout"):
            if name in meta["config"]:
                setattr(args, name, meta["config"][name])
        if hasattr(args, "season") and args.season in meta["train_years"]:
            raise ValueError("Evaluation season is in ABCD training data")
    return meta


def head_kwargs(args, maps, metadata=None):
    result = dict(n_pitchers=maps["n_pitcher"], n_batters=maps["n_batter"],
                  n_parks=maps["n_park"], d_model=args.d_model,
                  n_layers=args.layers, n_heads=args.heads)
    if metadata:
        result.update(metadata["model_options"])
    return result


def add_skill_season(arrays, metadata):
    if metadata:
        arrays["skill_season"] = np.clip(
            arrays["season"] - metadata["skill_season_base"], 0,
            metadata["model_options"]["skill_seasons"] - 1).astype(np.int32)
    return arrays
