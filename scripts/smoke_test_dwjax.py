"""Smoke test for DiamondWorldJAX: verify end-to-end forward pass on CPU.

Loads 50 games from 2015, builds batch + player table, runs one SVI init
step, one SVI update step, and one free-rollout predictive sample.
Prints shapes and ELBO at each stage.
"""
import sys
sys.path.insert(0, "/scratch/lblommes/diamondworld")

import numpy as np
import polars as pl
from pathlib import Path

DATA_ROOT = Path("/scratch/lblommes/diamondworld/data/processed")
N_GAMES   = 50   # small enough to fit in CPU memory comfortably

print("=" * 60)
print("DiamondWorldJAX smoke test")
print("=" * 60)

# ------------------------------------------------------------------
# 1. Load small data slice
# ------------------------------------------------------------------
print("\n[1] Loading data...", flush=True)
df = pl.read_parquet(DATA_ROOT / "pitches_2015.parquet")
game_ids = df["game_pk"].unique()[:N_GAMES].to_list()
df = df.filter(pl.col("game_pk").is_in(game_ids))
print(f"    {len(df):,} pitches from {N_GAMES} games")

# ------------------------------------------------------------------
# 2. Data pipeline
# ------------------------------------------------------------------
print("\n[2] Running data pipeline...", flush=True)
from diamondworldjax.data.pipeline import load_season
enriched = load_season(2015, data_root=DATA_ROOT)
enriched = enriched.filter(pl.col("game_pk").is_in(game_ids))
print(f"    {len(enriched):,} enriched pitches")
print(f"    columns: {enriched.columns[:10]}...")

# ------------------------------------------------------------------
# 3. Build batch
# ------------------------------------------------------------------
print("\n[3] Building batch...", flush=True)
from diamondworldjax.data.batching import build_batch
batch = build_batch(enriched)
B, T = batch["pitch_valid"].shape
print(f"    batch shape: B={B} games, T={T} pitches")
print(f"    pitch_valid: {int(batch['pitch_valid'].sum())} valid positions")
for k in ["pitch_type", "runs_scored", "base_state", "in_play_mask"]:
    print(f"    {k}: shape={batch[k].shape}, dtype={batch[k].dtype}")

# ------------------------------------------------------------------
# 4. Build player table
# ------------------------------------------------------------------
print("\n[4] Building player table...", flush=True)
pitcher_col = "pitcher_id" if "pitcher_id" in enriched.columns else "pitcher_idx"
batter_col  = "batter_id"  if "batter_id"  in enriched.columns else "batter_idx"

all_ids = np.unique(np.concatenate([
    enriched[pitcher_col].to_numpy(),
    enriched[batter_col].to_numpy(),
])).astype(np.int32)
P = len(all_ids)
id_to_idx = {int(pid): i for i, pid in enumerate(all_ids)}
F_PLAYER = 16

import jax.numpy as jnp
player_table = {
    "stats":  jnp.zeros((P, F_PLAYER), dtype=jnp.float32),
    "league": jnp.zeros((P,), dtype=jnp.int32),
    "hand":   jnp.zeros((P,), dtype=jnp.int32),
}
print(f"    P={P} unique players, F={F_PLAYER}")

# Remap player IDs in batch
def remap(arr):
    arr_np = np.array(arr)
    out = np.vectorize(lambda x: id_to_idx.get(int(x), 0))(arr_np)
    return jnp.array(out.astype(np.int32))

batch["pitcher_ids"] = remap(batch["pitcher_ids"])
batch["batter_ids"]  = remap(batch["batter_ids"])
print(f"    pitcher_ids range: {int(batch['pitcher_ids'].min())}..{int(batch['pitcher_ids'].max())}")
print(f"    batter_ids range:  {int(batch['batter_ids'].min())}..{int(batch['batter_ids'].max())}")

# ------------------------------------------------------------------
# 5. Trace model (teacher_force=True) — no SVI, just check site names
# ------------------------------------------------------------------
print("\n[5] Tracing model sample sites...", flush=True)
import jax
import numpyro
import numpyro.handlers as handlers
from diamondworldjax.model.joint import diamondworld_model

rng_key = jax.random.PRNGKey(0)
model_trace = handlers.trace(handlers.seed(diamondworld_model, rng_key)).get_trace(
    batch, player_table, teacher_force=True
)

sites = {k: v for k, v in model_trace.items() if v["type"] == "sample"}
print(f"    {len(sites)} sample sites found:")
for name, site in sorted(sites.items()):
    val = site["value"]
    shape = val.shape if hasattr(val, "shape") else "scalar"
    fn   = site.get("fn", None)
    dist_name = type(fn).__name__ if fn is not None else "?"
    print(f"      {name:<35s} shape={str(shape):<20s} dist={dist_name}")

# ------------------------------------------------------------------
# 6. One SVI init + update step
# ------------------------------------------------------------------
print("\n[6] SVI init...", flush=True)
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal
from numpyro.optim import Adam

guide     = AutoLowRankMultivariateNormal(diamondworld_model, rank=20)
optimizer = Adam(1e-3)
svi       = SVI(diamondworld_model, guide, optimizer, loss=Trace_ELBO(num_particles=1))

rng_key, init_key = jax.random.split(rng_key)
svi_state = svi.init(init_key, batch, player_table, teacher_force=True)
print("    SVI initialised OK")

print("\n[7] SVI update step...", flush=True)
rng_key, step_key = jax.random.split(rng_key)
svi_state, loss = svi.update(svi_state, batch, player_table, teacher_force=True)
print(f"    ELBO = {-float(loss):.2f}  (loss={float(loss):.2f})")

# ------------------------------------------------------------------
# 8. Posterior predictive (free rollout)
# ------------------------------------------------------------------
print("\n[8] Free rollout (posterior predictive)...", flush=True)
from numpyro.infer import Predictive
params = svi.get_params(svi_state)

predictive = Predictive(
    diamondworld_model,
    guide=guide, params=params,
    num_samples=2,
    return_sites=["pitch_type", "swing", "contact", "runs_scored", "base_state_after"],
)
rng_key, pred_key = jax.random.split(rng_key)
pred_samples = predictive(pred_key, batch, player_table, teacher_force=False)

print("    Sample shapes:")
for k, v in pred_samples.items():
    print(f"      {k}: {v.shape}")

# ------------------------------------------------------------------
# 9. Done
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("SMOKE TEST PASSED")
print("=" * 60)
