"""Pre-flight for the walk prior's EVAL path.

The walk latent's site is `player_skill_eps`, not `player_skills`, so
prod_playercorr's `--skill-mode mean` substitution had to be special-cased. If it
misses, the latent is silently sampled from the prior at eval and the scoreboard
is meaningless: exactly the failure mode of the guide-coverage bug, one layer
down. Cheap to check against the 6-step smoke checkpoint before spending 8 GPU
hours on the real run.
"""
import pickle

import numpy as np
import jax.numpy as jnp
import numpyro.handlers as nh
from functools import partial

from diamondworldjax.model.pa_model import pa_model

ck = pickle.load(open("checkpoints/dwjax_pa_smoke_v19w/dwjax_step_0000006.pkl", "rb"))["params"]
params = {**ck, "player_skill_eps": ck["player_mu"]}  # what prod_playercorr does

P, S, D = np.asarray(ck["player_mu"]).shape
B, T, F = 2, 4, 16
batch = {k: jnp.zeros((B, T)) for k in
         ["inning", "half", "outs", "base_state", "score_diff", "tto",
          "shift_restricted", "pitch_clock", "pitch_count_game", "bat_side", "pit_hand"]}
batch.update({
    "pa_valid": jnp.ones((B, T), bool),
    "pa_outcome": jnp.zeros((B, T), jnp.int32),
    "pitcher_ids": jnp.zeros((B, T), jnp.int32),
    "batter_ids": jnp.zeros((B, T), jnp.int32),
    "park_ids": jnp.zeros((B, T), jnp.int32),
    "season": jnp.full((B, T), 2024, jnp.int32),  # TEST season, beyond training
})
ptab = {"stats": jnp.zeros((P, F)), "league": jnp.zeros(P, jnp.int32),
        "hand": jnp.zeros(P, jnp.int32),
        "bat_hand": jnp.full(P, .5), "pit_hand": jnp.full(P, .5)}

m = partial(pa_model, outcome_only=True, fatigue=True, skill_prior="walk",
            season_base=2015, n_seasons=S)
with nh.seed(rng_seed=0), nh.substitute(data=params), nh.trace() as tr:
    m(batch, ptab, teacher_force=False)

took = np.allclose(np.asarray(tr["player_skill_eps"]["value"]),
                   np.asarray(ck["player_mu"]))
finite = bool(np.isfinite(np.asarray(tr["pa_outcome"]["fn"].logits)).all())
clamp = min(2024 - 2015, S - 1)

print(f"season axis S={S}")
print(f"player_skill_eps substituted from checkpoint: {took}")
print(f"2024 clamps to season index {clamp} (expect {S - 1})")
print(f"logits finite: {finite}")

ok = took and finite and clamp == S - 1
print("WALK EVAL PATH OK" if ok else "WALK EVAL PATH BROKEN")
raise SystemExit(0 if ok else 1)
