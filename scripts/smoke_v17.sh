#!/usr/bin/env bash
# Smoke test for the v17 structural variants before committing 5.2h/run of GPU.
#
# Checks, in order:
#   0. the DEFAULT model still produces exactly the v16 parameter set, so the
#      baseline is provably unchanged and any later delta is attributable;
#   1. each variant trains without NaN for a few hundred steps.
#
# Step 0 is the important one. Every variant is gated on a paired comparison
# against v16, and that comparison is meaningless if the flags-off path drifted.
set -u
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
source ~/dwjax-venv/bin/activate
cd ~/DiamondWorld

LOG=data/smoke_v17.log
: > "$LOG"

echo "=== 0. baseline parameter-set check vs the v16 checkpoint ===" | tee -a "$LOG"
python - >> "$LOG" 2>&1 <<'PY'
import pickle, sys
import numpy as np
import jax, jax.numpy as jnp
import numpyro, numpyro.handlers as nh
from functools import partial
from diamondworldjax.model.pa_model import pa_model

ck = pickle.load(open("checkpoints/dwjax_pa_v16/dwjax_step_0050000.pkl", "rb"))["params"]
# Drop only the variational latent params (player_mu / player_sigma), which the
# guide owns and which a traced model does not expose as params. The flax module
# params, including player_encoder, must match exactly.
ck_keys = {k for k in ck if k not in ("player_mu", "player_sigma", "player_skills")}

B, T, P, F = 2, 4, 50, 16
batch = {
    "pa_valid": jnp.ones((B, T), bool),
    "pa_outcome": jnp.zeros((B, T), jnp.int32),
    "pitcher_ids": jnp.zeros((B, T), jnp.int32),
    "batter_ids": jnp.zeros((B, T), jnp.int32),
    "park_ids": jnp.zeros((B, T), jnp.int32),
}
for k in ("inning", "half", "outs", "base_state", "score_diff", "tto",
          "shift_restricted", "pitch_clock", "pitch_count_game", "bat_side", "pit_hand"):
    batch[k] = jnp.zeros((B, T))
ptab = {"stats": jnp.zeros((P, F)), "league": jnp.zeros(P, jnp.int32),
        "hand": jnp.zeros(P, jnp.int32),
        "bat_hand": jnp.full(P, .5), "pit_hand": jnp.full(P, .5)}

def param_keys(**kw):
    fn = partial(pa_model, outcome_only=True, fatigue=True, **kw)
    with nh.seed(rng_seed=0), nh.trace() as tr:
        fn(batch, ptab, teacher_force=False)
    return {n for n, s in tr.items() if s["type"] == "param"}, tr

base, _ = param_keys()
print("checkpoint non-player param keys:", sorted(ck_keys))
print("default-model param keys       :", sorted(base))
missing, extra = ck_keys - base, base - ck_keys
if missing or extra:
    print(f"FAIL baseline drift: missing={sorted(missing)} extra={sorted(extra)}")
    sys.exit(1)
print("PASS default model matches the v16 parameter set exactly")

for label, kw in [("bilinear8", dict(bilinear_rank=8)),
                  ("nested", dict(nested=True)),
                  ("learned", dict(skill_prior="learned")),
                  ("lkj", dict(skill_prior="lkj"))]:
    ks, tr = param_keys(**kw)
    new = sorted(ks - base)
    sites = sorted(n for n, s in tr.items() if s["type"] == "sample")
    print(f"{label:10s} adds params {new} ; sample sites {sites}")
PY
rc=$?
if [ $rc -ne 0 ]; then echo "BASELINE CHECK FAILED (rc=$rc), aborting smoke" | tee -a "$LOG"; exit $rc; fi

run () {
  local tag="$1"; shift
  echo "=== 1. smoke-train $tag: $* ===" | tee -a "$LOG"
  python -m diamondworldjax.scripts.train_pa --steps 300 \
    --outcome-only --fatigue --recency-halflife 2.0 \
    --train-end 2023 --contact-quality --tag "smoke_$tag" "$@" >> "$LOG" 2>&1
  local r=$?
  echo "--- $tag rc=$r" | tee -a "$LOG"
  grep -iE "nan|inf|error|Traceback" "$LOG" | tail -3
}

run bilinear --bilinear-rank 8
run nested   --nested
run learned  --skill-prior learned
run lkj      --skill-prior lkj

echo "=== ELBO summary (must be finite and negative) ===" | tee -a "$LOG"
grep -E "step .*ELBO|Final ELBO" "$LOG" | tail -20
echo "SMOKE DONE" | tee -a "$LOG"
