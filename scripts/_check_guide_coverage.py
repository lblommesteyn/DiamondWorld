"""Does the hand-written SVI guide cover every latent the model samples?

A guide that misses a latent site does not error. Under Trace_ELBO the missing
site is simply drawn from its PRIOR at every step, so it is never learned. That
turns "give the model a learned prior parameter" into "inject fresh noise into
the prior each step", which is a completely different experiment.
"""
import jax.numpy as jnp
import numpyro.handlers as nh
from functools import partial

from diamondworldjax.model.pa_model import pa_model
from diamondworldjax.train.svi import make_player_skills_guide

B, T, P, F = 2, 4, 30, 16
batch = {k: jnp.zeros((B, T)) for k in
         ["inning", "half", "outs", "base_state", "score_diff", "tto",
          "shift_restricted", "pitch_clock", "pitch_count_game", "bat_side", "pit_hand"]}
batch.update({
    "pa_valid": jnp.ones((B, T), bool),
    "pa_outcome": jnp.zeros((B, T), jnp.int32),
    "pitcher_ids": jnp.zeros((B, T), jnp.int32),
    "batter_ids": jnp.zeros((B, T), jnp.int32),
    "park_ids": jnp.zeros((B, T), jnp.int32),
    "season": jnp.full((B, T), 2015, jnp.int32),
})
ptab = {"stats": jnp.zeros((P, F)), "league": jnp.zeros(P, jnp.int32),
        "hand": jnp.zeros(P, jnp.int32),
        "bat_hand": jnp.full(P, .5), "pit_hand": jnp.full(P, .5)}

ok = True
for sp, ns in [("iso", 1), ("learned", 1), ("lkj", 1), ("walk", 9)]:
    # Build the guide for THIS prior. Comparing every model against the iso guide
    # (an earlier version of this script did exactly that) reports a false failure
    # for every non-iso prior, because the iso guide is not the one that would be
    # used. The pairing has to match how train() constructs them.
    guide = make_player_skills_guide(P, skill_prior=sp, n_seasons=ns)
    m = partial(pa_model, outcome_only=True, fatigue=True,
                skill_prior=sp, n_seasons=ns)
    with nh.seed(rng_seed=0), nh.trace() as mt:
        m(batch, ptab)
    with nh.seed(rng_seed=0), nh.trace() as gt:
        guide(batch, ptab)
    msites = {k for k, v in mt.items()
              if v["type"] == "sample" and not v.get("is_observed")}
    gsites = {k for k, v in gt.items() if v["type"] == "sample"}
    missing = sorted(msites - gsites)
    if missing:
        ok = False
    flag = "OK" if not missing else "UNCOVERED (sampled from prior, never learned)"
    print(f"{sp:8s} latents={sorted(msites)}")
    print(f"{'':8s} guide={sorted(gsites)} -> {flag} {missing if missing else ''}")

print("ALL PRIORS COVERED" if ok else "COVERAGE FAILURE")
raise SystemExit(0 if ok else 1)
