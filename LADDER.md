# The ladder

How to find out which of our changes actually make the simulator better, one change at
a time, scored the same way every time.

This branch (`ladder`) is Jaden's `pitchformer_integration` merged with `main`, plus a
fix that makes `run_pregame_sim` take an explicit checkpoint. Every command below runs
from this branch as it is.

## The rule that matters more than any command

**Change one thing per rung. Score every rung on the same two metrics. Nothing under
the MDE is a result.**

The minimum detectable effect on the PA gate is **+0.030 average correlation** at 80%
power (RESULTS.md, power analysis). A step of +0.015 clears the paired bootstrap only
31% of the time. So a rung that does nothing and a rung that is genuinely worth +0.015
look identical on one seed. Two consequences:

- A point estimate is not a verdict. The paired bootstrap CI is the verdict.
- Read the ladder cumulatively as well as rung by rung. R1 to R2 may be unresolvable on
  its own while R0 to R3 is not.

And one more, decided now rather than after the numbers arrive: **a rung whose PA gate
goes up and whose game-level score goes down is not an improvement.** v1 through v5
improved components and made the simulator worse. That is the failure the second
metric exists to catch.

## The rungs

| rung | what changes from the rung below | isolates |
|---|---|---|
| R0 | v16 as it exists on main (0.624) | the reference |
| R1 | retrain on this branch with the label and mask fixes, unknown-player slot reserved | the bug fixes |
| R2 | R1 + `--pitchformer` (causal attention over the PAs of a game) | attention inside the PA model |
| R3 | R2 + `train_shared_skills` (one player latent shared across PA and pitch tasks) | the shared embedding |
| R4 | R3 with the A/B/C/D pitch-level stack as the PA sampler | pitch-level simulation |

R4 is conditional. The pitch-level stack cannot be scored fairly until its rollout feeds
the simulated count back into the model (RESULTS.md, "Simulating every pitch does NOT
beat the PA-level model"). Scoring it before then re-measures that bug. It joins when it
can be compared, not before.

## Where we already are

R1 and R2 have been trained once each (Jaden, seed 0). Point estimates on the PA gate:

| rung | tag | K | BB | Hit | HR | AVG |
|---|---|---|---|---|---|---|
| R0 | v16 | .792 | .651 | .445 | .610 | 0.624 |
| R1 | v22 | .799 | .660 | .483 | .633 | **0.644** |
| R2 | v22pf | .779 | .693 | .474 | .592 | **0.635** |

So the bug fixes look like +0.020 and the attention looks like -0.009. Both are under the
MDE, neither has a bootstrap CI, and neither has a game-level score. **Nothing above is a
result yet.** Steps 1 and 2 below turn them into one.

## Step 1: bootstrap what already exists

Cheap, CPU only, minutes. Turns the point estimates into verdicts.

    python -m diamondworldjax.scripts.prod_playercorr \
      --ckpt checkpoints/dwjax_pa_v22/dwjax_step_0050000.pkl \
      --contact-quality --per-stat-shrink --skill-prior walk       --train-end 2023 --test-seasons 2024 --tag v22

    python -m diamondworldjax.scripts.prod_playercorr \
      --ckpt checkpoints/dwjax_pa_v22pf/dwjax_step_0050000.pkl \
      --contact-quality --per-stat-shrink --skill-prior walk --pitchformer       --train-end 2023 --test-seasons 2024 --tag v22pf

    python -m diamondworldjax.scripts.bootstrap_playercorr \
      --rates data/eval2/prod_rates_v16.npz \
      --rates data/eval2/prod_rates_v22.npz \
      --rates data/eval2/prod_rates_v22pf.npz \
      --baseline data/eval2/prod_rates_v16.npz \
      --out data/eval2/bootstrap_ladder.txt --json-out data/eval2/bootstrap_ladder.json

`prod_rates_v16.npz` is in the Hugging Face `data/eval2` tier if it is not local. The
paired CI against v16 for each rung is the number to report. If the checkpoint step
number differs from 50000, use whatever `ls checkpoints/dwjax_pa_v22/` shows last.

## Step 2: game-level score for R1 and R2

About 2 hours each on a 3080. Checkpoints every 250 games, so a death costs 12 minutes
and rerunning the same command resumes.

    python -m diamondworldjax.scripts.run_pregame_sim --pregame-staff --r 100 \
      --ckpt checkpoints/dwjax_pa_v22/dwjax_step_0050000.pkl \
      --contact-quality --per-stat-shrink --recal data/eval2/v22_cal_params.npz       --tag v22-pregame-leakfree

    python -m diamondworldjax.scripts.run_pregame_sim --pregame-staff --r 100 \
      --ckpt checkpoints/dwjax_pa_v22pf/dwjax_step_0050000.pkl \
      --contact-quality --per-stat-shrink --pitchformer       --recal data/eval2/v22pf_cal_params.npz --tag v22pf-pregame-leakfree

    python -m diamondworldjax.scripts.simulator_benchmarks \
      --arrays data/eval2/calib_v22-pregame-leakfree_arrays.npz
    python -m diamondworldjax.scripts.simulator_benchmarks \
      --arrays data/eval2/calib_v22pf-pregame-leakfree_arrays.npz

Compare against `data/eval2/simulator_benchmarks_v16-pregame-leakfree.txt` (which, note,
was actually run on v15; RESULTS.md explains). The two numbers that matter: win
probability log-loss against the 0.6923 home-field base rate, and run-total KS.

How to read it. Win probability: the sim must beat the base rate (0.6923); v15/v16
leak-free did not (0.6985). Run totals: the sim mean must sit near the real 8.63 and
KS near the v16 value (0.058). Overdispersion near 2.1x means the *shape* is right; a
right shape with a mean 1.5 runs high is a scale problem, not a variance problem, and
the first suspect is a rung trained without `--outcome-only` (see the recipe section).

First reading, v22 seeds 42 and 97 (leak-free, r=100): win log-loss 0.6920 / 0.6873
(beats base rate, first time leak-free); run mean 10.23 / 10.06 (real 8.63), KS
0.146 / 0.133 (v16 0.058); overdispersion 2.12x / 2.08x (real 2.11x). Win-prob up,
run scale regressed. Not a pass until the recipe question is settled.

`--contact-quality` and `--pitchformer` must match how the checkpoint was trained or the
parameter tree will not load. That is what the flags are for.

## The recipe (read this before training anything)

R0 (v16) was trained by `scripts/run_train_v16.sh` with

    --outcome-only --fatigue --recency-halflife 2.0 --train-end 2023 --contact-quality

Every rung must use all five, or it is not one change from the rung below. An earlier
version of this file listed only the last two; that was a mistake. Consequences of
dropping each:

- `--outcome-only` off: the neural runs/bases heads come back (the v1-v5 design v6
  removed). Symptom: PA gate fine, sim run totals inflated by 1-2 runs/game.
- `--fatigue` off: `prod_playercorr` builds the model with `fatigue=True` regardless,
  so the parameter tree does not match the checkpoint.
- `--recency-halflife 2.0` off: the player table at train time is unweighted while
  `prod_playercorr` weights it (its default is 2.0). Input distribution shift.

`scripts/ladder_train.sh <rung> <seed>` bakes the recipe in; prefer it over typing flags.

## What v22 actually is, and what the eval tools must be told

Jaden's v22pf_s42 command:

    train_pa --train-end 2023 --seed 42 --steps 50000 --tag v22pf_s42       --outcome-only --fatigue --recency-halflife 2.0       --contact-quality --per-stat-shrink --skill-prior walk --pitchformer

So the full v16 recipe is on (outcome-only is not the run-inflation cause), but v22 adds
`--per-stat-shrink` and `--skill-prior walk` on top of the bug fixes. R1 is therefore
"v16 + bug fixes + per-stat shrink + seasonal random-walk skills", three changes. If
the rung is meant to isolate the bug fixes, those two flags come off; otherwise
relabel R1 honestly.

Every eval tool must be told all of these, and before this commit most could not be:

- `run_pregame_sim` / `Sim` had no `--per-stat-shrink`, so the sim read a player table
  built differently from training. Fixed: `--per-stat-shrink`.
- `simulate_games` substituted a walk checkpoint's `player_mu` (P, seasons, D) into the
  iso site, which the encoder rejects. Fixed: for a held-out season it now takes the
  last trained season, which is exactly what the model's own clamp does.
- The sim always applied v13's recalibration vector. A recal vector is a per-outcome
  logit shift fit to one checkpoint; on another checkpoint it shifts the outcome mix
  arbitrarily, and run totals move with it. This is the first suspect for the 10.1 vs
  8.63 run mean. Fixed: `--recal`. Build one per checkpoint:

      python -m diamondworldjax.scripts.diag_outcomes --ckpt <ckpt> --outcome-only         --fatigue --use-park --recency-halflife 2.0 --skill-mode mean         --train-end 2023 --contact-quality --per-stat-shrink [--pitchformer]         --dump-logits data/eval2/<tag>_logits.npz
      python -m diamondworldjax.scripts.fit_calibration         --logits data/eval2/<tag>_logits.npz --out data/eval2/<tag>_cal.txt

  then pass the resulting `<tag>_cal_params.npz` to `run_pregame_sim --recal`.
- `prod_playercorr` had no `--pitchformer`, so a v22pf checkpoint was scored with its
  attention parameters ignored. Fixed. Any v22pf PA-gate number from before this
  commit should be rerun.

## Step 3: seeds

One seed cannot separate a null from +0.015. Two more of each:

    bash scripts/ladder_train.sh v22 1
    bash scripts/ladder_train.sh v22 2
    bash scripts/ladder_train.sh v22pf 1
    bash scripts/ladder_train.sh v22pf 2

About 4 hours each on a 3080, ~3 on a 3090. With two GPUs run two at once, pinning
each with `CUDA_VISIBLE_DEVICES=0` / `=1`, because JAX preallocates every visible card.
Then Step 1 on each, and report the mean across seeds with the spread.

If v22_s42 / v22_s97 were trained without the full recipe they are not R1 and need
retraining. Check the first lines of their training log: it should print
`[outcome-only v6]` on the "Starting SVI" line.

## Step 4: R3

    bash scripts/ladder_train.sh v23 0

`train_shared_skills` now takes `--train-end`, `--contact-quality`, `--recency-halflife`,
`--outcome-only`, `--fatigue` and passes them the way `train_pa` does, so R3 differs
from R2 by the shared latent only. Then Steps 1 and 2 against it. Only start this after
Step 1 has said whether R2 is a regression, because if attention is genuinely hurting
the PA model there is no point sharing its embedding.

## Reporting

One table in RESULTS.md, rows are rungs, columns are: AVG (mean over seeds, spread),
paired CI vs the rung below, paired CI vs R0, win-prob log-loss, run-total KS. A rung
earns a bold number only when its paired CI excludes zero AND its game-level score did
not get worse.

## What not to do

- Do not compare a new run against `prod_rates_v16.npz` from before the index-0 metric
  fix. The corrected v16 file is the one that gives 0.624.
- Do not run bare on pcslurm. Use `scripts/run_detached.sh`. Three sweeps died silently
  before it existed. On a personal machine this does not apply.
- Do not change two things in one rung to save time. That is how the project ended up
  unable to say whether the bullpen leak or the version change moved the number.
