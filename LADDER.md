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
      --contact-quality --train-end 2023 --test-seasons 2024 --tag v22

    python -m diamondworldjax.scripts.prod_playercorr \
      --ckpt checkpoints/dwjax_pa_v22pf/dwjax_step_0050000.pkl \
      --contact-quality --train-end 2023 --test-seasons 2024 --tag v22pf

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
      --contact-quality --tag v22-pregame-leakfree

    python -m diamondworldjax.scripts.run_pregame_sim --pregame-staff --r 100 \
      --ckpt checkpoints/dwjax_pa_v22pf/dwjax_step_0050000.pkl \
      --contact-quality --pitchformer --tag v22pf-pregame-leakfree

    python -m diamondworldjax.scripts.simulator_benchmarks \
      --arrays data/eval2/calib_v22-pregame-leakfree_arrays.npz
    python -m diamondworldjax.scripts.simulator_benchmarks \
      --arrays data/eval2/calib_v22pf-pregame-leakfree_arrays.npz

Compare against `data/eval2/simulator_benchmarks_v16-pregame-leakfree.txt` (which, note,
was actually run on v15; RESULTS.md explains). The two numbers that matter: win
probability log-loss against the 0.6923 home-field base rate, and run-total KS.

`--contact-quality` and `--pitchformer` must match how the checkpoint was trained or the
parameter tree will not load. That is what the flags are for.

## Step 3: seeds

One seed cannot separate a null from +0.015. Two more of each:

    python -m diamondworldjax.scripts.train_pa --contact-quality --train-end 2023 \
      --seed 1 --tag v22_s1
    python -m diamondworldjax.scripts.train_pa --contact-quality --train-end 2023 \
      --seed 2 --tag v22_s2
    python -m diamondworldjax.scripts.train_pa --contact-quality --train-end 2023 \
      --pitchformer --seed 1 --tag v22pf_s1
    python -m diamondworldjax.scripts.train_pa --contact-quality --train-end 2023 \
      --pitchformer --seed 2 --tag v22pf_s2

About 4 hours each on a 3080, ~3 on a 3090. With two GPUs run two at once, pinning
each with `CUDA_VISIBLE_DEVICES=0` / `=1`, because JAX preallocates every visible card.
Then Step 1 on each, and report the mean across seeds with the spread.

## Step 4: R3

    python -m diamondworldjax.scripts.train_shared_skills --pa-pitchformer --seed 0 \
      --tag v23_shared

Then Steps 1 and 2 against it. Only start this after Step 1 has said whether R2 is
a regression, because if attention is genuinely hurting the PA model there is no point
sharing its embedding.

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
