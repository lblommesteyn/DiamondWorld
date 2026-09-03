# Next up

Three tasks that are independently valuable and do not block each other. Each has a
clear answer at the end of it, so none of them can quietly turn into open-ended work.

Two rules that apply to all of them, both learned the expensive way:

- **Submit through `scripts/run_detached.sh`, never bare.** pcslurm dispatches
  Windows to PowerShell to WSL, so work run directly sits in the submitting process
  tree and dies with it. Three sweeps died that way with no traceback and no non-zero
  exit (jobs 337, 393, 395).
- **A job showing RUNNING with a log that stopped advancing is dead. `scancel` it.**
  The holder used to wait forever on a DONE marker a killed job never got to write.
  That cost 22 hours of held GPU on job 399. It now self-detects, but check anyway.

---

## 1. Populate the park geometry table

The one piece of the super-state design that is completely untested. It needs a data
source and a human, not a GPU: park dimensions are not in the Statcast schema, and
writing thirty stadiums' dimensions from memory is the kind of plausible fabrication
this project has already been bitten by.

Create `data/parks/geometry.csv` with exactly these columns:

    park_id,lf_line_ft,lcf_ft,cf_ft,rcf_ft,rf_line_ft,lf_wall_ft,cf_wall_ft,rf_wall_ft,elevation_ft

`park_id` is the string team code as it appears in the parquet (LAA, PHI, WSH, SD).

Partial is genuinely fine. Any park without a row degrades to exactly today's
behaviour, because `geometry_features` returns zeros plus `has_geometry = 0` and the
flag lets a head tell "no data" from "a park whose dimensions happen to be zero". Ten
parks is enough to learn something. Nothing trained so far becomes invalid.

No code change needed once the file exists; the loader picks it up.

## 2. Capacity sweep on A and B

Cheap, and it answers a question that has been asked. Roughly 5 minutes each.

    pcslurm submit --name pf-small --shared -- bash /home/luke/DiamondWorld/scripts/run_detached.sh pfs \
      /home/luke/dwjax-venv/bin/python -m diamondworldjax.scripts.train_pitchformer \
      --d-model 96 --layers 2 --heads 4 --steps 6000 --tag cap_small

    pcslurm submit --name pf-big --shared -- bash /home/luke/DiamondWorld/scripts/run_detached.sh pfb \
      /home/luke/dwjax-venv/bin/python -m diamondworldjax.scripts.train_pitchformer \
      --d-model 384 --layers 8 --heads 8 --steps 6000 --tag cap_big

    pcslurm submit --name pf-long --shared -- bash /home/luke/DiamondWorld/scripts/run_detached.sh pfl \
      /home/luke/dwjax-venv/bin/python -m diamondworldjax.scripts.train_pitchformer \
      --d-model 192 --layers 4 --heads 6 --steps 20000 --tag cap_long

Baseline to beat is `data/eval2/pitchformer_v2_maskfix.json`: type **+0.2463**,
swing **+0.2227**, contact **+0.1124**, foul **+0.0313**.

**Do not compare against `pitchformer_ab_v1.json`.** Those numbers (type +0.269, swing
+0.230, contact +0.127, foul +0.147) were produced before the causal-mask fix, when a
fully-masked attention row at position 0 softmaxed uniformly over every key including
future pitches, leaking lookahead through the whole stack. Any run made after commit
1ea0258 uses the fixed mask, so scoring it against v1 measures that bug rather than
capacity, and every config would look like a regression.

The foul head is at +0.031 post-fix, which is close to no skill at all. If a capacity
arm appears to make foul dramatically better, suspect the harness before believing it.

`cap_long` is the interesting one. Training loss was still drifting down at 6k steps,
so some of what would look like a capacity win may just be undertraining, and those
two are worth being able to tell apart.

Do not expect this to fix the simulator. The simulator failure is structural (see 3
and the RESULTS.md writeup), and a larger model conditioned on the wrong count only
becomes confidently wrong faster.

## 3. Isolate the bullpen leak cleanly -- ALREADY DONE, DO NOT RUN

**Superseded. Running this wastes about two hours reproducing an existing result.**

`run_pregame_sim.py` calls `Sim(hook_model=True)` with no checkpoint argument, so it
takes the `V15` default. The sweep tagged `v16-pregame-leakfree` therefore ran the
**v15** model, not v16, and `v15-pregame-hook` ran v15 as well. Both sides of that
comparison are the same model, which makes it the clean isolation this task was asking
for. The leak's cost is 0.6881 -> 0.6985 log-loss, already measured.

The tag is simply wrong, and the earlier claim that the comparison "conflates the leak
with the version change" was wrong with it.

What would still be worth running is the same sweep with the model the tag claims:

    python -m diamondworldjax.scripts.run_pregame_sim --tag v16-real --r 100 --pregame-staff

and that needs `Sim` to be passed `ckpt=V16, contact_quality=True` first, because v16
was trained with contact quality on and the default is off. Until that is wired
through as a flag, this is a code change, not a run.

    pcslurm submit --name dw-realized --shared -- bash /home/luke/DiamondWorld/scripts/run_detached.sh realized \
      /home/luke/dwjax-venv/bin/python -m diamondworldjax.scripts.run_pregame_sim \
      --tag v16-pregame-realized --r 100

Note the deliberate absence of `--pregame-staff`: that is what makes this the
realized (leaky) bullpen with the v16 model. Then

    python -m diamondworldjax.scripts.simulator_benchmarks \
      --arrays data/eval2/calib_v16-pregame-realized_arrays.npz

and compare against `data/eval2/simulator_benchmarks_v16-pregame-leakfree.txt`. The
difference between them is the leak's actual cost.

About 2 hours. It checkpoints every 250 games, so a death costs about 12 minutes
rather than the whole sweep, and re-running the same command resumes.

---

## Not in this list, and why

The **autoregressive rollout** is the critical path for the simulator claim and is
being handled separately. It is the expensive one: today a single forward pass covers
a whole 160-pitch sequence, and recomputing after every pitch makes that 160
sequential passes.

The **wild pitch and balk label undercount** (0.55/game against a real ~0.8, and
0.048 against ~0.1) is real but low priority. Some are recorded inside pitch details
rather than as separate playEvents. Steals, errors and pickoffs already match real
rates, so C is trainable as it stands.
