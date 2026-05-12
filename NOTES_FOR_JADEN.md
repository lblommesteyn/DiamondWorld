# DiamondWorld — Research Notes for Jaden

## Current Results Summary

| Model | kl_run_dist | wasserstein | mean_rg_err | variance_err | p5+_err | p8+_err |
|---|---|---|---|---|---|---|
| B0_markov_re24 | 0.00830 | 0.05151 | 0.03301 | 0.39970 | 0.00492 | 0.00754 |
| B1_negbinom | 0.00667 | 0.08908 | 0.00917 | 0.66679 | 0.00513 | 0.00363 |
| B2_naive_pa | 0.00944 | 0.07538 | 0.02954 | 0.41901 | 0.00392 | 0.00919 |
| B3_lgbm_outcome | 0.01055 | 0.07859 | 0.04312 | 0.49256 | 0.00722 | 0.00610 |
| **B4_bayesian_hurdle** | 0.01788 | 0.07417 | 0.02707 | 0.70400 | **0.00040** | **0.00069** |
| **Phase3_MLP_rand** | **0.00284** | 0.10193 | 0.07213 | **0.34076** | 0.01792 | 0.00398 |
| Phase4_GCT_rand | 0.01050 | 0.09630 | 0.10400 | 0.50139 | 0.00801 | 0.00466 |

---

## Your Three Suggestions -- What We Tried

### 1. SVI + Non-linear ARMA Dynamics

**What it is:** `diamondworld/baselines/svi_arma.py` (B5)

Each game is driven by a pair of latent offense states (home, away) that
evolve across innings via a non-linear ARMA transition:

```
z_t = tanh(W @ z_{t-1} + b) + eps_t,   eps_t ~ Normal(0, sigma)
runs_t ~ NegativeBinomial(softplus(v * z_t), phi)
```

Global params (W, b, v, phi, sigma) are fit from 2015-2022 inning run data
using **Stochastic Variational Inference** via NumPyro/JAX. At simulation
time, a fresh latent trajectory is sampled per game, giving each game its
own momentum profile (hot streak in innings 4-6, etc.).

**Why it's interesting:** The ARMA dynamics capture within-game momentum
effects -- a high-scoring inning creates a higher latent state, making the
next inning more likely to also score. This is more principled than the
per-game Beta draw in B4, and the non-linearity (tanh) prevents the state
from exploding.

**Run it:**
```bash
sbatch scripts/slurm_eval_svi_arma.sh
```

---

### 2. JAX/NumPyro for GPU Acceleration

Both installed in the venv:
```bash
pip install "jax[cuda12]" numpyro
# jax 0.9.1, numpyro 0.21.0
```

The B5 SVI+ARMA model uses JAX natively. Key design choices:
- `AutoNormal` guide (mean-field Gaussian variational family)
- `Adam` optimizer, 5000 SVI steps, lr=0.01
- `Predictive` for posterior predictive sampling at simulation time
- Falls back to CPU transparently if no GPU found

The existing PyTorch models (Phase 3/4) are unaffected. JAX and PyTorch
coexist fine in the same venv.

**Custom implementations:** The ARMA transition in `_arma_model` is a
hand-rolled JAX function using `jnp.einsum` and `jax.nn.softplus`. Not
using any pre-built time-series modules -- the whole dynamics kernel is
~10 lines of JAX.

---

### 3. Fielding Errors -- Yes, We Added Them

**Decision: yes.** Fielding errors happen in ~1.2% of MLB plate appearances.
They matter because:
- Batter reaches base (like a walk for base-state purposes)
- No hit is credited (pitcher's ERA unaffected in real baseball)
- They inflate inning run potential non-trivially over 9 innings

**What changed:**
- `diamondworld/baselines/base.py`: added `"E"` to `PA_OUTCOMES` (now 9 outcomes)
  and `REACH_OUTCOMES = {"BB", "HBP", "1B", "2B", "3B", "HR", "E"}`
- `SVIARMASimulator._fit_error_rate()`: fits error rate from data if available,
  defaults to 0.012 otherwise
- Error outcome advances batter to 1B via the standard transition table
  (same base-state effect as a single, but recorded as "E")

**Note for future work:** The MLP and GCT models don't currently predict "E"
as an outcome -- they'd need a training data update and new output head.
For now errors are injected stochastically at the baseline level only.

---

## What's Still Running / Next Steps

### Phase 4 GCT Retraining (40 epochs)
Original training stopped at 9 epochs -- almost certainly undertrained.
The model's val PA NLL was still improving at epoch 9. Retraining with 40
epochs should close the gap with Phase 3.

```bash
sbatch scripts/slurm_retrain_phase4.sh
```

After it finishes, re-run eval:
```bash
sbatch scripts/slurm_eval_phase4.sh
```

### B4 Bayesian Hurdle -- Tighter Priors
Original B4 had variance_error=0.704 (worst of all models) because the
per-game Beta sampling was creating too many extreme games. Fixed by
re-parameterising the game-level draw as `Beta(mean*C, (1-mean)*C)` with
`C=50` (50 pseudo-observations concentration). This preserves the posterior
mean but reduces game-level variance by ~50x. Re-run:

```bash
sbatch scripts/slurm_eval_bayesian_hurdle.sh
```

### Lineup-Aware Simulator (Phase 3 + lineup)
The lineup simulator over-scores because the Phase 3 MLP was calibrated
against a random-pool setup (always tto=1, pitch_count resets each
half-inning). Naively adding real lineup cycling + TTO accumulation pushes
the model's fatigue predictions into over-scoring territory.

**This is actually a good negative result for the paper.** It directly
motivates why Phase 4 (GCT) is needed: a context-free model can't correctly
simulate accumulated game state. The Phase 4 GCT was specifically designed
to condition on game history, so lineup simulation with Phase 4 should work
correctly once the GCT is properly trained.

Recommended: after Phase 4 retraining, wire up the lineup simulator to use
`GameContextPASimulator` instead of `PASimulator`.

---

## File Map

```
diamondworld/
  baselines/
    base.py               -- PA_OUTCOMES now includes "E" (fielding errors)
    bayesian_hurdle.py    -- B4: tightened priors (concentration=50)
    svi_arma.py           -- B5: NEW -- SVI + non-linear ARMA (JAX/NumPyro)
    markov_re24.py        -- B0 (unchanged)
    negbinom.py           -- B1 (unchanged)
    naive_pa.py           -- B2 (unchanged)
    lgbm_outcome.py       -- B3 (unchanged)
  models/
    no_memory_mlp.py      -- Phase 3 (unchanged)
    game_context_transformer.py  -- Phase 4 (unchanged, retrain with 40 epochs)
    train_phase4.py       -- Phase 4 training script
  simulate/
    lineup_simulator.py   -- Lineup-aware sim (structural limitation documented)
    game_simulator.py     -- Random-pool sim (unchanged)
    game_context_simulator.py -- Phase 4 sim (unchanged)
    pa_simulator.py       -- PA-level sim (unchanged)
  scripts/
    eval_svi_arma.py      -- NEW -- B5 eval
    eval_bayesian_hurdle.py -- B4 eval (re-run with tighter priors)
    eval_lineup.py        -- Phase 3 + lineup eval
    eval_phase4.py        -- Phase 4 eval

scripts/  (SLURM)
  slurm_eval_svi_arma.sh       -- NEW
  slurm_retrain_phase4.sh      -- NEW (40 epoch retrain)
  slurm_eval_bayesian_hurdle.sh
  slurm_eval_phase4.sh
  slurm_eval_lineup.sh
```
