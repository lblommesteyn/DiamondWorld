# DiamondWorld: Reproduction Guide

This document makes the paper's results reviewer-reproducible. It lists the data sources
and as-of dates, the train/validation/test split, the environment, the exact scripts that
produce each figure and table, and the one item (market prices) that cannot be
redistributed, with the derived inputs and transformation logic provided instead.

## Environment

- Python 3.12, JAX 0.10.x on CUDA (CPU also works for all non-simulation analysis).
- Exact pinned versions: `requirements.lock` (`pip install -r requirements.lock`).
- Fixed seeds are set in every script (`--seed`, and NumPy `default_rng(seed)`); the
  simulator's per-game replica seeds are set from `seed` in `simulate()`.

## Data sources and as-of dates

| Input | Source | As-of / window | Redistributable |
|---|---|---|---|
| Pitch/PA play-by-play (2015-2024) | MLB Statcast via Baseball Savant | full seasons | derived Parquet in `data/processed/` |
| Game schedule, boxscores, lineups, starters | MLB Stats API (`statsapi.mlb.com`) | pulled 2026-07; cached in `data/cache/` | yes (public API) |
| Market-implied win probabilities (game moneylines, 2021-2025) | consensus pregame closing prices | closing, pre-first-pitch | **no** (see below) |
| Steamer preseason-2024 projections | public archived release | preseason 2024 | derived rates only |

**Market data (non-redistributable).** The validation ground truth is de-vigged
pregame market-implied win probabilities. We cannot redistribute the raw prices. We
release: (a) the derived per-game pregame home-win probability keyed by `game_pk`
(`data/eval2/odds_2023_2024.csv`, `odds_2025.csv`, schema `game_pk, ml_home, ml_away`);
(b) the exact de-vigging logic (`american_implied` in `simulator_benchmarks.py`:
normalize the two implied probabilities to sum to one); and (c) the join logic
(date + full team names to `game_pk`, `build_odds_2025.py`). The restriction is stated
rather than obscured.

## Train / validation / test manifest

- **Train**: 2015-2023 (rate features and skill latent). The locked model also uses 2024
  rates only for the later-season (2025) input-level check.
- **Model-selection (validation)**: held-out cross-player differentiation, used to pick
  the locked specification (contact-quality features, recency half-life, KL scaling).
  Documented in `RESULTS.md`; not the reported test season.
- **Test**: 2024 (primary). 2025 is an out-of-sample later-season check; 2026-to-date is a
  deliberate stale-input stress test.
- The magnitude-calibration slope is fit on a random half of 2024 series and evaluated
  frozen on the held-out half (`validation_stats.py`).

## One-command reproduction of the analysis (no GPU)

The paper's statistics and figures are reproduced from committed arrays:

```bash
pip install -r requirements.lock
python -m diamondworldjax.scripts.validation_stats        # Table 2 (CIs, sign acc, calib split)
python -m diamondworldjax.scripts.distributional_stats     # Table 3 (log score, CRPS, conditional coverage)
python -m diamondworldjax.scripts.simulator_benchmarks --arrays data/eval2/calib_v15-pregame-hook-r500_arrays.npz --tag main
python paper/make_figures.py                               # all figures -> paper/figs/
cd paper && tectonic diamondworld.tex                      # the PDF
```

## Regenerating the simulator arrays (GPU)

```bash
python -m diamondworldjax.scripts.run_pregame_sim --tag v15-pregame-hook-r500 --r 500
python -m diamondworldjax.scripts.lineup_backtest --games 90 --cands 24
```

## Which script produces what

| Artifact | Script |
|---|---|
| Within-series validation, CIs, calibration split | `validation_stats.py` |
| Distributional scoring, conditional coverage | `distributional_stats.py` |
| Run-distribution / WP / player benchmarks | `simulator_benchmarks.py` |
| Cross-season / later-season checks | `season_market_validation.py`, `build_odds_2025.py` |
| Pitching/hitting channel decomposition | `whatif_channels.py` |
| Counterfactual magnitude calibration | `counterfactual_validation.py` |
| Lineup decision study (winner's-curse corrected) | `lineup_backtest.py` |
| Contact-quality feature, hook model | `train_pa.py`, `game_extract.py` |
| All figures | `paper/make_figures.py` |

## Tests

Rules-engine and data-schema tests: `tests/` (`test_dwjax_sim.py`, `test_pa_encoding.py`,
`test_validate.py`, `test_enrich.py`). The PA-encoding test guards the outcome class order
against a silent index bug. Run with `pytest tests/`.

## Anonymization

For blind review, a name-stripped build is produced by removing the `\author`/`\affil`
lines from `paper/diamondworld.tex`; the repository link is supplied through the
submission system.
