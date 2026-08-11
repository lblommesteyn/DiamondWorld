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
| Sportsbook consensus moneylines (2021-2025) | reactiv/delphi public MLB odds dataset | pregame closing | derived probabilities only |
| Prediction-market prices (2026) | Kalshi (`KXMLBGAME`) and Polymarket game markets, public APIs | pregame, from price history | derived probabilities only |
| Steamer preseason-2024 projections | public archived release | preseason 2024 | derived rates only |

**Market sources (named; derived inputs released).** The validation target is a de-vigged
pregame market-implied win probability. The primary source (2024-2025) is the consensus of
sportsbook closing moneylines from the public reactiv/delphi MLB odds dataset; the
prediction-market cross-check (2026) is the Kalshi `KXMLBGAME` series (pregame implied
probability from its candlestick price history). Polymarket carries the same game markets
(median volume roughly \$0.5M/game) and its prices are retrievable via the CLOB, but its
event price histories span multiple days and do not pin cleanly to a single game's first
pitch, so it is assessed but not used as a within-series estimate. We release: (a) the
derived per-game pregame home-win probability keyed by `game_pk`
(`data/eval2/odds_2023_2024.csv`, `odds_2025.csv`); (b) the exact de-vigging logic
(`american_implied`: normalize the two implied probabilities to sum to one); (c) the join
logic (`build_odds_2025.py`, `season_market_validation.py`), including the Kalshi and
Polymarket API pipelines and their as-of timestamps. Raw prices are not redistributed;
the restriction is stated rather than obscured.

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
| Paired bootstrap CIs on the player-corr metric | `bootstrap_playercorr.py` |
| v17 structural variants (bilinear / nested / skill prior) | `scripts/run_v17.sh` |
| All figures | `paper/make_figures.py` |

Model comparisons are gated on `bootstrap_playercorr.py`, which resamples batters and
reports a PAIRED interval on the difference between two models. A variant counts as an
improvement only when that interval excludes zero; a better point estimate does not
qualify. Requires no GPU, since it reads the `prod_rates_<tag>.npz` arrays that
`prod_playercorr.py` writes:

```bash
python -m diamondworldjax.scripts.bootstrap_playercorr \
    --rates v15=data/eval2/prod_rates_v15_2024.npz \
    --rates v16=data/eval2/prod_rates_v16.npz \
    --baseline v15 --reps 20000
```

## Tests

Rules-engine and data-schema tests: `tests/` (`test_dwjax_sim.py`, `test_pa_encoding.py`,
`test_validate.py`, `test_enrich.py`, `test_pa_model_variants.py`). The PA-encoding test
guards the outcome class order against a silent index bug, and the variants test guards
the properties whose failure would be silent: that the nested head emits normalised
log-probabilities, that its two stages factorise as claimed, and that the bilinear term
is a genuine interaction rather than an additively separable one. Run with `pytest tests/`.

## Anonymization

For blind review, a name-stripped build is produced by removing the `\author`/`\affil`
lines from `paper/diamondworld.tex`; the repository link is supplied through the
submission system.
