# DiamondWorld

DiamondWorld is a pitch-level baseball data pipeline and simulator scaffold for testing
whether within-game context is necessary to recover the empirical tail of baseball run
scoring.

The repository is intentionally built in phase order. Phase 1 is implemented first:
season-level pitch Parquet generation from Statcast plus MLB Stats API play-by-play
enrichment, with cache-first IO and validation reports.

## Layout

```text
diamondworld/
  data/raw/          # cached API responses, raw pybaseball downloads
  data/processed/    # one Parquet file per season
  baselines/         # Phase 2 lives here after Phase 1 validation
  models/            # Phase 3/4 models
  eval/              # shared evaluation harness
  simulate/          # MC simulation and counterfactual interfaces
  scripts/           # runnable pipeline helpers
```

## Phase 1 quick start

Install locally:

```bash
cd diamondworld
python -m pip install -e .
```

If the cluster Python says `No module named pip`, create a local environment first:

```bash
cd /scratch/lblommes/diamondworld
bash scripts/setup_env.sh
source .venv/bin/activate
```

If `pip` exists but only after bootstrapping:

```bash
python -m ensurepip --user
python -m pip install -e .
```

On Nibi/SHARCNET, prefer the module-aware setup:

```bash
cd /scratch/lblommes/diamondworld
bash scripts/setup_nibi.sh
source scripts/use_nibi_env.sh
python scripts/build_season.py --season 2024
```

For a batch job:

```bash
cd /scratch/lblommes/diamondworld
mkdir -p logs
sbatch scripts/slurm_build_season.sh 2024
```

Build a season:

```bash
python scripts/build_season.py --season 2024
```

Validate an existing processed season:

```bash
python scripts/validate_season.py --season 2024
```

Outputs:

- `data/processed/pitches_YYYY.parquet`
- `data/processed/validation_YYYY.json`
- MLB play-by-play cache in `data/raw/mlb_api/play_by_play/{game_pk}.json`
- Statcast cache in `data/raw/statcast/statcast_YYYY.parquet`

## Notes

- Processing uses Polars. `pybaseball` internally returns pandas objects, which are
  converted immediately and cached as Parquet.
- MLB Stats API calls are cache-first because a full season is roughly 2,500 games.
- Splits are by season only: train `2015-2021`, calibration `2022`, held out
  `2023-2024`.
- `spin_rate` is deliberately not required.
