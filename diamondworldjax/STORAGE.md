# DiamondWorldJAX storage estimate

The repository checkout contains no local `data/` directory, so this is a capacity-planning
estimate rather than a measurement. It assumes MLB regular seasons from 2015 through 2024.
That is about 22,800 games and roughly 7-8 million pitches; 2020 is shorter than a normal
season.

| Data class | Estimated size |
| --- | ---: |
| Processed 44-column season Parquets | 1-2 GB |
| Raw Statcast season Parquets | 3-8 GB |
| MLB play-by-play and live-feed JSON cache | 14-41 GB |
| Validation files and small indexes | <1 GB |
| **Data-only working set** | **18-52 GB** |

The API cache dominates because the pipeline stores both play-by-play and live-feed payloads
for every game, or about 45,600 JSON files across this period. Payload size varies considerably
by game, so the upper end should be used for scratch-space planning.

Recommended capacity:

- 60 GB is a tight data-only minimum.
- 100 GB is a practical allocation for data plus rebuild headroom.
- 150 GB is safer when retaining multiple processed-schema versions, checkpoints, and
  evaluation artifacts.

Peak RAM is separate from disk. Loading every season into one Polars frame can require roughly
8-20 GB of RAM depending on string representation and selected columns. Prefer lazy scans or
season/game-batch streaming instead of concatenating all seasons eagerly.

On the cluster, replace the estimate with a measurement after the first complete build:

```bash
du -sh data/raw/statcast data/raw/mlb_api data/processed checkpoints eval/results
du -sh data
```
