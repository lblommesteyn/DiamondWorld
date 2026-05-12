#!/usr/bin/env python
from __future__ import annotations

import argparse

from _bootstrap import add_repo_root_to_path

add_repo_root_to_path()

from diamondworld.data.pipeline import build_season


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DiamondWorld Phase 1 seasons.")
    parser.add_argument("--start-season", type=int, default=2015)
    parser.add_argument("--end-season", type=int, default=2024)
    parser.add_argument("--api-sleep-seconds", type=float, default=0.0)
    parser.add_argument("--force-statcast", action="store_true")
    parser.add_argument("--force-mlb-api", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for season in range(args.start_season, args.end_season + 1):
        build_season(
            season,
            api_sleep_seconds=args.api_sleep_seconds,
            force_statcast=args.force_statcast,
            force_mlb_api=args.force_mlb_api,
            validate=True,
        )


if __name__ == "__main__":
    main()
