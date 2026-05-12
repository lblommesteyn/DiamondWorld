from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

from diamondworld.data.paths import ensure_data_dirs, raw_root

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"


def play_by_play_cache_path(game_pk: int) -> Path:
    return raw_root() / "mlb_api" / "play_by_play" / f"{game_pk}.json"


def live_feed_cache_path(game_pk: int) -> Path:
    return raw_root() / "mlb_api" / "feed_live" / f"{game_pk}.json"


def fetch_play_by_play(
    game_pk: int,
    *,
    force: bool = False,
    sleep_seconds: float = 0.0,
    timeout: int = 30,
) -> dict[str, Any]:
    """Fetch MLB Stats API playByPlay with an immutable local JSON cache."""
    ensure_data_dirs()
    path = play_by_play_cache_path(game_pk)
    if path.exists() and not force:
        return json.loads(path.read_text())

    url = f"{MLB_API_BASE}/game/{game_pk}/playByPlay"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    path.write_text(json.dumps(payload, sort_keys=True))
    if sleep_seconds:
        time.sleep(sleep_seconds)
    return payload


def fetch_live_feed(
    game_pk: int,
    *,
    force: bool = False,
    sleep_seconds: float = 0.0,
    timeout: int = 30,
) -> dict[str, Any]:
    """Fetch MLB Stats API feed/live with a local JSON cache."""
    ensure_data_dirs()
    path = live_feed_cache_path(game_pk)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return json.loads(path.read_text())

    url = f"{MLB_API_BASE}.1/game/{game_pk}/feed/live"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    path.write_text(json.dumps(payload, sort_keys=True))
    if sleep_seconds:
        time.sleep(sleep_seconds)
    return payload


def home_plate_umpire_id(live_feed: dict[str, Any]) -> int | None:
    officials = live_feed.get("liveData", {}).get("boxscore", {}).get("officials", [])
    for official in officials:
        if official.get("officialType") == "Home Plate":
            umpire_id = official.get("official", {}).get("id")
            return int(umpire_id) if umpire_id is not None else None
    return None
