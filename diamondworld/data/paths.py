from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    return project_root() / "data"


def raw_root() -> Path:
    return data_root() / "raw"


def processed_root() -> Path:
    return data_root() / "processed"


def ensure_data_dirs() -> None:
    for path in [
        raw_root() / "statcast",
        raw_root() / "mlb_api" / "play_by_play",
        raw_root() / "mlb_api" / "feed_live",
        processed_root(),
    ]:
        path.mkdir(parents=True, exist_ok=True)
