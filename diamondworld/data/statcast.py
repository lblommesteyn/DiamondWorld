from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import polars as pl
from pybaseball import statcast

from diamondworld.data.paths import ensure_data_dirs, raw_root


def season_dates(season: int) -> tuple[str, str]:
    return f"{season}-03-01", f"{season}-11-30"


def statcast_cache_path(season: int) -> Path:
    return raw_root() / "statcast" / f"statcast_{season}.parquet"


def monthly_ranges(start_dt: str, end_dt: str) -> list[tuple[str, str]]:
    start = date.fromisoformat(start_dt)
    end = date.fromisoformat(end_dt)
    ranges: list[tuple[str, str]] = []
    current = start
    while current <= end:
        next_month = date(current.year + int(current.month == 12), 1 if current.month == 12 else current.month + 1, 1)
        chunk_end = min(end, next_month - timedelta(days=1))
        ranges.append((current.isoformat(), chunk_end.isoformat()))
        current = chunk_end + timedelta(days=1)
    return ranges


def daily_ranges(start_dt: str, end_dt: str) -> list[tuple[str, str]]:
    start = date.fromisoformat(start_dt)
    end = date.fromisoformat(end_dt)
    return [(current.isoformat(), current.isoformat()) for current in _date_span(start, end)]


def _date_span(start: date, end: date) -> list[date]:
    days: list[date] = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def biweekly_ranges(start_dt: str, end_dt: str) -> list[tuple[str, str]]:
    start = date.fromisoformat(start_dt)
    end = date.fromisoformat(end_dt)
    ranges: list[tuple[str, str]] = []
    current = start
    while current <= end:
        chunk_end = min(end, current + timedelta(days=13))
        ranges.append((current.isoformat(), chunk_end.isoformat()))
        current = chunk_end + timedelta(days=1)
    return ranges


def fetch_statcast_range(start_dt: str, end_dt: str, *, attempts: int = 5) -> pl.DataFrame:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            pdf = statcast(start_dt=start_dt, end_dt=end_dt)
            return pl.from_pandas(pdf, include_index=False) if len(pdf) else pl.DataFrame()
        except Exception as exc:
            last_error = exc
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"Statcast fetch failed for {start_dt} through {end_dt}") from last_error


def fetch_statcast_season(season: int, *, force: bool = False) -> pl.DataFrame:
    """Fetch or load cached Statcast rows for one season.

    pybaseball returns pandas; the object is immediately converted to Polars and
    persisted as Parquet so downstream processing stays in Polars.
    Uses 2-week chunks to avoid connection resets from Baseball Savant.
    """
    ensure_data_dirs()
    cache_path = statcast_cache_path(season)
    if cache_path.exists() and not force:
        return pl.read_parquet(cache_path)

    frames = []
    for start_dt, end_dt in biweekly_ranges(*season_dates(season)):
        try:
            frame = fetch_statcast_range(start_dt, end_dt)
            if frame.height:
                frames.append(frame)
        except Exception:
            for day_start, day_end in daily_ranges(start_dt, end_dt):
                try:
                    frame = fetch_statcast_range(day_start, day_end)
                    if frame.height:
                        frames.append(frame)
                except Exception:
                    pass  # skip days that repeatedly fail (likely no games)
    frame = pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
    frame = frame.filter(pl.col("game_year") == season)
    frame.write_parquet(cache_path)
    return frame


def cache_status(season: int) -> dict[str, object]:
    path = statcast_cache_path(season)
    return {
        "season": season,
        "path": str(path),
        "exists": path.exists(),
        "checked_on": date.today().isoformat(),
    }
