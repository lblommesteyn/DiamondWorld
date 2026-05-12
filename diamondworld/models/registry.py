from __future__ import annotations
import json
from pathlib import Path
import polars as pl


class PlayerRegistry:
    """Maps MLBAM integer IDs to contiguous 0-indexed integers. Index 0 = UNK."""

    def __init__(self):
        self._pitcher: dict[int, int] = {}
        self._batter: dict[int, int] = {}
        self._umpire: dict[int, int] = {}
        self._park: dict[str, int] = {}
        self._pitch_type: dict[str, int] = {}

    def fit(self, pitches: pl.DataFrame) -> None:
        """Fit on training data. Call with concatenated 2015-2021 data."""
        def build(col_vals, store):
            unique = sorted(set(v for v in col_vals if v is not None))
            store.clear()
            for i, v in enumerate(unique, start=1):  # 0 = UNK
                store[v] = i

        build(pitches["pitcher_id"].drop_nulls().to_list(), self._pitcher)
        build(pitches["batter_id"].drop_nulls().to_list(), self._batter)
        build(pitches["umpire_id"].drop_nulls().to_list(), self._umpire)
        build(pitches["park_id"].drop_nulls().to_list(), self._park)
        build(pitches["pitch_type"].drop_nulls().to_list(), self._pitch_type)

    def pitcher(self, v) -> int:
        return self._pitcher.get(v, 0)

    def batter(self, v) -> int:
        return self._batter.get(v, 0)

    def umpire(self, v) -> int:
        return self._umpire.get(v, 0)

    def park(self, v) -> int:
        return self._park.get(v, 0)

    def pitch_type(self, v) -> int:
        return self._pitch_type.get(v, 0)

    @property
    def n_pitchers(self) -> int:
        return len(self._pitcher) + 1

    @property
    def n_batters(self) -> int:
        return len(self._batter) + 1

    @property
    def n_umpires(self) -> int:
        return len(self._umpire) + 1

    @property
    def n_parks(self) -> int:
        return len(self._park) + 1

    @property
    def n_pitch_types(self) -> int:
        return len(self._pitch_type) + 1

    def save(self, path: Path) -> None:
        data = {
            "pitcher": {str(k): v for k, v in self._pitcher.items()},
            "batter": {str(k): v for k, v in self._batter.items()},
            "umpire": {str(k): v for k, v in self._umpire.items()},
            "park": self._park,
            "pitch_type": self._pitch_type,
        }
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path) -> PlayerRegistry:
        data = json.loads(path.read_text())
        reg = cls()
        reg._pitcher = {int(k): v for k, v in data["pitcher"].items()}
        reg._batter = {int(k): v for k, v in data["batter"].items()}
        reg._umpire = {int(k): v for k, v in data["umpire"].items()}
        reg._park = data["park"]
        reg._pitch_type = data["pitch_type"]
        return reg
