from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def processed_root() -> Path:
    return project_root() / "data" / "processed"


def checkpoints_root() -> Path:
    return project_root() / "checkpoints"


def results_root() -> Path:
    return project_root() / "eval" / "results"
