from __future__ import annotations

import sys
from pathlib import Path


def add_repo_root_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    root = str(repo_root)
    if root not in sys.path:
        sys.path.insert(0, root)
