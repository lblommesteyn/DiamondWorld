"""Export either task from a shared-skills checkpoint for standalone use."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax

from diamondworldjax.model.multitask import task_checkpoint_params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True,
                        help="Checkpoint produced by train_shared_skills.py")
    parser.add_argument("--task", choices=["pa", "pitch"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with args.ckpt.open("rb") as file:
        checkpoint = pickle.load(file)
    params = task_checkpoint_params(checkpoint["params"], args.task)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as file:
        pickle.dump({"step": checkpoint.get("step"), "params": jax.device_get(params)}, file)
    print(f"Exported {args.task} checkpoint with {len(params)} parameter sites to {args.out}")


if __name__ == "__main__":
    main()
