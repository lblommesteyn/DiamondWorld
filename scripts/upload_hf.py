"""Publish the DiamondWorld data to a Hugging Face dataset repo.

WHAT GETS UPLOADED, AND WHY IT IS NOT SIMPLY "EVERYTHING"

`data/` is 32 GB, but 31 GB of that is data/raw/mlb_api: 45,526 individual
feed_live JSON files, one per game. Uploading 45k small files to the Hub is the
wrong shape for a dataset repo. Every file becomes its own LFS pointer and its
own request, the push takes hours, and the result is awkward to consume. This
project has already shipped one Hub dataset that turned out to be unloadable, so
the raw tier is packed into one archive per source instead of 45k loose files.

Tiers:
  derived      (default, ~260 MB)  processed parquet, eval outputs, sim chunks
  checkpoints  (+631 MB)           trained model parameters
  raw          (+32 GB)            packed archives of the API and Statcast pulls
  all                              all three

The derived tier is the one that makes the results reproducible without a 32 GB
download: the parquet files are what training and evaluation actually read. The
raw tier is re-downloadable from the public MLB API with this project's own
downloader, so it is a convenience rather than a dependency.

BEFORE PUBLISHING THE RAW TIER, note that it is bulk MLBAM game data. Whether it
is yours to redistribute is a licensing question, not a technical one, and this
script does not answer it.

Usage:
  hf auth login                    # once, needs a WRITE token
  python scripts/upload_hf.py --repo <user>/diamondworld --tier derived --dry-run
  python scripts/upload_hf.py --repo <user>/diamondworld --tier derived
  python scripts/upload_hf.py --repo <user>/diamondworld --verify
"""
from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

ROOT = Path.home() / "DiamondWorld"

DERIVED = [
    ("data/processed", "data/processed"),
    ("data/eval2", "data/eval2"),
    ("data/chunks", "data/chunks"),
    ("data/projections_2024.csv", "data/projections_2024.csv"),
    ("RESULTS.md", "RESULTS.md"),
]


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def tree_size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def pack_raw(outdir: Path) -> list[tuple[str, str]]:
    """Pack data/raw into a few archives rather than 45k loose files."""
    outdir.mkdir(parents=True, exist_ok=True)
    out = []
    raw = ROOT / "data" / "raw"
    for sub in sorted(p for p in raw.iterdir() if p.is_dir()):
        tarpath = outdir / f"raw_{sub.name}.tar.gz"
        if not tarpath.exists():
            print(f"  packing {sub.name} -> {tarpath.name} (slow)", flush=True)
            with tarfile.open(tarpath, "w:gz") as tf:
                tf.add(sub, arcname=sub.name)
        out.append((str(tarpath), f"data/raw/{tarpath.name}"))
    return out


def verify(repo: str) -> int:
    """Download one parquet back and read it.

    A dataset that uploaded without error but cannot be read back is not
    published, it is only uploaded. This project has shipped that failure before,
    so the check is part of the tool rather than a thing to remember to do.
    """
    from huggingface_hub import hf_hub_download
    import polars as pl
    print(f"verifying {repo} ...")
    f = hf_hub_download(repo_id=repo, repo_type="dataset",
                        filename="data/processed/pitches_2024.parquet")
    df = pl.read_parquet(f)
    print(f"OK: pitches_2024.parquet -> {df.height:,} rows, {df.width} cols")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. lblommesteyn/diamondworld")
    ap.add_argument("--tier", default="derived",
                    choices=["derived", "checkpoints", "raw", "all"])
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="download a file back and read it, then exit")
    args = ap.parse_args()

    if args.verify:
        return verify(args.repo)

    from huggingface_hub import HfApi
    api = HfApi()

    items: list[tuple[str, str]] = []
    for src, dst in DERIVED:
        p = ROOT / src
        if p.exists():
            items.append((str(p), dst))
    if args.tier in ("checkpoints", "all"):
        p = ROOT / "checkpoints"
        if p.exists():
            items.append((str(p), "checkpoints"))
    if args.tier in ("raw", "all"):
        items += pack_raw(ROOT / "data" / "_hf_raw_archives")

    total = sum(tree_size(Path(s)) for s, _ in items)
    print(f"\n{len(items)} paths, {human(total)} total:")
    for s, d in items:
        print(f"  {human(tree_size(Path(s))):>9}  {d}")

    # Authentication is checked AFTER the inventory, so --dry-run shows what would
    # be published without needing a token. Seeing the size and the file list is
    # what you want before authenticating, not after.
    if args.dry_run:
        print("\ndry run, nothing uploaded")
        return 0

    try:
        who = api.whoami()
        print(f"\nauthenticated as {who.get('name')}")
    except Exception as e:
        print("NOT AUTHENTICATED. Run: hf auth login  (needs a WRITE token)",
              file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        return 2

    api.create_repo(args.repo, repo_type="dataset", private=args.private,
                    exist_ok=True)
    card = ROOT / "HF_DATASET_CARD.md"
    if card.exists():
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md",
                        repo_id=args.repo, repo_type="dataset")
    for src, dst in items:
        p = Path(src)
        print(f"uploading {dst} ...", flush=True)
        if p.is_dir():
            api.upload_folder(folder_path=str(p), path_in_repo=dst,
                              repo_id=args.repo, repo_type="dataset")
        else:
            api.upload_file(path_or_fileobj=str(p), path_in_repo=dst,
                            repo_id=args.repo, repo_type="dataset")

    print(f"\ndone: https://huggingface.co/datasets/{args.repo}")
    print("Now verify it actually loads before telling anyone it works:")
    print(f"  python scripts/upload_hf.py --repo {args.repo} --verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
