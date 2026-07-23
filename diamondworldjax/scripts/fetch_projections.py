"""Fetch real preseason-2024 projections so the benchmark can be a direct
head-to-head instead of a Marcel proxy.

The obstacle documented in SSAC.md was that FanGraphs' projection leaderboards
are Cloudflare-gated. That is only true of the HTML pages: `/api/projections`
answers fine, but it always serves the CURRENT season (a `season=` parameter is
accepted and ignored), so it cannot give us 2024 preseason numbers today.

The way in is the Wayback Machine, which holds captures of both the FanGraphs
API and Razzball's public mirror of Steamer:

  hitters   Razzball's preseason Steamer table, captured 2024-04-13. The page is
            the frozen preseason set, not rest-of-season: it shows full-season
            playing time (Betts 150 G / 672 PA) and "Rest of Season" appears only
            as a link to Razzball's separate ROS page. Razzball adjusts Steamer's
            playing-time estimates but not its rates, and every metric here is a
            per-PA rate, so the adjustment does not touch what we measure.
  pitchers  FanGraphs' own Steamer pitcher projections, captured 2024-01-23,
            unambiguously preseason.

ZiPS and THE BAT are NOT available: no preseason-2024 capture of either exists in
the archive (the one THE BAT capture, 2024-01-25, is a 654-byte error response).
Steamer is the most widely used of the three and is the one we can source
honestly, so it is the head-to-head; do not fabricate the other two.

Writes data/projections_2024.csv with per-PA rates keyed by MLBAM id, which is
what the rest of the pipeline uses. Names are resolved to ids through the MLB
StatsAPI against the players actually in our 2024 test set.

  python -m diamondworldjax.scripts.fetch_projections
"""
from __future__ import annotations

import io
import json
import re
import subprocess
import unicodedata
import urllib.request
from pathlib import Path

import polars as pl

from diamondworldjax.data.pipeline import load_seasons
from diamondworldjax.paths import processed_root

WB = "https://web.archive.org/web"
SOURCES = {
    # (system, group): (wayback timestamp, original url, parser)
    ("steamer", "bat"): ("20240413220330", "https://razzball.com/steamer-hitter-projections/",
                         "razzball"),
    ("steamer", "pit"): ("20240123163225", "https://www.fangraphs.com/api/projections"
                                           "?type=steamer&stats=pit&pos=all&team=0&players=0&lg=all",
                         "fangraphs"),
}
OUT = Path("data/projections_2024.csv")
CACHE = Path("data/cache/projections")
_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv)\b")


def _fetch(url: str, cache_name: str) -> bytes:
    """Download via curl. The archive replays the original response bytes, which
    for FanGraphs are brotli-encoded; curl --compressed handles that, urllib does
    not (and `brotli` is not a project dependency)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / cache_name
    if path.exists() and path.stat().st_size > 1000:
        return path.read_bytes()
    out = subprocess.run(["curl", "-sSL", "--compressed", "--max-time", "180", url],
                         capture_output=True, check=True).stdout
    if len(out) < 1000:
        raise RuntimeError(f"{url} returned {len(out)} bytes; capture is probably an error page")
    path.write_bytes(out)
    return out


def norm(name: str) -> str:
    """Fold a display name to a match key: no accents, no punctuation, no suffix."""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^a-z ]", " ", s)
    s = _SUFFIX.sub(" ", s)
    return " ".join(s.split())


def parse_razzball(raw: bytes) -> list[dict]:
    import pandas as pd
    html = raw.decode("utf-8", errors="replace")
    tables = [t for t in pd.read_html(io.StringIO(html)) if len(t) > 50 and "PA" in t.columns]
    if not tables:
        raise RuntimeError("no projection table found in the Razzball capture")
    df = max(tables, key=len)
    rows = []
    for r in df.to_dict("records"):
        pa = float(r.get("PA") or 0)
        if pa < 1:
            continue
        rows.append(dict(name=str(r["Name"]), pa=pa,
                         so=float(r.get("SO") or 0), bb=float(r.get("BB") or 0),
                         hbp=float(r.get("HBP") or 0), h=float(r.get("H") or 0),
                         hr=float(r.get("HR") or 0)))
    return rows


def parse_fangraphs(raw: bytes) -> list[dict]:
    recs = json.loads(raw.decode("utf-8", errors="replace"))
    rows = []
    for r in recs:
        # TBF for pitchers, PA for hitters; the name field carries an <a> wrapper.
        pa = float(r.get("TBF") or r.get("PA") or 0)
        if pa < 1:
            continue
        name = r.get("PlayerName") or re.sub(r"<[^>]+>", "", str(r.get("Name", "")))
        rows.append(dict(name=name, pa=pa,
                         so=float(r.get("SO") or 0), bb=float(r.get("BB") or 0),
                         hbp=float(r.get("HBP") or 0), h=float(r.get("H") or 0),
                         hr=float(r.get("HR") or 0)))
    return rows


def id_lookup(ids: list[int]) -> dict[str, int]:
    """MLBAM id -> name for the players in our test set, inverted to a match key.

    Resolving in this direction (ours -> names) rather than scraping ids off the
    projection sources keeps the join anchored to the players we actually score.
    """
    cache = CACHE / "mlb_names.json"
    known = json.loads(cache.read_text()) if cache.exists() else {}
    miss = [i for i in ids if str(i) not in known]
    for j in range(0, len(miss), 100):
        chunk = miss[j:j + 100]
        url = "https://statsapi.mlb.com/api/v1/people?personIds=" + ",".join(map(str, chunk))
        try:
            d = json.loads(urllib.request.urlopen(url, timeout=30).read())
            for p in d.get("people", []):
                known[str(p["id"])] = p["fullName"]
        except Exception as e:  # partial resolution is fine; we report coverage
            print(f"  statsapi chunk failed ({e})")
    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(known))

    by_key: dict[str, int] = {}
    dupes = set()
    for sid, name in known.items():
        k = norm(name)
        if k in by_key and by_key[k] != int(sid):
            dupes.add(k)          # two real players share a name; refuse to guess
        by_key[k] = int(sid)
    for k in dupes:
        by_key.pop(k, None)
    if dupes:
        print(f"  dropped {len(dupes)} ambiguous name(s): {sorted(dupes)[:5]}")
    return by_key


def main():
    # the id space we care about: everyone with a real 2024 sample, both roles
    df24 = (load_seasons([2024], data_root=processed_root())
            .filter(pl.col("pa_terminal") & pl.col("pa_outcome").is_not_null()))
    ids = set()
    for col in ("batter_id", "pitcher_id"):
        if col in df24.columns:
            ids |= {int(x) for x in df24[col].unique().to_list()}
    print(f"resolving names for {len(ids)} players in the 2024 data ...")
    key2id = id_lookup(sorted(ids))

    out = []
    parsers = {"razzball": parse_razzball, "fangraphs": parse_fangraphs}
    for (system, group), (ts, url, parser) in SOURCES.items():
        print(f"fetching {system}/{group} ({ts}) ...")
        raw = _fetch(f"{WB}/{ts}id_/{url}", f"{system}_{group}_{ts}.raw")
        rows = parsers[parser](raw)
        hit = 0
        for r in rows:
            pid = key2id.get(norm(r["name"]))
            if pid is None:
                continue
            hit += 1
            pa = r["pa"]
            out.append(dict(system=system, group=group, mlbam_id=pid, name=r["name"], pa=pa,
                            k_rate=r["so"] / pa,
                            bb_rate=(r["bb"] + r["hbp"]) / pa,
                            hit_rate=r["h"] / pa,
                            hr_rate=r["hr"] / pa))
        print(f"  {len(rows)} projected, {hit} matched to MLBAM ids ({hit / max(len(rows), 1):.0%})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(out).write_csv(OUT)
    print(f"wrote {len(out)} rows -> {OUT}")


if __name__ == "__main__":
    main()
