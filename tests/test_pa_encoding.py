"""Guard the plate-appearance outcome encoding.

Why this file exists: an eval script once set `KIDX = 2` and called it strikeouts.
Class 2 is HBP. Nothing crashed, nothing looked wrong, and an entire architecture
comparison was scored against the wrong column, producing a published-quality
finding ("sequence models collapse on K") that was pure indexing. The class order
is a silent contract between the data pipeline, the rules engine, the model head,
and every eval script, so it gets a test.

Three layers of defence:
  1. the canonical order is pinned to a literal list (any reorder fails loudly);
  2. every independent definition of that order agrees;
  3. every eval script's hardcoded index constants match the canonical order,
     checked statically so this test never imports JAX or touches the GPU.

Plus an empirical check that the encoded integers land on realistic MLB rates,
which is what actually catches a mislabelled column.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# The contract. Do not "fix" this list to match code; fix the code.
CANONICAL = ["K", "BB", "HBP", "1B", "2B", "3B", "HR", "out", "E"]
CANON_IDX = {o: i for i, o in enumerate(CANONICAL)}


# ---------------------------------------------------------------- 1. pinned order

def test_canonical_order_is_pinned():
    from diamondworldjax.sim.rules_engine import PA_OUTCOMES, PA_OUTCOME_IDX

    assert list(PA_OUTCOMES) == CANONICAL
    assert PA_OUTCOME_IDX["K"] == 0, "K must be class 0"
    assert PA_OUTCOME_IDX["HBP"] == 2, "class 2 is HBP, not K (the original bug)"
    assert PA_OUTCOME_IDX["HR"] == 6
    assert len(PA_OUTCOMES) == 9


# ------------------------------------------------- 2. independent definitions agree

def test_rules_engine_and_baselines_agree():
    """Two packages define the order separately; they must not drift apart."""
    from diamondworld.baselines.base import PA_OUTCOMES as BASE_OUTCOMES
    from diamondworldjax.sim.rules_engine import PA_OUTCOMES as JAX_OUTCOMES

    assert list(BASE_OUTCOMES) == list(JAX_OUTCOMES) == CANONICAL


def test_pipeline_encoder_matches_canonical():
    """The string -> int encoder every parquet is built with."""
    from diamondworldjax.data.pipeline import _encode_outcome

    for name, idx in CANON_IDX.items():
        assert _encode_outcome(name) == idx, f"{name} encodes to the wrong class"
    assert _encode_outcome(None) == -1
    assert _encode_outcome("not_an_outcome") == -1, "unknown outcomes must sentinel, not alias class 0"


# ------------------------------------------------- 3. eval scripts' index constants

# name -> expected value, in terms of the canonical order.
EXPECTED_CONSTS = {
    "KIDX": CANON_IDX["K"],
    "HRIDX": CANON_IDX["HR"],
    "HIT_IDX": [CANON_IDX[o] for o in ("1B", "2B", "3B", "HR")],
    "BB_IDX": [CANON_IDX[o] for o in ("BB", "HBP")],
}

# Every script that scores players on per-outcome rates and therefore needs the
# class indices. Add new ones here; a script that slices probs[:, i] belongs here.
SCORING_SCRIPTS = [
    "diamondworldjax/scripts/wm_sweep.py",
    "diamondworldjax/scripts/seq_models.py",
    "diamondworldjax/scripts/prod_playercorr.py",
]


def _resolve_index_consts(path: Path) -> dict[str, object]:
    """Evaluate only the outcome-index assignments in a module, nothing else.

    Static on purpose: importing these modules pulls in JAX/torch and allocates
    GPU memory, which a unit test must never do. We parse the file, keep the
    assignment statements that bind the names we care about, and exec just those
    with PA_OUTCOME_IDX in scope, so both literal forms (`KIDX = 0`) and derived
    forms (`KIDX = PA_OUTCOME_IDX["K"]`) resolve.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted = set(EXPECTED_CONSTS)
    keep = []
    for node in tree.body:  # module level only; these are module constants
        if not isinstance(node, ast.Assign):
            continue
        bound = set()
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                bound.add(tgt.id)
            elif isinstance(tgt, ast.Tuple):  # KIDX, HRIDX, ... = 0, 6, ...
                bound |= {e.id for e in tgt.elts if isinstance(e, ast.Name)}
        if bound & wanted:
            keep.append(node)

    ns: dict[str, object] = {"PA_OUTCOME_IDX": dict(CANON_IDX), "PA_OUTCOMES": list(CANONICAL)}
    if keep:
        exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), ns)
    return {k: v for k, v in ns.items() if k in wanted}


@pytest.mark.parametrize("rel", SCORING_SCRIPTS)
def test_script_index_constants_match_canonical(rel):
    path = REPO / rel
    assert path.exists(), f"{rel} moved; update SCORING_SCRIPTS"
    found = _resolve_index_consts(path)
    assert found, f"{rel} defines no outcome-index constants; did the metric move?"
    for name, value in found.items():
        expected = EXPECTED_CONSTS[name]
        got = list(value) if isinstance(value, (list, tuple)) else value
        assert got == expected, (
            f"{rel}: {name} = {got}, canonical is {expected}. "
            f"Class order is {CANONICAL}."
        )


def test_hit_and_bb_buckets_are_disjoint_and_complete():
    """The rate buckets must partition cleanly, or a PA gets double-counted."""
    hit, bb = set(EXPECTED_CONSTS["HIT_IDX"]), set(EXPECTED_CONSTS["BB_IDX"])
    assert not hit & bb
    assert CANON_IDX["K"] not in hit | bb
    # everything not in a bucket is K, out, or E
    rest = set(range(9)) - hit - bb - {CANON_IDX["K"]}
    assert rest == {CANON_IDX["out"], CANON_IDX["E"]}


# ------------------------------------------------- 4. the encoded data looks like baseball

# 2024 terminal-PA frequencies, from the data itself. Tolerances are wide enough
# for a season of drift but far tighter than the gap between any two classes, so
# a swapped column (K 0.226 vs HBP 0.011) fails by a mile.
REFERENCE_2024 = {
    "K": (0.226, 0.03), "BB": (0.082, 0.02), "HBP": (0.011, 0.008),
    "1B": (0.142, 0.03), "2B": (0.043, 0.015), "3B": (0.004, 0.004),
    "HR": (0.030, 0.012), "out": (0.463, 0.05),
}


def _parquet_2024():
    from diamondworldjax.paths import processed_root
    p = processed_root() / "pitches_2024.parquet"
    return p if p.exists() else None


@pytest.mark.skipif(_parquet_2024() is None, reason="processed 2024 parquet not present")
def test_encoded_classes_have_realistic_frequencies():
    """Catches the failure mode a pure-constant test cannot: an integer column
    whose values do not mean what the label says."""
    import polars as pl
    from diamondworldjax.data.pipeline import _encode_outcome

    df = (pl.scan_parquet(_parquet_2024())
          .filter(pl.col("pa_terminal"))
          .select("pa_outcome")
          .collect())
    counts = df["pa_outcome"].value_counts()
    total = int(counts["count"].sum())
    freq = {r["pa_outcome"]: r["count"] / total for r in counts.iter_rows(named=True)}

    for name, (expect, tol) in REFERENCE_2024.items():
        got = freq.get(name, 0.0)
        assert abs(got - expect) < tol, f"{name}: {got:.4f} vs expected ~{expect}"

    # and the integers those strings encode to are the canonical ones
    by_idx = {_encode_outcome(k): v for k, v in freq.items()}
    assert abs(by_idx[CANON_IDX["K"]] - REFERENCE_2024["K"][0]) < REFERENCE_2024["K"][1]
    assert by_idx[CANON_IDX["K"]] > 10 * by_idx[CANON_IDX["HBP"]], (
        "class 0 should be strikeouts (~0.23), not hit-by-pitch (~0.01)"
    )
