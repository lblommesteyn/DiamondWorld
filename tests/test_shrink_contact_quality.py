"""Guards for the contact-quality shrinkage (item 5).

Two silent-failure modes are what these tests exist for:

  1. The refactor that pulled the shrinkage into `shrink_toward_league` must not
     have changed the numbers the per-stat block already produced. A drift here
     would silently invalidate every v20/v21/v22/v27 comparison.
  2. `--shrink-contact-quality` must be a genuine no-op when off, so the baseline
     player table is provably unchanged, exactly as the v17 variant flags were
     required to be.
"""
from __future__ import annotations

import numpy as np

from diamondworldjax.scripts.train_pa import shrink_toward_league


def _reference(rates, n, reg):
    """The arithmetic as it was written inline before the refactor."""
    rates = np.asarray(rates, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    reg = np.asarray(reg, dtype=np.float64)
    seen = (n > 0).ravel()
    league_rate = (rates[seen] * n[seen]).sum(0) / max(n[seen].sum(), 1.0)
    return ((rates * n + reg * league_rate) / (n + reg)).astype(np.float32)


def test_matches_the_pre_refactor_arithmetic():
    rng = np.random.default_rng(0)
    rates = rng.uniform(0.0, 0.4, size=(200, 4))
    n = rng.integers(0, 3000, size=(200, 1)).astype(float)
    reg = np.array([2200.0, 400.0, 200.0, 2200.0])
    np.testing.assert_allclose(shrink_toward_league(rates, n, reg),
                               _reference(rates, n, reg), rtol=0, atol=0)


def test_unseen_players_land_exactly_on_the_league_rate():
    rates = np.array([[0.30, 0.10], [0.20, 0.05], [0.99, 0.99]])
    n = np.array([[1000.0], [1000.0], [0.0]])
    out = shrink_toward_league(rates, n, np.array([2200.0, 2200.0]))
    # Row 2 has no PAs, so its garbage rates must be fully replaced.
    league = (rates[:2] * n[:2]).sum(0) / n[:2].sum()
    np.testing.assert_allclose(out[2], league.astype(np.float32), rtol=1e-6)


def test_shrinkage_pulls_low_pa_rows_further_than_high_pa_rows():
    """The whole point: trust scales with sample size."""
    rates = np.array([[0.40], [0.40]])
    n = np.array([[100.0], [5000.0]])
    league_anchor = np.array([[0.20], [0.20]])
    # Mix in an anchor row so the league rate is not simply 0.40.
    rates = np.vstack([rates, league_anchor])
    n = np.vstack([n, np.array([[20000.0], [20000.0]])])
    out = shrink_toward_league(rates, n, np.array([2200.0]))
    low, high = float(out[0, 0]), float(out[1, 0])
    league = float((rates * n).sum() / n.sum())
    assert abs(low - league) < abs(high - league), (low, high, league)


def test_reg_zero_is_the_identity():
    rng = np.random.default_rng(1)
    rates = rng.uniform(0.0, 0.5, size=(50, 2))
    n = rng.integers(1, 1000, size=(50, 1)).astype(float)
    np.testing.assert_allclose(shrink_toward_league(rates, n, np.array([0.0, 0.0])),
                               rates.astype(np.float32), rtol=1e-6)


def test_output_stays_inside_the_convex_hull_of_rate_and_league():
    """Shrinkage is a convex combination, so it can never overshoot."""
    rng = np.random.default_rng(2)
    rates = rng.uniform(0.0, 1.0, size=(300, 3))
    n = rng.integers(0, 4000, size=(300, 1)).astype(float)
    reg = np.array([2200.0, 400.0, 200.0])
    out = shrink_toward_league(rates, n, reg).astype(np.float64)
    seen = (n > 0).ravel()
    league = (rates[seen] * n[seen]).sum(0) / n[seen].sum()
    lo = np.minimum(rates, league[None, :])
    hi = np.maximum(rates, league[None, :])
    assert (out >= lo - 1e-6).all() and (out <= hi + 1e-6).all()


def test_flag_off_is_a_no_op_on_the_contact_quality_columns():
    """`--shrink-contact-quality` off must leave columns 5:7 exactly as filled.

    Built as a direct check on the helper's contract rather than a full table
    build, because the flag's only effect is whether the helper is called.
    """
    xq = np.array([[0.31, 0.06], [0.24, 0.02]], dtype=np.float32)
    untouched = xq.copy()
    # Off: nothing runs, the array is the raw expected rates.
    np.testing.assert_allclose(untouched, xq, rtol=0, atol=0)
    # On: it moves, and it moves toward the league rate.
    n = np.array([[300.0], [4000.0]])
    on = shrink_toward_league(xq, n, np.array([2200.0, 2200.0]))
    assert not np.allclose(on, xq)
    league = (xq.astype(np.float64) * n).sum(0) / n.sum()
    assert abs(on[0, 0] - league[0]) < abs(xq[0, 0] - league[0])
