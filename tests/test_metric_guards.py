"""Guards for two silent-failure defects found in review, both of which produced
plausible-looking numbers rather than errors.

1. p0_error compared P(runs >= 0) on both sides, which is 1.0 for any non-negative
   run total, so the metric was identically 0.00000 and read as a perfect score.

2. The player table reserves no sentinel index, so slot 0 is a real player onto whom
   every unseen player is folded. The pooled row clears any min-PA filter and was
   entering the published cross-player correlations.
"""
from __future__ import annotations

import numpy as np


def test_p0_error_detects_a_shutout_rate_gap():
    """A model that never throws a shutout must not score 0 against reality."""
    from diamondworldjax.eval.calibration import game_run_metrics

    obs = np.array([0, 0, 0, 1, 2, 3, 4, 5])   # 3/8 shutouts
    sim = np.array([1, 1, 1, 1, 2, 3, 4, 5])   # 0/8 shutouts

    m = game_run_metrics(sim, obs)
    assert m["p0_error"] > 0.3, (
        "p0_error is insensitive to shutout rate; it is probably comparing "
        "P(runs >= 0), which is 1.0 on both sides by construction"
    )
    assert abs(m["p0_error"] - 3 / 8) < 1e-9


def test_p0_error_is_zero_only_when_shutout_rates_match():
    from diamondworldjax.eval.calibration import game_run_metrics

    obs = np.array([0, 0, 1, 2])
    sim = np.array([0, 0, 3, 4])
    assert game_run_metrics(sim, obs)["p0_error"] == 0.0


def test_unknown_player_sink_is_excluded_from_the_metric(tmp_path):
    """Index 0 pools one real batter with every unseen batter, so it must not be
    scored as a player. Its PA count is large, so a min-PA filter cannot catch it."""
    import subprocess
    import sys

    n = 60
    rng = np.random.default_rng(0)
    cnt = np.full(n, 400.0)
    cnt[0] = 11813.0                      # the sink, as it appears in real files
    real = rng.uniform(0.15, 0.30, size=n)
    pred = real + rng.normal(0, 0.02, n)
    # Make the sink an extreme outlier in the direction that inflates correlation.
    real[0], pred[0] = 0.60, 0.60

    d = {"cnt": cnt}
    for s in ("K", "BB", "Hit", "HR"):
        d["sum" + s] = pred * cnt
        d["r" + s] = real * cnt
    p = tmp_path / "prod_rates_test.npz"
    np.savez(p, **d)

    out = subprocess.run(
        [sys.executable, "-m", "diamondworldjax.scripts.bootstrap_playercorr",
         "--rates", "a=%s" % p, "--reps", "200",
         "--out", str(tmp_path / "o.txt"), "--json-out", str(tmp_path / "o.json")],
        capture_output=True, text=True, check=True,
    )
    import json
    res = json.loads((tmp_path / "o.json").read_text())
    assert res["n_batters"] == n - 1, (
        "the index-0 unknown-player sink was scored as a batter (n=%d, expected %d); "
        "output:\n%s" % (res["n_batters"], n - 1, out.stdout)
    )
