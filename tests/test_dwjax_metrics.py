import numpy as np

from diamondworldjax.domain import N_PA_OUTCOMES, PA_OUTCOME_IDX
from diamondworldjax.eval.metrics import game_run_metrics


def test_zero_run_error_uses_equality_not_greater_than_zero():
    report = game_run_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
    assert report["p0_error"] == 0.25


def test_outcome_contract_has_exactly_one_error_class():
    assert N_PA_OUTCOMES == 9
    assert PA_OUTCOME_IDX["E"] == 8
    assert sorted(PA_OUTCOME_IDX.values()) == list(range(N_PA_OUTCOMES))
