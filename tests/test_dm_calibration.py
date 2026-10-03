import pytest

from dm.calibration import (
    brier_score,
    calibration_in_the_large,
    calibration_summary,
    pairs_from_records,
    reliability_table,
)
from dm.types import AttemptRecord


PAIRS = [(0.9, True), (0.7, False), (0.2, False), (0.2, True), (1.0, True)]


def _record(attempt_id, stated_p, passed, solver="s"):
    return AttemptRecord(
        attempt_id=attempt_id,
        source="live",
        protocol="forcebench_menu",
        venue="forcebench",
        world="gravity",
        solver=solver,
        seed=0,
        stated_p_success=stated_p,
        rounds=1,
        experiments=1,
        lab_cost=1.0,
        verdict={"passed": passed},
    )


def test_hand_computed_scores_and_reliability_bins():
    assert brier_score(PAIRS) == pytest.approx(0.236)
    assert calibration_in_the_large(PAIRS) == pytest.approx(0.0)
    table = reliability_table(PAIRS)
    assert table[9] == {"lo": 0.9, "hi": 1.0, "count": 2,
                        "mean_p": 0.95, "pass_rate": 1.0}
    assert table[7] == {"lo": 0.7, "hi": 0.8, "count": 1,
                        "mean_p": 0.7, "pass_rate": 0.0}
    assert table[2] == {"lo": 0.2, "hi": 0.3, "count": 2,
                        "mean_p": 0.2, "pass_rate": 0.5}
    assert table[0]["count"] == 0
    assert table[0]["mean_p"] is table[0]["pass_rate"] is None


def test_nonzero_calibration_in_the_large():
    assert calibration_in_the_large([(0.9, False), (0.8, False)]) == pytest.approx(0.85)


def test_empty_pairs_and_probability_endpoints():
    assert brier_score([]) is None
    assert calibration_in_the_large([]) is None
    assert reliability_table([])[0]["count"] == 0
    table = reliability_table([(0.0, False), (1.0, True)])
    assert table[0]["count"] == 1
    assert table[-1]["count"] == 1


def test_records_without_stated_probability_are_excluded_and_counted():
    records = [
        _record("a", 0.6, True),
        _record("b", None, False),
        _record("c", None, True, solver="other"),
    ]
    pairs, excluded = pairs_from_records(records)
    assert pairs == [(0.6, True)]
    assert excluded == 2
    summary = calibration_summary(records)
    assert summary["n"] == 1
    assert summary["n_excluded_no_stated_p"] == 2
    assert summary["by_solver"]["other"]["n"] == 0
    assert summary["by_solver"]["other"]["n_excluded_no_stated_p"] == 1
