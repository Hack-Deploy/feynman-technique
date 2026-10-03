"""Dashboard data must agree with the simulation's own summary."""

import pytest

import report


@pytest.fixture(scope="module")
def outputs():
    return report.load_outputs()


def test_final_balances_match_summary(outputs):
    events, ledger, summary = outputs
    series = report.balance_series(events)
    for run_id, run in series.items():
        final = summary["run_summaries"][run_id]["final_balances"]
        for agent, pts in run["points"].items():
            assert pts[-1][1] == pytest.approx(final[f"agent:{agent}"])


def test_ledger_rows_match_summary(outputs):
    events, ledger, summary = outputs
    rows = report.ledger_rows(ledger)
    for run_id, s in summary["run_summaries"].items():
        if run_id.startswith("track_a_"):
            assert len(rows.get(run_id, [])) == s["ledger_size"]
