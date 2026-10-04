"""Archive visualizations preserve observations and recorded outcomes."""
import json
from pathlib import Path

from poc import recorded_replay as replay

SOURCE = Path(__file__).resolve().parents[1] / 'attempts/fixtures/live/runs.jsonl'


def entries():
    return [json.loads(line) for line in SOURCE.read_text().splitlines()]


def test_every_archive_run_preserves_outcome_and_original_criteria():
    for entry in entries():
        result = replay.build(entry)
        record = entry['record']
        assert result['passed'] == record['verdict']['passed']
        assert result['resolution_criteria'] == record['extra']['resolution_criteria']
        assert len(result['rounds']) == len(entry['rounds'])
        for trace in result['traces']:
            assert len(trace['times']) == len(trace['observed']) == len(trace['positions'])
            original = record['extra']['runs'][trace['experiment'] - 1]
            assert trace['times'] == original['output']['measurement_times']


def test_individual_probes_are_preserved_instead_of_averaged():
    entry = next(e for e in entries() if e['key']['hypothesis_id'] == 'dark-matter-unseen-pull'
                 and e['key']['model'] == 'claude-sonnet-5')
    result = replay.build(entry)
    assert len(result['traces']) == 5
    assert result['stats']['points'] == 50
    positions = entry['record']['extra']['runs'][0]['output']['positions']
    for i, trace in enumerate(result['traces']):
        assert trace['positions'] == [frame[-5 + i] for frame in positions]
    assert result['passed'] is False


def test_sparse_coulomb_readings_do_not_become_fabricated_curves():
    entry = next(e for e in entries() if e['key']['hypothesis_id'] == 'coulomb-source-strength')
    result = replay.build(entry)
    assert result['stats'] == {'traces': 2, 'curves': 0, 'points': 2, 'fits': 0}
    assert all(len(t['times']) == 1 for t in result['traces'])


def test_recommendations_select_rich_failure_and_disclose_missing_outcomes():
    rows = [{'hypothesis_id': e['key']['hypothesis_id'], 'passed': e['record']['verdict']['passed'],
             'attempt_id': e['record']['attempt_id'], 'visual_stats': replay.stats(e)} for e in entries()]
    selected = replay.recommendations(rows)
    by_id = {e['record']['attempt_id']: e for e in entries()}
    dark = selected['dark-matter-unseen-pull']
    assert dark['success'] is None
    assert by_id[dark['failure']]['key']['model'] == 'claude-sonnet-5'
    assert selected['circle-ordinary-gravity']['failure'] is None
    for choices in selected.values():
        for kind, passed in [('success', True), ('failure', False)]:
            if choices[kind]:
                assert by_id[choices[kind]]['record']['verdict']['passed'] == passed


def test_fitted_numpy_scalars_are_read_without_evaluating_code():
    assert replay.fit_params("{'fitted_params': {'n': np.float64(2.0)}, 'error': None}") == {'n': {'value': 2.0}}
    assert replay.fit_params("__import__('os').system('false')") == {}
    assert replay.fit_params({'error': 'fit failed', 'fitted_params': {'n': 2.0}}) == {}
