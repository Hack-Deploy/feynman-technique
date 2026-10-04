"""Presentation of archived measurements; never re-score a recorded verdict."""
from __future__ import annotations

import ast
import math
import re


def traces(runs: list[dict]) -> list[dict]:
    result = []
    for experiment, run in enumerate(runs, 1):
        inp, out = run.get('input') or {}, run.get('output') or {}
        times = out.get('measurement_times') or inp.get('measurement_times') or []
        if 'pos2' in out and 'pos2' in inp:
            starts = [inp['pos2']]
            frames = [[p] for p in out['pos2']]
            origin = inp.get('pos1', [0, 0])
        elif 'probe_positions' in inp and 'positions' in out:
            starts = inp['probe_positions']
            frames = [frame[-len(starts):] for frame in out['positions']]
            origin = [0, 0]
        elif 'ring_radius' in inp and out.get('positions'):
            count = len(out['positions'][0]) - 1
            starts = [[inp['ring_radius'] * math.cos(2 * math.pi * i / count),
                       inp['ring_radius'] * math.sin(2 * math.pi * i / count)] for i in range(count)]
            frames = [frame[1:] for frame in out['positions']]
            origin = [0, 0]
        else:
            continue
        if not times or len(times) != len(frames):
            continue
        for probe, start in enumerate(starts):
            try:
                xy = [[float(frame[probe][0]), float(frame[probe][1])] for frame in frames]
                values = [math.hypot(start[0] - origin[0], start[1] - origin[1])
                          - math.hypot(p[0] - origin[0], p[1] - origin[1]) for p in xy]
                ts = [float(t) for t in times]
                if not all(math.isfinite(v) for v in [*ts, *values, *(v for p in xy for v in p)]):
                    continue
            except (IndexError, ValueError, TypeError):
                continue
            result.append({'round': run.get('round', 1), 'experiment': experiment,
                           'probe': probe + 1, 'label': f'Experiment {experiment} · probe {probe + 1}',
                           'times': ts, 'observed': values, 'positions': xy})
    return result


def stats(entry: dict) -> dict:
    series = traces((entry['record'].get('extra') or {}).get('runs') or [])
    return {'traces': len(series), 'curves': sum(len(t['times']) > 1 for t in series),
            'points': sum(len(t['times']) for t in series),
            'fits': sum(bool(r.get('mse_fit')) for r in entry.get('rounds', []))}


def fit_params(raw) -> dict:
    if isinstance(raw, str):
        # Some archived NumPy scalar representations are not literal Python.
        raw = re.sub(r'np\.float(?:32|64)\(([-+\d.eE]+)\)', r'\1', raw)
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return {}
    if not isinstance(raw, dict) or raw.get('error'):
        return {}
    return {key: {'value': float(value)} for key, value in (raw.get('fitted_params') or {}).items()
            if isinstance(value, (int, float)) and math.isfinite(value)}


def build(entry: dict) -> dict:
    record = entry['record']
    x = record.get('extra') or {}
    estimates = {}
    rounds = []
    for rd in entry.get('rounds') or x.get('round_log') or []:
        estimates.update(fit_params(rd.get('mse_fit')))
        estimates.update(rd.get('estimates') or {})
        rounds.append({**rd, 'estimates': estimates.copy(),
                       'experiments': rd.get('n_experiments', rd.get('experiments', 0)), 'points': [],
                       'experiment_results': [{'input': run.get('input'), 'output': run.get('output')}
                           for run in x.get('runs') or [] if run.get('round') == rd.get('round')]})
    paid = x.get('prize_paid', 0)
    return {'kind': 'recorded', 'id': record['attempt_id'], 'hypothesis_id': x.get('hypothesis_id'),
            'agent': entry.get('model_label') or record['solver'], 'model': record['solver'],
            'hypothesis': x.get('hypothesis'), 'resolution_criteria': x.get('resolution_criteria'),
            'world': record['world'], 'seed': record['seed'], 'rounds': rounds,
            'traces': traces(x.get('runs') or []), 'stats': stats(entry),
            'max_rounds': x.get('max_rounds', entry.get('settings', {}).get('max_rounds')),
            'judge': record['verdict'].get('resolved_by'), 'answer': record['verdict'].get('answer'),
            'verdict': x.get('agent_verdict'), 'passed': record['verdict'].get('passed', False),
            'outcome': 'confirmed' if record['verdict'].get('passed') else 'false_claim',
            'prize': x.get('prize'), 'prize_paid': paid, 'spent': record['lab_cost'],
            'profit': paid - record['lab_cost'], 'usd': entry.get('usd', record.get('llm_usage', {}).get('usd', 0)),
            'source_url': 'https://huggingface.co/datasets/arushisinha98/discovery-market-live'}


def recommendations(rows: list[dict]) -> dict:
    result = {}
    for claim in dict.fromkeys(r['hypothesis_id'] for r in rows):
        choices = {}
        for name, passed in [('success', True), ('failure', False)]:
            candidates = [r for r in rows if r['hypothesis_id'] == claim and r['passed'] == passed]
            if candidates:
                best = max(candidates, key=lambda r: (r['visual_stats']['curves'] > 0,
                           r['visual_stats']['points'], r['visual_stats']['curves'], r['visual_stats']['fits']))
                choices[name] = best['attempt_id']
            else:
                choices[name] = None
        result[claim] = choices
    return result
