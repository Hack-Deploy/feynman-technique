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
            'profit': paid - record['lab_cost'], 'usd': entry.get('usd', record.get('llm_usage', {}).get('usd', 0))}


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


TIME_VARYING = ('oscillator',)


def _support_guesses() -> dict:
    from poc.baselines import SUPPORT_GUESS
    return {k: {'n': v['n'], 'a3': v['a3']} for k, v in SUPPORT_GUESS.items() if 'n' in v and 'a3' in v}


SUPPORT_GUESS_LAW = _support_guesses()


def law(entry: dict) -> dict | None:
    """Pull readings by round for runs that released a probe at rest near one source, with the
    same launches replayed on the noise-free simulator as the reference."""
    import numpy as np
    from scienceagent.worlds import get_world

    from poc import animate
    from poc import config as C

    record = entry['record']
    if record['world'] in TIME_VARYING:
        return None  # the pull changes over time, so one curve against distance would mislead
    runs = (record.get('extra') or {}).get('runs') or []
    rounds: dict = {}
    reference = []
    executor = None
    for run in runs:
        reading = animate._reading(run)
        if not reading:
            continue
        rounds.setdefault(run.get('round', 1), []).append(reading)
        try:
            executor = executor or get_world(record['world'], engine=C.ENGINE, noise_std=0.0,
                                             noise_seed=record['seed'])['executor']
            clean = animate._reading({'input': run['input'],
                                      'output': executor.run([run['input']])[0]})
        except Exception:  # a reference that fails to render never hides the readings
            clean = None
        if clean and clean['a'] > 0:
            reference.append(clean)
    if len({p['r'] for ps in rounds.values() for p in ps}) < 2:
        return None
    result = {'rounds': [{'round': k, 'points': v} for k, v in sorted(rounds.items())],
              'reference_points': reference}
    if len({p['r'] for p in reference}) >= 2:
        # a ≈ a3 (3/r)^n through the noise-free readings, so the reference draws as a curve.
        x = np.log([3 / p['r'] for p in reference])
        n, log_a3 = np.polyfit(x, np.log([p['a'] for p in reference]), 1)
        result['true'] = {'n': round(float(n), 4), 'a3': round(float(np.exp(log_a3)), 5)}
    guess = SUPPORT_GUESS_LAW.get((record.get('extra') or {}).get('hypothesis_id'))
    if guess:
        result['expected'] = guess  # the law the claim says holds, drawn as the hypothesis
    return result
