#!/usr/bin/env python3
"""Small user-workflow evaluation; no training or extra dependencies beyond NaLa."""
import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import nala


FIXTURES = Path(__file__).resolve().parent / 'evals' / 'value_sanity.json'
NOISE_FLOOR = 1e-4  # nats per weighted target token, used only for sign agreement
THRESHOLDS = {'mean_spearman_min': 0.7, 'sign_agreement_min': 0.8,
              'material_edits_min': 10, 'greedy_random_tolerance': 1e-6}


def average_ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def ranking_metrics(rows, normalizer):
    """Deletion/suppression contribution: larger actual loss increase is better."""
    if not rows or not math.isfinite(normalizer) or normalizer <= 0:
        raise ValueError('Metrics require candidates and a positive token normalizer.')
    rows = sorted(rows, key=lambda r: r['span_index'])
    pred = np.array([r['predicted_loss_change'] for r in rows]) / normalizer
    actual = np.array([r['actual_loss_change'] for r in rows]) / normalizer
    if not np.isfinite(pred).all() or not np.isfinite(actual).all():
        raise ValueError('Nonfinite candidate measurements.')
    rp, ra = average_ranks(pred), average_ranks(actual)
    rho = float(np.corrcoef(rp, ra)[0, 1]) if np.ptp(rp) and np.ptp(ra) else None
    material = np.abs(actual) >= NOISE_FLOOR
    correct = int(np.sum((np.sign(pred) == np.sign(actual)) & material))
    chosen = int(np.argmax(pred))
    return {'spearman': rho, 'sign_correct': correct, 'material_edits': int(material.sum()),
            'sign_agreement': correct / int(material.sum()) if material.any() else None,
            'mae_nats_per_token': float(np.mean(np.abs(pred-actual))),
            'top1_regret_nats_per_token': float(actual.max()-actual[chosen]),
            'predicted_top_span': rows[chosen]['span_index'],
            'actual_top_span': rows[int(np.argmax(actual))]['span_index']}


def subset_metrics(subsets, greedy, static, measured_greedy_loss):
    """Use executed JOINT subset losses, never sums of singleton effects."""
    losses = {tuple(sorted(row['retained'])): row['nll_per_token'] for row in subsets}
    greedy, static = tuple(sorted(greedy)), tuple(sorted(static))
    if len(losses) != len(subsets) or not all(math.isfinite(v) for v in losses.values()):
        raise ValueError('Subset measurements must be unique and finite.')
    oracle = min(losses, key=lambda key: (losses[key], key))
    random_loss = math.fsum(losses.values()) / len(losses)
    return {'greedy_retained': list(greedy), 'static_retained': list(static),
            'oracle_retained': list(oracle), 'greedy_nll_per_token': measured_greedy_loss,
            'static_nll_per_token': losses[static], 'random_mean_nll_per_token': random_loss,
            'oracle_nll_per_token': losses[oracle],
            'greedy_gain_over_random': random_loss-measured_greedy_loss,
            'greedy_gain_over_static': losses[static]-measured_greedy_loss,
            'greedy_regret_nats_per_token': measured_greedy_loss-losses[oracle],
            'independent_greedy_replay_passed': abs(losses[greedy]-measured_greedy_loss) <= 1e-6,
            'all_subsets': subsets}


def report_checks(report, expected_candidates, expected_executed):
    counts = report['counts']
    items = report['heldout_items']
    capture = report['capture_checks']
    executions = report['verification']
    return {
        'one_backward': counts['capture_backwards'] == counts['backward'] == 1,
        'no_scoring_model_calls': counts['model_calls_for_analytic_scoring'] ==
                                  counts['model_calls_for_greedy_rescoring'] == 0,
        'candidate_coverage': len(report['ranked']) == report['candidate_count'] == expected_candidates,
        'execution_coverage': len(executions) == counts['candidates_executed'] == expected_executed,
        'all_query_rows': counts['capture_query_rows'] == sum(len(i['token_ids']) for i in items),
        'capture_audits': len(capture) == len(items) and
                          all(c['verified'] and c['causal_mask_verified'] for c in capture),
        'edit_audits': all(e['local_verified'] and len(e['audits']) == len(items) and
                           all(a['verified'] and a['causal_mask_verified'] for a in e['audits'])
                           for e in executions),
        'noop': bool(report['noop']['verified']) if expected_executed else report['noop'] is None,
        'finite_scores': all(math.isfinite(r['predicted_loss_change']) and
                             (r['verification'] is None or math.isfinite(r['actual_loss_change']))
                             for r in report['ranked']),
    }


def run_case(lab, case, layer, destination):
    primary = lab.value(case['prompt'], heldout=case['heldout'], spans=case['spans'],
                        factors=[0.5], top_k=8, select=2, layers=[layer],
                        out=destination/'ranking')
    pool = primary['selection']['pool']
    pairs = list(itertools.combinations(range(4), 2))
    complements = [{'indices': sorted(i for j, rows in enumerate(pool) if j not in kept for i in rows)}
                   for kept in pairs]
    oracle = lab.value(case['prompt'], heldout=case['heldout'], spans=complements,
                       top_k=6, layers=[layer], out=destination/'subsets')
    normalizer = math.fsum(i['weight']*len(i['token_ids']) for i in primary['heldout_items'])
    groups = {kind: [r for r in primary['ranked'] if r['edit']['kind'] == kind]
              for kind in ['delete', 'reweight']}
    ranking = {kind: ranking_metrics(rows, normalizer) for kind, rows in groups.items()}
    static = [r['span_index'] for r in sorted(groups['delete'],
              key=lambda r: (-r['predicted_loss_change'], r['span_index']))[:2]]
    subsets = [{'retained': list(pairs[r['span_index']]),
                'nll_per_token': r['verification']['actual_loss']/normalizer}
               for r in sorted(oracle['ranked'], key=lambda r: r['span_index'])]
    selection = primary['selection']
    selected = subset_metrics(subsets, selection['selected_span_indices'], static,
                              selection['verification']['actual_loss']/normalizer)
    selected['full_context_nll_per_token'] = primary['baseline_loss']/normalizer
    selected['support_spans_retained'] = len(set(selected['greedy_retained']) & set(case['support']))
    checks = {**{'ranking/'+k: v for k, v in report_checks(primary, 8, 8).items()},
              **{'subsets/'+k: v for k, v in report_checks(oracle, 6, 6).items()}}
    remaining = set(range(4))
    rescoring = len(selection['steps']) == 2
    for step in selection['steps']:
        rescoring &= {r['span_index'] for r in step['rescored_candidates']} == remaining
        remaining.discard(step['retained_span_index'])
    checks.update(greedy_rescore_coverage=bool(rescoring),
                  greedy_local_audit=selection['verification']['local_verified'],
                  greedy_replay=selected['independent_greedy_replay_passed'],
                  same_prompt=primary['prompt_sha256'] == oracle['prompt_sha256'],
                  same_targets=primary['heldout_sha256'] == oracle['heldout_sha256'],
                  same_baseline=abs(primary['baseline_loss']-oracle['baseline_loss']) <= 1e-5)
    return {'id': case['id'], 'checks': checks, 'ranking': ranking, 'selection': selected,
            'weighted_target_tokens': normalizer, 'prompt_tokens': primary['prompt_tokens'],
            'support_top1': ranking['delete']['predicted_top_span'] in case['support'],
            'prompt_sha256': primary['prompt_sha256'], 'heldout_sha256': primary['heldout_sha256'],
            'counts': {'ranking': primary['counts'], 'subsets': oracle['counts']}}


def aggregate(cases, token_checks):
    metrics = [m for case in cases for m in case['ranking'].values()]
    rhos = [m['spearman'] for m in metrics if m['spearman'] is not None]
    material = sum(m['material_edits'] for m in metrics)
    signs = sum(m['sign_correct'] for m in metrics)
    mean = lambda xs: math.fsum(xs)/len(xs)
    quality = {'mean_spearman': mean(rhos) if rhos else None,
               'rank_groups_scored': len(rhos), 'rank_groups_total': len(metrics),
               'sign_agreement': signs/material if material else None, 'material_edits': material,
               'mean_mae_nats_per_token': mean([m['mae_nats_per_token'] for m in metrics]),
               'mean_top1_regret_nats_per_token': mean([m['top1_regret_nats_per_token'] for m in metrics]),
               'mean_greedy_gain_over_random': mean([c['selection']['greedy_gain_over_random'] for c in cases]),
               'mean_greedy_gain_over_static': mean([c['selection']['greedy_gain_over_static'] for c in cases]),
               'mean_greedy_regret_nats_per_token': mean([c['selection']['greedy_regret_nats_per_token'] for c in cases]),
               'support_top1_rate': mean([float(c['support_top1']) for c in cases])}
    gates = {'rank_agreement': bool(rhos) and quality['mean_spearman'] >= THRESHOLDS['mean_spearman_min'],
             'sign_agreement': material >= THRESHOLDS['material_edits_min'] and
                               quality['sign_agreement'] >= THRESHOLDS['sign_agreement_min'],
             'greedy_vs_random': quality['mean_greedy_gain_over_random'] >= -THRESHOLDS['greedy_random_tolerance']}
    correctness = all(token_checks.values()) and all(all(c['checks'].values()) for c in cases)
    return {'passed': correctness and all(gates.values()), 'correctness_passed': correctness,
            'quality_passed': all(gates.values()), 'quality_gates': gates, 'quality': quality}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('runs')/('value-sanity-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')))
    parser.add_argument('--model', default=nala.MODEL_ID)
    parser.add_argument('--revision', help='Full immutable model commit; resolved and recorded when omitted.')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    parser.add_argument('--layer', type=int, default=18)
    parser.add_argument('--context-limit', type=int, default=4096)
    parser.add_argument('--local-only', action='store_true', help='Require cached model weights; never download.')
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('Choose a new --out directory; existing results are preserved.')
    if args.layer < 0:
        parser.error('--layer must be nonnegative.')
    if args.context_limit < 2:
        parser.error('--context-limit must be at least 2.')
    raw = FIXTURES.read_bytes()
    fixture = json.loads(raw)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        lab = nala.load(model_id=args.model, revision=args.revision,
                        device=args.device, local_only=args.local_only,
                        context_limit=args.context_limit)
        loaded = time.perf_counter()
        results = []
        for case in fixture['cases']:
            result = run_case(lab, case, args.layer, args.out/case['id'])
            results.append(result)
            print(f"{case['id']}: delete rho={result['ranking']['delete']['spearman']}; "
                  f"greedy gain over random={result['selection']['greedy_gain_over_random']:+.6f} nats/token", flush=True)
        first = fixture['cases'][0]
        token = lab.value(first['prompt'], heldout=first['heldout'], top_k=0,
                          layers=[args.layer], out=args.out/'token_coverage')
        token_checks = report_checks(token, token['prompt_tokens'], 0)
        token_checks['all_tokens_ranked'] = {tuple(r['indices']) for r in token['ranked']} == {
            (i,) for i in range(token['prompt_tokens'])}
        report = {**aggregate(results, token_checks), 'protocol': fixture['protocol'],
                  'fixture_sha256': hashlib.sha256(raw).hexdigest(), 'identity': lab.identity,
                  'layer': args.layer, 'context_limit': lab.context_limit,
                  'thresholds': THRESHOLDS, 'sign_noise_floor': NOISE_FLOOR,
                  'selection_budget': 2, 'spans_per_case': 4, 'cases': results,
                  'token_coverage_checks': token_checks,
                  'capture_count': len(results)*2+1,
                  'target_swap': {'same_prompt': results[0]['prompt_sha256'] == results[1]['prompt_sha256'],
                                  'different_targets': results[0]['heldout_sha256'] != results[1]['heldout_sha256'],
                                  'top_span_changed': results[0]['ranking']['delete']['predicted_top_span'] !=
                                                      results[1]['ranking']['delete']['predicted_top_span']},
                  'timing_seconds': {'model_load': loaded-started, 'evaluation': time.perf_counter()-loaded,
                                     'total': time.perf_counter()-started},
                  'scope': 'Fixed-layer pooled attention deletion; no text removal, training or generalization benchmark.',
                  'run_directory': args.out.name}
        nala.save_json(args.out/'summary.json', report)
        print(json.dumps({k: report[k] for k in ['passed', 'correctness_passed', 'quality_passed', 'quality', 'timing_seconds']}, indent=2))
        print('Full results:', args.out/'summary.json')
        return 0 if report['passed'] else 1
    except Exception as exc:
        nala.save_json(args.out/'error.json', {'passed': False, 'error': type(exc).__name__, 'message': str(exc)})
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
