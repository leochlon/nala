import itertools

import pytest

from eval_value import aggregate, average_ranks, ranking_metrics, subset_metrics


def test_rank_ties_and_noise_do_not_create_fake_success():
    assert average_ranks([4, 1, 4, 2]).tolist() == [2.5, 0., 2.5, 1.]
    rows = [{'span_index': i, 'predicted_loss_change': p, 'actual_loss_change': a}
            for i, (p, a) in enumerate([(0., 0.), (0., 1e-7), (0., -.1)])]
    metrics = ranking_metrics(rows, 1.)
    assert metrics['spearman'] is None
    assert metrics['material_edits'] == 1
    assert metrics['sign_correct'] == 0
    assert metrics['top1_regret_nats_per_token'] == pytest.approx(1e-7)


def test_subset_baselines_use_executed_joint_losses():
    # Static singleton ordering picked (0, 1), yet their JOINT outcome is worst.
    pairs = list(itertools.combinations(range(4), 2))
    rows = [{'retained': list(pair), 'nll_per_token': loss}
            for pair, loss in zip(pairs, [10., 2., 4., 6., 8., 0.])]
    result = subset_metrics(rows, greedy=[2, 0], static=[1, 0], measured_greedy_loss=2.)
    assert result['oracle_retained'] == [2, 3]
    assert result['random_mean_nll_per_token'] == 5.
    assert result['greedy_gain_over_random'] == 3.
    assert result['greedy_gain_over_static'] == 8.
    assert result['greedy_regret_nats_per_token'] == 2.
    assert result['independent_greedy_replay_passed']
    assert not subset_metrics(rows, [0, 2], [0, 1], 1.)['independent_greedy_replay_passed']


def test_quality_success_cannot_hide_failed_execution_audit():
    metric = {'spearman': 1., 'material_edits': 12, 'sign_correct': 12,
              'mae_nats_per_token': 0., 'top1_regret_nats_per_token': 0.}
    case = {'checks': {'native_execution': False}, 'ranking': {'delete': metric},
            'selection': {'greedy_gain_over_random': 1., 'greedy_gain_over_static': 0.,
                          'greedy_regret_nats_per_token': 0.}, 'support_top1': True}
    report = aggregate([case], {'all_tokens_ranked': True})
    assert report['quality_passed']
    assert not report['correctness_passed'] and not report['passed']
