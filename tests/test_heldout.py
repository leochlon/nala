"""Independent mathematical and native FP32 regression checks for held-out value."""
import copy
import json
from dataclasses import replace
from contextlib import nullcontext

import numpy as np
import pytest

import nala


@pytest.fixture
def native():
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    if transformers.__version__ != nala.TRANSFORMERS_VERSION:
        pytest.skip('Native tests require transformers==' + nala.TRANSFORMERS_VERSION)
    from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers
    torch.set_num_threads(2)
    torch.manual_seed(73)
    words = ['[PAD]', '[EOS]', '[UNK]', 'user', 'assistant', 'system', ':', '.',
             'red', 'green', 'blue', 'sky', 'ocean', 'is', 'bright', 'dark',
             'Paris', 'a', 'city', 'The', 'and', 'context', 'first', 'second']
    raw = Tokenizer(models.WordLevel({s: i for i, s in enumerate(words)}, unk_token='[UNK]'))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]',
                                       eos_token='[EOS]', pad_token='[PAD]')
    tokenizer.chat_template = ("{% for message in messages %}{{ message['role'] + ': ' + message['content'] + ' ' }}"
                               "{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}")
    config = Qwen2Config(vocab_size=61, hidden_size=32, intermediate_size=48,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        max_position_embeddings=256, attention_dropout=0., eos_token_id=1)
    config._attn_implementation = 'eager'
    model = Qwen2ForCausalLM(config).float().eval()
    return model, tokenizer


def specification():
    return {'kind': nala.HELD_OUT_LOG_LIKELIHOOD, 'items': [
        {'token_ids': [11, 12, 13], 'weight': 1.7}, {'token_ids': [14, 15], 'weight': 0.4}]}


def full_prefix_oracle(model, ids, targets, candidate=None):
    """Independent dense native execution, without NaLa capture/scoring/editors.

    Prefix query rows are kept original. Only rows that predict held-out tokens
    receive the edit. This recomputes the entire prompt without a detached cache.
    """
    import torch
    from transformers.models.qwen2 import modeling_qwen2 as qwen
    def forward(module, hidden_states, position_embeddings, attention_mask, **kwargs):
        shape = (*hidden_states.shape[:-1], -1, module.head_dim)
        q = module.q_proj(hidden_states).view(shape).transpose(1, 2)
        k = module.k_proj(hidden_states).view(shape).transpose(1, 2)
        v = module.v_proj(hidden_states).view(shape).transpose(1, 2)
        q, k = qwen.apply_rotary_pos_emb(q, k, *position_embeddings)
        mask = attention_mask.clone()
        rows, edit = candidate['indices'], candidate['edit']
        mask[:, :, len(ids)-1:, rows] += (-float('inf') if edit['kind'] == 'delete'
                                        else np.log(edit['factor']))
        raw, probabilities = qwen.eager_attention_forward(module, q, k, v, mask,
            scaling=module.scaling, dropout=0., sliding_window=None)
        return module.o_proj(raw.reshape(*hidden_states.shape[:-1], -1)), probabilities
    manager = (nala.patched_forward(model.model.layers[candidate['layer']].self_attn, forward)
               if candidate is not None else nullcontext())
    with torch.no_grad(), manager:
        logits = model(input_ids=torch.tensor([ids+targets[:-1]]), use_cache=False).logits[0, len(ids)-1:]
        probabilities = torch.log_softmax(logits, dim=-1)
        return np.array([float(probabilities[t, target]) for t, target in enumerate(targets)])


def test_capture_is_one_backward_and_items_are_independent(native, monkeypatch):
    import torch
    model, _ = native
    ids, spec = [5, 6, 7, 8], specification()
    before = [(p.requires_grad, p.grad) for p in model.parameters()]
    calls, forwards = [], []
    original = torch.autograd.grad
    def tracked(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(torch.autograd, 'grad', tracked)
    hook = model.register_forward_pre_hook(lambda *_: forwards.append(1))
    capture = nala.capture_objective(model, ids, spec)
    hook.remove()
    assert len(calls) == 1
    assert len(forwards) == 3  # one shared prefix and two independent query forwards
    assert capture.counts['capture_query_rows'] == 5
    assert capture.counts['capture_backwards'] == 1
    for parameter, (requires_grad, grad) in zip(model.parameters(), before):
        assert parameter.requires_grad == requires_grad and parameter.grad is grad
    expected = 0.
    for i, item in enumerate(spec['items']):
        logs = full_prefix_oracle(model, ids, item['token_ids'])
        np.testing.assert_allclose(capture.item_results[i]['token_log_probabilities'], logs, atol=2e-6)
        expected += item['weight']*logs.sum()
        single = nala.capture_objective(model, ids, {'kind': spec['kind'], 'items': [item]})
        for bank, reference in zip([b for b in capture.banks if b.item == i], single.banks):
            np.testing.assert_allclose(bank.gradient, reference.gradient, atol=1e-7, rtol=1e-6)
            np.testing.assert_array_equal(bank.keys, reference.keys)
            assert bank.query.shape[1] == len(item['token_ids'])
            np.testing.assert_array_equal(bank.mask, np.arange(bank.n)[None, :] <=
                                          np.arange(len(ids)-1, bank.n)[:, None])
    assert capture.value == pytest.approx(expected, abs=3e-6)
    assert all(check['verified'] for check in capture.checks)


@pytest.mark.parametrize('layer', [0, 1])
@pytest.mark.parametrize('edit', [{'kind': 'delete'}, {'kind': 'reweight', 'factor': 2.},
                                  {'kind': 'reweight', 'factor': .5}, {'kind': 'reweight', 'factor': 1.}])
def test_execution_matches_full_prefix_oracle(native, layer, edit):
    model, _ = native
    ids, spec = [5, 6, 7, 8], specification()
    capture = nala.capture_objective(model, ids, spec, layers=[layer])
    gain = sum(nala.Scorer(b).local(edit, [0, 2])['prediction'] for b in capture.banks)
    candidate = {'layer': layer, 'indices': [0, 2], 'edit': edit, 'predicted_loss_change': -gain}
    executed = nala.execute_candidate(model, capture, candidate)
    expected = 0.
    for i, item in enumerate(spec['items']):
        logs = full_prefix_oracle(model, ids, item['token_ids'], candidate)
        np.testing.assert_allclose(executed['items'][i]['token_log_probabilities'], logs, atol=2e-6)
        expected -= item['weight']*logs.sum()
    assert executed['actual_loss'] == pytest.approx(expected, abs=3e-6)
    assert executed['actual_loss_change'] == pytest.approx(expected-capture.loss, abs=3e-6)
    assert executed['actual_log_likelihood_change'] == -executed['actual_loss_change']
    assert executed['local_verified'] and len(executed['audits']) == 2
    assert [a['query_count'] for a in executed['audits']] == [3, 2]
    if edit.get('factor') == 1:
        assert gain == 0 and executed['actual_loss_change'] == 0


def test_multiquery_gradient_matches_finite_difference(native):
    import torch
    model, _ = native
    ids, target = [5, 6, 7, 8], [11, 12, 13]
    capture = nala.capture_objective(model, ids,
        {'kind': nala.HELD_OUT_LOG_LIKELIHOOD, 'items': [{'token_ids': target}]}, layers=[0])
    epsilon = 0.01
    def likelihood(delta):
        def inject(module, args):
            value = args[0].clone()
            value[0, len(ids), 2*8+3] += delta  # query row 1, head 2, value coordinate 3
            return (value,)
        hook = model.model.layers[0].self_attn.o_proj.register_forward_pre_hook(inject)
        try:
            return full_prefix_oracle(model, ids, target).sum()
        finally:
            hook.remove()
    numerical = (likelihood(epsilon)-likelihood(-epsilon))/(2*epsilon)
    assert capture.banks[0].gradient[2, 1, 3] == pytest.approx(numerical, abs=8e-5, rel=.03)


def banks_for_pool():
    rng = np.random.default_rng(107)
    banks = []
    for layer in (0, 1):
        for item, qcount in enumerate((3, 2)):
            n = 4+qcount
            mask = np.arange(n)[None, :] <= np.arange(4, n)[:, None]
            banks.append(nala.Bank(layer, rng.normal(size=(4, qcount, 8)), rng.normal(size=(2, n, 8)),
                rng.normal(size=(2, n, 5)), rng.normal(size=(4, qcount, 5)),
                np.array([1., .1, .01, .001]), 8**-.5, mask=mask, item=item))
    return banks


def test_rank_sums_items_and_queries_but_keeps_layers_separate():
    banks = banks_for_pool()
    templates = [{'kind': 'delete'}, {'kind': 'reweight', 'factor': 2.}]
    ranked, arrays, count = nala.rank_value_banks(banks, 5, templates)
    assert count == 2*2*5 and len(ranked) == count
    for row in ranked:
        gain = sum(nala._dense_oracle(b, row['edit'], row['indices'])[0]
                   for b in banks if b.layer == row['layer'])
        assert row['predicted_loss_change'] == pytest.approx(-gain, abs=1e-12)
        assert row['predicted_log_likelihood_change'] == pytest.approx(gain, abs=1e-12)
    assert all(a['ranking_score'] >= b['ranking_score'] for a, b in zip(ranked, ranked[1:]))
    improvement, _, _ = nala.rank_value_banks(banks, 5, templates, rank_by='improvement')
    assert all(a['predicted_loss_change'] <= b['predicted_loss_change'] for a, b in zip(improvement, improvement[1:]))


@pytest.mark.parametrize('budget', [1, 2, 3])
def test_greedy_rescores_against_dense_cumulative_deletion(budget):
    banks, pool = banks_for_pool(), [[0], [1], [2, 3]]
    result = nala.greedy_value_selection(banks, 5, pool, budget)
    layer_banks = [b for b in banks if b.layer == result['layer']]
    def gain(deleted):
        if not deleted:
            return 0.
        return sum(nala._dense_oracle(b, {'kind': 'delete'}, sorted(deleted))[0] for b in layer_banks)
    deleted = {0, 1, 2, 3}
    before = gain(deleted)
    for step in result['steps']:
        for candidate in step['rescored_candidates']:
            expected = gain(deleted-set(pool[candidate['span_index']]))-before
            assert candidate['predicted_log_likelihood_change'] == pytest.approx(expected, abs=3e-12)
        best = step['rescored_candidates'][0]
        assert step['retained_span_index'] == best['span_index']
        deleted -= set(pool[best['span_index']])
        before = gain(deleted)
        assert step['predicted_loss_change'] == pytest.approx(-before, abs=3e-12)
    assert result['predicted_loss_change'] == pytest.approx(-gain(deleted), abs=3e-12)
    assert result['capture_backwards'] == result['model_calls_for_rescoring'] == 0
    assert not result['gradient_refreshed']
    assert result['protected_indices'] == [4]


def test_causal_masks_zero_invisible_edits_and_refuse_empty_rows():
    bank = banks_for_pool()[0]
    early_only = replace(bank, gradient=bank.gradient.copy())
    early_only.gradient[:, 1:] = 0
    scorer = nala.Scorer(early_only)
    assert scorer.local({'kind': 'delete'}, [6])['prediction'] == 0
    with pytest.raises(nala.UserError, match='every attended key'):
        scorer.local({'kind': 'delete'}, [0, 1, 2, 3, 4])
    with pytest.raises(nala.VerificationError, match='nonempty attended bank'):
        replace(bank, mask=np.zeros_like(bank.mask))


def test_portable_capture_roundtrip_and_legacy_margin(native, tmp_path):
    model, _ = native
    capture = nala.capture_objective(model, [5, 6, 7], specification())
    path = tmp_path/'capture.npz'
    capture.save(path)
    with np.load(path, allow_pickle=False) as data:
        assert json.loads(str(data['objective']))['kind'] == nala.HELD_OUT_LOG_LIKELIHOOD
    for old, new in zip(capture.banks, nala.load_banks(path)):
        assert old.layer == new.layer and old.item == new.item
        for name in ('query', 'keys', 'values', 'gradient', 'mask'):
            np.testing.assert_array_equal(getattr(old, name), getattr(new, name))
    margin = nala.capture_margin(model, [5, 6, 7], 11, 12)
    margin.save(tmp_path/'margin.npz')
    with np.load(tmp_path/'margin.npz', allow_pickle=False) as data:
        assert 'layers' in data and float(data['margin']) == margin.margin
    assert all(b.query.ndim == 2 for b in margin.banks)
    ranked, _, _ = nala.rank_banks(margin.banks, templates=[{'kind': 'reweight', 'factor': 1.}], limit=1)
    assert nala.execute_candidate(model, margin, ranked[0])['actual_margin_change'] == 0
    # Old format still loads, independently of the version-two writer.
    bank = margin.banks[0]
    old = {'layers': np.array([bank.layer])}
    old.update({f'layer_{bank.layer}_{k}': getattr(bank, k)
                for k in ('query', 'keys', 'values', 'gradient', 'frequencies', 'scale')})
    np.savez(tmp_path/'legacy.npz', **old)
    assert nala.load_banks(tmp_path/'legacy.npz')[0].query.ndim == 2


def test_single_token_prompt_and_continuation(native):
    model, _ = native
    capture = nala.capture_objective(model, [5],
        {'kind': nala.HELD_OUT_LOG_LIKELIHOOD, 'items': [{'token_ids': [11]}]})
    assert capture.counts['capture_prefix_forwards'] == 0
    assert capture.banks[0].query_count == 1
    assert nala.measure_heldout(model, capture)['loss'] == capture.loss


@pytest.mark.parametrize('items', [[], '', [''], [{'text': 'ok', 'weight': 0}],
    [{'text': 'ok', 'weight': float('nan')}], [{'text': 'ok', 'weight': True}],
    [{'text': 'ok', 'unknown': 3}], {'items': ['ok'], 'extra': True}])
def test_invalid_heldout_items(items):
    with pytest.raises(nala.UserError):
        nala.heldout_items(items)


def test_cancellation_restores_model_state(native):
    import threading
    model, _ = native
    parameters = [(p, p.requires_grad, p.grad) for p in model.parameters()]
    event = threading.Event()
    event.set()
    with pytest.raises(nala.UserError, match='cancelled'):
        nala.capture_objective(model, [5, 6, 7], specification(), cancel=event)
    assert all(p.requires_grad == r and p.grad is g for p, r, g in parameters)
    assert all('forward' not in layer.self_attn.__dict__ for layer in model.model.layers)


@pytest.fixture
def lab(native):
    model, tokenizer = native
    return nala.NaLa(model, tokenizer, model_id='tiny-qwen-test', revision='0'*40, context_limit=128)


def test_value_report_and_cli_roundtrip(lab, tmp_path, monkeypatch, capsys):
    heldout = tmp_path/'items.json'
    heldout.write_text(json.dumps([' sky is bright', {'text': ' ocean is dark', 'weight': .5}]))
    spans = tmp_path/'spans.json'
    spans.write_text(json.dumps(['red', 'blue', 'green']))
    monkeypatch.setattr(nala, 'load', lambda *a, **k: lab)
    destination = tmp_path/'report'
    code = nala.main(['value', '--prompt', 'red blue green context', '--heldout', str(heldout),
        '--spans', str(spans), '--top-k', '2', '--select', '2', '--factors', '.5', '2',
        '--out', str(destination), '--json'])
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report['objective']['kind'] == nala.HELD_OUT_LOG_LIKELIHOOD
    assert report['candidate_count'] == 2*3*3
    assert len(report['ranked']) == report['candidate_count']
    assert report['counts']['capture_backwards'] == 1
    assert report['counts']['capture_query_rows'] == 6
    assert report['counts']['candidate_verification_forwards'] == 4
    assert report['counts']['model_calls_for_analytic_scoring'] == 0
    assert len(report['verification']) == 2 and report['selection']['verification']['local_verified']
    assert all('actual_loss_change' in r for r in report['ranked'][:2])
    assert all(r['verification'] is None for r in report['ranked'][2:])
    assert report['preflight']['passed'] and report['noop']['verified']
    assert report == json.loads((destination/'report.json').read_text())
    assert len(nala.load_banks(destination/'capture.npz')) == 4


def test_zero_top_k_still_scores_every_prompt_token(lab):
    report = lab.value('red blue green', heldout=[' sky'], top_k=0)
    assert report['status'] == 'scored_only' and report['verification'] == []
    assert report['candidate_count'] == report['prompt_tokens']*2
    assert {r['indices'][0] for r in report['ranked']} == set(range(report['prompt_tokens']))
    assert report['counts']['candidate_verification_forwards'] == 0
    assert report['baseline'] is None and report['noop'] is None


def test_token_selection_preserves_anchor_and_executes_final_subset(lab):
    report = lab.value('red blue green', heldout=[' sky is bright', ' ocean'], top_k=1, select=2, layers=[0])
    selection = report['selection']
    assert selection['protected_indices'] == [report['prompt_tokens']-1]
    assert len(selection['retained_indices']) == 2
    assert selection['verification']['local_verified']
    assert report['counts']['capture_backwards'] == 1
    assert report['counts']['model_calls_for_greedy_rescoring'] == 0


def test_spans_boundaries_and_selection_input_errors(lab):
    with pytest.raises(nala.UserError, match='occurs 2 times'):
        lab.value('red red', heldout=[' sky'], spans=['red'], top_k=0)
    with pytest.raises(nala.UserError, match='cuts a token'):
        lab.value('green', heldout=[' sky'], spans=['reen'], top_k=0)
    with pytest.raises(nala.UserError, match='non-overlapping'):
        lab.value('red blue green', heldout=[' sky'], spans=['red blue', 'blue green'], select=1)
    with pytest.raises(nala.UserError, match='configured limit'):
        lab.value('red '*126, heldout=[' sky'])


def test_cli_rejects_invalid_json_before_loading(tmp_path, monkeypatch, capsys):
    path = tmp_path/'invalid.json'
    path.write_text('[{"text":"sky", "weight": NaN}]')
    def forbidden(*a, **k):
        raise AssertionError('Invalid input reached model loading')
    monkeypatch.setattr(nala, 'load', forbidden)
    assert nala.main(['value', '--prompt', 'red', '--heldout', str(path)]) == 2
    assert 'Nonfinite JSON value' in capsys.readouterr().err


def test_mcp_schema_and_live_value(lab, monkeypatch, capsys):
    import asyncio
    pytest.importorskip('mcp')
    import nala_mcp as server
    monkeypatch.setattr(server, '_engine', lab)
    tools = asyncio.run(server.mcp.list_tools())
    tool = next(t for t in tools if t.name == 'nala_value')
    assert tool.inputSchema['required'] == ['prompt', 'heldout']
    assert 'prefer' not in tool.inputSchema['properties']
    report = server.nala_value('red blue green', [' sky is bright'], spans=['red', 'blue'], top_k=1, select=1)
    assert report['status'] == 'verified'
    assert report['counts']['capture_backwards'] == 1
    assert report['ranked'][0]['verification']['local_verified']
    assert capsys.readouterr().out == ''
    assert 'nala_value' in server.nala_capabilities()['tools']
