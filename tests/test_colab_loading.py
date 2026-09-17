from types import SimpleNamespace

import pytest

import eval_value
import nala


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_loader_preserves_fp32_and_uses_one_device_without_cpu_staging(monkeypatch, device):
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    monkeypatch.setattr(nala, 'model_dependencies', lambda: (torch, transformers))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(transformers.utils, 'is_accelerate_available', lambda: True)
    revision = 'a' * 40
    calls = []

    class Model:
        def to(self, destination):
            calls.append(('to', destination))
            return self

        def eval(self):
            calls.append(('eval',))
            return self

    model = Model()
    monkeypatch.setattr(transformers.AutoConfig, 'from_pretrained', lambda *a, **kw:
                        SimpleNamespace(model_type='qwen2', _commit_hash=revision))
    tokenizer = object()
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **kw: tokenizer)

    def pretrained(model_id, **kwargs):
        calls.append(('load', model_id, kwargs))
        return model

    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', pretrained)
    monkeypatch.setattr(nala, 'NaLa', lambda m, t, **kw: (m, t, kw))
    loaded, _, identity = nala.load('Qwen/Qwen2.5-7B-Instruct', revision,
                                   device=device, context_limit=512)
    assert loaded is model and identity['revision'] == revision
    options = calls[0][2]
    assert options['torch_dtype'] is torch.float32
    assert options['attn_implementation'] == 'eager'
    assert options['trust_remote_code'] is False
    if device == 'cuda':
        assert options['device_map'] == {'': 'cuda'}
        assert not any(c[0] == 'to' for c in calls)
    else:
        assert 'device_map' not in options
        assert ('to', 'cpu') in calls
    assert calls[-1] == ('eval',)


def test_evaluator_forwards_model_revision_and_context(monkeypatch, tmp_path):
    received = {}

    def stop_after_load(**kwargs):
        received.update(kwargs)
        raise RuntimeError('test stops before allocating model weights')

    monkeypatch.setattr(nala, 'load', stop_after_load)
    revision = 'b' * 40
    status = eval_value.main(['--model', 'Qwen/Qwen2.5-7B-Instruct', '--revision', revision,
                              '--device', 'cuda', '--context-limit', '512',
                              '--out', str(tmp_path/'eval')])
    assert status == 2
    assert received == {'model_id': 'Qwen/Qwen2.5-7B-Instruct', 'revision': revision,
                        'device': 'cuda', 'local_only': False, 'context_limit': 512}
    assert (tmp_path/'eval'/'error.json').exists()
