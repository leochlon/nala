import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def notebook():
    return json.loads((ROOT/'notebooks/NaLa_7B_Colab.ipynb').read_text())


def cell_source(cell_id):
    return ''.join(next(c for c in notebook()['cells'] if c['id'] == cell_id)['source'])


def test_notebook_schema_code_and_embedded_source_are_current():
    nbformat = pytest.importorskip('nbformat')
    nb = notebook()
    nbformat.validate(nb)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
    literals = {}
    for node in ast.parse(cell_source('setup')).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {'PAYLOAD_BASE64', 'PAYLOAD_SHA256'}:
                literals[node.targets[0].id] = ast.literal_eval(node.value)
    payload = base64.b64decode(literals['PAYLOAD_BASE64'], validate=True)
    assert hashlib.sha256(payload).hexdigest() == literals['PAYLOAD_SHA256']
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        assert z.testzip() is None
        assert {'nala.py', 'eval_value.py', 'evals/value_sanity.json', 'requirements.txt'} <= set(z.namelist())
        for name in z.namelist():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            assert z.read(name) == (ROOT/name).read_bytes()


@pytest.mark.parametrize('available,free,total,passes', [
    (False, 0, 0, False),
    (True, 16, 16, False),
    (True, 32, 40, False),  # Sufficient total capacity, insufficient FREE memory.
    (True, 36, 40, True),
])
def test_notebook_checks_free_vram_before_download(monkeypatch, available, free, total, passes):
    fake = SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: available,
        mem_get_info=lambda: (free*1024**3, total*1024**3),
        get_device_name=lambda: 'test GPU',
    ))
    monkeypatch.setitem(sys.modules, 'torch', fake)
    monkeypatch.setattr(shutil, 'disk_usage', lambda p: SimpleNamespace(free=100*1024**3))
    context = {}
    exec(compile(cell_source('settings'), '<settings>', 'exec'), context)
    guard = compile(cell_source('hardware'), '<hardware>', 'exec')
    if passes:
        exec(guard, context)
        assert context['hardware']['fp32_weight_gib'] == pytest.approx(28.37038230895996)
    else:
        with pytest.raises(RuntimeError):
            exec(guard, context)


def test_notebook_retains_logs_and_continues_after_failed_quality_gate(tmp_path):
    commands = []

    def popen(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout=iter(['quality gate failed\n']), wait=lambda: 1)

    context = {'WORKDIR': tmp_path, 'MODEL_ID': 'Qwen/Qwen2.5-7B-Instruct',
               'MODEL_REVISION': 'a09a35458c702b33eeacc393d103063234e8bc28',
               'LAYER': 18, 'CONTEXT_LIMIT': 512, 'PAYLOAD_SHA256': 'a'*64,
               'hardware': {}, 'json': json, 'sys': sys,
               'subprocess': SimpleNamespace(Popen=popen, PIPE=-1, STDOUT=-2)}
    exec(compile(cell_source('evaluate'), '<evaluate>', 'exec'), context)
    assert context['evaluation_exit'] == 1
    assert (context['JOB_DIR']/'evaluation.log').read_text() == 'quality gate failed\n'
    command = commands[0]
    assert command[command.index('--model')+1] == context['MODEL_ID']
    assert command[command.index('--revision')+1] == context['MODEL_REVISION']
    assert command[command.index('--device')+1] == 'cuda'
