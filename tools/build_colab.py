#!/usr/bin/env python3
"""Build the self-contained Colab notebook from the current tested source."""
import base64
import hashlib
import io
import json
from pathlib import Path
import textwrap
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_FILES = [
    'nala.py', 'nala_cases.json', 'eval_value.py', 'requirements.txt',
    'evals/value_sanity.json', 'examples/prompt.txt', 'examples/items.json',
    'examples/spans.json', 'verification/value_sanity.json',
]


def markdown(cell_id, source):
    return {'cell_type': 'markdown', 'id': cell_id, 'metadata': {'id': cell_id},
            'source': textwrap.dedent(source).strip().splitlines(keepends=True)}


def code(cell_id, source):
    return {'cell_type': 'code', 'id': cell_id, 'execution_count': None, 'outputs': [],
            'metadata': {'id': cell_id, 'cellView': 'form'},
            'source': (textwrap.dedent(source).strip()+'\n').splitlines(keepends=True)}


def build_notebook():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in BUNDLE_FILES:
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (ROOT/name).read_bytes())
    payload = buffer.getvalue()
    encoded = base64.b64encode(payload).decode('ascii')
    digest = hashlib.sha256(payload).hexdigest()
    cells = [
        markdown('intro', '''
            # NaLa held-out likelihood on Qwen2.5-7B

            Upload this notebook to Colab, select **Runtime → Change runtime type → GPU → A100**,
            then **Run all**. Use a GPU with at least 40 GB; the notebook checks actual free memory.
            The source, fixed evaluation cases, and recorded 0.5B comparison are included here.
            No separate ZIP upload or repository checkout is needed.

            This preserves **FP32 eager attention**, one layer per edit, and the original numerical
            audits. The 7B checkpoint has 7.62 billion parameters: FP32 weights need 28.37 GiB,
            plus working memory. A 16 GB T4 or 24 GB L4 cannot hold these FP32 weights.

            The notebook runs five cases, scores deletion/reweight edits, and compares two-span
            retention with static selection, uniform random and all six possible pairs. GPU/7B
            results are produced by your run; the included 0.5B CPU result is only a comparison.
            The first run downloads about 15.2 GB of checkpoint files. The model is public.
        '''),
        code('settings', '''
            #@title 1. Model and evaluation settings
            MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
            MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
            LAYER = 18 #@param {type:"integer"}
            CONTEXT_LIMIT = 512
            PARAMETER_COUNT = 7_615_616_512
            CHECKPOINT_BYTES = 15_231_233_024
            WORKING_HEADROOM_GIB = 6
            print(f"{MODEL_ID}, layer {LAYER}, FP32; revision {MODEL_REVISION}")
        '''),
        code('hardware', '''
            #@title 2. Check GPU and disk before downloading weights
            import shutil
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("Select a GPU runtime with an A100 (40 GB or larger), then reconnect.")
            if not 0 <= LAYER < 28:
                raise ValueError("This 7B model has layers 0 through 27.")
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            weight_gib = PARAMETER_COUNT * 4 / 1024**3
            required_gib = weight_gib + WORKING_HEADROOM_GIB
            gpu_name = torch.cuda.get_device_name()
            print(f"GPU: {gpu_name}; free {free_bytes / 1024**3:.2f} GiB of {total_bytes / 1024**3:.2f} GiB")
            print(f"FP32 weights: {weight_gib:.2f} GiB; short-evaluation memory allowance: {required_gib:.2f} GiB")
            if free_bytes < required_gib * 1024**3:
                raise RuntimeError(f"Need at least {required_gib:.2f} GiB free VRAM for this FP32 run. "
                                   "Select an A100 with 40 GB or more, or release other GPU models first.")
            if shutil.disk_usage('/content').free < CHECKPOINT_BYTES + 5 * 1024**3:
                raise RuntimeError("Need about 20 GiB free disk for the checkpoint and results.")
            hardware = {'gpu': gpu_name, 'free_vram_gib': free_bytes / 1024**3,
                        'total_vram_gib': total_bytes / 1024**3, 'fp32_weight_gib': weight_gib,
                        'working_headroom_gib': WORKING_HEADROOM_GIB}
        '''),
        code('setup', '''
            #@title 3. Prepare the bundled source and pinned dependencies
            import base64
            import hashlib
            import io
            import json
            import os
            from pathlib import Path
            import subprocess
            import sys
            import zipfile

            PAYLOAD_BASE64 = "__PAYLOAD__"
            PAYLOAD_SHA256 = "__DIGEST__"
            payload = base64.b64decode(PAYLOAD_BASE64, validate=True)
            if hashlib.sha256(payload).hexdigest() != PAYLOAD_SHA256:
                raise RuntimeError("Notebook source bundle failed its checksum.")
            WORKDIR = Path('/content') / ('nala-7b-' + PAYLOAD_SHA256[:12])
            WORKDIR.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
                for entry in bundle.infolist():
                    path = Path(entry.filename)
                    if path.is_absolute() or '..' in path.parts:
                        raise RuntimeError("Unexpected bundle path.")
                bundle.extractall(WORKDIR)
            os.chdir(WORKDIR)
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)
            subprocess.run([sys.executable, 'nala.py', 'self-test'], check=True)
            print('Ready:', WORKDIR)
        '''.replace('__PAYLOAD__', encoded).replace('__DIGEST__', digest)),
        code('evaluate', '''
            #@title 4. Run the same five-case evaluation on 7B
            from datetime import datetime, timezone

            run_name = 'qwen7b-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
            JOB_DIR = WORKDIR / 'runs' / run_name
            JOB_DIR.mkdir(parents=True, exist_ok=False)
            RUN_DIR = JOB_DIR / 'evaluation'
            metadata = {'model': MODEL_ID, 'revision': MODEL_REVISION, 'layer': LAYER,
                        'context_limit': CONTEXT_LIMIT, 'bundle_sha256': PAYLOAD_SHA256,
                        'hardware_before_load': hardware}
            (JOB_DIR / 'colab.json').write_text(json.dumps(metadata, indent=2))

            def run_logged(command, log_path):
                with log_path.open('w', encoding='utf-8') as log:
                    process = subprocess.Popen(command, cwd=WORKDIR, stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT, text=True, bufsize=1)
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                        log.flush()
                    return process.wait()

            evaluation_exit = run_logged([
                sys.executable, '-u', 'eval_value.py', '--model', MODEL_ID,
                '--revision', MODEL_REVISION, '--device', 'cuda', '--layer', str(LAYER),
                '--context-limit', str(CONTEXT_LIMIT), '--out', str(RUN_DIR),
            ], JOB_DIR / 'evaluation.log')
            print(f"Evaluation exit code: {evaluation_exit} (0=passed, 1=gate failed, 2=execution error)")
            print('Reports and logs are retained even when a check fails.')
        '''),
        code('results', '''
            #@title 5. Inspect results and the recorded 0.5B comparison
            from IPython.display import Markdown, display

            summary_file = RUN_DIR / 'summary.json'
            if summary_file.exists():
                result = json.loads(summary_file.read_text())
                reference = json.loads((WORKDIR / 'verification/value_sanity.json').read_text())
                print('Actual model identity:', json.dumps(result['identity'], indent=2))
                print('Correctness passed:', result['correctness_passed'])
                print('Quality gates:', result['quality_gates'])
                print('Timing (includes checkpoint loading/download):', result['timing_seconds'])
                comparable = (result['fixture_sha256'] == reference['fixture_sha256']
                              and result['layer'] == reference['layer'])
                def number(value):
                    return 'undefined' if value is None else f'{value:.6g}'
                fields = [
                    ('Mean ranking correlation', 'mean_spearman'),
                    ('Sign agreement', 'sign_agreement'),
                    ('Material edits', 'material_edits'),
                    ('Prediction MAE (nats/token)', 'mean_mae_nats_per_token'),
                    ('Greedy gain over random (nats/token)', 'mean_greedy_gain_over_random'),
                    ('Greedy gain over static (nats/token)', 'mean_greedy_gain_over_static'),
                    ('Greedy oracle regret (nats/token)', 'mean_greedy_regret_nats_per_token'),
                    ('Semantic top-span support rate', 'support_top1_rate'),
                ]
                rows = ['| Metric | Current 7B run | Recorded 0.5B CPU |', '|---|---:|---:|']
                for label, field in fields:
                    baseline = number(reference['quality'][field]) if comparable else 'different fixture/layer'
                    rows.append(f"| {label} | {number(result['quality'][field])} | {baseline} |")
                display(Markdown('\\n'.join(rows)))
                for case in result['cases']:
                    print(case['id'], 'greedy:', case['selection']['greedy_retained'],
                          'oracle:', case['selection']['oracle_retained'],
                          'all native checks:', all(case['checks'].values()))
            else:
                error_file = RUN_DIR / 'error.json'
                print(error_file.read_text() if error_file.exists() else 'No summary was produced; inspect evaluation.log.')
                print('Download the logs below to inspect the failed run.')
        '''),
        markdown('custom-intro', '''
            ## Optional: try your own prompt

            Enable the next cell and edit the prompt, held-out continuations and quoted spans.
            It loads the same cached 7B checkpoint in a fresh process. Weights stay FP32;
            deletion masks attention at the chosen layer and preserves the original token positions.
        '''),
        code('custom', '''
            #@title 6. Optional custom example (disabled by default)
            RUN_CUSTOM = False #@param {type:"boolean"}
            if RUN_CUSTOM:
                prompt = (WORKDIR / 'examples/prompt.txt').read_text()
                heldout = json.loads((WORKDIR / 'examples/items.json').read_text())
                spans = json.loads((WORKDIR / 'examples/spans.json').read_text())
                # Replace the three values above with your own text/lists.
                (JOB_DIR / 'custom-prompt.txt').write_text(prompt)
                (JOB_DIR / 'custom-items.json').write_text(json.dumps(heldout))
                (JOB_DIR / 'custom-spans.json').write_text(json.dumps(spans))
                custom_exit = run_logged([
                    sys.executable, '-u', 'nala.py', 'value', '--model', MODEL_ID,
                    '--revision', MODEL_REVISION, '--device', 'cuda', '--layers', str(LAYER),
                    '--context-limit', str(CONTEXT_LIMIT), '--top-k', '3', '--select', '2',
                    '--prompt-file', str(JOB_DIR / 'custom-prompt.txt'),
                    '--heldout', str(JOB_DIR / 'custom-items.json'),
                    '--spans', str(JOB_DIR / 'custom-spans.json'), '--out', str(JOB_DIR / 'custom'),
                ], JOB_DIR / 'custom.log')
                print('Custom run exit code:', custom_exit)
            else:
                print('Custom example skipped; the five-case evaluation above is complete.')
        '''),
        code('download', '''
            #@title 7. Download results, captures and logs
            from google.colab import files

            results_zip = shutil.make_archive(str(WORKDIR / (run_name + '-results')), 'zip', root_dir=JOB_DIR)
            print('Downloading:', results_zip)
            files.download(results_zip)
        '''),
        markdown('scope', '''
            The evaluation tests the same finite attention interventions as the feature.
            Local responses are exact; downstream predictions use a frozen gradient. The recorded
            0.5B run tied greedy and static selection and matched semantic support in 3/5 cases.
            Inspect those diagnostics at 7B too. A passing five-case check is not evidence of
            training progress or broad retrieval quality.

            This notebook was prepared and validated locally for syntax, bundled-source integrity,
            loader routing and CPU regressions. 7B GPU execution has not been validated;
            the actual 7B GPU measurements come from your Colab run.
        '''),
    ]
    return {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'colab': {'name': 'NaLa_7B_Colab.ipynb', 'provenance': []},
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
        'language_info': {'name': 'python'}, 'accelerator': 'GPU',
    }, 'cells': cells}


def main():
    target = ROOT / 'notebooks' / 'NaLa_7B_Colab.ipynb'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build_notebook(), indent=1, ensure_ascii=False)+'\n')
    print(target)


if __name__ == '__main__':
    main()
