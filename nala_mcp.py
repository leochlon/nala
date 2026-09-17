#!/usr/bin/env python3
"""NaLa MCP server. Keep this file next to nala.py. Python >=3.10.

Install server dependencies:
    python -m pip install 'numpy>=1.26,<3' 'mcp>=1.28,<2'
Install live-model dependencies when needed:
    python -m pip install 'torch>=2.6' 'transformers==4.57.3' 'jinja2>=3.1'
Start locally over stdio:
    python nala_mcp.py

Four tools: capabilities, recorded demo, a controlled trial, and held-out value.
The model loads on the first live trial. Requests are serialized; logs go to
stderr so stdout stays reserved for MCP. No arbitrary shell/check commands are
accepted. Python callers can supply their own behavioral checker to nala.py.

Official SDK v1 documentation: https://py.sdk.modelcontextprotocol.io/v1/
The <2 bound is intentional: this server uses the SDK's v1 FastMCP API.

Original compact wrapper; distributed under the MIT license embedded in nala.py.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import threading
from contextlib import redirect_stdout
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    raise SystemExit("Install the MCP dependency: python -m pip install 'mcp>=1.28,<2'") from exc

import nala

mcp = FastMCP('NaLa', instructions=(
    'Test whether attention interventions help a supplied known model failure. '
    'Runs on its own local FP32 Qwen2/Qwen2.5 checkpoint. '
    'For diagnose, request preferred and competing continuations; supply expected when an exact answer is known. '
    'For value, use caller-supplied held-out continuations without prefer/avoid labels. '
    'Distinguish recorded evidence, predicted changes, measured likelihoods or margins, and task-check passes. '
    'Never describe a failed bounded search as proof that attention cannot help.'))

_engine = None
_lock = threading.Lock()


@mcp.tool()
def nala_capabilities() -> dict:
    """Explain scope, setup, and current readiness without loading model weights."""
    return {
        'value': 'Rank attention edits, execute the top few, and compare checked output against controls.',
        'tools': ['nala_capabilities', 'nala_demo', 'nala_diagnose', 'nala_value'],
        'model': os.environ.get('NALA_MODEL', nala.MODEL_ID),
        'revision': os.environ.get('NALA_REVISION') or 'resolved to an immutable commit at load',
        'device': os.environ.get('NALA_DEVICE', 'auto'),
        'model_loaded': _engine is not None,
        'model_dependencies_present': all(importlib.util.find_spec(x) for x in ('torch', 'transformers')),
        'supported': 'Local unquantized FP32 Qwen2/Qwen2.5; eager attention; fixed unscaled RoPE.',
        'edits': 'Boost, suppress, or mask token/span attention at one layer; value edits all held-out query rows.',
        'required_input': 'diagnose: prompt/prefer/avoid; value: prompt and held-out continuations.',
        'correctness_check': 'Optional stripped exact match against caller-supplied expected text.',
        'limits': 'The target is this server\'s local checkpoint. Hosted-model attention is inaccessible. '
                  'Objectives are next-token margin or summed held-out log-likelihood. Local response is exact; '
                  'downstream prediction is approximate. Value measures in-context contribution; weights are unchanged.',
        'live_install': "python -m pip install 'torch>=2.6' 'transformers==4.57.3' 'jinja2>=3.1'"}


@mcp.tool()
def nala_demo() -> dict:
    """Show the recorded 0/6 → 6/6 API example and recompute its six-case behavior.

    Instant, with no model download or fresh inference. The recorded ranking
    and executed margins come from the original supplied case, not this server.
    """
    report = nala.demo()
    return {key: report[key] for key in (
        'mode', 'fresh_model_inference', 'scope', 'recorded_candidate_count',
        'selected_edit', 'predicted_margin_change', 'actual_margin_change',
        'replayed_predicted_margin_change', 'branches')}


@mcp.tool()
def nala_diagnose(prompt: str, prefer: str, avoid: str, expected: str | None = None,
                  span: str | None = None, occurrence: int | None = None,
                  top_k: int = 3, max_tokens: int = 32, layers: list[int] | None = None) -> dict:
    """Run a live controlled trial on the server's local model.

    prefer and avoid are supplied continuations; shared token prefixes are
    explicitly conditioned on. expected, when supplied, uses stripped exact
    match of the complete response. span restricts edits to an exact quote in
    the prompt; repeated quotes need a 1-based occurrence. top_k is 1–5.
    First call can download model weights. Each trial has independent caches,
    baseline/no-op controls, native local verification, and actual continuations.
    A check_passed status only establishes the configured check, not general repair.
    """
    if not prompt.strip() or not prefer or not avoid or not 1 <= top_k <= 5 or not 1 <= max_tokens <= 512:
        raise ValueError('Supply nonempty prompt/prefer/avoid, top_k 1–5, and max_tokens 1–512.')
    global _engine
    with _lock, redirect_stdout(sys.stderr):
        if _engine is None:
            _engine = nala.load(
                model_id=os.environ.get('NALA_MODEL', nala.MODEL_ID),
                revision=os.environ.get('NALA_REVISION') or None,
                device=os.environ.get('NALA_DEVICE', 'auto'),
                local_only=os.environ.get('NALA_LOCAL_ONLY', '0') == '1',
                context_limit=int(os.environ.get('NALA_CONTEXT_LIMIT', '4096')))
        return _engine.diagnose(prompt, prefer=prefer, avoid=avoid, expected=expected,
            span=span, occurrence=occurrence, top_k=top_k, max_tokens=max_tokens, layers=layers)


@mcp.tool()
def nala_value(prompt: str, heldout: list[str | dict[str, Any]],
               spans: list[str | dict[str, Any]] | None = None, top_k: int = 3,
               select: int | None = None, layers: list[int] | None = None,
               factors: list[float] | None = None, rank_by: str = 'contribution') -> dict:
    """Rank every prompt token/span by its contribution to held-out likelihood.

    heldout contains exact continuation strings or {text, weight?} objects.
    Each item is independently teacher-forced after the prompt; likelihood is
    summed over tokens and items without length normalization or implicit EOS.
    spans optionally contains quotes, {quote, occurrence?}, or {indices: [...]}.
    Deletion is the default family; factors adds reweight candidates.

    One capture and backward score all candidates. top_k (0–200) selects edits
    for actual teacher-forced measurement. Positive loss change means worse
    prediction. contribution ranks useful spans first; improvement ranks edits
    that improve likelihood first. select greedily retains K disjoint spans,
    re-scoring after each retention with the same captured gradient and verifying
    the final subset. Insertion is implemented by deleting unretained pool rows.
    Returns the full ranking, likelihood/loss changes, capture and edit audits.
    """
    if not prompt.strip() or type(top_k) is not int or not 0 <= top_k <= 200:
        raise ValueError('Supply nonempty prompt text and top_k 0–200.')
    items = nala.heldout_items(heldout)
    if select is not None and (type(select) is not int or select < 1):
        raise ValueError('select must be a positive integer.')
    if rank_by not in {'contribution', 'improvement'}:
        raise ValueError('rank_by must be contribution or improvement.')
    global _engine
    with _lock, redirect_stdout(sys.stderr):
        if _engine is None:
            _engine = nala.load(
                model_id=os.environ.get('NALA_MODEL', nala.MODEL_ID),
                revision=os.environ.get('NALA_REVISION') or None,
                device=os.environ.get('NALA_DEVICE', 'auto'),
                local_only=os.environ.get('NALA_LOCAL_ONLY', '0') == '1',
                context_limit=int(os.environ.get('NALA_CONTEXT_LIMIT', '4096')))
        return _engine.value(prompt, heldout=items, spans=spans, top_k=top_k, select=select,
                             layers=layers, factors=factors, rank_by=rank_by)


if __name__ == '__main__':
    mcp.run(transport='stdio')
