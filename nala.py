"""NaLa — specify attention relationships, rank finite edits, verify repairs.

One readable Python core, one Colab notebook, one MCP server. Python >=3.10.
Offline: pip install 'numpy>=1.26,<3'
Live: pip install 'torch>=2.6' 'transformers==4.57.3' 'jinja2>=3.1'

Start: python nala.py demo --recorded
Construct a query edit: python nala.py solve --demo
Inspect token rows: python nala.py tokens --prompt '...'
Execute a requested ratio: python nala.py run --prompt '...' --attend Rome --over Paris --ratio 4
Search: python nala.py diagnose --prompt '...' --prefer A --avoid B --try-repair
Value: python nala.py value --prompt '...' --heldout items.json --top-k 3
Live fixed-check case: python nala.py repair-case --out runs/discovery
Fresh model confirmation: python nala.py repair-case --confirm runs/discovery --out runs/confirmation
Offline regression checks: python nala.py self-test

Included:
- Snapshot, solve_edit, effective_step, probe, rotate_rows and run_request;
  affine log-odds constraints, minimum-norm solutions and numerical certificates.
- Seven edit kinds: reweight, delete, position, value_scale, key, value, joint.
- Native query intervention, derivative ranking, isolated cached generations.
- repair(), run_study(), confirm_case(), assess(), immutable-check/source-manifest
  gates and evidence hashing. Callable output checks are a separate weaker API.
- Exact greedy restore_cache against supplied reference K/V and a fixed query.

Run as python nala.py COMMAND or python -m nala COMMAND; there is no installed CLI.
Recorded evidence is ordinary JSON in nala_cases.json.

Scope: local unquantized FP32 Qwen2/Qwen2.5, eager full causal attention,
fixed unscaled RoPE, one sequence. Local identities are numerically verified;
downstream rankings are estimates. No claim of automatic diagnosis, universal
repair, benchmark gains, or access to a hosted assistant's internal attention.
Workspace copies isolate content, not operating-system privileges; fixed checks
execute trusted repository code. Precision restoration needs a known reference.

Native adapter uses Transformers Qwen2 primitives
(Apache-2.0; https://github.com/huggingface/transformers).
MIT License

Copyright (c) 2026 Hassana Labs Ltd

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any
import numpy as np
from contextlib import contextmanager
from datetime import datetime, timezone
import html
import platform
import re
import shlex
import types
import copy
from dataclasses import replace
from dataclasses import dataclass, field
import time
import uuid
from dataclasses import asdict
from pathlib import Path, PurePosixPath
import shutil
import signal
import subprocess
import difflib
import ast
import threading
from contextlib import nullcontext

__version__ = "0.4.0"
VERSION = __version__

# Recorded evidence is data, not code: it lives in nala_cases.json beside this
# file. Keep the two together; nothing else in this module reads that file.
_CASES_PATH = Path(__file__).resolve().with_name("nala_cases.json")
_CASES_CACHE = None


def _recorded_evidence(name):
    """Return one recorded evidence record, loaded once and copied per caller."""
    global _CASES_CACHE
    if _CASES_CACHE is None:
        try:
            _CASES_CACHE = json.loads(_CASES_PATH.read_text(encoding="utf-8"))
        except OSError as exc:
            raise UserError(
                f"Recorded evidence not found at {_CASES_PATH}. "
                "Keep nala_cases.json in the same directory as nala.py.") from exc
    if name not in _CASES_CACHE:
        raise UserError(f"Recorded evidence file has no {name!r} record.")
    return copy.deepcopy(_CASES_CACHE[name])


# ========================================================================
# EDITS
# ========================================================================

"""Controlled attention interventions, with mathematical verification.

Runtime: Python >= 3.10 and NumPy. NumPy core; no downloads or model-specific hooks.
Run: nala solve --demo --output report.json
     python nala.py solve --snapshot head.json --request edit.json --output report.json

CONTRACT
For one head with its permitted K/V bank fixed, scores are s(x)=A x+b and
attention is softmax(s). Specify exact pairwise attention ratios, optionally
preserve other ratios, fix a destination RoPE position, restrict editable
directions, and set a Euclidean edit budget. The solver produces a minimum-norm
query edit or a qualified failure record. Verification is numerical; tolerances
and SVD cutoffs are part of every report, never a proof about exact machine reals.

The budget measures the compensating edit in the supplied query coordinates at
the DESTINATION position. It does not include a separately requested positional
change. With q=x before RoPE, rotation preserves this compensation norm.
Queries, masks, values, and other heads are not silently recomputed by this file.

MATHEMATICAL MAP (the corresponding comments appear at each critical operation)
  M1  log(p_j/p_k) = (a_j-a_k)^T x + b_j-b_k, exactly at finite x.
  M2  F=(I-11^T/N)A; positive feature curvature H=F^T D F has ker H=ker F.
  M3  Editable h=Qz, Q^TQ=I; hence ||h||_2=||z||_2.
  M4  Requested ratios give Tz=d, T=CQ. SVD yields minimum-norm z=T^+d.
  M5  Dual g(lambda)=lambda^T d-||T^T lambda||^2/2 certifies the minimum.
  M6  w^T T=0, w^T d!=0 is an inconsistency witness. Truncation alone is not.
  M7  R(p)=exp(pA_rot); finite rotations use sin/cos, not a Taylor truncation.
  M8  Positive phi_1 coefficients give the mean-score-gauge effective step.
  M9  Delta y=(p'-p)V + p Delta V + (p'-p)Delta V.
  M10 A runtime patch changes exactly one selected query row; casting is rechecked.

M1/M4 also follow directly from affine logits. This program makes no priority
claim for attention inversion or pseudoinverses, and no downstream-task claim.
"""


















class InputError(ValueError):
    """The input does not describe the supported mathematical problem."""


class NumericalError(ArithmeticError):
    """The requested floating-point verification could not be completed."""


def _number(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise InputError(f"{name} must be a finite real number")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise InputError(f"{name} is outside float64 range") from exc
    if not math.isfinite(value):
        raise InputError(f"{name} must be finite")
    return value


def _array(value: Any, name: str, ndim: int) -> np.ndarray:
    """Copy inputs; reject complex, boolean, string, nonfinite, and ragged data."""
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise InputError(f"{name} must contain real numbers, not {raw.dtype}")
        a = np.array(raw, dtype=np.float64, copy=True)
    except (ValueError, TypeError, OverflowError) as exc:
        raise InputError(f"{name} must be a rectangular real array") from exc
    if a.ndim != ndim or not np.isfinite(a).all():
        raise InputError(f"{name} must be a finite {ndim}-dimensional array")
    a.setflags(write=False)
    return a


def _index(value: Any, size: int, name: str = "token index") -> int:
    # Validate on the host BEFORE indexing any array, including runtime buffers.
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise InputError(f"{name} must be an integer")
    if not 0 <= value < size:
        raise InputError(f"{name} {value} is outside [0, {size})")
    return int(value)


def _finite(value: Any, name: str) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if not np.isfinite(a).all():
        raise NumericalError(f"{name} exceeded finite float64 range")
    return a


def _norm(a: Any) -> float:
    """Scaled norm avoids spurious overflow/underflow when squaring entries."""
    a = _finite(a, "norm input")
    if a.size == 0:
        return 0.0
    scale = float(np.max(np.abs(a)))
    if scale == 0:
        return 0.0
    return float(_finite(scale * np.sqrt(np.sum((a / scale) ** 2)), "norm"))


def _maxabs(a: Any) -> float:
    return float(np.max(np.abs(a))) if np.size(a) else 0.0


def _center(a: np.ndarray) -> np.ndarray:
    """M2: P a, P=I-11^T/N, evaluated after removing a common reference row.

    (a-a_0)-mean(a-a_0) equals a-mean(a) in exact arithmetic. Removing the
    common part before the mean/dot product avoids needless large-offset loss.
    Information already lost when input tensors were rounded cannot be recovered.
    """
    with np.errstate(over="raise", invalid="raise"):
        shifted = a - a[0]
        return _finite(shifted - shifted.mean(axis=0), "centering")


def _softmax(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract max(s) without changing probabilities; keep log probabilities.

    exp(s-max(s)) is in [0,1]. Underflowed probabilities are reported, and ratio
    constraints are checked from score differences rather than dividing zeros.
    """
    s = _finite(s, "scores")
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        shifted = _finite(s - np.max(s), "shifted scores")
        logp = shifted - np.log(np.exp(shifted).sum())
        return np.exp(logp), logp


@dataclass(frozen=True)
class Tolerances:
    """Per-constraint tolerance = atol + rtol*abs(target log-odds).

    Neither another constraint's magnitude nor the size of the required change
    loosens this tolerance. rcond defines the reported SVD resolution.
    """
    atol: float = 1e-10
    rtol: float = 1e-10
    rcond: float = 1e-12

    def __post_init__(self) -> None:
        for name in ("atol", "rtol", "rcond"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if self.atol <= 0 or self.rtol < 0 or not 0 < self.rcond < 1:
            raise InputError("Require atol > 0, rtol >= 0, and 0 < rcond < 1")

    def for_target(self, target: Any) -> np.ndarray:
        return _finite(self.atol + self.rtol * np.abs(target), "tolerance")


def _svd(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        u, s, vt = np.linalg.svd(_finite(a, "SVD input"), full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise NumericalError("SVD did not converge") from exc
    return u, _finite(s, "singular values"), vt


def _orthonormal_basis(b: np.ndarray, rcond: float) -> tuple[np.ndarray, dict[str, Any]]:
    """M3: Orthonormalise an allowed span, so column scaling is not a cost metric.

    If B=U S V^T, retained U columns span the resolved editable directions.
    h=Qz then satisfies ||h||=||z||. The discarded spectrum is disclosed.
    """
    if b.shape[1] == 0:
        return np.empty((b.shape[0], 0)), {"rank": 0, "singular_values": [], "cutoff": 0.0}
    u, s, _ = _svd(b)
    cutoff = rcond * float(s[0]) if s.size else 0.0
    rank = int(np.count_nonzero(s > cutoff))
    return u[:, :rank], {"rank": rank, "singular_values": s.tolist(), "cutoff": cutoff}


@dataclass(frozen=True)
class Snapshot:
    """Single permitted bank: A=features [N,d], b=bias [N], V=values [N,r].

    query is x [d]. Values may already include a linear output projection.
    Arrays are copied and made read-only to isolate the intervention state.
    """
    query: np.ndarray
    features: np.ndarray
    values: np.ndarray
    bias: np.ndarray | None = None
    labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        x = _array(self.query, "query", 1)
        a = _array(self.features, "features", 2)
        v = _array(self.values, "values", 2)
        if x.size == 0 or a.shape[0] == 0 or a.shape[1] != x.size:
            raise InputError("Expected query [d] and a nonempty features [N,d]")
        if v.shape[0] != a.shape[0] or v.shape[1] == 0:
            raise InputError("values must have shape [N,value_width], value_width > 0")
        b = _array(np.zeros(a.shape[0]) if self.bias is None else self.bias, "bias", 1)
        if b.shape != (a.shape[0],):
            raise InputError("bias must have shape [N]")
        labels = tuple(str(i) for i in range(a.shape[0])) if self.labels is None else tuple(self.labels)
        if len(labels) != a.shape[0] or any(not isinstance(label, str) for label in labels):
            raise InputError("labels must contain one string per permitted key")
        for key, value in (("query", x), ("features", a), ("values", v), ("bias", b), ("labels", labels)):
            object.__setattr__(self, key, value)
        _finite(self.scores, "snapshot scores")

    @property
    def scores(self) -> np.ndarray:
        """M2: Canonical mean-zero scores P(Ax+b), with gauge removed FIRST."""
        with np.errstate(over="raise", invalid="raise"):
            return _center(_finite(_center(self.features) @ self.query + _center(self.bias), "scores"))

    @property
    def probabilities(self) -> np.ndarray:
        return _softmax(self.scores)[0]

    @property
    def output(self) -> np.ndarray:
        return _finite(self.probabilities @ self.values, "attention output")

    def log_odds(self, j: int, k: int) -> float:
        """M1: Contrast BEFORE the dot product; no probability division."""
        j, k = _index(j, len(self.labels)), _index(k, len(self.labels))
        return float(_finite((self.features[j] - self.features[k]) @ self.query
                             + (self.bias[j] - self.bias[k]), "log-odds"))

    def with_query(self, query: Any) -> "Snapshot":
        return Snapshot(query, self.features, self.values, self.bias, self.labels)

    def fingerprint(self) -> str:
        """Bind a report to its array values, shapes, ordering, and token labels."""
        digest = hashlib.sha256()
        for a in (self.query, self.features, self.bias, self.values):
            digest.update(json.dumps(a.shape).encode())
            digest.update(np.asarray(a, dtype="<f8").tobytes(order="C"))
        digest.update(json.dumps(self.labels, ensure_ascii=False).encode())
        return digest.hexdigest()

    def geometry(self, rcond: float = 1e-12) -> dict[str, Any]:
        """M2: ker(F^T D F)=ker(F) for positive diagonal D.

        h^T H h=sum_j D_jj (f_j^T h)^2 vanishes iff Fh=0. Thus the
        numerical contrast rank identifies editable query directions. We use
        unweighted F, avoiding false nullspaces from underflowed positive weights.
        This is not the query Jacobian, and values can add output degeneracies.
        """
        cfg = Tolerances(rcond=rcond)
        _, info = _orthonormal_basis(_center(self.features).T, cfg.rcond)
        return {"dimension": self.query.size, "contrast_rank": info["rank"],
                "invisible_dimension": self.query.size - info["rank"], **info}

    def probe(self, direction: Any, atol: float = 1e-10) -> dict[str, Any]:
        """M2: P A h=0 iff the finite edit changes every score equally."""
        atol = Tolerances(atol=atol).atol
        h = _array(direction, "direction", 1)
        if h.shape != self.query.shape:
            raise InputError("direction must have the query shape")
        contrasts = _finite(_center(self.features) @ h, "direction contrasts")
        after = self.with_query(self.query + h)
        return {"invisible_at_tolerance": _maxabs(contrasts) <= atol,
                "max_abs_centred_score_change": _maxabs(contrasts),
                "max_abs_probability_change": _maxabs(after.probabilities - self.probabilities),
                "atol": atol}

    def effective_step(self) -> tuple[np.ndarray, np.ndarray, float]:
        """M8: Delta M=sum_j c_j f_j (v_j-mu)^T = -grad_B E(0).

        a_aug=[a,b], u=[x,1], f=P a_aug, t_j=u^T f_j, Z=sum exp(t_j),
        c_j=phi_1(t_j)/Z>0, mu=mean(V), and
          E(B)=1/2 sum_j c_j ||f_j^T B-(v_j-mu)||^2.
        Since sum(V-mu)=0 and exp(t)-1=t phi_1(t),
          y=mu+u^T Delta M.
        Stable coefficients are evaluated without exp(large positive t):
          t>0:  c=p*(-expm1(-t))/t;
          t<0:  c=exp(-logZ)*expm1(t)/t;
          t=0:  c=exp(-logZ).
        Underflow can round a positive c to zero; geometry() never uses those
        rounded weights to decide rank. Readout reconstruction is checked.
        """
        f = _center(np.column_stack((self.features, self.bias)))
        u = np.append(self.query, 1.0)
        t = _finite(f @ u, "effective-step scores")
        p, logp = _softmax(t)
        largest = int(np.argmax(t))
        logz = float(t[largest] - logp[largest])
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            invz = np.exp(-logz)
            c = np.empty_like(t)
            pos, neg, zero = t > 0, t < 0, t == 0
            c[pos] = p[pos] * (-np.expm1(-t[pos])) / t[pos]
            c[neg] = invz * np.expm1(t[neg]) / t[neg]
            c[zero] = invz
            centred_values = _center(self.values)
            mu = _finite(self.values[0] + (self.values - self.values[0]).mean(axis=0), "value mean")
            matrix = _finite(f.T @ (c[:, None] * centred_values), "effective matrix")
        error = _maxabs(mu + u @ matrix - self.output)
        return matrix, mu, error


def rotate_rows(vectors: Any, positions: Any, frequencies: Any, layout: str) -> np.ndarray:
    """M7: Exact finite fixed-frequency RoPE, including a nonrotating tail.

    For J=[[0,-1],[1,0]], J^2=-I:
      exp(theta J)=cos(theta)I+sin(theta)J,
      exp(theta J)-I=theta J phi_1(theta J).
    We evaluate the full rotation; this is not a first-order positional model.
    Rows transform as v R(p)^T. interleaved pairs (0,1),(2,3),...;
    half_split pairs (0,m),(1,m+1),... in the leading rotary width 2m.
    Frequencies must be the actual fixed frequencies used by the captured head.
    """
    v = _array(vectors, "vectors", 2)
    w = _array(frequencies, "frequencies", 1)
    if v.shape[0] == 0 or v.shape[1] == 0 or 2 * w.size > v.shape[1]:
        raise InputError("Rotary width must fit nonempty vectors")
    raw_positions = np.asarray(positions)
    if raw_positions.ndim == 0:
        p = np.full(v.shape[0], _number(raw_positions.item(), "position"))
    else:
        p = _array(positions, "positions", 1)
    if p.shape != (v.shape[0],):
        raise InputError("positions must be scalar or one per vector")
    if layout == "interleaved":
        first, second = np.arange(0, 2*w.size, 2), np.arange(1, 2*w.size, 2)
    elif layout == "half_split":
        first, second = np.arange(w.size), np.arange(w.size, 2*w.size)
    else:
        raise InputError("layout must be 'interleaved' or 'half_split'")
    angle = _finite(p[:, None] * w, "rotary angles")
    co, si = np.cos(angle), np.sin(angle)
    out = v.copy()
    out[:, first] = co*v[:, first] - si*v[:, second]
    out[:, second] = si*v[:, first] + co*v[:, second]
    return _finite(out, "rotated vectors")


def from_rotated_cache(query: Any, rotated_keys: Any, values: Any, *, position: float,
                       frequencies: Any, layout: str, scale: float,
                       query_projection: Any = None, query_bias: Any = None,
                       labels: Any = None) -> Snapshot:
    """M7: Convert a PRE-RoPE query and POST-RoPE permitted keys to A,b.

    Row convention q=x^T W+bq and q_rot=q R(p)^T. For cached row k_rot,
      score=(q R(p)^T) k_rot^T / scale
           =(k_rot R(p)) W^T x / scale + (k_rot R(p)) bq^T / scale.
    Thus rotate the cached key by -p, then apply W^T. This sign is essential.
    scale is the score DIVISOR, explicitly supplied. No GQA mapping or mask is
    inferred: supply the KV bank for the chosen query head and permitted keys.
    """
    x, k = _array(query, "query", 1), _array(rotated_keys, "rotated_keys", 2)
    scale, position = _number(scale, "scale"), _number(position, "position")
    if scale <= 0:
        raise InputError("scale must be positive")
    relative = rotate_rows(k, -position, frequencies, layout)
    if query_projection is None:
        if x.size != k.shape[1]:
            raise InputError("Query/key widths differ; supply query_projection [input_width,head_width]")
        features = relative / scale
    else:
        projection = _array(query_projection, "query_projection", 2)
        if projection.shape != (x.size, k.shape[1]):
            raise InputError("query_projection must have shape [input_width,head_width]")
        features = relative @ projection.T / scale
    bq = _array(np.zeros(k.shape[1]) if query_bias is None else query_bias, "query_bias", 1)
    if bq.shape != (k.shape[1],):
        raise InputError("query_bias must have shape [head_width]")
    return Snapshot(x, features, values, relative @ bq / scale, labels)


def _keys(obj: Any, allowed: set[str], required: set[str], name: str) -> None:
    if not isinstance(obj, dict):
        raise InputError(f"{name} must be an object")
    unknown, missing = set(obj) - allowed, required - set(obj)
    if unknown or missing:
        raise InputError(f"{name}: unknown fields {sorted(unknown)}; missing fields {sorted(missing)}")


def _targets(baseline: Snapshot, constraints: Any) -> list[tuple[int, int, float]]:
    if not isinstance(constraints, list) or not constraints:
        raise InputError("constraints must be a nonempty list")
    out = []
    for spec in constraints:
        _keys(spec, {"numerator", "denominator", "ratio", "log_odds", "preserve"},
              {"numerator", "denominator"}, "constraint")
        j, k = _index(spec["numerator"], len(baseline.labels)), _index(spec["denominator"], len(baseline.labels))
        if j == k:
            raise InputError("A ratio must compare different keys")
        options = set(spec) & {"ratio", "log_odds", "preserve"}
        if len(options) != 1:
            raise InputError("Specify exactly one of ratio, log_odds, or preserve")
        if "preserve" in spec:
            if spec["preserve"] is not True:
                raise InputError("preserve must be true")
            target = baseline.log_odds(j, k)
        elif "ratio" in spec:
            ratio = _number(spec["ratio"], "ratio")
            if ratio <= 0:
                raise InputError("Finite softmax ratios must be strictly positive")
            target = math.log(ratio)
        else:
            target = _number(spec["log_odds"], "log_odds")
        out.append((j, k, target))
    return out


def _cycle_witness(targets: list[tuple[int, int, float]], tolerances: np.ndarray) -> dict | None:
    """M6: A closed cycle of log-ratios must sum to zero, independently of A.

    A graph path expresses a token potential as an integer combination of the
    requested differences. Two inconsistent paths give w with w^T D=0 for
    the token-incidence matrix D, yet w^T target != 0. This catches genuine
    target contradictions before any numerical rank truncation is attempted.
    """
    m = len(targets)
    adjacency: dict[int, list[tuple[int, int, int]]] = {}
    for edge, (j, k, _) in enumerate(targets):
        adjacency.setdefault(j, []).append((k, edge, -1))
        adjacency.setdefault(k, []).append((j, edge, 1))
    path: dict[int, np.ndarray] = {}
    desired = np.array([v for _, _, v in targets])
    for root in adjacency:
        if root in path:
            continue
        path[root] = np.zeros(m)
        queue = [root]
        for node in queue:
            for other, edge, sign in adjacency[node]:
                proposed = path[node].copy()
                proposed[edge] += sign
                if other not in path:
                    path[other] = proposed
                    queue.append(other)
                else:
                    w = proposed - path[other]
                    signal = float(w @ desired)
                    tolerance = float(np.abs(w) @ tolerances)
                    if abs(signal) > tolerance:
                        return {"weights": w.tolist(), "target_combination": signal,
                                "combination_tolerance": tolerance,
                                "meaning": "A closed cycle of requested log-odds has nonzero sum."}
    return None


def solve_edit(baseline: Snapshot, constraints: list[dict[str, Any]], *,
               working: Snapshot | None = None, edit_basis: Any = None,
               budget: float | None = None, atol: float = 1e-10,
               rtol: float = 1e-10, rcond: float = 1e-12) -> dict[str, Any]:
    """Solve M1-M6, then independently replay the finite attention and M8-M9.

    'preserve' refers to baseline, before any requested positional/cache change.
    The optimisation starts from working.query. Numerical tolerances verify
    exact equality targets; they do not define a least-change tolerance-band QP.
    Only status 'verified' emits an applicable edit. Other statuses retain
    diagnostic evidence and, when available, a candidate for inspection.
    """
    cfg = Tolerances(atol, rtol, rcond)
    work = baseline if working is None else working
    if (work.query.shape != baseline.query.shape or work.values.shape != baseline.values.shape
            or work.labels != baseline.labels):
        raise InputError("Snapshots must use the same indexed keys, labels, and coordinate widths")
    if budget is not None:
        budget = _number(budget, "budget")
        if budget < 0:
            raise InputError("budget must be nonnegative")
    targets = _targets(baseline, constraints)
    desired = np.array([t for _, _, t in targets])
    tolerance = cfg.for_target(desired)
    common: dict[str, Any] = {
        "version": VERSION, "status": None, "can_apply": False, "edit": None,
        "scope": "one head, declared permitted bank, Euclidean query-coordinate compensation",
        "baseline_fingerprint": baseline.fingerprint(), "working_fingerprint": work.fingerprint(),
        "labels": list(work.labels), "budget": budget,
        "tolerances": {"atol": cfg.atol, "rtol": cfg.rtol, "rcond": cfg.rcond},
        "constraint_tolerances": tolerance.tolist(),
        "targets": [{"numerator": j, "denominator": k, "log_odds": t} for j, k, t in targets],
    }
    cycle = _cycle_witness(targets, tolerance)
    if cycle is not None:
        return {**common, "status": "inconsistent_targets", "witness": cycle}

    # M1: [a_j-a_k]^T h = target - current log-odds. This is finite and exact;
    # no softmax Jacobian or first-order approximation occurs in this system.
    c = _finite(np.stack([work.features[j] - work.features[k] for j, k, _ in targets]), "constraint matrix")
    current = np.array([work.log_odds(j, k) for j, k, _ in targets])
    d = _finite(desired - current, "constraint RHS")
    if edit_basis is None:
        q, t, editable_rank = None, c, work.query.size
        basis_info = {"rank": editable_rank, "cutoff": 0.0, "singular_values": None}
    else:
        supplied = _array(edit_basis, "edit_basis", 2)
        if supplied.shape[0] != work.query.size:
            raise InputError("edit_basis must have one row per query coordinate")
        q, basis_info = _orthonormal_basis(supplied, cfg.rcond)
        t, editable_rank = _finite(c @ q, "restricted constraints"), q.shape[1]

    # M4: T=U diag(s) V^T, z*=V_r diag(1/s_r) U_r^T d.
    # Other exact feasible solutions differ by n in ker(T), orthogonal to z*:
    # ||z*+n||^2=||z*||^2+||n||^2. Normal equations would square conditioning.
    if t.shape[1]:
        u, s, vt = _svd(t)
        cutoff = cfg.rcond * float(s[0]) if s.size else 0.0
        rank = int(np.count_nonzero(s > cutoff))
        coefficients = (u[:, :rank].T @ d) / s[:rank]
        z = _finite(vt[:rank].T @ coefficients, "minimum-norm edit")
        dual = _finite(u[:, :rank] @ (coefficients / s[:rank]), "dual variables")
    else:
        s, rank, cutoff = np.array([]), 0, 0.0
        z, dual = np.zeros(0), np.zeros(len(d))
    delta = z if q is None else _finite(q @ z, "query edit")
    residual = _finite(d - t @ z, "constraint residual")
    common.update({"geometry": work.geometry(cfg.rcond), "editable_basis": basis_info,
                   "constraint_rank": rank, "singular_values": s.tolist(), "singular_cutoff": cutoff,
                   "constraint_matrix": c.tolist(), "constraint_rhs": d.tolist(),
                   "resolved_edit_basis": None if q is None else q.tolist(),
                   "constraint_residual": residual.tolist()})
    if np.any(np.abs(residual) > tolerance):
        # M6: A truncated residual is NOT automatically an exact left-null vector
        # of the original T. Disclose w^T T and qualify every rank-based decision.
        w = residual / _norm(residual)
        unresolved_basis = (basis_info["singular_values"] is not None
                            and any(0 < v <= basis_info["cutoff"] for v in basis_info["singular_values"]))
        unresolved = bool(np.any((s > 0) & (s <= cutoff))) or unresolved_basis
        return {**common, "status": "rank_limited" if unresolved else "inconsistent_at_tolerance",
                "witness": {"weights": w.tolist(), "left_null_residual": (t.T @ w).tolist(),
                            "left_null_residual_norm": _norm(t.T @ w), "rhs_combination": float(w @ d)},
                "explanation": "Discarded nonzero singular directions may change feasibility; no exact impossibility claim is made."
                               if unresolved else "The resolved constraint system is inconsistent at the reported tolerance."}

    # M5: For min ||z||^2/2 subject to Tz=d, the Lagrange dual is
    # g(lambda)=lambda^T d-||T^T lambda||^2/2. At the minimum T^T lambda=z.
    # Report primal feasibility, stationarity, and primal-dual gap explicitly.
    stationarity = _finite(t.T @ dual - z, "stationarity")
    norm = _norm(delta)
    primal = float(_finite(0.5 * _norm(z) ** 2, "primal cost"))
    dual_cost = float(_finite(dual @ d - 0.5 * _norm(t.T @ dual) ** 2, "dual cost"))
    gap = float(_finite(primal - dual_cost, "duality gap"))
    stationarity_tol = float(cfg.for_target(_norm(z)))
    gap_tol = float(cfg.for_target(max(abs(primal), abs(dual_cost))))
    optimality = {"dual_variables": dual.tolist(), "primal_cost": primal, "dual_cost": dual_cost,
                  "primal_dual_gap": gap, "gap_tolerance": gap_tol,
                  "stationarity_norm": _norm(stationarity), "stationarity_tolerance": stationarity_tol,
                  "numerical_norm_lower_bound": math.sqrt(max(0.0, 2*dual_cost)),
                  "scope": "equality-constrained Euclidean optimum at declared numerical resolution"}
    if _norm(stationarity) > stationarity_tol or abs(gap) > gap_tol:
        return {**common, "status": "numerical_verification_failed", "optimality": optimality,
                "explanation": "Primal/dual optimality checks did not meet the declared tolerances."}

    after = work.with_query(_finite(work.query + delta, "edited query"))
    applied_delta = _finite(after.query - work.query, "representable edit")
    rounding_error = _norm(applied_delta - delta)
    # Replaying x+h in actual float64 coordinates is essential: a sub-ULP edit
    # can be correct algebraically yet disappear when added to a large x.
    affine_achieved = np.array([after.log_odds(j, k) for j, k, _ in targets])
    replay_scores = after.scores
    achieved = np.array([replay_scores[j] - replay_scores[k] for j, k, _ in targets])
    if (np.any(np.abs(achieved - desired) > tolerance)
            or np.any(np.abs(affine_achieved - desired) > tolerance)
            or rounding_error > float(cfg.for_target(norm))):
        return {**common, "status": "numerical_verification_failed", "optimality": optimality,
                "achieved_log_odds": achieved.tolist(), "application_rounding_error": rounding_error,
                "explanation": "The represented query edit failed direct finite score replay."}

    # M9: Expand p'V'-pV with V'=V+dV. The last term is exactly the interaction.
    # This is local vector algebra. No downstream gradient enters this check.
    p0, pw, p1 = baseline.probabilities, work.probabilities, after.probabilities
    dv = _finite(after.values - baseline.values, "value change")
    probability_part = (p1 - p0) @ baseline.values
    value_part, interaction = p0 @ dv, (p1 - p0) @ dv
    local_change = _finite(after.output - baseline.output, "local change")
    decomposition_error = _maxabs(local_change - probability_part - value_part - interaction)
    _, _, effective_error = after.effective_step()
    output_tol = float(cfg.for_target(max(1.0, _maxabs(baseline.values), _maxabs(after.values))))
    if max(effective_error, decomposition_error) > output_tol:
        return {**common, "status": "numerical_verification_failed", "optimality": optimality,
                "effective_readout_error": effective_error, "decomposition_error": decomposition_error,
                "output_tolerance": output_tol, "explanation": "Local readout verification failed."}

    within_budget = budget is None or max(norm, _norm(applied_delta)) <= budget
    result = {**common, "status": "verified" if within_budget else "budget_exceeded",
              "can_apply": within_budget, "edit": applied_delta.tolist() if within_budget else None,
              "candidate_edit": applied_delta.tolist(), "candidate_query": after.query.tolist(),
              "minimum_edit_norm": norm, "applied_edit_norm": _norm(applied_delta),
              "application_rounding_error": rounding_error, "optimality": optimality,
              "within_budget": within_budget, "achieved_log_odds": achieved.tolist(),
              "max_abs_log_odds_error": _maxabs(achieved - desired),
              "baseline_probabilities": p0.tolist(), "working_probabilities": pw.tolist(),
              "edited_probabilities": p1.tolist(), "edited_scores": replay_scores.tolist(),
              "probability_underflow_count": int(np.count_nonzero(p1 == 0)),
              "baseline_output": baseline.output.tolist(), "working_output": work.output.tolist(),
              "edited_output": after.output.tolist(), "local_output_change": local_change.tolist(),
              "probability_contribution": probability_part.tolist(), "value_contribution": value_part.tolist(),
              "interaction_contribution": interaction.tolist(), "decomposition_error": decomposition_error,
              "effective_readout_error": effective_error, "output_tolerance": output_tol,
              "total_variation_from_baseline": float(0.5*np.abs(p1-p0).sum())}
    # Fail on nonfinite diagnostic values as well as on nonfinite edit values.
    json.dumps(result, allow_nan=False)
    return result


def _construct(data: dict[str, Any], position: float | None = None) -> Snapshot:
    common = {"schema_version", "source", "description", "query", "values", "labels", "native_reference"}
    if "rotated_keys" in data:
        _keys(data, common | {"rotated_keys", "rope", "query_projection", "query_bias"},
              {"schema_version", "query", "values", "rotated_keys", "rope"}, "snapshot")
        rope = data["rope"]
        _keys(rope, {"query_position", "frequencies", "layout", "scale"},
              {"query_position", "frequencies", "layout", "scale"}, "rope")
        return from_rotated_cache(data["query"], data["rotated_keys"], data["values"],
            position=rope["query_position"] if position is None else position,
            frequencies=rope["frequencies"], layout=rope["layout"], scale=rope["scale"],
            query_projection=data.get("query_projection"), query_bias=data.get("query_bias"), labels=data.get("labels"))
    _keys(data, common | {"features", "bias"}, {"schema_version", "query", "values", "features"}, "snapshot")
    if position is not None:
        raise InputError("Positional edits require rotated_keys and explicit RoPE settings")
    return Snapshot(data["query"], data["features"], data["values"], data.get("bias"), data.get("labels"))


def load_snapshot(data: dict[str, Any], position: float | None = None) -> Snapshot:
    """Replay the ORIGINAL capture before returning even a position-edited state.

    Compare centred reference scores, since common offsets are a softmax gauge.
    This also avoids allowing a large common offset to loosen relative tolerance.
    Output replay checks the declared value/output-projection convention.
    """
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise InputError("snapshot schema_version must be integer 1")
    if data.get("source", "manual") not in {"manual", "constructed", "native_capture"}:
        raise InputError("source must be manual, constructed, or native_capture")
    base = _construct(data)
    reference = data.get("native_reference")
    if data.get("source") == "native_capture" and reference is None:
        raise InputError("A native_capture requires native_reference scores, output, atol, and rtol")
    if reference is not None:
        _keys(reference, {"scores", "output", "atol", "rtol"}, {"scores", "output", "atol", "rtol"}, "native_reference")
        cfg = Tolerances(atol=reference["atol"], rtol=reference["rtol"])
        for key, actual in (("scores", base.scores), ("output", base.output)):
            expected = _array(reference[key], f"native_reference.{key}", 1)
            if expected.shape != actual.shape:
                raise InputError(f"Native {key} has the wrong shape")
            if key == "scores":
                expected = _center(expected)
            if np.any(np.abs(actual - expected) > cfg.for_target(expected)):
                raise InputError(f"Native {key} replay failed: check head/KV mapping, permitted keys, RoPE, scale, and query coordinates")
    return base if position is None else _construct(data, _number(position, "query_position"))


def _native_query(data: dict[str, Any], x: Any, position: float) -> np.ndarray:
    x = _array(x, "query", 1)
    q = x if data.get("query_projection") is None else x @ np.asarray(data["query_projection"], dtype=np.float64)
    if data.get("query_bias") is not None:
        q = q + np.asarray(data["query_bias"], dtype=np.float64)
    rope = data["rope"]
    return rotate_rows(q[None, :], position, rope["frequencies"], rope["layout"])[0]


def run_request(data: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    _keys(request, {"constraints", "query_position", "edit_basis", "budget", "tolerances"}, {"constraints"}, "request")
    tolerance_data = request.get("tolerances", {})
    _keys(tolerance_data, {"atol", "rtol", "rcond"}, set(), "tolerances")
    cfg = Tolerances(**tolerance_data)
    base = load_snapshot(data)
    work = base if "query_position" not in request else load_snapshot(data, request["query_position"])
    result = solve_edit(base, request["constraints"], working=work, edit_basis=request.get("edit_basis"),
                        budget=request.get("budget"), atol=cfg.atol, rtol=cfg.rtol, rcond=cfg.rcond)
    if "rotated_keys" in data and result["can_apply"]:
        original_position = _number(data["rope"]["query_position"], "query_position")
        destination = _number(request.get("query_position", original_position), "query_position")
        original = _native_query(data, base.query, original_position)
        working = _native_query(data, work.query, destination)
        edited = _native_query(data, result["candidate_query"], destination)
        # M7: A separate direct post-RoPE dot product checks the adapter's sign,
        # scale, projection convention, and the vector that will be exported.
        k = np.asarray(data["rotated_keys"], dtype=np.float64)
        direct_scores = _center(_finite(k @ edited / data["rope"]["scale"], "native-query scores"))
        p, _ = _softmax(direct_scores)
        direct_output = p @ base.values
        native_audit = verify_backend_result(result, direct_scores, direct_output, atol=cfg.atol, rtol=cfg.rtol)
        if not native_audit["verified"]:
            result.update(status="numerical_verification_failed", can_apply=False, edit=None,
                          explanation="Exported post-RoPE query failed direct attention replay.")
        else:
            result["native_query_patch"] = {
                "original_rotated_query": original.tolist(), "working_rotated_query": working.tolist(),
                "edited_rotated_query": edited.tolist(), "original_position": original_position,
                "destination_position": destination,
                "compensation_query_norm": _norm(edited-working),
                "total_query_change_norm": _norm(edited-original),
                "verification": "float64 direct dot-product attention; backend arithmetic requires its own replay",
                "audit": native_audit}
    json.dumps(result, allow_nan=False)
    return result


def apply_query_patch(query_states: np.ndarray, data: dict[str, Any], report: dict[str, Any], *,
                      batch: int, head: int, token: int, from_state: str,
                      atol: float, rtol: float) -> tuple[np.ndarray, dict[str, Any]]:
    """M10: Copy a NumPy [batch,query_heads,query_tokens,head_width] query buffer.

    Only row (batch,head,token) changes; K/V are never mutated. from_state is
    explicitly 'original' (position+content intervention) or 'working' (content
    compensation after the position change). This prevents stage ambiguity.

    The replacement is cast to the buffer dtype BEFORE checking its resulting
    log-odds and local output. If casting misses requested tolerances, raise
    without modifying the original. These checks replay the cast tensor in
    float64; verify_backend_result() checks actual backend scores/output later.
    This array helper requires direct pre-RoPE query coordinates. Reports using
    an affine input projection still export their query vector; installing those
    requires a hook that respects the declared input-coordinate intervention.
    """
    cfg = Tolerances(atol=atol, rtol=rtol)
    if not isinstance(query_states, np.ndarray) or query_states.ndim != 4 or query_states.dtype.kind != "f":
        raise InputError("query_states must be a floating NumPy [batch,heads,tokens,width] array")
    b, h, t = (_index(batch, query_states.shape[0], "batch"), _index(head, query_states.shape[1], "head"),
               _index(token, query_states.shape[2], "token"))
    if from_state not in {"original", "working"}:
        raise InputError("from_state must explicitly be original or working")
    if report.get("status") != "verified" or report.get("can_apply") is not True or "native_query_patch" not in report:
        raise InputError("Only a verified rotated-cache report can be applied")
    base = load_snapshot(data)
    if base.fingerprint() != report.get("baseline_fingerprint"):
        raise InputError("Report belongs to a different snapshot")
    if data.get("query_projection") is not None:
        raise InputError("Array application requires direct pre-RoPE query coordinates; use a projection-aware hook for an affine-input report")
    patch = report["native_query_patch"]
    expected = _array(patch[f"{from_state}_rotated_query"], "expected query", 1)
    replacement = _array(patch["edited_rotated_query"], "replacement query", 1)
    actual = np.asarray(query_states[b, h, t], dtype=np.float64)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise InputError("Selected runtime query has the wrong width or is nonfinite")
    if np.any(np.abs(actual-expected) > cfg.for_target(expected)):
        raise InputError("Runtime query does not match the declared intervention stage")
    with np.errstate(over="ignore", invalid="ignore"):
        cast = replacement.astype(query_states.dtype).astype(np.float64)
    if not np.isfinite(cast).all():
        raise NumericalError("Replacement is not finite in the runtime query dtype")
    # M3/M7: Pull the cast vector back through the orthogonal rotation. Its
    # compensation must still lie in the declared editable pre-RoPE span.
    # Ratio checks alone would miss rounding in directions invisible to K.
    rope = data["rope"]
    before_rope = rotate_rows(cast[None, :], -patch["destination_position"],
                              rope["frequencies"], rope["layout"])[0]
    working_before_rope = np.asarray(data["query"], dtype=np.float64)
    if data.get("query_bias") is not None:
        working_before_rope = working_before_rope + np.asarray(data["query_bias"], dtype=np.float64)
    cast_edit = before_rope-working_before_rope
    basis = report.get("resolved_edit_basis")
    outside_span = 0.0
    if basis is not None:
        q = np.asarray(basis, dtype=np.float64)
        outside_span = _norm(cast_edit-q @ (q.T @ cast_edit))
        if outside_span > float(cfg.for_target(_norm(cast_edit))):
            raise NumericalError("Casting moved the query outside the allowed edit subspace")
    scores = _center(np.asarray(data["rotated_keys"], dtype=np.float64) @ cast / data["rope"]["scale"])
    p, _ = _softmax(scores)
    output = p @ base.values
    audit = verify_backend_result(report, scores, output, atol=atol, rtol=rtol)
    if not audit["verified"]:
        raise NumericalError("Casting the query failed the requested ratio/output tolerances; original buffer was not changed")
    # Orthogonality preserves compensation norm for direct query coordinates.
    # Check both evaluations so rounding never silently increases the budget.
    cast_compensation = _norm(cast - np.asarray(patch["working_rotated_query"]))
    if report.get("budget") is not None and max(cast_compensation, _norm(cast_edit)) > report["budget"]:
        raise NumericalError("The cast query exceeds the compensation budget")
    out = query_states.copy()
    out[b, h, t] = cast
    audit.update(query_dtype=str(query_states.dtype), query_cast_error=_norm(cast-replacement),
                 cast_compensation_norm=cast_compensation, allowed_span_residual=outside_span,
                 verification="float64 replay of the dtype-cast query; run verify_backend_result on actual backend outputs")
    return out, audit


def verify_backend_result(report: dict[str, Any], scores: Any, output: Any, *,
                           atol: float, rtol: float) -> dict[str, Any]:
    """Verify actual selected-head scores and output AFTER applying the edit.

    Common score shifts are removed by M2. M1 checks each requested log-ratio;
    the full score-pattern and output comparisons also detect unintended changes
    outside the explicitly protected ratios. Supply raw or projected output to
    match the input snapshot's values convention. No downstream logits are used.
    """
    cfg = Tolerances(atol=atol, rtol=rtol)
    if report.get("status") != "verified" or report.get("can_apply") is not True:
        raise InputError("Backend verification requires a verified report")
    s, y = _array(scores, "backend scores", 1), _array(output, "backend output", 1)
    if len(s) != len(report["labels"]) or y.shape != np.shape(report["edited_output"]):
        raise InputError("Backend score/output shapes do not match the selected bank")
    centred = _center(s)
    targets = report["targets"]
    desired = np.array([row["log_odds"] for row in targets])
    got = np.array([centred[row["numerator"]] - centred[row["denominator"]] for row in targets])
    tolerances = cfg.for_target(desired)
    expected_output = np.asarray(report["edited_output"])
    expected_scores = np.asarray(report["edited_scores"])
    probabilities, _ = _softmax(centred)
    p_expected = np.asarray(report["edited_probabilities"])
    ratio_ok = bool(np.all(np.abs(got-desired) <= tolerances))
    score_ok = bool(np.all(np.abs(centred-expected_scores) <= cfg.for_target(expected_scores)))
    p_ok = bool(np.all(np.abs(probabilities-p_expected) <= cfg.for_target(p_expected)))
    y_ok = bool(np.all(np.abs(y-expected_output) <= cfg.for_target(expected_output)))
    return {"verified": ratio_ok and score_ok and p_ok and y_ok, "ratios_verified": ratio_ok,
            "scores_verified": score_ok, "probabilities_verified": p_ok, "output_verified": y_ok,
            "max_abs_log_odds_error": _maxabs(got-desired),
            "max_abs_centred_score_error": _maxabs(centred-expected_scores),
            "max_abs_probability_error": _maxabs(probabilities-p_expected),
            "max_abs_output_error": _maxabs(y-expected_output), "atol": atol, "rtol": rtol}


def example() -> tuple[dict[str, Any], dict[str, Any]]:
    """Constructed four-key demonstration, explicitly not a trained-model result."""
    return ({"schema_version": 1, "source": "constructed",
             "query": [0.2, 0.1, 0.6, -0.2],
             "rotated_keys": [[1,0,0,0], [-1,0,0,0], [0,0,1,0], [0,0,-1,0]],
             "values": [[1,0], [-1,0], [0,1], [0,-1]],
             "labels": ["Target", "Distractor", "Protected A", "Protected B"],
             "rope": {"query_position": 0, "frequencies": [1,0.25], "layout": "interleaved", "scale": 1}},
            {"query_position": math.pi/2,
             "constraints": [{"numerator": 0, "denominator": 1, "ratio": 4},
                             {"numerator": 2, "denominator": 3, "preserve": True}], "budget": 1})


def read_json(path: Path) -> dict[str, Any]:
    def unique_pairs(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise InputError(f"Duplicate JSON key: {key}")
            out[key] = value
        return out
    def reject_constant(value):
        raise InputError(f"Nonfinite JSON number: {value}")
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs, parse_constant=reject_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"Cannot read JSON {path}: {exc}") from exc


def _write_json(path: Path | None, result: dict[str, Any]) -> None:
    """Atomic replacement prevents a failed write from leaving a partial report."""
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if path is None:
        sys.stdout.write(text)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def solve_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Construct a minimum-norm, verified local attention intervention.")
    parser.add_argument("--demo", action="store_true", help="Use the built-in constructed example")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output", type=Path, help="JSON report; defaults to stdout")
    args = parser.parse_args(argv)
    if args.demo and (args.snapshot is not None or args.request is not None):
        parser.error("--demo cannot be combined with --snapshot/--request")
    if not args.demo and (args.snapshot is None or args.request is None):
        parser.error("Supply --demo or both --snapshot and --request")
    if args.output is not None and any(path is not None and path.resolve() == args.output.resolve()
                                        for path in (args.snapshot, args.request)):
        parser.error("The report output must not overwrite an input file")
    try:
        data, request = example() if args.demo else (read_json(args.snapshot), read_json(args.request))
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            result = run_request(data, request)
        code = 0 if result["status"] == "verified" else (4 if result["status"] == "numerical_verification_failed" else 3)
    except (InputError, KeyError, TypeError) as exc:
        result, code = {"version": VERSION, "status": "invalid_input", "can_apply": False, "edit": None, "error": str(exc)}, 2
    except (NumericalError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        result, code = {"version": VERSION, "status": "numerical_failure", "can_apply": False, "edit": None, "error": str(exc)}, 4
    _write_json(args.output, result)
    print(result["status"], file=sys.stderr)
    return code


# ========================================================================
# DEBUGGER
# ========================================================================

"""A controlled attention experiment, from named tokens to measured answer changes.

Quick start: nala demo --offline
Real model:  nala demo
Help:        nala run --help

The mathematical solver lives in nala/edits.py. This file owns token
selection, native Qwen2 execution, and the experiment report. Critical contracts
are documented where implemented. No gradient proxy is used for final logits.
"""






















MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
TRANSFORMERS_VERSION = "4.57.3"
DEMO_PROMPT = "Maya lives in Rome. Noah lives in Paris. Where does Maya live? Answer with just the city."
NATIVE_ATOL = 3e-5
NATIVE_RTOL = 3e-5


class UserError(ValueError):
    """An actionable problem with the requested experiment."""


class VerificationError(RuntimeError):
    """A native numerical gate failed. Do not report the intervention as verified."""


def say(message: str) -> None:
    print(message, flush=True)


def finite_positive(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use a positive number, for example 4.") from exc
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("Use a finite number greater than zero.")
    return result


def nonnegative_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use a zero-based integer index.") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("Indices start at zero.")
    return result


@dataclass
class PromptTokens:
    text: str
    ids: list[int]
    offsets: list[tuple[int, int]]
    pieces: list[str]
    body_start: int = 0
    body_end: int | None = None

    def select(self, selector: str) -> int:
        """Token constraints are row constraints, not sums over a text span.

        Exact text must identify exactly one tokenizer row. Repeated text must
        use NAME#occurrence or @index. Multi-token text is refused; silently
        reducing a span to its last token would change the user's experiment.
        Only occurrences inside the user's prompt are matched by text.
        """
        if re.fullmatch(r"@\d+", selector):
            index = int(selector[1:])
            if index < len(self.ids):
                return index
            raise UserError(f"{selector} is outside this prompt ({len(self.ids)} tokens). Run the tokens command.")
        occurrence = None
        match = re.fullmatch(r"(.+)#([1-9]\d*)", selector, re.S)
        text = selector
        if match:
            text, occurrence = match[1], int(match[2])
        if not text:
            raise UserError("A token selector cannot be empty.")
        end = len(self.text) if self.body_end is None else self.body_end
        positions = []
        cursor = self.body_start
        while True:
            pos = self.text.find(text, cursor, end)
            if pos < 0:
                break
            positions.append(pos)
            cursor = pos + len(text)
        if not positions:
            raise UserError(f"{selector!r} was not found in the prompt. Matching is case-sensitive; run tokens to see exact indices.")
        if occurrence is None and len(positions) != 1:
            raise UserError(f"{text!r} occurs {len(positions)} times. Use {text}#1, {text}#2, or @index from the tokens command.")
        which = 0 if occurrence is None else occurrence-1
        if which >= len(positions):
            raise UserError(f"{text!r} has only {len(positions)} occurrence(s).")
        start, stop = positions[which], positions[which] + len(text)
        rows = [i for i, (a,b) in enumerate(self.offsets) if a < stop and b > start]
        if len(rows) != 1:
            choices = ", ".join(f"@{i}={self.pieces[i]!r}" for i in rows)
            raise UserError(f"{selector!r} spans {len(rows)} tokens ({choices}). Select one @index; this version controls token-pair ratios, not span totals.")
        i = rows[0]
        a,b = self.offsets[i]
        if self.text[a:b].strip() != text.strip():
            raise UserError(f"{selector!r} is only part of token @{i}={self.pieces[i]!r}. Use @{i} explicitly if that is the intended token.")
        return i


def encode_prompt(tokenizer, prompt: str, raw: bool = False) -> PromptTokens:
    if not prompt.strip():
        raise UserError("The prompt is empty.")
    if raw:
        rendered = prompt
    else:
        if not tokenizer.chat_template:
            raise UserError("This tokenizer has no chat template. Use --raw for an exact text-completion prompt.")
        rendered = tokenizer.apply_chat_template([{"role":"user", "content":prompt}],
                                                 tokenize=False, add_generation_prompt=True)
    first = rendered.find(prompt)
    if first < 0 or rendered.find(prompt, first+1) != -1:
        raise UserError("Cannot map the prompt unambiguously through this chat template. Use --raw.")
    enc = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    pieces = [tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in ids]
    return PromptTokens(rendered, ids, [tuple(x) for x in enc["offset_mapping"]], pieces, first, first+len(prompt))


def answer_token(tokenizer, text: str) -> tuple[int, str]:
    """The metric compares explicitly disclosed next-token candidates.

    Candidates are literal decoded token strings, not sequence probabilities.
    Requiring round-trip equality prevents silent token/word mismatches. A
    leading space is significant; users can write --answer ' Rome'.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1 or tokenizer.decode(ids, clean_up_tokenization_spaces=False) != text:
        raise UserError(f"Answer candidate {text!r} is not one exact token. This release measures next-token preference; supply a single token, including any intended leading space.")
    return int(ids[0]), text


def answer_sequence(tokenizer, text: str) -> list[int]:
    """Encode a literal answer continuation without retokenizing the prompt.

    Unlike an attention selector, an answer may contain several tokens. Its
    score sums conditional log probabilities under teacher forcing. Preserve
    significant spaces and disclose every token ID in the saved report.
    """
    ids=tokenizer.encode(text,add_special_tokens=False)
    if not ids or len(ids)>32 or tokenizer.decode(ids,clean_up_tokenization_spaces=False)!=text:
        raise UserError(f"Answer {text!r} must round-trip as 1–32 tokens. Spaces are significant; use --answer and --alternative for literal continuations.")
    return [int(i) for i in ids]


def model_dependencies():
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise UserError("Live runs need: pip install 'torch>=2.6' 'transformers==4.57.3' 'jinja2>=3.1'. "
                        "For the offline example, run: python nala.py demo") from exc
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise UserError(f"Requires transformers=={TRANSFORMERS_VERSION}; found {transformers.__version__}.")
    version = re.match(r"(\d+)\.(\d+)", torch.__version__)
    if not version or tuple(map(int, version.groups())) < (2, 6):
        raise UserError("Requires torch>=2.6.")
    return torch, transformers


def model_support_reason():
    """Reason the native adapter is unusable here, or None when it is usable.

    Reports rather than raises so suites can skip instead of erroring. Skips are
    never silent: the reason is printed by the runner and by unittest itself.
    """
    try:
        model_dependencies()
    except UserError as exc:
        return str(exc)
    return None


@contextmanager
def precise_inference(torch):
    """Native verification needs the declared FP32 arithmetic, including matmul.

    Disable TF32 during this scoped experiment. Restore settings even after a
    refused intervention, so use from another notebook does not leak settings.
    """
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.inference_mode():
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def to_numpy(tensor) -> np.ndarray:
    return tensor.detach().double().cpu().numpy().copy()


def assert_close(actual, expected, name: str, atol=NATIVE_ATOL, rtol=NATIVE_RTOL) -> float:
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise VerificationError(f"{name}: shape mismatch or nonfinite result.")
    error = float(np.max(np.abs(a-b))) if a.size else 0.0
    if np.any(np.abs(a-b) > atol + rtol*np.abs(b)):
        raise VerificationError(f"{name}: error {error:.3g} exceeded the declared tolerance. The experiment was stopped.")
    return error


class QwenAdapter:
    """A temporary, single-layer Qwen2 adapter using the pinned native primitives.

    Critical contract: after native RoPE, replace exactly q[0,head,query,:].
    K, V, the native mask, all other query rows, and the output projection follow
    their original path. A full unmodified-model comparison and an installed
    no-op check precede experimental runs. No global function is monkey-patched.

    Only dense full-prompt, batch-one, FP32 eager Qwen2 inference is supported.
    Forward caching, dropout, sliding windows, quantization, and altered RoPE
    schedules are rejected instead of silently changing the intervention.
    """
    def __init__(self, model, layer: int, head: int, query: int):
        torch, _ = model_dependencies()
        from transformers.models.qwen2 import modeling_qwen2 as native
        if model.config.model_type != "qwen2" or model.training:
            raise UserError("Use an eval-mode Qwen2/Qwen2.5 model.")
        if model.config._attn_implementation != "eager":
            raise UserError("Load the model with attn_implementation='eager'.")
        if getattr(model, "is_quantized", False) or any(p.dtype != torch.float32 for p in model.parameters()):
            raise UserError("The verified adapter currently requires unquantized float32 weights.")
        if any(x.self_attn.sliding_window is not None for x in model.model.layers):
            raise UserError("This adapter supports full causal attention, not sliding windows.")
        rotary = model.model.rotary_emb
        if rotary.rope_type != "default" or rotary.attention_scaling != 1:
            raise UserError("This adapter requires fixed-frequency, unscaled RoPE.")
        if not 0 <= layer < len(model.model.layers) or not 0 <= head < model.config.num_attention_heads or query < 0:
            raise UserError("Layer/head/query indices are outside the model.")
        self.model, self.layer, self.head, self.query = model, layer, head, query
        self.module = model.model.layers[layer].self_attn
        self.torch, self.native = torch, native
        self.frequencies = to_numpy(rotary.inv_freq)
        self.capture = None
        self.expected = None
        self.replacement = None
        self.expected_report = None
        self.calls = 0

    @contextmanager
    def installed(self):
        module = self.module
        had_local = "forward" in module.__dict__
        old_local = module.__dict__.get("forward")
        old = module.forward
        adapter = self
        def forward(mod, hidden_states, position_embeddings, attention_mask,
                    past_key_values=None, cache_position=None, **kwargs):
            return adapter._forward(hidden_states, position_embeddings, attention_mask,
                                    past_key_values=past_key_values, **kwargs)
        module.forward = types.MethodType(forward, module)
        try:
            yield self
        finally:
            if had_local:
                module.forward = old_local
            else:
                delattr(module, "forward")
            self.replacement = None
            self.expected_report = None
            # The bound method identity is ephemeral; underlying function must match.
            assert module.forward.__func__ is old.__func__

    def _forward(self, hidden_states, position_embeddings, attention_mask,
                 past_key_values=None, **kwargs):
        t, m, native = self.torch, self.module, self.native
        if past_key_values is not None or kwargs.get("past_key_value") is not None:
            raise UserError("KV-cache decoding is outside this full-prompt adapter. Use use_cache=False.")
        if hidden_states.shape[0] != 1 or not 0 <= self.query < hidden_states.shape[1]:
            raise UserError("Expected one prompt and a query index inside it.")
        shape = (*hidden_states.shape[:-1], -1, m.head_dim)
        # Same projection and native rotary operations as Qwen2Attention.forward.
        q = m.q_proj(hidden_states).view(shape).transpose(1,2)
        k = m.k_proj(hidden_states).view(shape).transpose(1,2)
        v = m.v_proj(hidden_states).view(shape).transpose(1,2)
        q, k = native.apply_rotary_pos_emb(q, k, *position_embeddings)
        n = k.shape[2]
        permitted = t.arange(n, device=q.device) <= self.query
        if attention_mask is None:
            if n != 1:
                raise VerificationError("A multi-token causal mask was unexpectedly absent.")
        else:
            if attention_mask.ndim != 4 or attention_mask.shape[1] != 1:
                raise VerificationError("Unexpected native mask layout.")
            row = attention_mask[0,0,self.query,:n]
            floor = t.finfo(row.dtype).min / 2
            if not t.equal(row > floor, permitted) or not t.all(row[permitted] == 0):
                raise VerificationError("The native permitted bank differs from the declared full causal bank.")
        kv = self.head // m.num_key_value_groups  # Native repeat_kv maps contiguous groups.
        query_before = to_numpy(q[0,self.head,self.query])
        keys, values = to_numpy(k[0,kv,permitted]), to_numpy(v[0,kv,permitted])
        if self.expected is not None:
            for name, current in (("query",query_before),("keys",keys),("values",values)):
                # Teacher forcing can change GEMM shapes when a suffix is added.
                # Causality preserves these prefix states in real arithmetic;
                # verify the actual FP32 difference rather than assume bit identity.
                assert_close(current,self.expected[name],f"Pre-intervention {name}")
        cast_norm = 0.0
        if self.replacement is not None:
            # Validate shape/range on the CPU before any CUDA index or assignment.
            patch = np.asarray(self.replacement, dtype=np.float64)
            if patch.shape != query_before.shape or not np.isfinite(patch).all():
                raise VerificationError("Invalid replacement query.")
            cast = t.as_tensor(patch.copy(), dtype=q.dtype, device=q.device)
            cast_np = to_numpy(cast)
            if not np.isfinite(cast_np).all():
                raise VerificationError("The query cannot be represented in the native dtype.")
            if self.expected_report is not None:
                work = np.asarray(self.expected_report["debug_working_query"])
                cast_norm = float(np.linalg.norm(cast_np-work))
                budget = self.expected_report.get("budget")
                if budget is not None and cast_norm > budget:
                    raise VerificationError("Casting the query exceeds the requested edit budget.")
            q = q.clone()
            q[0,self.head,self.query] = cast  # The sole intervention assignment.
        # Execute the pinned native attention, with the ORIGINAL mask and K/V.
        raw, probabilities = native.eager_attention_forward(
            m,q,k,v,attention_mask,scaling=m.scaling,dropout=0.0,
            sliding_window=m.sliding_window)
        actual_p = to_numpy(probabilities[0,self.head,self.query,permitted])
        actual_y = to_numpy(raw[0,self.query,self.head])
        # Same scaled dot-product operands, checked against native probabilities.
        scores_t = (t.matmul(q, native.repeat_kv(k,m.num_key_value_groups).transpose(2,3))*m.scaling)
        scores = to_numpy(scores_t[0,self.head,self.query,permitted])
        reference_p = np.exp(scores-scores.max()); reference_p /= reference_p.sum()
        assert_close(actual_p, reference_p, "Native probability replay")
        audit = None
        if self.expected_report is not None:
            audit = verify_backend_result(self.expected_report,scores,actual_y,
                                             atol=NATIVE_ATOL,rtol=NATIVE_RTOL)
            assert_close(actual_p,self.expected_report["edited_probabilities"],"Executed attention probabilities")
            if not audit["verified"]:
                raise VerificationError(f"The executed query missed its target: {audit}")
        self.capture = {"query":query_before,"keys":keys,"values":values,
                        "actual_query":to_numpy(q[0,self.head,self.query]),
                        "scores":scores,"probabilities":actual_p,"output":actual_y,
                        "scaling":float(m.scaling),"kv_head":kv,"audit":audit,
                        "cast_compensation_norm":cast_norm}
        self.calls += 1
        # Preserve the native reshape/projection; subsequent layers run normally.
        return m.o_proj(raw.reshape(*hidden_states.shape[:-1],-1).contiguous()), probabilities

    def run(self, inputs, replacement=None, report=None, logits_to_keep=1):
        self.replacement, self.expected_report, self.capture, self.calls = replacement, report, None, 0
        with precise_inference(self.torch):
            result = self.model(**inputs,use_cache=False,logits_to_keep=logits_to_keep)
        if self.calls != 1 or self.capture is None:
            raise VerificationError("The selected attention module did not execute exactly once.")
        return to_numpy(result.logits[0,-1] if logits_to_keep==1 else result.logits[0]), self.capture


def observe_heads(model, inputs, query: int, a: int, b: int, layer=None, head=None):
    """Choose by baseline attention mass, before any intervention outcome exists.

    This is a convenience heuristic, not a causal head detector. Return its full
    ranking so automatic selection is visible and can be overridden/reproduced.
    Hooks store only two probabilities per head, not full attention matrices.
    """
    torch, _ = model_dependencies()
    rows, handles = [], []
    if layer is not None and not 0 <= layer < len(model.model.layers):
        raise UserError(f"Layer must be between 0 and {len(model.model.layers)-1}.")
    if head is not None and not 0 <= head < model.config.num_attention_heads:
        raise UserError(f"Head must be between 0 and {model.config.num_attention_heads-1}.")
    def callback(index):
        def hook(module,args,output):
            weights=output[1]
            if weights is None:
                raise VerificationError("Native eager attention did not return probabilities.")
            for h in range(weights.shape[1]):
                if head is None or h == head:
                    pa,pb=float(weights[0,h,query,a]),float(weights[0,h,query,b])
                    rows.append({"layer":index,"head":h,"attention_to_pair":pa+pb,"p_a":pa,"p_b":pb})
        return hook
    try:
        for i,block in enumerate(model.model.layers):
            if layer is None or layer == i:
                handles.append(block.self_attn.register_forward_hook(callback(i)))
        with precise_inference(torch):
            logits=to_numpy(model(**inputs,use_cache=False,logits_to_keep=1).logits[0,-1])
    finally:
        for handle in handles:
            handle.remove()
    if not rows:
        raise VerificationError("No candidate heads were observed.")
    rows.sort(key=lambda x:(-x["attention_to_pair"],x["layer"],x["head"]))
    return rows,logits


def metric(logits: np.ndarray, answer: int, alternative: int) -> dict:
    """Measured log-odds = logit(answer)-logit(alternative); softmax cancels.

    This is the NEXT-token metric at the end of the original prompt. The whole
    remaining transformer is executed. It is not a frozen downstream gradient,
    a generation trajectory, or a multi-token answer probability.
    """
    s=np.asarray(logits,dtype=np.float64)
    if s.ndim!=1 or not np.isfinite(s).all() or answer==alternative or not 0<=min(answer,alternative)<=max(answer,alternative)<len(s):
        raise VerificationError("Invalid output logits or answer candidates.")
    p=np.exp(s-s.max()); p/=p.sum()
    top=np.argsort(s)[-5:][::-1]
    return {"log_odds":float(s[answer]-s[alternative]),"answer_probability":float(p[answer]),
            "alternative_probability":float(p[alternative]),
            "top_tokens":[{"id":int(i),"probability":float(p[i])} for i in top]}


def log_probabilities(logits):
    s=np.asarray(logits,dtype=np.float64)
    if not np.isfinite(s).all():
        raise VerificationError("Nonfinite answer logits.")
    z=s-np.max(s,axis=-1,keepdims=True)
    return z-np.log(np.exp(z).sum(axis=-1,keepdims=True))


def continuation_metric(adapter, inputs, first_logits, answer_ids, alternative_ids,
                        replacement=None, report=None):
    """Execute the autoregressive chain rule, including the fixed intervention.

    log P(t_1...t_m|prompt)=sum_i log P(t_i|prompt,t_<i). Later candidates are
    appended as teacher-forced inputs; the SAME original query row is edited in
    every full forward. Native verification reruns for each suffix. This is an
    exact executed likelihood comparison up to arithmetic, not a gradient proxy.
    It does not sample text or claim free-generation accuracy.
    """
    t=adapter.torch
    first_lp=log_probabilities(first_logits)
    def score(ids):
        if len(ids)==1:
            return float(first_lp[ids[0]])
        suffix=t.tensor([ids[:-1]],device=inputs["input_ids"].device,dtype=t.long)
        full=t.cat((inputs["input_ids"],suffix),dim=1)
        forced={"input_ids":full,"attention_mask":t.ones_like(full)}
        logits,_=adapter.run(forced,replacement,report,logits_to_keep=len(ids))
        assert_close(logits[0],first_logits,"Teacher-forced prefix logits")
        lp=log_probabilities(logits)
        return float(lp[np.arange(len(ids)),ids].sum())
    positive,negative=score(answer_ids),score(alternative_ids)
    p=np.exp(first_lp);top=np.argsort(first_lp)[-5:][::-1]
    return {"log_odds":positive-negative,"answer_log_probability":positive,
            "alternative_log_probability":negative,"answer_probability":math.exp(positive),
            "alternative_probability":math.exp(negative),
            "top_tokens":[{"id":int(i),"probability":float(p[i])} for i in top]}


def execute_experiment(model, inputs, tokens: PromptTokens, a: int, b: int,
                       answer_id: int, alternative_id: int, ratios: list[float], *,
                       query=None, layer=None, head=None, preserve=(), shift=0.0, budget=None,
                       progress=say, answer_sequences=None) -> dict:
    """Public API for an existing Qwen notebook; no CLI or downloads required."""
    query=len(tokens.ids)-1 if query is None else query
    n=len(tokens.ids)
    if set(inputs)-{"input_ids","attention_mask"}:
        raise UserError("This adapter accepts input_ids and attention_mask only; custom position IDs or embeddings need a separate validated adapter.")
    indices=[query,a,b]+[i for pair in preserve for i in pair]
    if any(type(i) is not int or not 0<=i<n for i in indices) or a==b:
        raise UserError("Select distinct in-range attention tokens and a valid query row.")
    if any(i>query for i in [a,b]+[i for pair in preserve for i in pair]):
        raise UserError("A selected key is after the query and is causally masked. Choose an earlier key or a later --query.")
    if not ratios or any(not math.isfinite(r) or r<=0 for r in ratios) or not math.isfinite(shift):
        raise UserError("Ratios must be finite and positive; the query shift must be finite.")
    if budget is not None and (not math.isfinite(budget) or budget<0):
        raise UserError("The edit budget must be finite and nonnegative.")
    sequences=answer_sequences or ([answer_id],[alternative_id])
    if (len(sequences)!=2 or any(not seq or len(seq)>32 or any(type(i) is not int or not 0<=i<model.config.vocab_size for i in seq) for seq in sequences)
            or list(sequences[0])==list(sequences[1])):
        raise UserError("Provide two distinct candidate answer sequences of 1–32 valid token IDs.")
    ids=to_numpy(inputs["input_ids"])
    if ids.shape!=(1,n) or not np.array_equal(ids[0],tokens.ids):
        raise UserError("Token metadata does not match the model input.")
    mask=inputs.get("attention_mask")
    if mask is not None and (tuple(mask.shape)!=(1,n) or not bool((mask==1).all())):
        raise UserError("Use one unpadded prompt with its full causal mask.")
    QwenAdapter(model,0,0,query)  # Validate architecture/arithmetic before observing any heads.
    progress("Reading baseline attention and answer preference...")
    ranking,vanilla=observe_heads(model,inputs,query,a,b,layer,head)
    selected=ranking[0]
    layer,head=selected["layer"],selected["head"]
    progress(f"Selected layer {layer}, head {head} (zero-based): {100*selected['attention_to_pair']:.2f}% baseline attention to the selected pair.")
    adapter=QwenAdapter(model,layer,head,query)
    with adapter.installed():
        baseline_logits,capture=adapter.run(inputs)
        replay_error=assert_close(baseline_logits,vanilla,"Unmodified-model versus adapter logits")
        adapter.expected={key:capture[key].copy() for key in ("query","keys","values")}
        labels=tuple(f"@{i} {tokens.pieces[i]}" for i in range(query+1))
        base=Snapshot(capture["query"],capture["keys"]*capture["scaling"],capture["values"],labels=labels)
        assert_close(base.probabilities,capture["probabilities"],"Captured-bank attention replay")
        assert_close(base.output,capture["output"],"Captured-bank output replay")
        baseline_metric=continuation_metric(adapter,inputs,baseline_logits,*sequences)
        noop_logits,noop=adapter.run(inputs,replacement=capture["query"])
        noop_error=assert_close(noop_logits,baseline_logits,"Installed no-op logits",atol=1e-6,rtol=1e-6)
        assert_close(noop["probabilities"],capture["probabilities"],"Installed no-op attention",atol=1e-7,rtol=1e-7)
        noop_metric=continuation_metric(adapter,inputs,noop_logits,*sequences,replacement=capture["query"])
        assert_close(noop_metric["log_odds"],baseline_metric["log_odds"],"Installed no-op answer score",atol=1e-6,rtol=1e-6)
        baseline_row={"name":"baseline","status":"verified","metric":baseline_metric,
                      "attention_log_odds":float(capture["scores"][a]-capture["scores"][b]),
                      "probabilities":capture["probabilities"].tolist(),"edit_norm":0.0}
        # Exact finite positional intervention R(delta)q in post-RoPE coordinates.
        # Fixed K/V and the causal bank remain unchanged. This is a QUERY phase
        # intervention, not moving text or rebuilding an entire shifted context.
        work=base.with_query(rotate_rows(base.query[None,:],shift,adapter.frequencies,"half_split")[0]) if shift else base
        result={"schema_version":1,"version":VERSION,"source":"native_qwen2",
                "scope":"one post-RoPE query row; fixed native prefix K/V and causal mask; measured candidate continuation likelihoods",
                "metric_scope":"sum of teacher-forced answer-token log probabilities; no length normalization; no sampled generation",
                "prompt":tokens.text,"input_ids":tokens.ids,"token_pieces":tokens.pieces,
                "selection":{"layer":layer,"head":head,"kv_head":capture["kv_head"],"query":query,"a":a,"b":b,
                             "method":"largest baseline attention mass on selected pair within requested filters",
                             "ranking":ranking},
                "answer_id":answer_id,"alternative_id":alternative_id,"answer_token_ids":list(sequences[0]),
                "alternative_token_ids":list(sequences[1]),"preserve":[list(x) for x in preserve],
                "query_shift":shift,"budget":budget,"requested_ratios":ratios,
                "verification":{"adapter_logits_max_error":replay_error,"noop_logits_max_error":noop_error,
                                 "native_atol":NATIVE_ATOL,"native_rtol":NATIVE_RTOL,"dtype":"float32"},
                "snapshot":{"schema_version":1,"source":"native_capture","query":base.query.tolist(),
                            "features":base.features.tolist(),"values":base.values.tolist(),"labels":list(labels),
                            "native_reference":{"scores":capture["scores"].tolist(),"output":capture["output"].tolist(),
                                                "atol":NATIVE_ATOL,"rtol":NATIVE_RTOL}},
                "baseline":baseline_row,"trials":[]}
        if shift:
            shifted_logits,shifted=adapter.run(inputs,replacement=work.query)
            assert_close(work.probabilities,shifted["probabilities"],"Shift-only attention replay")
            assert_close(work.output,shifted["output"],"Shift-only output replay")
            shifted_metric=continuation_metric(adapter,inputs,shifted_logits,*sequences,replacement=work.query)
            result["shift_only"]={"name":"shift only","status":"verified","metric":shifted_metric,
                                   "attention_log_odds":float(shifted["scores"][a]-shifted["scores"][b]),
                                   "probabilities":shifted["probabilities"].tolist(),"edit_norm":0.0}
        for ratio in ratios:
            progress(f"Testing attention ratio {ratio:g}:1...")
            if math.log(ratio)<baseline_row["attention_log_odds"]-1e-6:
                progress("  This target lowers A relative to B compared with baseline.")
            constraints=[{"numerator":a,"denominator":b,"ratio":ratio}]
            constraints += [{"numerator":c,"denominator":d,"preserve":True} for c,d in preserve]
            plan=solve_edit(base,constraints,working=work,budget=budget)
            row={"name":f"{ratio:g}:1","requested_ratio":ratio,"status":plan["status"],"plan":plan}
            if plan["can_apply"]:
                plan["debug_working_query"]=work.query.tolist()
                try:
                    changed_logits,changed=adapter.run(inputs,np.asarray(plan["candidate_query"]),plan)
                    changed_metric=continuation_metric(adapter,inputs,changed_logits,*sequences,
                                                        replacement=np.asarray(plan["candidate_query"]),report=plan)
                    row.update(metric=changed_metric,
                               attention_log_odds=float(changed["scores"][a]-changed["scores"][b]),
                               probabilities=changed["probabilities"].tolist(),edit_norm=plan["minimum_edit_norm"],
                               cast_edit_norm=changed["cast_compensation_norm"],
                               query_norm=float(np.linalg.norm(base.query)),
                               total_query_change_norm=float(np.linalg.norm(changed["actual_query"]-base.query)),
                               installed_query=changed["actual_query"].tolist(),audit=changed["audit"],
                               local_output=changed["output"].tolist())
                except VerificationError as exc:
                    row.update(status="native_verification_failed",error=str(exc))
            result["trials"].append(row)
        result["status"]="verified" if all(t["status"]=="verified" for t in result["trials"]) else "incomplete"
    return result


def offline_demo() -> dict:
    """Constructed attention + nonlinear readout. Never label this an LLM result.

    q controls two independent odds. The downstream scalar is tanh(y_0+0.2y_1).
    Execute this map afresh after every edit; local exactness does not assert a
    linear downstream response. This fast example needs only NumPy.
    """
    data,request=example()
    base=load_snapshot(data)
    labels=["Rome","Paris","Protected A","Protected B"]
    def downstream(p):
        y=p@base.values
        score=2*math.tanh(float(y[0]+0.2*y[1]))
        return metric(np.array([score/2,-score/2]),0,1)
    result={"schema_version":1,"version":VERSION,"source":"constructed_offline_demo","status":"verified",
            "prompt":"Constructed four-token attention with a nonlinear readout; no trained model.",
            "input_ids":[0,1,2,3],"token_pieces":labels,"answer":"Rome","alternative":"Paris",
            "selection":{"layer":0,"head":0,"query":3,"a":0,"b":1,"method":"fixed constructed example"},
            "preserve":[[2,3]],"query_shift":0.0,"baseline":{"name":"baseline","status":"verified",
            "metric":downstream(base.probabilities),"attention_log_odds":base.log_odds(0,1),
            "probabilities":base.probabilities.tolist(),"edit_norm":0.0},"trials":[],
            "verification":{"scope":"NumPy float64 constructed attention and nonlinear readout"}}
    for ratio in [0.25,1,4,16]:
        plan=solve_edit(base,[{"numerator":0,"denominator":1,"ratio":ratio},
                                 {"numerator":2,"denominator":3,"preserve":True}])
        if not plan["can_apply"]:
            raise VerificationError("The constructed example failed its solver gate.")
        q=np.asarray(plan["candidate_query"])
        s=base.features@q+base.bias
        p=np.exp(s-s.max());p/=p.sum()
        audit=verify_backend_result(plan,s,p@base.values,atol=1e-10,rtol=1e-10)
        if not audit["verified"]:
            raise VerificationError("The constructed example failed direct attention replay.")
        result["trials"].append({"name":f"{ratio:g}:1","requested_ratio":ratio,"status":"verified",
             "metric":downstream(p),"attention_log_odds":float(s[0]-s[1]),"probabilities":p.tolist(),
             "edit_norm":plan["minimum_edit_norm"],"query_norm":float(np.linalg.norm(base.query)),"plan":plan,"audit":audit})
    return result


def display_ratio(log_odds: float) -> str:
    return f"{math.exp(log_odds):.5g}:1" if abs(log_odds)<600 else f"exp({log_odds:.4g}):1"


def print_result(result: dict) -> None:
    a,b=result["selection"]["a"],result["selection"]["b"]
    tokens=result["token_pieces"]
    answer,alternative=result.get("answer","answer"),result.get("alternative","alternative")
    say("")
    if result["source"]=="constructed_offline_demo":
        say("OFFLINE DEMO — constructed attention and nonlinear readout; no trained model.")
    else:
        s=result["selection"]
        say(f"Layer {s['layer']}, head {s['head']}, query @{s['query']}  |  Native replay and no-op checks passed")
    say(f"Attention: @{a} {tokens[a]!r} / @{b} {tokens[b]!r}")
    say(f"Answer preference: {answer!r} versus {alternative!r}; positive favors {answer!r}.")
    say("")
    say(f"{'Run':<12} {'Achieved':>12} {'P(A)':>8} {'P(B)':>8} {'Answer score*':>13} {'Change':>10} {'Edit norm':>11}")
    baseline=result["baseline"]["metric"]["log_odds"]
    rows=[result["baseline"]]+([result["shift_only"]] if "shift_only" in result else [])+result["trials"]
    for row in rows:
        if row["status"]!="verified":
            explanations={"budget_exceeded":"The smallest computed edit exceeds your budget.",
                          "inconsistent_targets":"The requested ratios contradict each other.",
                          "rank_limited":"Numerical resolution is insufficient to verify this request.",
                          "inconsistent_at_tolerance":"This head cannot meet all targets at the declared numerical resolution.",
                          "native_verification_failed":"The executed attention failed verification; no result is accepted."}
            say(f"{row['name']:<12} BLOCKED: {explanations.get(row['status'],'Numerical verification failed; inspect the saved report.')}")
            if row.get("error"):
                say(f"  {row['error']}")
            continue
        p=row["probabilities"];m=row["metric"]["log_odds"]
        say(f"{row['name']:<12} {display_ratio(row['attention_log_odds']):>12} {p[a]:>8.2%} {p[b]:>8.2%} {m:>+13.5f} {m-baseline:>+10.5f} {row['edit_norm']:>11.5g}")
    say("* Answer score = log P(answer) - log P(alternative), in nats; positive favors the desired answer.")
    valid=[r for r in result["trials"] if r["status"]=="verified"]
    if len(valid)==1:
        change=valid[0]["metric"]["log_odds"]-baseline
        word="increased" if change>1e-6 else "decreased" if change < -1e-6 else "was unchanged"
        say(f"\nThe intervention was verified. Preference for {answer!r} {word} (change {change:+.5f} nats).")
    if valid:
        flips=sum((r["metric"]["log_odds"]>0)!=(baseline>0) for r in valid)
        if abs(baseline)>1e-6:
            if flips:
                say(f"Preferred answer changed in {flips} of {len(valid)} verified runs.")
            else:
                favored=answer if baseline>0 else alternative
                say(f"All {len(valid)} verified runs still favor {favored!r}. These edits did not reverse answer preference.")
    if valid:
        row=valid[-1];delta=np.asarray(row["probabilities"])-np.asarray(result["baseline"]["probabilities"])
        top=np.argsort(np.abs(delta))[-3:][::-1]
        say("Largest attention changes in the last verified run: "+", ".join(f"@{i} {tokens[i]!r} {100*delta[i]:+.2f} pp" for i in top)+".")
    say("Protected ratios do not freeze absolute probabilities. Full attention changes are saved.")
    if result.get("query_shift"):
        say("The shift changes only this query's RoPE phase; keys, text, and the mask stay fixed.")


def response_scatter(result: dict) -> str:
    """Plot only measured points: log attention ratio versus answer score.

    No fitted curve or interpolation is implied. Include the zero answer-score
    line so a reader can see whether candidate preference actually reversed.
    """
    rows=[result["baseline"]]+[r for r in result["trials"] if r["status"]=="verified"]
    x=np.array([r["attention_log_odds"] for r in rows]);y=np.array([r["metric"]["log_odds"] for r in rows])
    xmin,xmax=float(x.min()),float(x.max());ymin,ymax=min(0.,float(y.min())),max(0.,float(y.max()))
    dx=max(.25,xmax-xmin);dy=max(.1,ymax-ymin)
    xmin-=dx*.08;xmax+=dx*.08;ymin-=dy*.10;ymax+=dy*.10
    px=lambda v:85+660*(v-xmin)/(xmax-xmin)
    py=lambda v:270-225*(v-ymin)/(ymax-ymin)
    elements=[]
    for v in np.linspace(ymin,ymax,5):
        yy=py(v)
        elements.append(f"<line x1='85' y1='{yy:.2f}' x2='745' y2='{yy:.2f}' stroke='#e2e8f0'/><text x='73' y='{yy+4:.2f}' text-anchor='end'>{v:.2f}</text>")
    elements.append(f"<line x1='85' x2='745' y1='{py(0):.2f}' y2='{py(0):.2f}' stroke='#8495a8' stroke-dasharray='4 4'/>")
    for v in np.linspace(xmin,xmax,5):
        label=f"{math.exp(v):.3g}" if abs(v)<600 else f"exp({v:.2g})"
        elements.append(f"<text x='{px(v):.2f}' y='294' text-anchor='middle'>{label}</text>")
    for i,r in enumerate(rows):
        tip=html.escape(f"{r['name']}: attention {display_ratio(x[i])}; answer score {y[i]:+.6f}")
        color='#172638' if i==0 else '#147b79'
        elements.append(f"<circle cx='{px(x[i]):.2f}' cy='{py(y[i]):.2f}' r='6' fill='{color}'><title>{tip}</title></circle>")
    return "<svg viewBox='0 0 800 350' role='img' aria-label='Measured attention intervention response' style='width:100%;background:white;border-radius:12px;font:12px system-ui;fill:#334155'><text x='85' y='23'>Answer score (nats)</text>"+"".join(elements)+"<text x='415' y='328' text-anchor='middle'>Achieved A:B attention ratio (logarithmic axis)</text><circle cx='520' cy='20' r='5' fill='#172638'/><text x='531' y='24'>Baseline</text><circle cx='622' cy='20' r='5' fill='#147b79'/><text x='633' y='24'>Verified edits</text></svg>"


def report_html(result: dict) -> str:
    """Portable report; all user/model text is escaped and no remote assets load."""
    esc=lambda x:html.escape(str(x),quote=True)
    a,b=result["selection"]["a"],result["selection"]["b"]
    base=result["baseline"]["metric"]["log_odds"]
    rows=[result["baseline"]]+([result["shift_only"]] if "shift_only" in result else [])+result["trials"]
    table=[]
    details=[]
    for row in rows:
        if row["status"]!="verified":
            table.append(f"<tr><td>{esc(row['name'])}</td><td colspan='6'>Blocked: {esc(row['status'])} {esc(row.get('error',''))}</td></tr>")
            continue
        m=row["metric"]["log_odds"]; p=row["probabilities"]
        cells=[row["name"],display_ratio(row["attention_log_odds"]),f"{p[a]:.3%}",f"{p[b]:.3%}",f"{m:+.6f}",f"{m-base:+.6f}",f"{row['edit_norm']:.6g}"]
        table.append("<tr>"+"".join(f"<td>{esc(c)}</td>" for c in cells)+"</tr>")
        if row is result["baseline"]:
            continue
        delta=np.asarray(p)-np.asarray(result["baseline"]["probabilities"])
        changes="".join(f"<tr><td>@{i}</td><td>{esc(result['token_pieces'][i])}</td><td>{p[i]:.6%}</td><td>{100*delta[i]:+.6f} pp</td></tr>" for i in np.argsort(-np.abs(delta)))
        details.append(f"<details><summary>{esc(row['name'])}: all attention changes</summary><table><tr><th>Index</th><th>Token</th><th>Attention</th><th>Change</th></tr>{changes}</table></details>")
    source="Constructed offline example; no trained model" if result["source"]=="constructed_offline_demo" else "Executed Qwen2 model experiment"
    s=result["selection"]
    chart=response_scatter(result)
    return """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Attention experiment</title><style>body{font:16px/1.55 system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;color:#172638;background:#f7f9fc}h1{font-size:32px}table{border-collapse:collapse;width:100%;background:white;margin:20px 0}th,td{text-align:right;padding:12px;border-bottom:1px solid #dae2ec}th:first-child,td:first-child{text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:20px;border-left:4px solid #457ca5}details{margin:16px 0}summary{cursor:pointer;font-weight:600}.tag{color:#315f87;font-weight:600}.scroll{overflow-x:auto}</style>
"""+f"<h1>Attention experiment</h1><p class='tag'>{esc(source)}</p><p>Layer {s['layer']}, head {s['head']}, query @{s['query']}.</p><p>A: @{a} {esc(result['token_pieces'][a])}; B: @{b} {esc(result['token_pieces'][b])}.</p><p>Positive answer log-odds favor <b>{esc(result.get('answer','answer'))}</b> over <b>{esc(result.get('alternative','alternative'))}</b>.</p>{chart}<div class='scroll'><table><tr><th>Run</th><th>Achieved A:B</th><th>P(A)</th><th>P(B)</th><th>Answer log-odds</th><th>Change (nats)</th><th>Edit norm</th></tr>{''.join(table)}</table></div><p>The answer score is log P(answer) minus log P(alternative), summing teacher-forced token log probabilities without length normalization. Spaces are significant. This is not sampled-generation accuracy. Ratios are verified locally; the downstream response is measured by execution. Minimum norm is in the declared query coordinates and does not imply minimum semantic disruption.</p><p>Pair-ratio constraints can change other attention probabilities. Expand a run to inspect every change.</p>{''.join(details)}<h2>Prompt</h2><pre>{esc(result['prompt'])}</pre><h2>Repeat this experiment</h2><pre>{esc(shlex.join(result.get('reproduce_argv',['nala','demo','--offline'])))}</pre><h2>Verification</h2><pre>{esc(json.dumps(result.get('verification',{}),indent=2))}</pre><p>The companion report.json contains model and token provenance, the captured attention bank, per-run solver records, and native verification.</p></html>"


def save_result(result: dict, directory: Path | None) -> Path:
    if directory is None:
        stamp=datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        directory=Path("nala-results")/stamp
    if directory.exists():
        raise UserError(f"Output already exists: {directory}. Choose a new --out directory to preserve earlier experiments.")
    directory.mkdir(parents=True)
    _write_json(directory/"report.json",result)
    (directory/"report.html").write_text(report_html(result),encoding="utf-8")
    return directory


def build_parser():
    parser=argparse.ArgumentParser(prog="nala",description="Change an attention relationship. Verify the edit. Measure the answer change.",
        epilog="Start instantly: nala demo --offline\nRun Qwen: nala demo\nUse your prompt: nala run --prompt '...' --attend Rome --over Paris",formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version",action="version",version=VERSION)
    subs=parser.add_subparsers(dest="command")
    def common(p):
        p.add_argument("--model",default="qwen",help="qwen (pinned Qwen2.5-0.5B-Instruct), a Qwen2 repo, or a local model directory")
        p.add_argument("--revision",help="Immutable checkpoint revision; qwen already has a pinned default")
        p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
        p.add_argument("--raw",action="store_true",help="Use literal completion text instead of the model's chat template")
        p.add_argument("--local-only",action="store_true",help="Use cached files only; never download")
    def experiment(p):
        targets=p.add_mutually_exclusive_group()
        targets.add_argument("--ratio",type=finite_positive,help="Desired A:B attention ratio (run default: 4, meaning 4:1)")
        targets.add_argument("--sweep",nargs="+",type=finite_positive,help="Test several absolute ratios, for example: --sweep 0.25 1 4 16")
        p.add_argument("--layer",type=nonnegative_int,help="Zero-based layer; otherwise selected from baseline attention")
        p.add_argument("--head",type=nonnegative_int,help="Zero-based head; otherwise selected from baseline attention")
        p.add_argument("--query",type=nonnegative_int,help="Query token index; default: last token of the rendered prompt")
        p.add_argument("--preserve",nargs=2,action="append",default=[],metavar=("C","D"),help="Keep C:D at its original ratio; repeat for more pairs")
        p.add_argument("--shift-query",type=float,default=0,help="Finite RoPE shift of this query only, before the compensating edit")
        p.add_argument("--budget",type=float,help="Maximum compensating query-edit norm")
        p.add_argument("--out",type=Path,help="New report directory; default: a fresh timestamped directory")
        p.add_argument("--max-tokens",type=nonnegative_int,default=512,help="Refuse longer prompts before model loading (default: 512)")
    demo=subs.add_parser("demo",help="Run the included experiment; add --offline for an instant constructed example")
    common(demo);experiment(demo)
    demo.add_argument("--offline",action="store_true",help="Constructed NumPy example: seconds, no weights or model dependencies")
    run=subs.add_parser("run",help="Test an attention change on your prompt")
    common(run);experiment(run)
    prompt=run.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt",help="Your prompt text")
    prompt.add_argument("--prompt-file",type=Path,help="Read a UTF-8 prompt from a file")
    run.add_argument("--attend",required=True,help="A: exact token text, NAME#occurrence, or @index")
    run.add_argument("--over",required=True,help="B: token to compare attention against")
    run.add_argument("--answer",help="Desired answer text (1–32 tokens); defaults to selected A text without leading space")
    run.add_argument("--alternative",help="Competing answer text (1–32 tokens); defaults to selected B text without leading space")
    tokens=subs.add_parser("tokens",help="Show exact token indices; loads the tokenizer only")
    common(tokens)
    prompt=tokens.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file",type=Path)
    return parser


def debug_main(argv=None) -> int:
    parser=build_parser()
    args=parser.parse_args(argv)
    if args.command is None:
        parser.print_help(); return 0
    try:
        if getattr(args,"out",None) is not None and args.out.exists():
            raise UserError(f"Output already exists: {args.out}. Choose a new --out directory.")
        if args.command=="demo" and args.offline:
            # Do not silently ignore a requested real-model/intervention option.
            if args.model!="qwen" or args.revision or args.layer is not None or args.head is not None or args.query is not None or args.preserve or args.shift_query or args.budget is not None or args.sweep or args.ratio is not None or args.raw or args.device!="auto":
                raise UserError("The offline demo is a fixed constructed example. Use run or demo without --offline to configure a model experiment.")
            result=offline_demo()
        else:
            torch,transformers=model_dependencies()
            from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
            preset=args.model in {"qwen",MODEL_ID}
            model_id=MODEL_ID if preset else args.model
            revision=args.revision or (MODEL_REVISION if preset else None)
            if revision is not None and not re.fullmatch(r"[0-9a-fA-F]{40}",revision):
                raise UserError("--revision must be the checkpoint's full 40-character commit hash.")
            config=AutoConfig.from_pretrained(model_id,revision=revision,local_files_only=args.local_only,trust_remote_code=False)
            if config.model_type!="qwen2":
                raise UserError("This first adapter supports Qwen2/Qwen2.5 only. Use --model qwen for the verified example.")
            # Resolve a custom remote repo once, then pin tokenizer and weights
            # to that same immutable commit. Never mix moving-branch snapshots.
            if not Path(model_id).is_dir():
                resolved=getattr(config,"_commit_hash",None)
                if not resolved or not re.fullmatch(r"[0-9a-fA-F]{40}",resolved):
                    raise UserError("Could not resolve an immutable model revision; supply --revision explicitly.")
                revision=resolved
            if args.command=="demo":
                prompt,attend,over,answer,alternative=DEMO_PROMPT,"Rome","Paris","Rome","Paris"
            else:
                prompt=args.prompt if args.prompt is not None else args.prompt_file.read_text(encoding="utf-8")
            say("Loading the tokenizer...")
            tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision,local_files_only=args.local_only,trust_remote_code=False,use_fast=True)
            tokens=encode_prompt(tokenizer,prompt,args.raw)
            if args.command=="tokens":
                say("Indices include the chat template. The default query is the final row.")
                for i,(piece,span) in enumerate(zip(tokens.pieces,tokens.offsets)):
                    say(f"@{i:<5} {piece!r}")
                return 0
            if len(tokens.ids)>args.max_tokens:
                raise UserError(f"Rendered prompt has {len(tokens.ids)} tokens; limit is {args.max_tokens}. Shorten it or explicitly raise --max-tokens (eager attention uses quadratic memory).")
            if args.command=="run":
                attend,over=args.attend,args.over
            a,b=tokens.select(attend),tokens.select(over)
            if args.command=="run":
                answer=args.answer if args.answer is not None else tokens.pieces[a].strip()
                alternative=args.alternative if args.alternative is not None else tokens.pieces[b].strip()
            protected=[(tokens.select(c),tokens.select(d)) for c,d in args.preserve]
            query=len(tokens.ids)-1 if args.query is None else args.query
            if not 0<=query<len(tokens.ids) or any(i>query for i in [a,b]+[i for pair in protected for i in pair]):
                raise UserError("The query must be inside the prompt and at or after every selected key; later keys are causally masked.")
            if a==b:
                raise UserError("--attend and --over must select different tokens.")
            if not math.isfinite(args.shift_query) or (args.budget is not None and (not math.isfinite(args.budget) or args.budget<0)):
                raise UserError("The query shift must be finite and the edit budget must be finite and nonnegative.")
            answer_ids=answer_sequence(tokenizer,answer)
            alternative_ids=answer_sequence(tokenizer,alternative)
            if answer_ids==alternative_ids:
                raise UserError("Answer and alternative resolve to the same token sequence.")
            if args.device=="cuda" and not torch.cuda.is_available():
                raise UserError("CUDA is unavailable. Use --device cpu or attach a GPU runtime.")
            device="cuda" if args.device=="auto" and torch.cuda.is_available() else ("cpu" if args.device=="auto" else args.device)
            torch.set_num_threads(min(4,torch.get_num_threads()))
            say(f"Loading {model_id} on {device} in float32. The default checkpoint is about 1 GB to download once.")
            model=AutoModelForCausalLM.from_pretrained(model_id,revision=revision,local_files_only=args.local_only,
                       trust_remote_code=False,dtype=torch.float32,attn_implementation="eager").to(device).eval()
            if model.config.model_type!="qwen2":
                raise UserError("This first adapter supports Qwen2/Qwen2.5 only.")
            inputs={"input_ids":torch.tensor([tokens.ids],device=device),"attention_mask":torch.ones((1,len(tokens.ids)),dtype=torch.long,device=device)}
            ratios=args.sweep if args.sweep else ([args.ratio] if args.ratio is not None else ([.25,1,4,16] if args.command=="demo" else [4]))
            result=execute_experiment(model,inputs,tokens,a,b,answer_ids[0],alternative_ids[0],ratios,
                        query=args.query,layer=args.layer,head=args.head,preserve=protected,shift=args.shift_query,budget=args.budget,
                        answer_sequences=(answer_ids,alternative_ids))
            result.update(answer=answer,alternative=alternative,model={"id":model_id,"revision":revision,
                  "resolved_commit":getattr(model.config,"_commit_hash",None),"device":device,
                  "config":model.config.to_dict()},environment={"python":platform.python_version(),"numpy":np.__version__,
                  "torch":torch.__version__,"transformers":transformers.__version__})
            argv=["nala","run","--model",model_id,"--prompt",prompt,
                  "--attend",f"@{a}","--over",f"@{b}","--answer",answer,"--alternative",alternative,
                  "--layer",str(result["selection"]["layer"]),"--head",str(result["selection"]["head"]),
                  "--query",str(result["selection"]["query"]),"--device",device,"--max-tokens",str(args.max_tokens),
                  "--sweep",*[str(r) for r in ratios]]
            if revision:argv += ["--revision",revision]
            if args.raw:argv.append("--raw")
            if args.shift_query:argv += ["--shift-query",str(args.shift_query)]
            if args.budget is not None:argv += ["--budget",str(args.budget)]
            for c,d in protected:argv += ["--preserve",f"@{c}",f"@{d}"]
            result.update(reproduce_argv=argv,user_prompt=prompt,
                input_ids_sha256=hashlib.sha256(json.dumps(tokens.ids,separators=(",",":")).encode()).hexdigest())
            for row in [result["baseline"]]+result["trials"]+([result["shift_only"]] if "shift_only" in result else []):
                if "metric" in row:
                    for tok in row["metric"]["top_tokens"]:
                        tok["text"]=tokenizer.decode([tok["id"]],clean_up_tokenization_spaces=False)
        print_result(result)
        directory=save_result(result,getattr(args,"out",None))
        say(f"\nOpen {directory/'report.html'}\nReproduce or inspect: {directory/'report.json'}")
        return 0 if result["status"]=="verified" else (4 if any(r["status"]=="native_verification_failed" for r in result["trials"]) else 3)
    except (UserError,InputError,OSError) as exc:
        print(f"\nCannot run this experiment: {exc}",file=sys.stderr);return 2
    except (VerificationError,NumericalError,ArithmeticError) as exc:
        print(f"\nVerification stopped the experiment: {exc}",file=sys.stderr);return 4
    except RuntimeError as exc:
        print(f"\nModel execution stopped: {exc}\nCheck the [models] dependencies and the selected device. No result is marked verified.",file=sys.stderr);return 4
    except KeyboardInterrupt:
        print("\nStopped. No model weights were modified.",file=sys.stderr);return 130


# ========================================================================
# TEXT
# ========================================================================

"""Strict text serialization and exact token-span anchoring."""










def loads(text: str):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise UserError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result
    def invalid(value):
        raise UserError(f"Nonfinite JSON value: {value}")
    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)
    except (ValueError, TypeError) as exc:
        raise UserError(f"Invalid JSON: {exc}") from exc


def digest(value) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def normalized_messages(messages: list[dict]) -> list[dict]:
    """Preserve tool IDs/arguments and text through OpenAI→native serialization.

    Qwen's template expects function.arguments as a JSON object, whereas the
    OpenAI API transports it as a string. Normalize once; never eval model text.
    Reject unsupported multimodal content rather than silently dropping it.
    """
    if not isinstance(messages, list) or not messages:
        raise UserError("messages must be a nonempty list.")
    result = copy.deepcopy(messages)
    for message in result:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise UserError("Only system, user, assistant and tool text messages are supported.")
        content = message.get("content")
        if isinstance(content, list):
            if any(not isinstance(p, dict) or p.get("type") != "text" or not isinstance(p.get("text"), str) for p in content):
                raise UserError("This adapter accepts text-only messages.")
            content = "".join(p["text"] for p in content)
        if content is None and message["role"] == "assistant" and message.get("tool_calls"):
            content = ""
        if not isinstance(content, str):
            raise UserError("Message content must be text.")
        message["content"] = content
        for call in message.get("tool_calls", []):
            if call.get("type") != "function" or not isinstance(call.get("function"), dict):
                raise UserError("Only function tool calls are supported.")
            args = call["function"].get("arguments", {})
            if isinstance(args, str):
                args = loads(args)
            if not isinstance(args, dict):
                raise UserError("Function arguments must be a JSON object.")
            call["function"]["arguments"] = args
    return result


@dataclass
class Rendered:
    text: str
    ids: list[int]
    offsets: list[tuple[int, int]]


def render(tokenizer, messages: list[dict], tools=None) -> Rendered:
    normalized = normalized_messages(messages)
    text = tokenizer.apply_chat_template(normalized, tools=tools or None, tokenize=False,
                                         add_generation_prompt=True)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return Rendered(text, list(encoded["input_ids"]), [tuple(p) for p in encoded["offset_mapping"]])


def anchor(tokenizer, rendered: Rendered, spec: dict, *, end: int | None = None) -> dict:
    """Resolve an exact quoted occurrence to whole tokenizer rows.

    A source span is never a guessed token ID or a silent last-subtoken proxy.
    All its rows are disclosed and constrained (N1). The caller restricts 'end'
    to the original conversation so newly appended instructions cannot supply
    the supposedly pre-existing evidence.
    """
    if not isinstance(spec, dict) or set(spec) - {"quote", "focus", "occurrence", "file"}:
        raise UserError("Source anchors use quote, focus, occurrence and (for evidence) file.")
    quote, focus = spec.get("quote"), spec.get("focus")
    if not isinstance(quote, str) or not quote or len(quote) > 600 or not isinstance(focus, str) or not focus:
        raise UserError("Use an exact nonempty quote (at most 600 characters) and focus within it.")
    if quote.count(focus) != 1:
        raise UserError("The focus must occur exactly once inside its quote.")
    positions = [m.start() for m in re.finditer(re.escape(quote), rendered.text[:end])]
    occurrence = spec.get("occurrence")
    if not positions:
        raise UserError("The quoted source is not in the current model context. Read that file first.")
    if occurrence is None and len(positions) != 1:
        raise UserError(f"The quote occurs {len(positions)} times. Supply a 1-based occurrence.")
    occurrence = 1 if occurrence is None else occurrence
    if type(occurrence) is not int or not 1 <= occurrence <= len(positions):
        raise UserError("The source occurrence is outside the current context.")
    start = positions[occurrence - 1] + quote.index(focus)
    stop = start + len(focus)
    rows = [i for i, (a, b) in enumerate(rendered.offsets) if a < stop and b > start]
    if not rows or len(rows) > 8:
        raise UserError("Choose a focus spanning 1–8 tokens.")
    a, b = rendered.offsets[rows[0]][0], rendered.offsets[rows[-1]][1]
    if rendered.text[a:b].strip() != focus.strip():
        raise UserError(f"The focus cuts through a token. Use the complete token span {rendered.text[a:b]!r}.")
    return {"indices": rows, "pieces": [tokenizer.decode([rendered.ids[i]]) for i in rows],
            "character_range": [start, stop], "spec": spec}


# ========================================================================
# IO
# ========================================================================

"""Atomic, finite-valued experiment reports."""






def save_json(path, value):
    # Prepare JSON first: nonfinite or unserializable data never replaces a report.
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


# ========================================================================
# RANKING
# ========================================================================

"""Finite cache-edit scoring from captured tensors. No model calls in this file.

S1: exact local softmax response, including joint key/value interaction.
S2: its contraction with the baseline raw-head gradient is a downstream proxy.
S3: vectorized singleton search is linear in bank size, not one replay per token.
All algebra is FP64; native execution has a separate FP32 verification gate.
"""









@dataclass
class Bank:
    layer: int
    query: np.ndarray                 # H,Q,D; legacy H,D also accepted; post-RoPE
    keys: np.ndarray                  # G,N,D; physical GQA cache groups
    values: np.ndarray                # G,N,Dv
    gradient: np.ndarray              # H,Q,Dv; d objective / d raw head write
    frequencies: np.ndarray           # D/2, Qwen split-half rotary pairs
    scale: float
    mask: np.ndarray | None = None    # Q,N bool: True means causally attended
    item: int = 0                     # independent held-out continuation index

    def __post_init__(self):
        for name in ('query', 'keys', 'values', 'gradient', 'frequencies'):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if not np.isfinite(value).all():
                raise VerificationError(f'Nonfinite captured {name}.')
            setattr(self, name, value)
        q, k, v, g = self.query, self.keys, self.values, self.gradient
        if (q.ndim not in (2, 3) or k.ndim != 3 or v.ndim != 3 or g.ndim != q.ndim
                or k.shape[:2] != v.shape[:2] or k.shape[0] < 1 or k.shape[1] < 1
                or min(q.shape) < 1 or v.shape[2] < 1
                or q.shape[0] % k.shape[0] or q.shape[-1] != k.shape[2]
                or g.shape != (*q.shape[:-1], v.shape[2]) or q.shape[-1] % 2
                or self.frequencies.shape != (q.shape[-1] // 2,)
                or not math.isfinite(self.scale) or self.scale <= 0):
            raise VerificationError('Invalid captured head/GQA geometry.')
        if self.mask is None:
            self.mask = np.ones((self.query_count, self.n), dtype=bool)
        else:
            self.mask = np.asarray(self.mask)
        if (self.mask.dtype != np.bool_ or self.mask.shape != (self.query_count, self.n)
                or not self.mask.any(axis=-1).all()):
            raise VerificationError('Each query needs a boolean causal mask with a nonempty attended bank.')
        if type(self.item) is not int or self.item < 0:
            raise VerificationError('Invalid held-out item index.')

    @property
    def query_count(self):
        return self.query.shape[1] if self.query.ndim == 3 else 1

    @property
    def n(self):
        return self.keys.shape[1]

    @property
    def groups(self):
        return np.arange(self.query.shape[0]) // (self.query.shape[0] // self.keys.shape[0])


DEFAULT_TEMPLATES = [
    {'kind': 'reweight', 'factor': 2.0}, {'kind': 'reweight', 'factor': 0.5},
    # Include the finite odds range needed by the recorded repair in ordinary
    # searches as well as the case-specific catalog. Retain every edit family.
    *[{'kind': 'reweight', 'factor': factor} for factor in (4., 8., 16., 32., 64., 1/64)],
    {'kind': 'delete'}, {'kind': 'value_scale', 'factor': 0.0},
    *[{'kind': 'position', 'delta': delta, 'band': band}
      for band in ('fast', 'middle', 'slow') for delta in (-16.0, 16.0)],
]


def validate_rows(rows, n, *, deletion=False):
    if (not isinstance(rows, list) or not rows
            or any(type(i) is not int or not 0 <= i < n for i in rows)
            or len(rows) != len(set(rows))):
        raise UserError('Cache rows must be distinct, valid integer indices.')
    if deletion and len(rows) == n:
        raise UserError('Cannot delete the entire attended bank.')


def validate_template(template, n):
    if not isinstance(template, dict):
        raise UserError('Each candidate template must be an object.')
    kind = template.get('kind')
    fields = {'delete': set(), 'reweight': {'factor'}, 'position': {'delta', 'band'},
              'value_scale': {'factor'}, 'key': {'donor'}, 'value': {'donor'}, 'joint': {'donor'}}
    if kind not in fields or set(template) != fields[kind] | {'kind'}:
        raise UserError('Use delete, reweight, position, value_scale, key, value or joint with its declared fields.')
    if kind in {'reweight', 'value_scale'}:
        factor = template['factor']
        if type(factor) not in (int, float) or not math.isfinite(factor):
            raise UserError('The factor must be finite.')
        # A finite log-odds budget, not a small-perturbation assumption. The
        # expanded explicit range is exercised by the real repair study. The
        # exact exponential response and native local checks still apply.
        if kind == 'reweight' and not 1/64 <= factor <= 64:
            raise UserError('Reweight factors must be between 1/64 and 64.')
        if kind == 'value_scale' and not -2 <= factor <= 2:
            raise UserError('Value scaling must be between -2 and 2.')
    if kind == 'position':
        if (type(template['delta']) not in (int, float) or not math.isfinite(template['delta'])
                or abs(template['delta']) > 4096 or template['band'] not in {'full', 'fast', 'middle', 'slow'}):
            raise UserError('Use a finite positional displacement within ±4096 and a named RoPE band.')
    if kind in {'key', 'value', 'joint'} and (type(template['donor']) is not int or not 0 <= template['donor'] < n):
        raise UserError('A donor is one existing cache-row index, checked before device indexing.')


def rotate_keys(keys, frequencies, delta, band):
    """S4: R_b(delta)=exp(delta A_b), exact finite split-half RoPE rotation.

    Pair r is (r,r+D/2), not adjacent coordinates. Native inv_freq supplies
    frequencies; band thirds follow descending frequency, not a guessed base.
    This rotates cached keys. It does not move tokens or change causal order.
    """
    theta = np.zeros_like(frequencies)
    if band == 'full':
        selected = np.arange(len(theta))
    else:
        selected = np.array_split(np.argsort(-frequencies, kind='stable'), 3)[('fast', 'middle', 'slow').index(band)]
    theta[selected] = delta * frequencies[selected]
    c, s = np.cos(theta), np.sin(theta)
    a, b = np.split(keys, 2, axis=-1)
    return np.concatenate((c*a-s*b, s*a+c*b), axis=-1)


def logsumexp(x, axis=-1):
    maximum = np.max(x, axis=axis, keepdims=True)
    return (maximum + np.log(np.exp(x-maximum).sum(axis=axis, keepdims=True))).squeeze(axis)


class Scorer:
    def __init__(self, bank: Bank):
        self.bank = bank
        self.k, self.v = bank.keys[bank.groups], bank.values[bank.groups]
        self.s = np.einsum('h...d,hnd->h...n', bank.query, self.k) * bank.scale
        self.allowed = bank.mask if bank.query.ndim == 3 else bank.mask[0]
        self.s = np.where(self.allowed, self.s, -np.inf)
        self.logp = self.s - logsumexp(self.s)[..., None]
        self.p = np.exp(self.logp)
        self.y = np.einsum('h...n,hnd->h...d', self.p, self.v)
        self.a = np.einsum('h...d,hnd->h...n', bank.gradient, self.v)
        self.mean = np.sum(self.p*self.a, axis=-1)
        # S3: ordinary complement averages cost O(HN). Recompute only dominant
        # entries with a fresh log normalizer: 1-p may round to zero even though
        # the complement remains mathematically nonempty. At most one per head.
        remain = 1-self.p
        self.log_rest = np.full_like(remain, -np.inf)
        np.log(remain, out=self.log_rest, where=remain > 0)
        self.complement = np.divide(self.mean[..., None]-self.p*self.a, remain,
                                   out=np.zeros_like(remain), where=remain > 0)
        if bank.n > 1:
            for location in zip(*np.nonzero(remain < 1e-5)):
                prefix, j = location[:-1], location[-1]
                keep = np.arange(bank.n) != j
                scores = self.s[prefix]
                if not np.isfinite(scores[keep]).any():
                    continue
                total = logsumexp(scores[keep])
                self.log_rest[location] = total-logsumexp(scores)
                self.complement[location] = np.exp(scores[keep]-total) @ self.a[prefix][keep]

    def changes(self, template, indices=None):
        """Sparse score increments and value increments for the same joint edit."""
        b = self.bank
        validate_template(template, b.n)
        ix = np.arange(b.n) if indices is None else np.asarray(indices)
        k, v = self.k[:, ix], self.v[:, ix]
        ds = np.zeros(self.s[..., ix].shape); dv = np.zeros_like(v)
        kind = template['kind']
        if kind == 'delete':
            ds.fill(-np.inf)
        elif kind == 'reweight':
            ds.fill(math.log(template['factor']))
        elif kind == 'position':
            changed = rotate_keys(k, b.frequencies, template['delta'], template['band'])
            ds = np.einsum('h...d,hnd->h...n', b.query, changed-k) * b.scale
        elif kind == 'value_scale':
            dv = (template['factor']-1)*v
        else:
            donor = template['donor']
            if kind in {'key', 'joint'}:
                ds = np.einsum('h...d,hnd->h...n', b.query, self.k[:, donor:donor+1]-k) * b.scale
            if kind in {'value', 'joint'}:
                dv = self.v[:, donor:donor+1]-v
        return ds, dv

    def all_singletons(self, template):
        """S1–S3: score every one-token edit simultaneously, no dense replays.

        p'_j = sigmoid(log p_j - log(1-p_j) + ds_j).
        g·Δy = (p'_j-p_j)(g·v_j-g·y_-j) + p'_j g·dv_j.
        The second coefficient is p'_j, which retains the K/V cross term.
        Independent deletions are scored here, not simultaneous deletion of N.
        """
        if template.get('kind') == 'delete' and np.any(self.bank.mask.sum(axis=-1) == 1):
            raise UserError('A singleton attended bank cannot be deleted.')
        ds, dv = self.changes(template)
        changed = self.logp + ds
        denom = np.logaddexp(self.log_rest, changed)
        pnew = np.exp(changed-denom)
        gain = ((pnew-self.p)*(self.a-self.complement)
                + pnew*np.einsum('h...d,hnd->h...n', self.bank.gradient, dv))
        result = gain.sum(axis=tuple(range(gain.ndim-1)))
        if not np.isfinite(result).all():
            raise VerificationError('Nonfinite candidate scores; no ranking was returned.')
        return result

    def local(self, template, indices):
        """Exact group edit for verification. A span is evaluated jointly.

        Never add singleton scores to predict a multi-token deletion or edit:
        all changed entries share the new softmax normalizer.
        """
        b = self.bank
        validate_template(template, b.n)
        validate_rows(indices, b.n, deletion=template['kind'] == 'delete')
        if template['kind'] == 'delete':
            remaining = b.mask.copy()
            remaining[:, indices] = False
            if not remaining.any(axis=-1).all():
                raise UserError('An edit cannot delete every attended key of any query row.')
        ds, dv = self.changes(template, indices)
        scores, values = self.s.copy(), self.v.copy()
        scores[..., indices] += ds
        values[:, indices] += dv
        pnew = np.exp(scores-logsumexp(scores)[..., None])
        # Difference form is exactly zero for a zero edit and avoids subtracting
        # two complete readouts to obtain a small increment.
        delta = np.einsum('h...n,hnd->h...d', pnew-self.p, self.v)
        delta += np.einsum('h...n,hnd->h...d', pnew[..., indices], dv)
        prediction = float(np.sum(b.gradient * delta))
        return {'prediction': prediction, 'delta': delta, 'probabilities': pnew,
                'output': self.y+delta, 'scores': scores}


def rank_banks(banks, templates=None, spans=None, limit=20, cancel=None):
    """Return global top candidates plus dense score ARRAYS for offline replay.

    Memory for ranked dictionaries is O(limit), not O(layers×templates×tokens).
    Scores use the same frozen baseline throughout. They are not additive gains
    for greedy multi-edit selection, which needs re-scoring after each edit.
    """
    if type(limit) is not int or not 1 <= limit <= 200:
        raise UserError('Return between 1 and 200 ranked candidates.')
    templates = DEFAULT_TEMPLATES if templates is None else templates
    if not isinstance(templates, list) or not 1 <= len(templates) <= 32:
        raise UserError('Use between 1 and 32 edit templates.')
    if spans is not None and (not isinstance(spans, list) or not 1 <= len(spans) <= 64):
        raise UserError('Use up to 64 explicit spans, or omit spans to score every token.')
    top, arrays, total = [], {}, 0
    for bank in banks:
        if cancel is not None and cancel.is_set():
            raise UserError('Ranking cancelled.')
        scorer = Scorer(bank)
        for t, template in enumerate(templates):
            validate_template(template, bank.n)
            if spans is None:
                scores = scorer.all_singletons(template)
            else:
                scores = np.array([scorer.local(template, rows)['prediction'] for rows in spans])
            arrays[f'layer_{bank.layer}_template_{t}'] = scores
            total += scores.size
            # Stable ties use token/span order, then layer/template order below.
            for row in np.argsort(-scores, kind='stable')[:limit]:
                top.append({'layer': bank.layer, 'template_index': t, 'span_index': int(row),
                    'indices': [int(row)] if spans is None else spans[row], 'edit': dict(template),
                    'predicted_margin_change': float(scores[row])})
            top.sort(key=lambda r: (-r['predicted_margin_change'], r['layer'], r['template_index'], r['span_index']))
            del top[limit:]
    for i, row in enumerate(top):
        row['rank'] = i+1
    return top, arrays, total


# ========================================================================
# PRECISION
# ========================================================================

"""Exact greedy cache restoration against a supplied reference write.

P1–P4 correspond to the paper's single-entry restoration and repair-benefit
identities. This is a local, reference-based repair; it does not infer a correct
answer, recover an unavailable original cache, or promise a global optimum.
"""







def _read(bank):
    scorer = Scorer(bank)
    return scorer.s, scorer.logp, scorer.p, scorer.y, scorer.v


def restoration_deltas(bank, reference_keys, reference_values):
    """P1: every singleton restoration's exact raw-head delta, shape (N,H,Dv).

    For each j independently, p'_j=exp(log p_j+ds_j)/
    (exp(log(1-p_j))+exp(log p_j+ds_j)), and
    Δy_j=(p'_j-p_j)(v_j-y_-j)+p'_j Δv_j.
    The p'_j coefficient preserves the joint K/V interaction. Complement
    normalizers are recomputed for dominant tokens rather than trusting 1-p.
    Only one row is restored in each candidate, across all physical KV groups.
    """
    rk, rv = np.asarray(reference_keys, dtype=float), np.asarray(reference_values, dtype=float)
    if (rk.shape != bank.keys.shape or rv.shape != bank.values.shape
            or not np.isfinite(rk).all() or not np.isfinite(rv).all()):
        raise UserError('Reference K/V must be finite and match the physical cache shapes.')
    scores, logp, p, y, values = _read(bank)
    remain = 1-p
    log_rest = np.full_like(p, -np.inf)
    np.log(remain, out=log_rest, where=remain > 0)
    complement = np.divide(y[:, None, :]-p[:, :, None]*values, remain[:, :, None],
                           out=np.zeros_like(values), where=remain[:, :, None] > 0)
    if bank.n > 1:
        for h, j in zip(*np.nonzero(remain < 1e-5)):
            keep = np.arange(bank.n) != j
            normalizer = logsumexp(scores[h, keep])
            log_rest[h, j] = normalizer-logsumexp(scores[h])
            complement[h, j] = np.exp(scores[h, keep]-normalizer) @ values[h, keep]
    ds = np.einsum('hd,hnd->hn', bank.query, (rk-bank.keys)[bank.groups])*bank.scale
    dv = (rv-bank.values)[bank.groups]
    changed = logp+ds
    pnew = np.exp(changed-np.logaddexp(log_rest, changed))
    delta = (pnew-p)[:, :, None]*(values-complement)+pnew[:, :, None]*dv
    if not np.isfinite(delta).all():
        raise VerificationError('Nonfinite restoration deltas.')
    return delta.transpose(1, 0, 2)


def restore_cache(bank: Bank, reference_keys, reference_values, *, projection=None, budget=4):
    """Return a checked greedy plan and repaired copies, without mutating inputs.

    projection is W_o with shape (output_dim, H*Dv), e.g. the native PyTorch
    o_proj.weight. Omit it to measure the concatenated raw-head write. The same
    query, mask and reference apply throughout. A shared output bias cancels.
    budget counts distinct physical token rows, not individual head entries.
    """
    if type(budget) is not int or not 0 <= budget <= bank.n:
        raise UserError('Restoration budget must be an integer between zero and the bank size.')
    rk, rv = np.asarray(reference_keys, dtype=float), np.asarray(reference_values, dtype=float)
    if (rk.shape != bank.keys.shape or rv.shape != bank.values.shape
            or not np.isfinite(rk).all() or not np.isfinite(rv).all()):
        raise UserError('Reference K/V must be finite and match the physical cache shapes.')
    width = bank.query.shape[0]*bank.values.shape[-1]
    if projection is None:
        project = lambda x: x
    else:
        matrix = np.asarray(projection, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != width or matrix.shape[0] < 1 or not np.isfinite(matrix).all():
            raise UserError('Projection must have shape (output_dim, H*Dv) and finite entries.')
        project = lambda x: x @ matrix.T
    # P2: concatenate all heads, then project BEFORE taking a squared norm.
    # Summing per-head squared errors would discard output-projection cross terms.
    current = replace(bank, keys=bank.keys.copy(), values=bank.values.copy())
    reference = replace(bank, keys=rk.copy(), values=rv.copy())
    target = project(_read(reference)[3].reshape(-1))
    write = project(_read(current)[3].reshape(-1))
    initial_error = float(np.sum((write-target)**2))
    history, selected = [], []
    for step in range(budget):
        error = write-target
        deltas = project(restoration_deltas(current, rk, rv).reshape(bank.n, width))
        # P3: B_j=||e||²-||e+Δo_j||², evaluated without cancellation of norms.
        benefits = -2*(deltas @ error)-np.einsum('nd,nd->n', deltas, deltas)
        if not np.isfinite(benefits).all():
            raise VerificationError('Nonfinite restoration benefits.')
        if selected:
            benefits[selected] = -np.inf
        j = int(np.argmax(benefits))  # stable tie: lowest row index
        if benefits[j] <= 0:
            break
        before = float(error @ error)
        current.keys[:, j] = rk[:, j]
        current.values[:, j] = rv[:, j]
        # P4: independent dense execution checks the chosen candidate each step.
        # Recompute all candidate scores on this new state on the next iteration.
        actual = project(_read(current)[3].reshape(-1))
        if not np.allclose(actual-write, deltas[j], atol=1e-10, rtol=1e-9):
            raise VerificationError('Sparse restoration disagrees with dense attention.')
        after = float(np.sum((actual-target)**2))
        if after > before + 1e-10*max(1., before):
            raise VerificationError('A selected restoration increased the measured error.')
        history.append({'step': step+1, 'row': j, 'predicted_benefit': float(benefits[j]),
                        'actual_benefit': before-after, 'squared_error': after,
                        'max_local_delta_error': float(np.max(np.abs(actual-write-deltas[j])))})
        selected.append(j)
        write = actual
    return {'keys': current.keys, 'values': current.values, 'selected': selected,
            'initial_squared_error': initial_error, 'final_squared_error': float(np.sum((write-target)**2)),
            'history': history, 'scope': 'fixed-query local reference-write restoration',
            'stopped': 'budget_exhausted' if len(selected) == budget else 'no_positive_singleton_benefit'}


# ========================================================================
# NATIVE
# ========================================================================

"""Cached autoregressive Qwen2 execution; mathematical contracts N1–N5.

The native projection, RoPE, cache update and output projection follow
transformers 4.57.3's Qwen2Attention (Apache-2.0; see THIRD_PARTY_NOTICES.md).
Only a selected post-RoPE query is replaced. Each generation owns its cache.
"""












def eager_rows(native, module, q, k, v, mask, chunk_size=128):
    """N6: exact row partitioning, with bounded prefill probability storage.

    Softmax normalizes over KEYS independently for every query row. Partition
    query rows only; never partition the key normalizer. All native primitives
    remain FP32/eager. Return the full output and only the last probability row
    used by this instrument. Rounding can differ with GEMM shape and is tested.
    """
    import torch
    raw, last = [], None
    for start in range(0, q.shape[2], chunk_size):
        stop = min(start + chunk_size, q.shape[2])
        part_mask = mask[:, :, start:stop, :] if mask is not None else None
        output, weights = native.eager_attention_forward(module, q[:, :, start:stop], k, v, part_mask,
            dropout=0.0, scaling=module.scaling, sliding_window=module.sliding_window)
        raw.append(output)
        last = weights[:, :, -1:, :]
    return torch.cat(raw, dim=1), last


@contextmanager
def chunked_prefill(model):
    """Instance-local native adapters; avoid full H×N×N prefill storage.

    Research prompts can exceed 10k tokens before an experiment
    starts. This wrapper bounds per-head softmax storage without changing
    positions, the cache, the key bank or the softmax denominator. It is only
    used by this instrument, whose consumers require the last attention row.
    """
    from transformers.models.qwen2 import modeling_qwen2 as native
    previous = []
    def forward(module, hidden_states, position_embeddings, attention_mask,
                past_key_values=None, cache_position=None, **kwargs):
        shape = (*hidden_states.shape[:-1], -1, module.head_dim)
        q = module.q_proj(hidden_states).view(shape).transpose(1, 2)
        k = module.k_proj(hidden_states).view(shape).transpose(1, 2)
        v = module.v_proj(hidden_states).view(shape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = native.apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, module.layer_idx,
                                         {"sin": sin, "cos": cos, "cache_position": cache_position})
        raw, probabilities = eager_rows(native, module, q, k, v, attention_mask)
        return module.o_proj(raw.reshape(*hidden_states.shape[:-1], -1).contiguous()), probabilities
    try:
        for layer in model.model.layers:
            module = layer.self_attn
            previous.append((module, "forward" in module.__dict__, module.__dict__.get("forward")))
            module.forward = types.MethodType(forward, module)
        yield
    finally:
        for module, had_local, old in reversed(previous):
            if had_local:
                module.forward = old
            else:
                delattr(module, "forward")


class Refused(UserError):
    """A requested edit was not feasible within the declared protocol."""


@dataclass
class EditRecipe:
    good: list[int]
    bad: list[int]
    boost: float = 4.0
    window: int = 8
    budget: float = 5.0
    layer: int | None = None
    head: int | None = None
    preserve: list[tuple[int, int]] = field(default_factory=list)

    def validate(self, n: int) -> None:
        if any(value is not None and (type(value) is not int or value < 0) for value in (self.layer, self.head)):
            raise Refused("Layer and head must be nonnegative integer indices.")
        if (not self.good or not self.bad or max(len(self.good), len(self.bad)) > 8
                or len(set(self.good + self.bad)) != len(self.good + self.bad)):
            raise Refused("Select disjoint nonempty source spans of at most eight tokens each.")
        rows = self.good + self.bad + [i for pair in self.preserve for i in pair]
        if any(type(i) is not int or not 0 <= i < n for i in rows):
            raise Refused("An attention anchor is outside the current prompt.")
        if not math.isfinite(self.boost) or not 1 <= self.boost <= 16:
            raise Refused("Attention boost must be between 1 and 16.")
        if type(self.window) is not int or not 1 <= self.window <= 16:
            raise Refused("The edited generation window must be 1–16 tokens.")
        if not math.isfinite(self.budget) or not 0 <= self.budget <= 20:
            raise Refused("The per-query edit budget must be between 0 and 20.")


def span_constraints(snapshot: Snapshot, recipe: EditRecipe) -> list[dict]:
    """N1: exact span odds by retaining each span's internal distribution.

    log(p_j/p_k) = (a_j-a_k)^T q. Preserve all ratios within G and B;
    increase log(p_g/p_b) by log(boost) for one anchor in each. Consequently
    every cross-span odds ratio, AND sum_G p / sum_B p, increases by boost.
    This is a linear constraint system. The solver is minimum-norm subject
    to these additional within-span constraints, not to span totals alone.
    """
    good, bad = recipe.good, recipe.bad
    constraints = [{"numerator": good[0], "denominator": bad[0],
                    "log_odds": snapshot.log_odds(good[0], bad[0]) + math.log(recipe.boost)}]
    for rows in (good, bad):
        constraints.extend({"numerator": j, "denominator": rows[0], "preserve": True}
                           for j in rows[1:])
    constraints.extend({"numerator": a, "denominator": b, "preserve": True}
                       for a, b in recipe.preserve)
    return constraints


class CachedEditor:
    def __init__(self, model, recipe: EditRecipe, prompt_length: int):
        recipe.validate(prompt_length)
        if recipe.layer is None or recipe.head is None:
            raise Refused("Choose a head from baseline observations before editing.")
        # Reuse the already validated architecture and FP32 gates, not its
        # full-prompt-only forward implementation.
        validator = QwenAdapter(model, recipe.layer, recipe.head, prompt_length - 1)
        self.model, self.recipe, self.prompt_length = model, recipe, prompt_length
        self.torch, self.native, self.module = validator.torch, validator.native, validator.module
        self.audits: list[dict] = []

    @contextmanager
    def installed(self):
        module = self.module
        had_local, previous = "forward" in module.__dict__, module.__dict__.get("forward")
        owner = self
        def forward(mod, hidden_states, position_embeddings, attention_mask,
                    past_key_values=None, cache_position=None, **kwargs):
            return owner.forward(hidden_states, position_embeddings, attention_mask,
                                 past_key_values, cache_position, **kwargs)
        module.forward = types.MethodType(forward, module)
        try:
            yield self
        finally:
            if had_local:
                module.forward = previous
            else:
                delattr(module, "forward")

    def forward(self, hidden, position_embeddings, mask, cache, cache_position, **kwargs):
        t, m, native, recipe = self.torch, self.module, self.native, self.recipe
        if hidden.shape[0] != 1 or hidden.shape[1] < 1:
            raise VerificationError("Cached editing requires one unpadded sequence.")
        shape = (*hidden.shape[:-1], -1, m.head_dim)
        q = m.q_proj(hidden).view(shape).transpose(1, 2)
        k = m.k_proj(hidden).view(shape).transpose(1, 2)
        v = m.v_proj(hidden).view(shape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = native.apply_rotary_pos_emb(q, k, cos, sin)
        # N2: update the current layer's cache BEFORE the query replacement.
        # K and V do not depend on this layer's Q. Higher layers then consume
        # the changed write and maintain their own, genuinely changed caches.
        if cache is not None:
            k, v = cache.update(k, v, m.layer_idx,
                               {"sin": sin, "cos": cos, "cache_position": cache_position})
        n = k.shape[2]
        step = n - self.prompt_length
        if step < 0 or (step == 0 and hidden.shape[1] != n) or (step > 0 and hidden.shape[1] != 1):
            raise VerificationError("Unexpected prefill/decode order.")
        if cache_position is None or int(cache_position[-1].item()) != n - 1:
            raise VerificationError("Cache positions do not match the captured bank.")
        plan = snap = None
        cast_norm = 0.0
        if step < recipe.window:
            if step != len(self.audits):
                raise VerificationError("A query was repeated or skipped during the edit window.")
            if mask is not None:
                if mask.ndim != 4 or mask.shape[1] != 1 or mask.shape[-1] < n:
                    raise VerificationError("Unexpected native mask shape.")
                row = mask[0, 0, -1, :n]
                if not bool((row == 0).all()):
                    raise VerificationError("The selected last query must see the full causal bank.")
            elif step == 0 and n > 1:
                raise VerificationError("A multi-token prefill must have the native causal mask.")
            head, kv = recipe.head, recipe.head // m.num_key_value_groups
            before = to_numpy(q[0, head, -1])
            keys, values = to_numpy(k[0, kv]), to_numpy(v[0, kv])
            snap = Snapshot(before, keys * m.scaling, values)
            # N3: re-solve from the CURRENT query and CURRENT keys at every
            # generation step. No vector is reused across positions or prompts.
            plan = solve_edit(snap, span_constraints(snap, recipe), budget=recipe.budget)
            if not plan["can_apply"]:
                raise Refused(f"Generation step {step}: attention edit {plan['status']}.")
            replacement = t.as_tensor(plan["candidate_query"], dtype=q.dtype, device=q.device)
            cast_norm = float(np.linalg.norm(to_numpy(replacement) - before))
            if not math.isfinite(cast_norm) or cast_norm > recipe.budget:
                raise Refused("Casting the query exceeds the edit budget.")
            q = q.clone()
            q[0, head, -1] = replacement  # The ONLY native intervention assignment.
        raw, probabilities = eager_rows(native, m, q, k, v, mask)
        if plan is not None:
            head, kv = recipe.head, recipe.head // m.num_key_value_groups
            # N4: read actual native probabilities and raw head output AFTER
            # execution. The solver's own replay is not this verification.
            scores = to_numpy((q[0, head, -1] @ k[0, kv].T) * m.scaling)
            p = to_numpy(probabilities[0, head, -1, :n])
            y = to_numpy(raw[0, -1, head])
            audit = verify_backend_result(plan, scores, y, atol=NATIVE_ATOL, rtol=NATIVE_RTOL)
            assert_close(p, plan["edited_probabilities"], "Native generation probabilities")
            if not audit["verified"]:
                raise VerificationError(f"Generation step {step}: native verification failed: {audit}")
            self.audits.append({"step": step, "query_index": n - 1, "bank_size": n,
                "layer": recipe.layer, "head": head, "kv_head": kv,
                "edit_norm": cast_norm, "budget": recipe.budget, "audit": audit,
                "good_mass_before": float(snap.probabilities[recipe.good].sum()),
                "bad_mass_before": float(snap.probabilities[recipe.bad].sum()),
                "good_mass_after": float(p[recipe.good].sum()),
                "bad_mass_after": float(p[recipe.bad].sum()),
                "snapshot": {"query": snap.query.tolist(), "features": snap.features.tolist(),
                             "values": snap.values.tolist()},
                "plan": plan, "native_scores": scores.tolist(), "native_output": y.tolist(),
                "native_probabilities": p.tolist()})
        return m.o_proj(raw.reshape(*hidden.shape[:-1], -1).contiguous()), probabilities


def choose_head(model, ids: list[int], recipe: EditRecipe) -> tuple[EditRecipe, list[dict]]:
    """Choose the greatest baseline attention mass on the selected source spans.

    This observational heuristic is recorded and precedes every candidate
    outcome. It is not a causal diagnosis and may choose an unhelpful head.
    """
    import torch
    recipe.validate(len(ids))
    QwenAdapter(model, recipe.layer or 0, recipe.head or 0, len(ids) - 1)
    rows, handles = [], []
    def callback(layer):
        def hook(module, args, result):
            p = result[1]
            for head in range(p.shape[1]):
                if recipe.head is None or recipe.head == head:
                    good = float(p[0, head, -1, recipe.good].sum())
                    bad = float(p[0, head, -1, recipe.bad].sum())
                    rows.append({"layer": layer, "head": head, "mass": good + bad,
                                 "good_mass": good, "bad_mass": bad})
        return hook
    try:
        for layer, block in enumerate(model.model.layers):
            if recipe.layer is None or layer == recipe.layer:
                handles.append(block.self_attn.register_forward_hook(callback(layer)))
        with precise_inference(torch), chunked_prefill(model):
            model(input_ids=torch.tensor([ids], device=model.device), use_cache=False, logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()
    rows.sort(key=lambda x: (-x["mass"], x["layer"], x["head"]))
    if not rows:
        raise Refused("No heads meet the requested selection.")
    from dataclasses import replace
    return replace(recipe, layer=rows[0]["layer"], head=rows[0]["head"]), rows


def generate_ids(model, ids: list[int], *, max_tokens: int, eos_ids: set[int],
                 temperature: float = 0, top_p: float = 1, seed: int = 0,
                 recipe: EditRecipe | None = None, cache_edit=None, cancel=None, on_token=None, on_logits=None) -> dict:
    """N5: true autoregressive sampling, with a fresh DynamicCache per branch.

    Generation after the short edit window is native. Earlier edits remain in
    the higher-layer cache exactly as executed; there is no cache 'restoration'
    that would erase their consequences. Local checks never certify the text.
    """
    import torch
    from transformers import DynamicCache
    from contextlib import nullcontext
    if not ids or any(type(i) is not int or not 0 <= i < model.config.vocab_size for i in ids):
        raise Refused("Invalid prompt token IDs.")
    if type(max_tokens) is not int or max_tokens < 1:
        raise Refused("max_tokens must be a positive integer.")
    if not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1:
        raise Refused("Invalid sampling parameters.")
    if recipe is not None and cache_edit is not None:
        raise Refused('Choose a query recipe or a ranked cache edit, not both.')
    if cache_edit is not None:
        editor = CacheEditor(model, cache_edit, len(ids))
    else:
        editor = CachedEditor(model, recipe, len(ids)) if recipe else None
    cache = DynamicCache(config=model.config)
    generator = torch.Generator(device=model.device).manual_seed(seed)
    produced = []
    input_ids = torch.tensor([ids], device=model.device)
    reason = "length"
    with precise_inference(torch), chunked_prefill(model), (editor.installed() if editor else nullcontext()):
        for step in range(max_tokens):
            if cancel is not None and cancel.is_set():
                raise Refused("Generation cancelled; its cache was discarded.")
            start = 0 if step == 0 else len(ids) + step - 1
            positions = torch.arange(start, len(ids) + step, device=model.device)
            out = model(input_ids=input_ids, past_key_values=cache, cache_position=positions,
                        use_cache=True, logits_to_keep=1)
            logits = out.logits[0, -1].float()
            if not bool(torch.isfinite(logits).all()):
                raise VerificationError("Nonfinite generation logits.")
            if on_logits:
                on_logits(step, to_numpy(logits))
            if temperature == 0:
                token = int(logits.argmax())
            else:
                p = torch.softmax(logits / temperature, dim=-1)
                if top_p < 1:
                    sorted_p, order = torch.sort(p, descending=True)
                    remove = sorted_p.cumsum(-1) - sorted_p >= top_p
                    sorted_p[remove] = 0
                    p = torch.zeros_like(p).scatter(0, order, sorted_p)
                token = int(torch.multinomial(p, 1, generator=generator))
            produced.append(token)
            if on_token:
                on_token(produced)
            if token in eos_ids:
                reason = "stop"
                break
            input_ids = torch.tensor([[token]], device=model.device)
    return {"token_ids": produced, "finish_reason": reason,
            "audits": editor.audits if editor else []}


# ========================================================================
# MARGIN
# ========================================================================

"""One cached baseline + one backward pass, then a few native edit checks.

S5: g_h = d margin / d raw head write. This already includes W_o and all
downstream derivatives. Contracting g_h with Δy_h must NOT apply W_o again.
S6: detached prefix caches are valid because causal past rows cannot depend on
an intervention at the current final query. Parameters and their grads stay put.
"""



def validate_ids(model, ids, prefer=None, avoid=None):
    if not ids or any(type(i) is not int or not 0 <= i < model.config.vocab_size for i in ids):
        raise UserError('Invalid prompt token IDs.')
    if len(ids) > model.config.max_position_embeddings:
        raise UserError('Prompt exceeds the model position limit.')
    if prefer is not None or avoid is not None:
        if (any(type(i) is not int or not 0 <= i < model.config.vocab_size for i in (prefer, avoid))
                or prefer == avoid):
            raise UserError('Margin targets must be two distinct valid token IDs.')


def qkv(module, hidden, positions, cache, cache_position, native):
    shape = (*hidden.shape[:-1], -1, module.head_dim)
    q = module.q_proj(hidden).view(shape).transpose(1, 2)
    k = module.k_proj(hidden).view(shape).transpose(1, 2)
    v = module.v_proj(hidden).view(shape).transpose(1, 2)
    cos, sin = positions
    q, k = native.apply_rotary_pos_emb(q, k, cos, sin)
    if cache is not None:
        k, v = cache.update(k, v, module.layer_idx,
                            {'sin': sin, 'cos': cos, 'cache_position': cache_position})
    return q, k, v


def last_mask(mask, n):
    if mask is None:
        return None
    if mask.ndim != 4 or mask.shape[1] != 1 or mask.shape[-1] < n:
        raise VerificationError('Unexpected causal mask geometry.')
    row = mask[:, :, -1:, :n]
    if not bool((row == 0).all()):
        raise UserError('Ranking requires the unpadded final query and its full causal bank.')
    return row


@contextmanager
def patched_forward(module, function):
    had_local, old = 'forward' in module.__dict__, module.__dict__.get('forward')
    module.forward = types.MethodType(function, module)
    try:
        yield
    finally:
        if had_local:
            module.forward = old
        else:
            delattr(module, 'forward')


HELD_OUT_LOG_LIKELIHOOD = 'held_out_log_likelihood'
MARGIN_OBJECTIVE = 'first_distinct_next_token_logit_margin'


@dataclass
class ObjectiveCapture:
    banks: list[Bank]
    ids: list[int]
    prefer: int | None
    avoid: int | None
    value: float
    logits: np.ndarray | None
    prefix_cache: tuple             # detached native tensors, never mutated
    checks: list[dict]
    objective: dict = field(default_factory=lambda: {'kind': MARGIN_OBJECTIVE})
    item_results: list[dict] = field(default_factory=list)
    counts: dict = field(default_factory=dict)

    @property
    def margin(self):
        """Compatibility for next-token margin callers."""
        if self.objective['kind'] != MARGIN_OBJECTIVE:
            raise UserError('A likelihood capture has a value/loss, not a prefer/avoid margin.')
        return self.value

    @property
    def loss(self):
        if self.objective['kind'] != HELD_OUT_LOG_LIKELIHOOD:
            raise UserError('Loss is defined only for held-out likelihood captures.')
        return -self.value

    def fresh_cache(self, model):
        from transformers import DynamicCache
        return DynamicCache([(k.clone(), v.clone()) for k, v in self.prefix_cache], config=model.config)

    def save(self, path):
        """Portable capture has no PyTorch pickle or model object; allow_pickle=False."""
        if self.objective['kind'] == MARGIN_OBJECTIVE:
            # Keep existing margin artifacts byte-schema compatible.
            arrays = {'ids': np.array(self.ids), 'prefer': np.array(self.prefer), 'avoid': np.array(self.avoid),
                      'margin': np.array(self.margin), 'layers': np.array([b.layer for b in self.banks])}
            for bank in self.banks:
                for key in ('query', 'keys', 'values', 'gradient', 'frequencies', 'scale'):
                    arrays[f'layer_{bank.layer}_{key}'] = np.asarray(getattr(bank, key))
            np.savez_compressed(path, **arrays)
            return
        arrays = {'schema_version': np.array(2), 'ids': np.array(self.ids),
                  'objective': np.array(json.dumps(self.objective, allow_nan=False)),
                  'value': np.array(self.value), 'bank_count': np.array(len(self.banks)),
                  'item_results': np.array(json.dumps(self.item_results, allow_nan=False)),
                  'counts': np.array(json.dumps(self.counts, allow_nan=False))}
        for i, bank in enumerate(self.banks):
            for key in ('layer', 'item', 'query', 'keys', 'values', 'gradient', 'frequencies', 'scale', 'mask'):
                arrays[f'bank_{i}_{key}'] = np.asarray(getattr(bank, key))
        np.savez_compressed(path, **arrays)


MarginCapture = ObjectiveCapture


def load_banks(path):
    with np.load(path, allow_pickle=False) as data:
        if 'bank_count' in data:
            return [Bank(layer=int(data[f'bank_{i}_layer']), item=int(data[f'bank_{i}_item']),
                **{key: data[f'bank_{i}_{key}'].copy()
                   for key in ('query', 'keys', 'values', 'gradient', 'frequencies', 'mask')},
                scale=float(data[f'bank_{i}_scale'])) for i in range(int(data['bank_count']))]
        # Version-one single-query captures remain readable.
        return [Bank(layer=int(layer), **{key: data[f'layer_{layer}_{key}'].copy()
            for key in ('query', 'keys', 'values', 'gradient', 'frequencies')},
            scale=float(data[f'layer_{layer}_scale'])) for layer in data['layers']]


def causal_query_mask(mask, positions, n):
    """Validate native full-causal, batch-one rows; retain future-key exclusions."""
    positions = to_numpy(positions).astype(np.int64)
    allowed = np.arange(n)[None, :] <= positions[:, None]
    if mask is None:
        if not allowed.all():
            raise VerificationError('Missing causal mask for a multi-query forward.')
        return None, allowed
    if mask.shape != (1, 1, len(positions), n):
        raise VerificationError('Unexpected teacher-forced causal mask geometry.')
    actual = to_numpy(mask[0, 0])
    if (not np.array_equal(actual == 0, allowed)
            or np.any(actual[~allowed] > -1e20) or np.isnan(actual).any()):
        raise VerificationError('Teacher-forced rows must use the native unpadded causal mask.')
    return mask, allowed


def teacher_forced_log_probs(logits, token_ids):
    """Each row predicts exactly its supplied target, including the final token."""
    import torch
    if logits.ndim != 2 or logits.shape[0] != len(token_ids) or not token_ids:
        raise VerificationError('Teacher-forced targets and query rows do not align.')
    if not bool(torch.isfinite(logits).all()):
        raise VerificationError('Nonfinite teacher-forced logits.')
    targets = torch.tensor(token_ids, dtype=torch.long, device=logits.device)
    return torch.log_softmax(logits, dim=-1).gather(1, targets[:, None]).squeeze(1)


def _capture_specification(model, ids, objective):
    validate_ids(model, ids)
    if not isinstance(objective, dict):
        raise UserError('The capture objective must be an object with a kind.')
    result = copy.deepcopy(objective)
    if result.get('kind') == MARGIN_OBJECTIVE:
        validate_ids(model, ids, result.get('prefer_id'), result.get('avoid_id'))
        if any(type(result.get(k)) is not int for k in ('prefer_id', 'avoid_id')):
            raise UserError('Margin targets must be distinct valid token IDs.')
    elif result.get('kind') == HELD_OUT_LOG_LIKELIHOOD:
        items = result.get('items')
        if not isinstance(items, list) or not 1 <= len(items) <= 64:
            raise UserError('Supply 1–64 independent held-out continuations.')
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get('token_ids'), list) or not item['token_ids']:
                raise UserError('Each held-out item needs a nonempty token_ids list.')
            validate_ids(model, ids + item['token_ids'])
            weight = item.setdefault('weight', 1.0)
            if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
                raise UserError('Held-out weights must be positive finite numbers.')
        if result.get('reduction', 'sum') != 'sum':
            raise UserError('Held-out likelihood uses a weighted sum, without length normalization.')
        result['reduction'] = 'sum'
    else:
        raise UserError('Use held_out_log_likelihood or first_distinct_next_token_logit_margin.')
    return result


def capture_objective(model, ids, objective, layers=None, cancel=None):
    """One capture and ONE backward for all layers, query rows and held-out items.

    A shared no-grad prefix is cloned for each independent continuation. Item i
    feeds [last_prompt_token] + target_i[:-1]; every row predicts target_i[t].
    autograd.grad(sum_i weight_i * log P(target_i | prompt), all zero leaves)
    supplies the multi-query gradients without allocating parameter gradients.
    Counts distinguish one capture/backward from the item forwards it contains.
    """
    import torch
    from transformers import DynamicCache
    from transformers.models.qwen2 import modeling_qwen2 as native
    from contextlib import ExitStack
    objective = _capture_specification(model, ids, objective)
    likelihood = objective['kind'] == HELD_OUT_LOG_LIKELIHOOD
    items = objective['items'] if likelihood else [{'token_ids': [], 'weight': 1.0}]
    QwenAdapter(model, 0, 0, len(ids)-1)
    count = len(model.model.layers)
    layers = list(range(count)) if layers is None else layers
    if (not isinstance(layers, list) or not layers or len(set(layers)) != len(layers)
            or any(type(i) is not int or not 0 <= i < count for i in layers)):
        raise UserError('Choose distinct valid layer indices, or omit layers to capture all.')
    layers = sorted(layers)
    previous = [(p, p.requires_grad) for p in model.parameters()]
    tf32 = torch.backends.cuda.matmul.allow_tf32
    records, leaves, item_results, objectives = {}, {}, [], []
    current_item, query_count = 0, 1
    def forward(module, hidden_states, position_embeddings, attention_mask,
                past_key_values=None, cache_position=None, **kwargs):
        if cancel is not None and cancel.is_set():
            raise UserError('Objective capture cancelled.')
        if hidden_states.shape[:2] != (1, query_count):
            raise VerificationError('Unexpected objective query geometry.')
        q, k, v = qkv(module, hidden_states, position_embeddings, past_key_values, cache_position, native)
        mask, allowed = causal_query_mask(attention_mask, cache_position, k.shape[2])
        raw, p = native.eager_attention_forward(module, q, k, v, mask,
            dropout=0.0, scaling=module.scaling, sliding_window=module.sliding_window)
        delta = torch.zeros_like(raw, requires_grad=True)
        key = (current_item, module.layer_idx)
        leaves[key] = delta
        records[key] = (to_numpy(q[0]), to_numpy(k[0]), to_numpy(v[0]),
                        to_numpy(raw[0].transpose(0, 1)), to_numpy(p[0]), allowed)
        # The zero leaf has identity Jacobian to the raw attention write. Keep
        # upstream dependence too; detach-and-replace would sever earlier grads.
        raw = raw + delta
        return module.o_proj(raw.reshape(1, query_count, -1).contiguous()), p
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        for parameter, _ in previous:
            parameter.requires_grad_(False)
        with torch.inference_mode(False), chunked_prefill(model):
            cache = DynamicCache(config=model.config)
            with torch.no_grad():
                if len(ids) > 1:
                    model(input_ids=torch.tensor([ids[:-1]], device=model.device), past_key_values=cache,
                          use_cache=True, logits_to_keep=1)
            prefix = tuple((k.detach(), v.detach()) for k, v in cache.to_legacy_cache()) if len(ids) > 1 else ()
            with torch.enable_grad(), ExitStack() as stack:
                for layer in layers:
                    stack.enter_context(patched_forward(model.model.layers[layer].self_attn, forward))
                logits_value = None
                for current_item, item in enumerate(items):
                    if cancel is not None and cancel.is_set():
                        raise UserError('Objective capture cancelled.')
                    forced = [ids[-1]] + (item['token_ids'][:-1] if likelihood else [])
                    query_count = len(forced)
                    positions = torch.arange(len(ids)-1, len(ids)-1+query_count, device=model.device)
                    fresh = DynamicCache([(k.clone(), v.clone()) for k, v in prefix], config=model.config)
                    out = model(input_ids=torch.tensor([forced], device=model.device), past_key_values=fresh,
                        cache_position=positions, use_cache=True, logits_to_keep=0)
                    logits = out.logits[0]
                    if likelihood:
                        log_probs = teacher_forced_log_probs(logits, item['token_ids'])
                        score = log_probs.sum()
                        item_results.append({'item': current_item, 'token_ids': item['token_ids'],
                            'weight': item['weight'], 'query_indices': positions.tolist(),
                            'token_log_probabilities': to_numpy(log_probs).tolist(),
                            'log_likelihood': float(score.detach()),
                            'weighted_log_likelihood': float((score*item['weight']).detach())})
                        objectives.append(score*item['weight'])
                    else:
                        if not bool(torch.isfinite(logits).all()):
                            raise VerificationError('Nonfinite margin logits.')
                        objectives.append(logits[0, objective['prefer_id']]-logits[0, objective['avoid_id']])
                        logits_value = to_numpy(logits[0])
                total = torch.stack(objectives).sum()
                if not bool(torch.isfinite(total)):
                    raise VerificationError('Nonfinite objective; check held-out weights and logits.')
                order = [(i, layer) for i in range(len(items)) for layer in layers]
                gradients = torch.autograd.grad(total, tuple(leaves[key] for key in order))
                objective_value = math.fsum(float(score.detach()) for score in objectives)
            banks, checks = [], []
            for (item, layer), grad in zip(order, gradients):
                q, k, v, y, p, allowed = records[(item, layer)]
                g = to_numpy(grad[0].transpose(0, 1))
                if not likelihood:  # preserve the public single-query geometry
                    q, g, y, p = q[:, 0], g[:, 0], y[:, 0], p[:, 0]
                bank = Bank(layer, q, k, v, g, to_numpy(model.model.rotary_emb.inv_freq),
                            model.model.layers[layer].self_attn.scaling, mask=allowed, item=item)
                replay = Scorer(bank)
                checks.append({'layer': layer, 'item': item, 'query_count': bank.query_count,
                    'causal_mask_verified': True, 'causal_mask_sha256': digest(allowed.tolist()),
                    'probability_error': assert_close(p, replay.p, 'Captured native probabilities'),
                    'head_output_error': assert_close(y, replay.y, 'Captured native head output'),
                    'verified': True})
                banks.append(bank)
        counts = {'capture_prefix_forwards': int(len(ids) > 1), 'capture_query_forwards': len(items),
                  'capture_backwards': 1, 'capture_items': len(items),
                  'capture_query_rows': sum(len(i['token_ids']) if likelihood else 1 for i in items)}
        return ObjectiveCapture(banks, list(ids), objective.get('prefer_id'), objective.get('avoid_id'),
                                objective_value, logits_value, prefix, checks, objective, item_results, counts)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        for parameter, value in previous:
            parameter.requires_grad_(value)


def capture_margin(model, ids, prefer, avoid, layers=None, cancel=None):
    """Backward-compatible next-token margin wrapper around capture_objective."""
    return capture_objective(model, ids,
        {'kind': MARGIN_OBJECTIVE, 'prefer_id': prefer, 'avoid_id': avoid}, layers, cancel)


class CacheEditor:
    """One prescribed cache-bank edit, at the first generated query only.

    K/V changes affect all physical KV groups and their GQA consumer heads at
    one layer. They are local to this read: stored K/V are not overwritten.
    Higher-layer cached consequences of the changed write persist naturally.
    Deletion masks rows without removing positions or changing the causal mask.
    """
    def __init__(self, model, candidate, prompt_length):
        layer = candidate.get('layer')
        if type(layer) is not int:
            raise UserError('Cache edit layer must be an integer.')
        validator = QwenAdapter(model, layer, 0, prompt_length-1)
        self.model, self.module, self.native = model, validator.module, validator.native
        self.candidate, self.prompt_length = candidate, prompt_length
        validate_template(candidate['edit'], prompt_length)
        validate_rows(candidate['indices'], prompt_length, deletion=candidate['edit']['kind'] == 'delete')
        self.audits = []

    @contextmanager
    def installed(self):
        owner = self
        def forward(module, hidden_states, position_embeddings, attention_mask,
                    past_key_values=None, cache_position=None, **kwargs):
            return owner.forward(hidden_states, position_embeddings, attention_mask, past_key_values, cache_position)
        with patched_forward(self.module, forward):
            yield self

    def forward(self, hidden, positions, mask, cache, cache_position):
        import torch
        m, native = self.module, self.native
        q, k, v = qkv(m, hidden, positions, cache, cache_position, native)
        n = k.shape[2]
        if n != self.prompt_length or self.audits:
            raw, p = eager_rows(native, m, q, k, v, mask)
            return m.o_proj(raw.reshape(*hidden.shape[:-1], -1).contiguous()), p
        if hidden.shape[0] != 1 or cache_position is None or int(cache_position[-1]) != n-1:
            raise VerificationError('Unexpected cache-edit execution position.')
        indices, edit = self.candidate['indices'], self.candidate['edit']
        bank = Bank(m.layer_idx, to_numpy(q[0, :, -1]), to_numpy(k[0]), to_numpy(v[0]),
            np.zeros((q.shape[1], v.shape[-1])), to_numpy(self.model.model.rotary_emb.inv_freq), m.scaling)
        expected = Scorer(bank).local(edit, indices)
        newk, newv = k, v
        newmask = last_mask(mask, n)
        kind = edit['kind']
        if kind in {'position', 'key', 'joint'}:
            newk = k.clone()
            replacement = (rotate_keys(bank.keys[:, indices], bank.frequencies, edit['delta'], edit['band'])
                           if kind == 'position' else bank.keys[:, [edit['donor']]].repeat(len(indices), axis=1))
            newk[0, :, indices, :] = torch.as_tensor(replacement, dtype=k.dtype, device=k.device)
        if kind in {'value_scale', 'value', 'joint'}:
            newv = v.clone()
            replacement = (bank.values[:, indices]*edit['factor'] if kind == 'value_scale'
                           else bank.values[:, [edit['donor']]].repeat(len(indices), axis=1))
            newv[0, :, indices, :] = torch.as_tensor(replacement, dtype=v.dtype, device=v.device)
        if kind in {'reweight', 'delete'}:
            newmask = torch.zeros((1, 1, 1, n), device=q.device, dtype=q.dtype) if newmask is None else newmask.clone()
            newmask[:, :, :, indices] += -float('inf') if kind == 'delete' else np.log(edit['factor'])
        changed, p = native.eager_attention_forward(m, q[:, :, -1:, :], newk, newv, newmask,
            dropout=0.0, scaling=m.scaling, sliding_window=m.sliding_window)
        actual_p, actual_y = to_numpy(p[0, :, 0]), to_numpy(changed[0, 0])
        self.audits.append({'layer': m.layer_idx, 'query_index': n-1, 'indices': indices, 'edit': edit,
            'scope': 'all KV groups, one layer, current query only',
            'probability_error': assert_close(actual_p, expected['probabilities'], 'Edited native probabilities'),
            'head_output_error': assert_close(actual_y, expected['output'], 'Edited native head output'),
            'verified': True})
        if hidden.shape[1] > 1:
            # Earlier queries must remain exactly baseline. Editing the full
            # prefill bank for all rows would implement a different intervention.
            baseline, _ = eager_rows(native, m, q, k, v, mask)
            raw = torch.cat((baseline[:, :-1], changed), dim=1)
        else:
            raw = changed
        return m.o_proj(raw.reshape(*hidden.shape[:-1], -1).contiguous()), p


class LikelihoodEditor(CacheEditor):
    """One layer, all teacher-forced query rows; cached keys/values stay fixed.

    The prompt-row mask is changed for every continuation prediction, including
    the first and last. Future keys keep their native causal exclusions.
    """
    def __init__(self, model, candidate, prompt_length, query_count, item):
        super().__init__(model, candidate, prompt_length)
        if candidate['edit']['kind'] not in {'delete', 'reweight'}:
            raise UserError('Held-out likelihood initially supports delete and reweight edits.')
        self.query_count, self.item = query_count, item

    def forward(self, hidden, positions, mask, cache, cache_position):
        import torch
        m, native = self.module, self.native
        if hidden.shape[:2] != (1, self.query_count) or self.audits:
            raise VerificationError('Unexpected teacher-forced edit query geometry or repeated application.')
        q, k, v = qkv(m, hidden, positions, cache, cache_position, native)
        n = k.shape[2]
        if n != self.prompt_length + self.query_count - 1:
            raise VerificationError('Teacher-forced edit did not receive the expected independent prefix.')
        mask, allowed = causal_query_mask(mask, cache_position, n)
        bank = Bank(m.layer_idx, to_numpy(q[0]), to_numpy(k[0]), to_numpy(v[0]),
            np.zeros((q.shape[1], self.query_count, v.shape[-1])),
            to_numpy(self.model.model.rotary_emb.inv_freq), m.scaling, mask=allowed, item=self.item)
        indices, edit = self.candidate['indices'], self.candidate['edit']
        expected = Scorer(bank).local(edit, indices)
        changed_mask = (torch.zeros((1, 1, self.query_count, n), dtype=q.dtype, device=q.device)
                        if mask is None else mask.clone())
        changed_mask[..., indices] += -float('inf') if edit['kind'] == 'delete' else math.log(edit['factor'])
        raw, p = native.eager_attention_forward(m, q, k, v, changed_mask,
            dropout=0.0, scaling=m.scaling, sliding_window=m.sliding_window)
        probabilities, output = to_numpy(p[0]), to_numpy(raw[0].transpose(0, 1))
        if np.any(probabilities[:, ~allowed] != 0):
            raise VerificationError('A likelihood edit exposed future keys.')
        self.audits.append({'layer': m.layer_idx, 'item': self.item,
            'query_indices': cache_position.tolist(), 'query_count': self.query_count,
            'indices': indices, 'edit': edit, 'causal_mask_verified': True,
            'scope': 'all KV groups, one layer, every held-out teacher-forced query row',
            'probability_error': assert_close(probabilities, expected['probabilities'], 'Edited multi-row probabilities'),
            'head_output_error': assert_close(output, expected['output'], 'Edited multi-row head output'),
            'verified': True})
        return m.o_proj(raw.reshape(*hidden.shape[:-1], -1).contiguous()), p


def measure_heldout(model, capture, candidate=None):
    """Teacher-forced likelihood on fresh, independent copies of the capture prefix."""
    import torch
    if capture.objective['kind'] != HELD_OUT_LOG_LIKELIHOOD:
        raise UserError('measure_heldout requires a held_out_log_likelihood capture.')
    results, audits = [], []
    with precise_inference(torch), chunked_prefill(model):
        for index, item in enumerate(capture.objective['items']):
            forced = [capture.ids[-1]] + item['token_ids'][:-1]
            editor = (LikelihoodEditor(model, candidate, len(capture.ids), len(forced), index)
                      if candidate is not None else None)
            with editor.installed() if editor is not None else nullcontext():
                out = model(input_ids=torch.tensor([forced], device=model.device),
                    past_key_values=capture.fresh_cache(model),
                    cache_position=torch.arange(len(capture.ids)-1, len(capture.ids)-1+len(forced), device=model.device),
                    use_cache=True, logits_to_keep=0)
                log_probs = teacher_forced_log_probs(out.logits[0], item['token_ids'])
                score = float(log_probs.sum())
                results.append({'item': index, 'weight': item['weight'],
                    'token_log_probabilities': to_numpy(log_probs).tolist(), 'log_likelihood': score,
                    'weighted_log_likelihood': float(log_probs.sum()*item['weight'])})
            if editor is not None:
                if len(editor.audits) != 1:
                    raise VerificationError('A held-out execution missed its edited query rows.')
                audits.extend(editor.audits)
    total = math.fsum(row['weighted_log_likelihood'] for row in results)
    return {'log_likelihood': total, 'loss': -total, 'items': results, 'audits': audits,
            'teacher_forced_forwards': len(results)}


def execute_candidate(model, capture, candidate):
    """Execute ONE candidate over a fresh copy of the same baseline prefix."""
    if capture.objective['kind'] == HELD_OUT_LOG_LIKELIHOOD:
        measured = measure_heldout(model, capture, candidate)
        loss_change = measured['loss'] - capture.loss
        predicted = candidate.get('predicted_loss_change')
        if predicted is None:
            raise UserError('A held-out candidate needs predicted_loss_change.')
        return {'actual_loss_change': loss_change, 'predicted_loss_change': predicted,
            'actual_log_likelihood_change': -loss_change,
            'predicted_log_likelihood_change': -predicted, 'prediction_error': loss_change-predicted,
            'local_verified': all(a['verified'] for a in measured['audits']),
            'loss_improved': loss_change < 0, 'actual_loss': measured['loss'],
            'actual_log_likelihood': measured['log_likelihood'],
            'items': measured['items'], 'audits': measured['audits'],
            'teacher_forced_forwards': measured['teacher_forced_forwards']}
    import torch
    editor = CacheEditor(model, candidate, len(capture.ids))
    with precise_inference(torch), chunked_prefill(model), editor.installed():
        out = model(input_ids=torch.tensor([[capture.ids[-1]]], device=model.device),
            past_key_values=capture.fresh_cache(model), cache_position=torch.tensor([len(capture.ids)-1], device=model.device),
            use_cache=True, logits_to_keep=1)
        logits = out.logits[0, -1]
        if not bool(torch.isfinite(logits).all()) or len(editor.audits) != 1:
            raise VerificationError('Candidate execution missed its verified write or produced invalid logits.')
        actual = float(logits[capture.prefer]-logits[capture.avoid])-capture.margin
    return {'actual_margin_change': actual, 'predicted_margin_change': candidate['predicted_margin_change'],
        'prediction_error': actual-candidate['predicted_margin_change'], 'local_verified': True,
        'margin_improved': actual > 0, 'audit': editor.audits[0]}


def heldout_items(items):
    """Public JSON schema: strings or {text, weight?}, optionally {items: [...]}.

    Text is an exact continuation of the rendered assistant prefix. Leading
    spaces are significant. No implicit EOS, labels, or length normalization.
    """
    if isinstance(items, dict):
        if set(items) != {'items'}:
            raise UserError('The held-out JSON object must contain only items.')
        items = items['items']
    if not isinstance(items, list) or not 1 <= len(items) <= 64:
        raise UserError('heldout must contain 1–64 continuation strings or text/weight objects.')
    result = []
    for index, value in enumerate(items):
        item = {'text': value} if isinstance(value, str) else copy.deepcopy(value)
        if not isinstance(item, dict) or set(item)-{'text', 'weight'} or not isinstance(item.get('text'), str) or not item['text']:
            raise UserError(f'Held-out item {index} needs nonempty text and an optional weight.')
        weight = item.get('weight', 1.0)
        if type(weight) not in (float, int) or not math.isfinite(weight) or weight <= 0:
            raise UserError('Held-out weights must be positive finite numbers.')
        result.append({'text': item['text'], 'weight': weight})
    return result


def value_spans(tokenizer, rendered, spans):
    """Resolve whole source spans, allowing explicit token indices for replay."""
    if spans is None:
        return None, None
    if not isinstance(spans, list) or not 1 <= len(spans) <= 64:
        raise UserError('Supply 1–64 spans, or omit spans to rank every prompt token.')
    resolved, provenance = [], []
    for value in spans:
        spec = {'quote': value} if isinstance(value, str) else copy.deepcopy(value)
        if not isinstance(spec, dict):
            raise UserError('A span is a quoted string, a quote/occurrence object, or an indices object.')
        if set(spec) == {'indices'}:
            rows = spec['indices']
            validate_rows(rows, len(rendered.ids))
            rows = sorted(rows)
        else:
            if set(spec)-{'quote', 'occurrence'} or not isinstance(spec.get('quote'), str) or not spec['quote']:
                raise UserError('Quoted spans use a nonempty quote and optional 1-based occurrence.')
            quote = spec['quote']
            starts = [m.start() for m in re.finditer(re.escape(quote), rendered.text)]
            occurrence = spec.get('occurrence')
            if occurrence is None and len(starts) != 1:
                raise UserError(f'The quote occurs {len(starts)} times; supply its 1-based occurrence.')
            occurrence = 1 if occurrence is None else occurrence
            if type(occurrence) is not int or not 1 <= occurrence <= len(starts):
                raise UserError('The span occurrence is outside the prompt.')
            start, stop = starts[occurrence-1], starts[occurrence-1]+len(quote)
            rows = [i for i, (a, b) in enumerate(rendered.offsets) if a < stop and b > start]
            if not rows:
                raise UserError('The quoted span covers no token rows.')
            a, b = rendered.offsets[rows[0]][0], rendered.offsets[rows[-1]][1]
            if rendered.text[a:b].strip() != quote.strip():
                raise UserError(f'The span cuts a token. Use {rendered.text[a:b]!r}.')
        validate_rows(rows, len(rendered.ids))
        if rows in resolved:
            raise UserError('Duplicate spans resolve to the same token rows.')
        resolved.append(rows)
        provenance.append({'spec': spec, 'indices': rows,
                           'tokens': [tokenizer.decode([rendered.ids[i]]) for i in rows]})
    return resolved, provenance


def rank_value_banks(banks, prompt_length, templates=None, spans=None, *, rank_by='contribution', cancel=None):
    """Score every prompt token/span; sum queries AND independent items per layer.

    Layers remain alternative interventions, never summed into a multi-layer edit.
    Positive loss change means the edit makes prediction worse. Contribution
    reverses likelihood gain for deletion/suppression and keeps it for boosts.
    """
    if rank_by not in {'contribution', 'improvement'}:
        raise UserError('rank_by must be contribution or improvement.')
    templates = [{'kind': 'delete'}] if templates is None else templates
    if not isinstance(templates, list) or not 1 <= len(templates) <= 32:
        raise UserError('Use 1–32 delete/reweight templates.')
    for template in templates:
        validate_template(template, prompt_length)
        if template['kind'] not in {'delete', 'reweight'}:
            raise UserError('Held-out value supports delete and reweight templates.')
    if spans is not None:
        for rows in spans:
            validate_rows(rows, prompt_length)
    ranked, arrays = [], {}
    by_layer = {}
    for bank in banks:
        if bank.n < prompt_length:
            raise VerificationError('A held-out bank is shorter than its prompt.')
        by_layer.setdefault(bank.layer, []).append(Scorer(bank))
    for layer, scorers in sorted(by_layer.items()):
        for t, edit in enumerate(templates):
            if cancel is not None and cancel.is_set():
                raise UserError('Held-out ranking cancelled.')
            if spans is None:
                gains = sum(s.all_singletons(edit)[:prompt_length] for s in scorers)
            else:
                gains = np.array([sum(s.local(edit, rows)['prediction'] for s in scorers) for rows in spans])
            arrays[f'layer_{layer}_template_{t}'] = gains
            removal = edit['kind'] == 'delete' or edit['factor'] < 1
            for i, gain in enumerate(gains):
                contribution = -float(gain) if removal else float(gain)
                ranked.append({'layer': layer, 'template_index': t, 'span_index': i,
                    'indices': [i] if spans is None else list(spans[i]), 'edit': dict(edit),
                    'predicted_log_likelihood_change': float(gain), 'predicted_loss_change': -float(gain),
                    'predicted_value': contribution,
                    'ranking_score': contribution if rank_by == 'contribution' else float(gain)})
    ranked.sort(key=lambda r: (-r['ranking_score'], r['layer'], r['template_index'], r['span_index']))
    for i, row in enumerate(ranked):
        row['rank'] = i+1
    return ranked, arrays, len(ranked)


def selection_pool(prompt_length, spans, budget):
    # The final prompt key is the always-present anchor when selecting tokens.
    pool = [[i] for i in range(prompt_length-1)] if spans is None else [list(r) for r in spans]
    if type(budget) is not int or not 1 <= budget <= len(pool):
        raise UserError('select must be between 1 and the number of selectable spans/tokens.')
    for rows in pool:
        validate_rows(rows, prompt_length)
    flattened = [i for rows in pool for i in rows]
    if len(set(flattened)) != len(flattened):
        raise UserError('Greedy selection requires non-overlapping spans; ordinary ranking permits overlaps.')
    if not set(range(prompt_length))-set(flattened):
        raise UserError('The selection pool must leave at least one prompt key as an attended anchor.')
    return pool


def _group_statistics(scorer, indices):
    """Stable normalizer and gradient-contracted mean for one disjoint group."""
    scores = scorer.s[..., indices]
    maximum = scores.max(axis=-1, keepdims=True)
    valid = np.isfinite(maximum)
    weights = np.exp(scores-np.where(valid, maximum, 0.))
    total = weights.sum(axis=-1)
    normalizer = np.full_like(total, -np.inf)
    np.log(total, out=normalizer, where=total > 0)
    normalizer += np.where(valid[..., 0], maximum[..., 0], 0.)
    mean = np.divide(np.sum(weights*scorer.a[..., indices], axis=-1), total,
                     out=np.zeros_like(total), where=total > 0)
    return normalizer, mean


def greedy_value_selection(banks, prompt_length, pool, budget, cancel=None):
    """Forward greedy retention under the SAME frozen multi-query gradient.

    All candidate text is present in the captured pool. Insertion is represented
    by deleting the unretained pool entries at one layer, never by re-tokenizing
    or claiming to reconstruct the upstream features of a shorter prompt.
    Group means/normalizers are updated and every remaining span is re-scored
    after each retention. Each layer gets its own trajectory; the best final
    predicted likelihood selects one layer for execution.
    """
    pool = selection_pool(prompt_length, pool, budget)
    union = set(i for rows in pool for i in rows)
    trajectories = []
    for layer in sorted({b.layer for b in banks}):
        scorers = [Scorer(b) for b in banks if b.layer == layer]
        states, groups = [], []
        for scorer in scorers:
            outside = sorted(set(range(scorer.bank.n))-union)
            state = _group_statistics(scorer, outside)
            if not np.isfinite(state[0]).all():
                raise UserError('Selection leaves an empty attended query row.')
            states.append(state)
            groups.append([_group_statistics(scorer, rows) for rows in pool])
        baseline = sum(float(np.sum(s.mean)) for s in scorers)
        initial = sum(float(np.sum(state[1])) for state in states)
        selected, steps = [], []
        for step in range(budget):
            if cancel is not None and cancel.is_set():
                raise UserError('Greedy selection cancelled.')
            candidates = []
            for j in range(len(pool)):
                if j in selected:
                    continue
                marginal = 0.
                for state, item_groups in zip(states, groups):
                    log_mass, mean = item_groups[j]
                    fraction = np.exp(log_mass-np.logaddexp(state[0], log_mass))
                    marginal += float(np.sum(fraction*(mean-state[1])))
                candidates.append({'span_index': j, 'predicted_log_likelihood_change': marginal,
                                   'predicted_loss_change': -marginal})
            candidates.sort(key=lambda r: (-r['predicted_log_likelihood_change'], r['span_index']))
            best = candidates[0]
            j = best['span_index']
            selected.append(j)
            for i, (state, item_groups) in enumerate(zip(states, groups)):
                log_mass, mean = item_groups[j]
                merged = np.logaddexp(state[0], log_mass)
                fraction = np.exp(log_mass-merged)
                states[i] = (merged, state[1]+fraction*(mean-state[1]))
            current = sum(float(np.sum(state[1])) for state in states)
            steps.append({'step': step+1, 'retained_span_index': j, 'indices': pool[j],
                'predicted_marginal_loss_change': best['predicted_loss_change'],
                'predicted_loss_change': baseline-current, 'rescored_candidates': candidates})
        retained = sorted(i for j in selected for i in pool[j])
        deleted = sorted(union-set(retained))
        current = sum(float(np.sum(state[1])) for state in states)
        candidate = {'layer': layer, 'indices': deleted or [prompt_length-1],
            'edit': {'kind': 'delete'} if deleted else {'kind': 'reweight', 'factor': 1.},
            'predicted_loss_change': baseline-current,
            'predicted_log_likelihood_change': current-baseline}
        trajectories.append({'layer': layer, 'selected_span_indices': selected, 'retained_indices': retained,
            'masked_indices': deleted, 'steps': steps, 'candidate': candidate,
            'predicted_loss_change': baseline-current,
            'predicted_gain_from_empty_pool': current-initial})
    trajectories.sort(key=lambda t: (t['predicted_loss_change'], t['layer']))
    return {'budget': budget, **trajectories[0], 'pool': pool,
            'protected_indices': sorted(set(range(prompt_length))-union),
            'layer_summaries': [{'layer': t['layer'], 'predicted_loss_change': t['predicted_loss_change'],
                                 'selected_span_indices': t['selected_span_indices']} for t in trajectories],
            'capture_backwards': 0, 'model_calls_for_rescoring': 0,
            'gradient_refreshed': False, 'method': 'forward greedy retention through pooled-context deletion'}


# ========================================================================
# SEARCH
# ========================================================================

"""User-facing margin objectives and auditable ranked cache interventions."""













SEARCH_FIELDS = {'prefer', 'avoid', 'prefix', 'layers', 'spans', 'edits', 'top_k', 'verify'}


def prepare_probe(engine, base, specification):
    """S7: name the scalar objective before inspecting candidate outcomes.

    For multi-token alternatives, condition on their shared token prefix and
    compare the first distinct tokens. This is explicitly NOT a full-sequence
    likelihood comparison. A supplied assistant prefix is teacher-forced, with
    its own tokenization appended to the unchanged baseline token sequence.
    """
    if not isinstance(specification, dict) or set(specification)-SEARCH_FIELDS:
        raise UserError('Unknown ranking option.')
    preferred, avoided = specification.get('prefer'), specification.get('avoid')
    prefix = specification.get('prefix', '')
    if any(not isinstance(x, str) or not 1 <= len(x) <= 256 for x in (preferred, avoided)):
        raise UserError('Name the preferred and competing answer continuations (1–256 characters).')
    if not isinstance(prefix, str) or len(prefix) > 4000:
        raise UserError('The optional assistant prefix must contain at most 4000 characters.')
    good = engine.tokenizer.encode(preferred, add_special_tokens=False)
    bad = engine.tokenizer.encode(avoided, add_special_tokens=False)
    shared = 0
    while shared < min(len(good), len(bad)) and good[shared] == bad[shared]:
        shared += 1
    if shared == min(len(good), len(bad)):
        raise UserError('The alternatives need distinct next tokens; neither token sequence may be a prefix of the other.')
    forced_ids = engine.tokenizer.encode(prefix, add_special_tokens=False) + good[:shared]
    forced_text = engine.tokenizer.decode(forced_ids, skip_special_tokens=False)
    # New offsets are only for display; source anchoring below is limited to the
    # original base text. Token IDs, not re-tokenized concatenated text, define
    # the conditioning sequence and are preserved in the report.
    offsets = [*base.offsets, *[(len(base.text), len(base.text)+len(forced_text)) for _ in forced_ids]]
    probe = Rendered(base.text+forced_text, [*base.ids, *forced_ids], offsets)
    engine.validate_length(probe.ids, 1)
    objective = {'kind': 'first_distinct_next_token_logit_margin', 'prefer': preferred, 'avoid': avoided,
        'prefer_id': good[shared], 'avoid_id': bad[shared], 'shared_token_prefix': good[:shared],
        'prefix': prefix, 'forced_ids': forced_ids, 'forced_text': forced_text,
        'prefer_token': engine.tokenizer.decode([good[shared]]), 'avoid_token': engine.tokenizer.decode([bad[shared]]),
        'is_full_continuation_likelihood': False}
    top_k = specification.get('top_k', 3)
    if type(top_k) is not int or not 1 <= top_k <= 5:
        raise UserError('Verify or generate only the top 1–5 candidates.')
    if type(specification.get('verify', True)) is not bool:
        raise UserError('verify must be a boolean.')
    return probe, objective


def rank_prepared(engine, base, probe, objective, specification, run, cancel=None):
    templates = copy.deepcopy(specification.get('edits', DEFAULT_TEMPLATES))
    if not isinstance(templates, list) or not 1 <= len(templates) <= 32:
        raise UserError('Use between 1 and 32 edit templates.')
    for edit in templates:
        validate_template(edit, len(probe.ids))
    spans = None
    if 'spans' in specification:
        selected = specification['spans']
        if not isinstance(selected, list) or not 1 <= len(selected) <= 64:
            raise UserError('Use 1–64 quoted source spans, or omit spans to rank every token.')
        spans = [anchor(engine.tokenizer, base, item)['indices'] for item in selected]
    run = Path(run); run.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    capture = capture_margin(engine.model, probe.ids, objective['prefer_id'], objective['avoid_id'],
                             layers=specification.get('layers'), cancel=cancel)
    captured = time.perf_counter()
    ranked, arrays, total = rank_banks(capture.banks, templates, spans, limit=20, cancel=cancel)
    scored = time.perf_counter()
    for row in ranked:
        row['tokens'] = [engine.tokenizer.decode([probe.ids[i]], skip_special_tokens=False) for i in row['indices']]
    report = {'schema_version': 1, 'status': 'scored_only', 'identity': engine.identity, 'objective': objective,
        'prompt_hash': digest(probe.ids), 'baseline_margin': capture.margin, 'candidate_count': total,
        'templates': templates, 'spans': spans, 'scope': 'one layer, all physical KV groups, final query only',
        'prediction': 'exact local write change contracted with the baseline margin gradient',
        'downstream_prediction_is_exact': False, 'ranked': ranked, 'verification': [],
        'capture_checks': capture.checks, 'counts': {'prefix_forward': int(len(probe.ids) > 1),
            'baseline_query_forward': 1, 'backward': 1, 'model_calls_for_scoring': 0, 'candidate_forwards': 0},
        'timing_seconds': {'capture': captured-start, 'analytic_ranking': scored-captured},
        'files': {'capture': 'capture.npz', 'scores': 'scores.npz', 'prompt': 'prompt.json'}}
    capture.save(run/'capture.npz')
    np.savez_compressed(run/'scores.npz', **arrays)
    save_json(run/'prompt.json', {'text': probe.text, 'token_ids': probe.ids, 'specification': specification})
    save_json(run/'report.json', report)
    return capture, report


def search(engine, base, specification, run, *, cancel=None):
    """Score every candidate analytically, then execute only the top few."""
    probe, objective = prepare_probe(engine, base, specification)
    capture, report = rank_prepared(engine, base, probe, objective, specification, run, cancel)
    top_k = specification.get('top_k', 3)
    try:
        if specification.get('verify', True):
            for row in report['ranked'][:top_k]:
                if cancel is not None and cancel.is_set():
                    raise UserError('Ranking verification cancelled.')
                report['counts']['candidate_forwards'] += 1
                checked = execute_candidate(engine.model, capture, row)
                report['verification'].append({'rank': row['rank'], **checked})
        report['status'] = 'verified' if report['verification'] else 'scored_only'
    except (UserError, VerificationError) as exc:
        report.update(status='blocked', reason=str(exc))
        raise
    finally:
        save_json(Path(run)/'report.json', report)
    return compact_result(report, run, top_k)


def compact_result(report, run, top_k=3):
    verified = {r['rank']: r for r in report['verification']}
    return {'status': report['status'], 'objective': report['objective'], 'baseline_margin': report['baseline_margin'],
        'candidate_count': report['candidate_count'], 'counts': report['counts'], 'scope': report['scope'],
        'ranked': [{**r, 'verification': verified.get(r['rank'])} for r in report['ranked'][:top_k]],
        'report': str(Path(run)/'report.json'),
        'interpretation': 'Scores estimate the stated local decision margin. Native verification measures that margin; neither certifies whole-answer correctness.',
        'next_action': 'Use the verified candidates for an isolated intervention trial. Recompute after changing the prompt, model, objective, or cache.'}


def print_ranking(result):
    obj = result['objective']
    print(f"Next-token margin: {obj['prefer_token']!r} over {obj['avoid_token']!r}; baseline {result['baseline_margin']:+.6f}")
    if (obj['prefer_token'], obj['avoid_token']) != (obj['prefer'], obj['avoid']):
        print(f"These are the first distinct tokens of {obj['prefer']!r} versus {obj['avoid']!r}, not a whole-answer likelihood.")
    print(f"Scored {result['candidate_count']:,} edits with {result['counts']['model_calls_for_scoring']} model calls after capture.")
    print('Rank  Layer  Row(s)       Token/span       Edit                    Predicted    Executed')
    for row in result['ranked']:
        label = repr(''.join(row.get('tokens', [])))[:16]
        edit = row['edit']; name = edit['kind']
        if name == 'position': name += f" {edit['band']} {edit['delta']:+g}"
        elif 'factor' in edit: name += f" x{edit['factor']:g}"
        elif 'donor' in edit: name += f" <-{edit['donor']}"
        actual = row.get('verification')
        measured = f"{actual['actual_margin_change']:+.6f}" if actual else 'not run'
        indices = ','.join(str(i) for i in row['indices'])
        print(f"{row['rank']:>4}  {row['layer']:>5}  {indices:<12} {label:<16} {name:<23} {row['predicted_margin_change']:>+10.6f}  {measured}")
    print('Local attention changes are exact; downstream predictions use the baseline gradient.')
    print('Full ranking, replay tensors and native checks: '+result['report'])


# ========================================================================
# LAB
# ========================================================================

"""The short researcher workflow: diagnose → verify → try one measured repair."""










@dataclass
class Diagnosis:
    engine: object
    probe: object
    result: dict
    identity: dict

    def show(self):
        """Print token anchors, prescribed edits, predicted and executed gains."""
        print_ranking(self.result)

    def best_verified(self, min_gain=0.0):
        """Select only by measured margin gain, or return None.

        A negative native gain disqualifies even a highly ranked positive proxy.
        This does not establish that the user's stated answer is correct.
        """
        if not isinstance(min_gain, (int, float)) or not math.isfinite(min_gain) or min_gain < 0:
            raise UserError('min_gain must be a finite nonnegative number.')
        rows = [r for r in self.result['ranked'] if r.get('verification') is not None
                and r['verification']['actual_margin_change'] > min_gain]
        return max(rows, key=lambda r: r['verification']['actual_margin_change'], default=None)

    def try_repair(self, *, max_tokens=16, seed=0, min_gain=0.0):
        """Generate matched baseline/edited continuations in independent caches.

        The selected edit acts at the final prompt query at one layer. It is not
        reapplied to future queries. No model weights or persistent cache change.
        A stale baseline or a failed native local check blocks the result.
        Code repair with a fixed executable test is separately in nala.repair.
        """
        row = self.best_verified(min_gain)
        if row is None:
            return {'status': 'no_verified_improvement', 'selected': None}
        if self.engine.identity != self.identity:
            raise UserError('Model identity changed. Run a new diagnosis.')
        objective = self.result['objective']
        baseline = self.engine.generate(self.probe, max_tokens=max_tokens, seed=seed,
                                        margin_objective=objective)
        assert_close(baseline['first_token_margin'], self.result['baseline_margin'], 'Fresh baseline margin')
        edited = self.engine.generate(self.probe, max_tokens=max_tokens, seed=seed,
                                      cache_edit=row, margin_objective=objective)
        gain = edited['first_token_margin']-baseline['first_token_margin']
        assert_close(gain, row['verification']['actual_margin_change'], 'Repeated intervention margin')
        if not edited['audits']:
            raise VerificationError('The selected cache edit was not executed.')
        if gain <= min_gain:
            raise VerificationError('The repeated edit did not meet the measured gain threshold.')
        result = {'status': 'verified_margin_improvement', 'selected': row, 'actual_margin_change': gain,
                  'baseline': baseline, 'edited': edited,
                  'scope': 'one current-query cache read; whole-answer correctness is not certified'}
        # Preserve repeated trials rather than overwrite earlier evidence.
        path = Path(self.result['report']).parent / ('repair-'+uuid.uuid4().hex[:10]+'.json')
        save_json(path, result)
        result['report'] = str(path)
        return result


def diagnose(engine, prompt, *, prefer, avoid, out=None, top_k=3, **options):
    if not isinstance(prompt, str) or not prompt.strip():
        raise UserError('Supply a nonempty prompt string.')
    messages = [{'role': 'user', 'content': prompt}]
    base = engine.encode(messages)
    specification = {**options, 'prefer': prefer, 'avoid': avoid, 'top_k': top_k}
    probe, _ = prepare_probe(engine, base, specification)
    run = Path(out) if out is not None else Path('nala-results') / uuid.uuid4().hex[:12]
    result = search(engine, base, specification, run)
    return Diagnosis(engine, probe, result, dict(engine.identity))


# ========================================================================
# ENGINE
# ========================================================================

"""A pinned native model for attention diagnostics and isolated intervention trials."""











CODER_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
CODER_REVISION = "ea3f2471cf1b1f0db85067f1ef93848e38e88c25"
DEFAULT_MODEL = MODEL_ID
# Full-model FP32 cache replay accumulates rounding across layers. This gate
# is separate from the strict native probability/raw-head checks in debugger.
LOGIT_REPLAY_ATOL = 2e-4


class Engine:
    def __init__(self, model, tokenizer, *, model_id: str, revision: str,
                 context_limit: int = 16384, output_limit: int = 512):
        self.model, self.tokenizer = model, tokenizer
        self.model_id, self.revision = model_id, revision
        self.context_limit = min(context_limit, int(model.config.max_position_embeddings))
        self.output_limit = output_limit
        eos = model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos if eos is not None else tokenizer.eos_token_id])
        QwenAdapter(model, 0, 0, 0)

    @property
    def identity(self):
        import torch, transformers
        return {"model": self.model_id, "revision": self.revision,
                'instrument_version': __version__,
                "torch": torch.__version__, "transformers": transformers.__version__,
                "device": str(self.model.device), "dtype": "float32", "attention": "eager",
                "context_limit": self.context_limit, "output_limit": self.output_limit}

    def encode(self, messages, tools=None):
        return render(self.tokenizer, messages, tools)

    def diagnose(self, prompt, *, prefer, avoid, out=None, top_k=3, **options):
        """Rank prescribed cache edits, then natively verify only the top few.

        See nala.lab.Diagnosis.try_repair for an isolated continuation trial.
        A model instance must not be used concurrently or changed mid-experiment.
        """
        return diagnose(self, prompt, prefer=prefer, avoid=avoid, out=out, top_k=top_k, **options)

    def validate_length(self, ids, max_tokens):
        if len(ids) + max_tokens > self.context_limit:
            raise UserError(f"This request needs {len(ids)} prompt + {max_tokens} output tokens; "
                              f"configured context limit is {self.context_limit}. Shorten the prompt or raise context_limit.")

    def select(self, rendered, recipe):
        return choose_head(self.model, rendered.ids, recipe)

    def rank(self, base, probe, objective, specification, run, cancel=None):
        _, report = rank_prepared(self, base, probe, objective, specification, run, cancel)
        return report

    def generate(self, rendered, *, max_tokens=None, temperature=0.0, top_p=1.0, seed=0, recipe=None,
                 cache_edit=None, margin_objective=None, cancel=None):
        if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1):
            raise UserError("max_tokens must be a positive integer.")
        max_tokens = min(max_tokens or self.output_limit, self.output_limit)
        self.validate_length(rendered.ids, max_tokens)
        first_margin = []
        def observe(step, logits):
            if step == 0 and margin_objective is not None:
                first_margin.append(float(logits[margin_objective['prefer_id']]-logits[margin_objective['avoid_id']]))
        result = generate_ids(self.model, rendered.ids, max_tokens=max_tokens, eos_ids=self.eos_ids,
                              temperature=temperature, top_p=top_p, seed=seed, recipe=recipe,
                              cache_edit=cache_edit, cancel=cancel, on_logits=observe)
        result.update(text=self.tokenizer.decode(result["token_ids"], skip_special_tokens=True),
                      prompt_tokens=len(rendered.ids), output_budget=max_tokens,
                      seed=seed, temperature=temperature, top_p=top_p,
                      recipe=asdict(recipe) if recipe else None, cache_edit=cache_edit,
                      first_token_margin=first_margin[0] if first_margin else None)
        return result

    def preflight(self):
        """Independent native cache/full-forward and installed no-op gates."""
        import torch
        from transformers import DynamicCache
        ids = self.tokenizer.encode("def old(x): return x\ndef new(x): return x + 1\n", add_special_tokens=False)
        tensor = torch.tensor([ids], device=self.model.device)
        cache = DynamicCache(config=self.model.config)
        with precise_inference(torch):
            full = self.model(input_ids=tensor, use_cache=False, logits_to_keep=1).logits[0, -1]
            prefill = self.model(input_ids=tensor, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[0, -1]
            errors = {"prefill_logits": assert_close(to_numpy(prefill), to_numpy(full), "Native cache prefill", atol=LOGIT_REPLAY_ATOL)}
            next_id = full.argmax().reshape(1, 1)
            for _ in range(2):
                tensor = torch.cat((tensor, next_id), dim=1)
                exact = self.model(input_ids=tensor, use_cache=False, logits_to_keep=1).logits[0, -1]
                cached = self.model(input_ids=next_id, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[0, -1]
                errors[f"decode_logits_{tensor.shape[1]}"] = assert_close(to_numpy(cached), to_numpy(exact), "Cached versus full-prefix logits", atol=LOGIT_REPLAY_ATOL)
                next_id = exact.argmax().reshape(1, 1)
        recipe = EditRecipe([0], [1], boost=1, window=2, layer=0, head=0)
        plain = generate_ids(self.model, ids, max_tokens=3, eos_ids=set())
        noop = generate_ids(self.model, ids, max_tokens=3, eos_ids=set(), recipe=recipe)
        if plain["token_ids"] != noop["token_ids"] or len(noop["audits"]) != 2:
            raise VerificationError("Installed no-op changed native generation or missed a query.")
        errors["noop_audits"] = [row["audit"] for row in noop["audits"]]
        return {"passed": True, "checks": errors, "identity": self.identity,
                "tolerances": {"full_model_logits_atol": LOGIT_REPLAY_ATOL, "rtol": NATIVE_RTOL,
                               "local_attention_atol": NATIVE_ATOL}}


def load_engine(model_id=DEFAULT_MODEL, revision=None, *, device="auto", local_only=False,
                context_limit=16384, output_limit=512, progress=print):
    torch, transformers = model_dependencies()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    if revision is None and model_id == CODER_MODEL:
        revision = CODER_REVISION
    if revision is None and model_id == MODEL_ID:
        revision = MODEL_REVISION
    if revision is not None and not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
        raise UserError("--revision must be a full immutable 40-character commit hash.")
    config = AutoConfig.from_pretrained(model_id, revision=revision, local_files_only=local_only, trust_remote_code=False)
    revision = revision or getattr(config, "_commit_hash", None)
    if not revision or not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
        raise UserError("Could not resolve an immutable model revision.")
    if config.model_type != "qwen2":
        raise UserError("This release supports Qwen2/Qwen2.5 architecture only.")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise UserError("CUDA was requested but is unavailable.")
    if device == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    progress(f"Loading {model_id} at {revision[:12]} on {device} (FP32, eager attention)...")
    tokenizer_source = model_id
    if local_only:
        # Transformers 4.57.3's Mistral-regex probe calls Hub model_info for
        # remote-looking names even with local_files_only=True. Resolve the
        # pinned local snapshot so offline Qwen loading never takes that path.
        from transformers.utils.hub import cached_file
        tokenizer_source = str(Path(cached_file(model_id, "tokenizer_config.json",
            revision=revision, local_files_only=True)).parent)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, revision=revision, local_files_only=local_only,
                                              use_fast=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, local_files_only=local_only,
                                                torch_dtype=torch.float32, attn_implementation="eager",
                                                trust_remote_code=False).to(device).eval()
    return Engine(model, tokenizer, model_id=model_id, revision=revision,
                  context_limit=context_limit, output_limit=output_limit)


# ========================================================================
# WORKSPACE
# ========================================================================

"""Content snapshots and fixed executable checks; never apply to the live tree.

Workspace copies provide branch isolation, not an OS security sandbox. The
user-selected check executes repository code with the user's normal account.
"""
















EXCLUDED = {".git", ".nala", ".venv", "venv", "node_modules", "__pycache__",
            ".pytest_cache", ".mypy_cache", ".ruff_cache", "nala-runs"}


def safe_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise UserError("Use a repository-relative file path with forward slashes.")
    parts = PurePosixPath(relative).parts
    if PurePosixPath(relative).is_absolute() or any(p in {".", ".."} or p in EXCLUDED for p in parts):
        raise UserError("The requested file is outside the allowed repository tree.")
    path = root.joinpath(*parts)
    if root.resolve() not in path.resolve().parents or any(p.is_symlink() for p in [path, *path.parents] if root in p.parents):
        raise UserError("Symlinks and paths outside the repository are not supported.")
    if not path.is_file():
        raise UserError(f"No regular file at {relative}.")
    return path


def protected_file(relative: str) -> bool:
    p = PurePosixPath(relative)
    return (any(part.lower() in {"tests", "test", "__tests__", "fixtures", ".github"} for part in p.parts)
            or re.search(r"(^test_|^conftest\.|_test\.|\.(test|spec)\.)", p.name, re.I) is not None
            or p.name in {"pyproject.toml", "package.json", "pytest.ini", "setup.cfg", "tox.ini", "Makefile"})


def source_files(root: Path) -> list[Path]:
    """Copy tracked and nonignored working files, including uncommitted edits.

    git ls-files is only a file listing; no commits/checkouts are performed.
    Non-git folders use the disclosed directory exclusion set. Size limits
    refuse an incomplete snapshot rather than silently dropping big files.
    """
    p = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if p.returncode == 0 and Path(p.stdout.strip()).resolve() == root.resolve():
        result = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                                capture_output=True, check=True)
        candidates = [Path(os.fsdecode(x)) for x in result.stdout.split(b"\0") if x]
    else:
        candidates = []
        for base, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = sorted(x for x in dirs if x not in EXCLUDED)
            if any((Path(base) / x).is_symlink() for x in dirs):
                raise UserError("Snapshot contains a directory symlink. Use a repository without symlinks.")
            candidates.extend((Path(base) / name).relative_to(root) for name in names)
    files, total = [], 0
    for relative in sorted(set(candidates)):
        if any(p in EXCLUDED for p in relative.parts):
            continue
        path = root / relative
        if not path.exists() and not path.is_symlink():
            continue  # A tracked deletion is part of the current working state.
        safe_file(root, relative.as_posix())
        size = path.stat().st_size
        total += size
        if size > 5 * 1024**2 or total > 100 * 1024**2 or len(files) >= 10000:
            raise UserError("Workspace exceeds the 5 MB/file, 100 MB or 10,000-file snapshot limit.")
        files.append(relative)
    if not files:
        raise UserError("No source files were found in this workspace.")
    return files


def manifest(root: Path, files: list[Path]) -> dict[str, str]:
    return {p.as_posix(): hashlib.sha256(safe_file(root, p.as_posix()).read_bytes()).hexdigest() for p in files}


def snapshot(root: Path, destination: Path) -> dict[str, str]:
    files = source_files(root)
    before = manifest(root, files)
    destination.mkdir(parents=True, exist_ok=False)
    for relative in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, target)
    if manifest(destination, files) != before or manifest(root, files) != before:
        raise UserError("Workspace changed during capture. Retry after editing stops.")
    return before


def command_argv(command: str) -> list[str]:
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        raise UserError(f"Invalid check command: {exc}") from exc
    if not argv or any(token in {"&&", ";", "||", "|", ">", "<"} for token in argv):
        raise UserError("Use one executable check command, without shell operators.")
    executable = shutil.which(argv[0])
    if not executable:
        raise UserError(f"Check executable {argv[0]!r} is not on PATH.")
    return [str(Path(executable).absolute()), *argv[1:]]


def run_check(root: Path, argv: list[str], timeout: float, log_path: Path, cancel=None) -> dict:
    """Fixed command, shell=False, isolated cwd and bounded process lifetime.

    On POSIX a timeout kills the whole process group, not just the immediate
    shell child. No command is accepted from the model or from a generated patch.
    """
    start = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        proc = subprocess.Popen(argv, cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=(os.name == "posix"))
        timed_out = cancelled = False
        while proc.poll() is None:
            timed_out = time.monotonic() - start >= timeout
            cancelled = cancel is not None and cancel.is_set()
            if timed_out or cancelled:
                break
            try:
                proc.wait(timeout=min(.1, max(.001, timeout - (time.monotonic() - start))))
            except subprocess.TimeoutExpired:
                pass
        if timed_out or cancelled:
            try:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
            except ProcessLookupError:
                pass  # It exited between the deadline check and termination.
        code = proc.wait()
    with log_path.open("rb") as f:
        output = f.read(24000).decode("utf-8", errors="replace")
    return {"passed": code == 0 and not timed_out and not cancelled, "exit_code": code, "timeout": timed_out, "cancelled": cancelled,
            "seconds": time.monotonic() - start, "command": argv, "output": output,
            "output_truncated": log_path.stat().st_size > 24000}


def parse_edits(text: str, original: str) -> tuple[str, list[dict]]:
    """Accept only exact, unique string replacements in one preselected file.

    Refuse ambiguous replacements, no-ops and any extra commands/paths. Apply
    in listed order to a COPY. Returning a candidate is not live application.
    """
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    data = loads(text)
    if not isinstance(data, dict) or set(data) != {"edits"} or not isinstance(data["edits"], list) or not 1 <= len(data["edits"]) <= 8:
        raise UserError('Return only {"edits":[{"old":"exact existing text","new":"replacement"}]} (1–8 edits).')
    changed = original
    for edit in data["edits"]:
        if (not isinstance(edit, dict) or set(edit) != {"old", "new"}
                or not all(isinstance(edit[k], str) for k in ("old", "new"))
                or not edit["old"] or edit["old"] == edit["new"] or len(edit["new"]) > 32000):
            raise UserError("An edit is empty, unchanged or malformed.")
        if changed.count(edit["old"]) != 1:
            raise UserError("Each old string must match exactly once in the candidate file.")
        changed = changed.replace(edit["old"], edit["new"], 1)
    if changed == original or len(changed.encode()) > 128000:
        raise UserError("The candidate is unchanged or exceeds the file size limit.")
    return changed, data["edits"]


# ========================================================================
# REPAIR
# ========================================================================

"""Bounded matched repair branches, executed checks and auditable outcomes."""




















def repair(engine, request: dict, specification: dict, *, root: Path, runs: Path,
           check: list[str], timeout=60.0, max_tokens=384, seed=37, cancel=None,
           progress=lambda _: None) -> dict:
    """R1–R4: executable failure → grounded hypothesis → branches → fixed gate.

    No repair is accepted because an attention metric improved. All candidates
    are model-generated exact string edits to one implementation file, checked
    in independent copies of the same current working tree. The source tree
    remains untouched. The caller receives a proposed patch for review.
    """
    allowed = {"target_file", "evidence", "competing", "hypothesis", "boost", "window", "budget", "layer", "head", "search"}
    if runs.resolve() == root.resolve() or root.resolve() in runs.resolve().parents:
        raise UserError("Repair reports must be stored outside the source project.")
    if not isinstance(specification, dict) or set(specification) - allowed:
        raise UserError("Unknown repair option.")
    target = specification.get("target_file")
    evidence, competing = specification.get("evidence"), specification.get("competing")
    hypothesis = specification.get("hypothesis")
    if not isinstance(hypothesis, str) or not 8 <= len(hypothesis) <= 1200:
        raise UserError("State the concrete wrong-symbol/signature hypothesis (8–1200 characters).")
    target_path = safe_file(root, target)
    if protected_file(target):
        raise UserError("The repair tool cannot modify tests, fixtures or verification configuration.")
    if target_path.stat().st_size > 64000:
        raise UserError("The first repair protocol handles one implementation file of at most 64 KB.")
    if not isinstance(evidence, dict) or not isinstance(evidence.get("file"), str):
        raise UserError("Evidence must name the current source file plus an exact quote and focus.")
    source = safe_file(root, evidence["file"]).read_bytes().decode("utf-8")
    if not isinstance(evidence.get("quote"), str) or evidence["quote"] not in source:
        raise UserError("The evidence quote does not occur in the current source file.")
    original = target_path.read_bytes().decode("utf-8")
    original_prompt = engine.encode(request["messages"], request.get("tools"))
    # R1: evidence must have reached THIS model's actual rendered conversation.
    # Resolving here first rules out treating newly appended repair instructions
    # as proof that the model previously had access to the correct definition.
    original_good = anchor(engine.tokenizer, original_prompt, evidence)
    original_bad = anchor(engine.tokenizer, original_prompt, competing)
    instruction = ("A configured executable check is failing. Propose a small repair to the single file below. "
        "Return ONLY JSON with this schema: {\"edits\":[{\"old\":\"exact unique existing text\",\"new\":\"replacement\"}]}. "
        "Use 1–8 replacements. Do not call tools, change tests, or give shell commands. "
        "Use the authoritative definitions already present in the conversation.\n"
        f"Target file: {target}\nCurrent file contents follow as a JSON string:\n{json.dumps(original)}")
    messages = [*request["messages"], {"role": "user", "content": instruction}]
    common = engine.encode(messages, request.get("tools"))
    # Chat templates can alter prefixes when a message is appended. Determine
    # the common character prefix and require both sources to remain inside it.
    prefix = 0
    for a, b in zip(common.text, original_prompt.text):
        if a != b:
            break
        prefix += 1
    good = anchor(engine.tokenizer, common, evidence, end=prefix)
    bad = anchor(engine.tokenizer, common, competing, end=prefix)
    if set(good["indices"]) & set(bad["indices"]):
        raise UserError("Evidence and competing source spans overlap.")
    recipe = EditRecipe(good["indices"], bad["indices"], boost=specification.get("boost", 4.0),
                        window=specification.get("window", 8), budget=specification.get("budget", 5.0),
                        layer=specification.get("layer"), head=specification.get("head"))
    recipe.validate(len(common.ids))
    base_common, objective, search_spec = common, None, specification.get('search')
    if search_spec is not None:
        if any(key in specification for key in ('boost', 'window', 'budget', 'layer', 'head')):
            raise UserError('Ranked search has its own layers and edit templates; do not mix it with a query recipe.')
        if isinstance(search_spec, dict) and 'verify' in search_spec:
            raise UserError('Ranked repair always verifies executed writes and the fixed check; omit search.verify.')
        common, objective = prepare_probe(engine, base_common, search_spec)
    engine.validate_length(common.ids, min(max_tokens, engine.output_limit))
    runs.mkdir(parents=True, exist_ok=True)
    run = runs / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    run.mkdir()
    report = {"schema_version": 1, "id": run.name, "status": "running", "identity": engine.identity,
              "specification": specification, "request_hash": digest(request),
              "original_prompt_hash": digest(original_prompt.ids), "source_tree": str(root),
              "target_file": target, "branches": [], "selected": None,
              "protocol": {"check": check, "timeout_seconds": timeout, "seed": seed,
                           "max_tokens": min(max_tokens, engine.output_limit), "temperature": 0,
                           "order": ["ordinary", "source_emphasis", "ranked_search" if objective else "attention"],
                           "early_stop": "first passing candidate", "live_files_modified": False},
              "anchors": {"good": good, "bad": bad}, "recipe": asdict(recipe)}
    if objective is not None:
        report['objective'] = objective
    save_json(run / "request.json", request)
    save_json(run / "prompt.json", {"text": common.text, "token_ids": common.ids, "offsets": common.offsets})
    try:
        progress("Reproducing the configured check in a workspace copy...")
        tree = snapshot(root, run / "snapshot")
        if (tree.get(target) != hashlib.sha256(original.encode()).hexdigest()
                or tree.get(evidence["file"]) != hashlib.sha256(source.encode()).hexdigest()):
            raise UserError("Selected source files changed before the snapshot was captured.")
        report["workspace_manifest"] = tree
        report["workspace_hash"] = digest(tree)
        # Baseline check gets its own copy, just like candidates. Check-created
        # files and state cannot leak from this run into a candidate branch.
        shutil.copytree(run / "snapshot", run / "baseline")
        baseline = run_check(run / "baseline", check, timeout, run / "baseline.log", cancel=cancel)
        report["baseline_check"] = baseline
        if manifest(run / "baseline", [Path(p) for p in tree]) != tree:
            report.update(status="blocked", reason="The baseline check mutated source files; no repair was attempted.")
        elif baseline["passed"]:
            report["status"] = "no_failure"
            report["reason"] = "The configured check already passes; no repair was attempted."
        elif baseline["timeout"] or baseline["cancelled"]:
            report["status"] = "blocked"
            report["reason"] = "The baseline check timed out; this is not an identified code failure."
        else:
            # R2: ordinary and attention branches use byte-identical prompt
            # tokens, checkpoint, sampling settings and generation budget.
            # The explicit-emphasis comparator intentionally changes the prompt.
            # All branches start with fresh native caches and workspace copies.
            pending = ["ordinary", "source_emphasis", "ranked_search" if objective else "attention"]
            ranked_candidates = {}
            for name in pending:
                if cancel is not None and cancel.is_set():
                    raise Refused("Repair cancelled.")
                if name == 'ranked_search':
                    progress('Capturing the margin gradient and ranking all prescribed cache edits...')
                    ranking = engine.rank(base_common, common, objective, search_spec, run/'ranking', cancel=cancel)
                    report['selection'] = {'method': 'exact finite local changes contracted with baseline margin gradient',
                        'candidate_count': ranking['candidate_count'], 'ranking_report': 'ranking/report.json',
                        'baseline_margin': ranking['baseline_margin'], 'counts': ranking['counts']}
                    # Rank once at the fixed baseline. Generate only the top K
                    # positive candidates, each independently; never compose
                    # their scores as if a multi-edit effect were additive.
                    selected_rows = [row for row in ranking['ranked'] if row['predicted_margin_change'] > 0][:search_spec.get('top_k', 3)]
                    for row in selected_rows:
                        label = f"ranked_attention_{row['rank']}"
                        ranked_candidates[label] = row
                        pending.append(label)
                    continue
                branch = {"name": name, "status": "running"}
                report["branches"].append(branch)
                progress(f"Testing {name.replace('_', ' ')} candidate...")
                try:
                    selected = None
                    cache_candidate = ranked_candidates.get(name)
                    prompt = common
                    if name == "source_emphasis":
                        emphasized = {"role": "user", "content": instruction + "\nGive special attention to this current source definition: " + evidence["quote"]}
                        prompt = engine.encode([*request["messages"], emphasized], request.get("tools"))
                        if objective is not None:
                            prompt, _ = prepare_probe(engine, prompt, search_spec)
                    elif name == "attention":
                        selected, ranking = engine.select(common, recipe)
                        report["selection"] = {"method": "greatest baseline source-span mass", "ranking": ranking,
                                               "recipe": asdict(selected)}
                    branch["prompt_hash"] = digest(prompt.ids)
                    generated = engine.generate(prompt, max_tokens=max_tokens, temperature=0.0, seed=seed,
                                                recipe=selected, cache_edit=cache_candidate,
                                                margin_objective=objective, cancel=cancel)
                    save_json(run / f"{name}_generation.json", generated)
                    branch["generation_file"] = f"{name}_generation.json"
                    branch["generated_tokens"] = len(generated["token_ids"])
                    branch["verified_queries"] = len(generated["audits"])
                    if generated["finish_reason"] == "length":
                        raise UserError("The candidate was truncated at its token budget.")
                    if (name == "attention" or cache_candidate is not None) and not generated["audits"]:
                        raise VerificationError("Attention branch produced no verified intervention.")
                    if cache_candidate is not None:
                        actual = generated['first_token_margin']-report['selection']['baseline_margin']
                        branch['margin'] = {'predicted_change': cache_candidate['predicted_margin_change'],
                            'actual_change': actual, 'prediction_error': actual-cache_candidate['predicted_margin_change']}
                        branch['cache_edit'] = cache_candidate
                    # The same supplied/shared prefix is teacher-forced for
                    # ordinary, emphasized, and ranked branches. Include it in
                    # the JSON candidate; do not present it as generated text.
                    candidate_text = (objective['forced_text'] if objective else '') + generated['text']
                    changed, edits = parse_edits(candidate_text, original)
                    branch_path = run / name
                    shutil.copytree(run / "snapshot", branch_path)
                    (branch_path / target).write_bytes(changed.encode("utf-8"))
                    expected = manifest(branch_path, [Path(p) for p in tree])
                    checked = run_check(branch_path, check, timeout, run / f"{name}.log", cancel=cancel)
                    if manifest(branch_path, [Path(p) for p in tree]) != expected:
                        raise UserError("The check mutated source files; its result cannot be accepted.")
                    branch.update(status="passed" if checked["passed"] else "failed", check=checked, edits=edits)
                    if checked["passed"]:
                        # R3: a fixed executable check is the selection gate.
                        # Attention mass, output probability and local exactness
                        # do not enter this pass/fail decision.
                        diff = "".join(difflib.unified_diff(original.splitlines(True), changed.splitlines(True),
                                                          fromfile=f"a/{target}", tofile=f"b/{target}"))
                        (run / "proposed.patch").write_text(diff)
                        report.update(status="candidate_passed", selected=name, edits=edits, patch=diff)
                        break
                except (UserError, VerificationError) as exc:
                    branch.update(status="blocked", reason=str(exc), error_type=type(exc).__name__)
                save_json(run / "report.json", report)
            if report["status"] == "running":
                report.update(status="no_passing_candidate", reason="All bounded candidates failed or were refused. Continue ordinary debugging.")
        # R4: detect concurrent changes to existing AND added/deleted files.
        # The candidate cannot be represented as applicable to a different tree.
        current = manifest(root, source_files(root))
        if current != tree:
            report.update(status="stale_workspace", selected=None, reason="Workspace changed during repair. The candidate must be recomputed.")
            report.pop("edits", None)
            report.pop("patch", None)
    except Exception as exc:
        report.update(status="blocked", selected=None, reason=str(exc), error_type=type(exc).__name__)
        report.pop("edits", None)
        report.pop("patch", None)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
    save_json(run / "report.json", report)
    result = {key: report[key] for key in ("id", "status", "selected", "target_file", "branches")}
    for key in ("reason", "edits", "patch", "workspace_hash"):
        if key in report:
            result[key] = report[key]
    result["report"] = str(run / "report.json")
    result["next_action"] = ("Review the current target and proposed edits, apply the patch, then rerun the configured check in the live workspace."
                             if report["status"] == "candidate_passed" else "Continue ordinary debugging; no live code was changed.")
    return result


# ========================================================================
# CASE_STUDY
# ========================================================================

"""Matched, real-model API repair with an executable success definition.

This is an exploratory existence search over a committed synthetic task catalog,
not a benchmark success-rate estimate. The correct API definition and objective
are visible to all branches. The output contract fixes one API call site, never
the preferred identifier. The model generates that identifier.

All candidate selection uses the existing finite-write/gradient ranking. Tests
are the acceptance gate. A positive proxy, malformed control, changed test,
timeout, or scripted candidate never satisfies the requested repair criterion.
"""





















PROTOCOL = 'nala-real-repair-v3'
MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
TEST_COUNT = 6
MAX_TOKENS = 32


def catalog():
    """Freeze order and names before inference; moderate to stronger distraction."""
    names = [('resolve_17', 'resolve_71'), ('normalise_record', 'normalize_record'),
             ('decode_B', 'decode_A'), ('fetch_new', 'fetch_old'),
             ('read_item_v3', 'read_item'), ('process_safe', 'process')]
    return [{'id': f'{i:02d}-r{repeats}-{good}', 'correct': good, 'competing': bad,
             'legacy_examples': repeats}
            for i, (repeats, (good, bad)) in enumerate(
                (r, pair) for r in (8, 32, 64, 0) for pair in names)]


def case_files(case):
    """Define the implementation and fixed behavioral tests without model output."""
    good, bad = case['correct'], case['competing']
    if any(not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', s) for s in (good, bad)) or good == bad:
        raise UserError('Case API names must be distinct Python identifiers.')
    count = case['legacy_examples']
    if type(count) is not int or not 0 <= count <= 128:
        raise UserError('Use 0–128 legacy examples.')
    api = (f'def {good}(value):\n'
           '    """Current canonical format: trim surrounding whitespace, then uppercase."""\n'
           '    return value.strip().upper()\n\n'
           f'def {bad}(value):\n'
           '    """Legacy format: trim surrounding whitespace, then lowercase."""\n'
           '    return value.strip().lower()\n')
    style=case.get('api_style','direct')
    if style == 'indexed':
        api=('CURRENT = 1\n\n'+f'def {good}(value):\n'
             '    modes = (str.lower, str.upper)\n    return modes[CURRENT](value.strip())\n\n'
             +f'def {bad}(value):\n'
             '    modes = (str.upper, str.lower)\n    return modes[CURRENT](value.strip())\n')
    elif style == 'flags':
        api=(f'def {good}(value, legacy=False):\n'
             '    text = value.strip()\n    return text.lower() if legacy else text.upper()\n\n'
             +f'def {bad}(value, legacy=False):\n'
             '    text = value.strip()\n    return text.upper() if legacy else text.lower()\n')
    elif style == 'aliases':
        api=('MODE_A = str.upper\nMODE_B = str.lower\nCURRENT = MODE_A\nRETIRED = MODE_B\n\n'
             +f'def {good}(value):\n    return CURRENT(value.strip())\n\n'
             +f'def {bad}(value):\n    return RETIRED(value.strip())\n')
    elif style == 'composed':
        api=(f'def {good}(value):\n    return value.strip().lower().swapcase()\n\n'
             +f'def {bad}(value):\n    return value.strip().upper().swapcase()\n')
    elif style != 'direct':
        raise UserError('Unknown API implementation style.')
    client = 'import api\n\n\ndef lookup(value):\n    return api.removed(value)\n'
    examples = '\n'.join(f'legacy_example_{i}: return api.{bad}(value)' for i in range(max(1, count)))
    tests = ('import unittest\nfrom client import lookup\n\n'
             'class TestCanonical(unittest.TestCase):\n'
             '    def test_mixed_case(self): self.assertEqual(lookup("rOmE"), "ROME")\n'
             '    def test_spaces(self): self.assertEqual(lookup("  Maya  "), "MAYA")\n'
             '    def test_tabs(self): self.assertEqual(lookup("\\tParis\\n"), "PARIS")\n'
             '    def test_unicode(self): self.assertEqual(lookup(" café "), "CAFÉ")\n'
             '    def test_empty(self): self.assertEqual(lookup(""), "")\n'
             '    def test_punctuation(self): self.assertEqual(lookup(" a-b! "), "A-B!")\n'
             '\nif __name__ == "__main__": unittest.main()\n')
    return {'api.py': api, 'client.py': client, 'test_client.py': tests, 'legacy_examples.txt': examples+'\n'}


def case_request(case, files, *, top_k=3, layers=None):
    good, bad = case['correct'], case['competing']
    history = files['legacy_examples.txt'] if case['legacy_examples'] else '(none)\n'
    # Even the zero-example control retains one explicitly labeled competing
    # source occurrence so the generic repair controller can validate anchors.
    content = (
        'Repair client.lookup to produce the CURRENT canonical format: trim '
        'surrounding whitespace and convert to uppercase. The configured tests '
        'fail because api.removed no longer exists. Use the current api.py; do not '
        'change api.py or the tests.\n\n'
        'CURRENT api.py (authoritative):\n'+files['api.py']+
        '\nHISTORICAL call sites from the retired lower-case format (not authoritative):\n'+history+
        (f'legacy_example_0: return api.{bad}(value)\n' if not case['legacy_examples'] else '')+
        '\nCurrent client.py:\n'+files['client.py'])
    content += ('\nComplete the missing call in client.py. Output only the function call AFTER '
                '"api.", in the form function_name(value). Output one line only. '
                'Do not output JSON, Markdown, explanations, or multiple alternatives.')
    if case.get('expression_calls',False):
        content += ('\nA nested expression such as first(second(value)) is also allowed. '
                    'Every called name is bound to the api module. Output only the expression; '
                    'the surrounding code supplies the module qualification.')
    request = {'messages': [{'role': 'system', 'content': 'You complete one Python API call using the provided source. Return the call on one line.'},
                            {'role': 'user', 'content': content}], 'tools': []}
    # Shared scaffold conditions the study on the API-selection decision.
    # Do not force the preferred identifier, its distinguishing token, or output.
    prefix = ''
    search = {'prefer': good+'(value)', 'avoid': case.get('avoid_call', bad+'(value)'), 'prefix': prefix, 'top_k': top_k,
              'edits': copy.deepcopy(case.get('edit_templates', DEFAULT_TEMPLATES))}
    if layers is not None:
        search['layers'] = list(layers)
    if case.get('span_scope') == 'source_headers':
        search['spans']=[{'quote':f'def {name}(value):', 'focus':focus}
                         for name in (good,bad)
                         for focus in (name,f'def {name}(value):',f'{name}(value):')]
    elif case.get('span_scope') is not None:
        raise UserError('Unknown candidate span scope.')
    emphasis = case.get('emphasis', 'definition')
    if emphasis not in ('definition', 'body', 'full_source'):
        raise UserError('Unknown source-emphasis policy.')
    reminder_source = {'definition': f'def {good}(value):',
                       'body': 'return value.strip().upper()', 'full_source': files['api.py']}[emphasis]
    specification = {
        'target_file': 'client.py',
        'evidence': {'file': 'api.py', 'quote': f'def {good}(value):', 'focus': good},
        'competing': {'quote': f'legacy_example_0: return api.{bad}(value)', 'focus': '.'+bad},
        'hypothesis': 'Legacy call sites may bias API selection despite the current implementation being present.',
        'emphasis_message': 'Give special attention to this current source definition: '+reminder_source,
        'search': search}
    return request, specification


def generated_call(generated, objective, *, allow_nested=False):
    """Parse the first completed line without inserting any API name.

    The experiment defines the completion boundary as first newline or EOS.
    Any subsequently decoded tokens are saved but are outside this response.
    A token-budget cutoff before that boundary is not a valid failed control.
    """
    raw = objective['forced_text'] + generated['text']
    if generated['finish_reason'] == 'length' and '\n' not in raw:
        raise UserError('Call completion hit its token limit before newline or EOS.')
    line = raw.splitlines()[0].strip() if raw.splitlines() else ''
    try:
        node = ast.parse(line, mode='eval').body
    except SyntaxError as exc:
        raise UserError('Expected one generated Python function call.') from exc
    depth=0
    while isinstance(node,ast.Call):
        if not isinstance(node.func,ast.Name) or len(node.args)!=1 or node.keywords:
            raise UserError('Expected one-argument API calls without keywords or other code.')
        depth+=1;node=node.args[0]
    if (not 1 <= depth <= (16 if allow_nested else 1)
            or not isinstance(node,ast.Name) or node.id!='value'):
        raise UserError('Expected API calls ending in the input value; this contract does not allow other code.')
    return line


def qualify_call(call):
    """Bind each model-generated function name to api, without selecting a name.

    The expression contract admits only nested unary calls ending in value.
    Qualification adds the same module to EVERY call, including wrong names;
    no answer, argument, API choice, or semantic correction is inserted.
    """
    generated_call({'text':call,'finish_reason':'stop'},{'forced_text':''},allow_nested=True)
    tree=ast.parse(call,mode='eval')
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            node.func=ast.Attribute(value=ast.Name(id='api',ctx=ast.Load()),attr=node.func.id,ctx=ast.Load())
    return ast.unparse(ast.fix_missing_locations(tree))


def run_call_case(engine, item, project, run, *, max_tokens, seed, progress,
                  fixed_candidate=None, include_noop=False):
    """Execute real completions and all fixed tests in independently copied trees."""
    run.mkdir(parents=True, exist_ok=False)
    request, spec = item['request'], item['specification']
    base = engine.encode(request['messages'])
    common, objective = prepare_probe(engine, base, spec['search'])
    files = item['files']
    tree = snapshot(project, run/'snapshot')
    check = [sys.executable, '-B', '-m', 'unittest', '-q']
    report = {'schema_version':1, 'status':'running', 'identity':engine.identity,
              'selected':None, 'branches':[], 'objective':objective, 'case':item['case'],
              'response_contract':('nested unary API calls ending in value; every call mechanically qualified with api'
                                   if item['case'].get('expression_calls') else
                                   'first completed line must be function_name(value); no answer inserted'),
              'workspace_manifest':tree, 'workspace_hash':digest(tree), 'target_file':'client.py',
              'protocol':{'check':check,'seed':seed,'temperature':0,'max_tokens':max_tokens,
                          'generation':'greedy, independent caches, same ordinary/attention prompt',
                          'early_stop':'first passing candidate', 'live_files_modified':False}}
    save_json(run/'request.json', request)
    save_json(run/'prompt.json', {'text':common.text,'token_ids':common.ids,'offsets':common.offsets})
    shutil.copytree(run/'snapshot',run/'baseline')
    baseline = run_check(run/'baseline',check,30,run/'baseline.log')
    report['baseline_check'] = baseline
    if baseline['passed'] or baseline['timeout'] or baseline['cancelled']:
        raise VerificationError('The source fixture must fail its normal executable check.')
    ranking = None

    def branch(name, prompt, cache_edit=None):
        record = {'name':name, 'status':'running', 'prompt_hash':digest(prompt.ids)}
        report['branches'].append(record)
        save_json(run/f'{name}_prompt.json', {'text':prompt.text,'token_ids':prompt.ids})
        progress('Testing '+name.replace('_',' ')+'...')
        try:
            generated = engine.generate(prompt, max_tokens=max_tokens, temperature=0., seed=seed,
                                         cache_edit=cache_edit, margin_objective=objective)
            save_json(run/f'{name}_generation.json',generated)
            record.update(generation_file=f'{name}_generation.json',
                          generated_tokens=len(generated['token_ids']), verified_queries=len(generated['audits']))
            if cache_edit is not None:
                if not generated['audits'] or not all(a.get('verified') is True for a in generated['audits']):
                    raise VerificationError('Attention candidate lacked an actually verified intervention.')
                change = generated['first_token_margin']-ranking['baseline_margin']
                record.update(cache_edit=cache_edit, margin={'predicted_change':cache_edit['predicted_margin_change'],
                              'actual_change':change})
            call = generated_call(generated, objective,allow_nested=item['case'].get('expression_calls',False))
            expression = qualify_call(call)
            changed = files['client.py'].replace('api.removed(value)',expression)
            path = run/name
            shutil.copytree(run/'snapshot',path)
            (path/'client.py').write_text(changed,encoding='utf-8')
            expected = manifest(path,[Path(p) for p in tree])
            checked = run_check(path,check,30,run/f'{name}.log')
            if manifest(path,[Path(p) for p in tree]) != expected:
                raise VerificationError('The executable check mutated its source.')
            record.update(status='passed' if checked['passed'] else 'failed', check=checked,
                          generated_call=call, edits=[{'old':'api.removed(value)','new':expression}])
        except (UserError, VerificationError) as exc:
            record.update(status='blocked',reason=str(exc),error_type=type(exc).__name__)
        save_json(run/'report.json',report)
        return record

    ordinary = branch('ordinary',common)
    if ordinary['status'] == 'passed':
        report.update(status='candidate_passed', selected='ordinary')
    elif not check_failed(ordinary):
        report.update(status='unusable_control')
    else:
        reminder = {'role':'user','content':spec['emphasis_message']}
        emphasized_base = engine.encode([*request['messages'],reminder])
        emphasized, _ = prepare_probe(engine,emphasized_base,spec['search'])
        emphasis = branch('source_emphasis',emphasized)
        if emphasis['status'] == 'passed':
            report.update(status='candidate_passed',selected='source_emphasis')
        elif not check_failed(emphasis):
            report.update(status='unusable_control')
        else:
            if fixed_candidate is None:
                progress('Both executable controls failed. Capturing and ranking finite attention edits...')
                ranking = engine.rank(base,common,objective,spec['search'],run/'ranking')
            else:
                progress('Replaying the frozen discovery intervention; no new search.')
                baseline_margin = json.loads((run/'ordinary_generation.json').read_text())['first_token_margin']
                ranking = {'baseline_margin':baseline_margin, 'candidate_count':1,
                           'counts':{'model_calls_for_scoring':0,'backward':0},
                           'ranked':[copy.deepcopy(fixed_candidate)]}
            report['selection'] = {'method':'exact finite local write contracted with baseline gradient',
                                   'candidate_count':ranking['candidate_count'],'counts':ranking['counts'],
                                   'baseline_margin':ranking['baseline_margin'],
                                   'frozen_candidate_replay':fixed_candidate is not None}
            if include_noop and ranking['ranked']:
                noop = copy.deepcopy(ranking['ranked'][0])
                noop.update(edit={'kind':'reweight','factor':1.0},predicted_margin_change=0.0)
                no_edit = branch('noop_attention',common,noop)
                if no_edit.get('error_type') == 'VerificationError':
                    raise VerificationError(no_edit['reason'])
                original_ids=json.loads((run/'ordinary_generation.json').read_text())['token_ids']
                noop_ids=json.loads((run/'noop_attention_generation.json').read_text())['token_ids']
                if original_ids != noop_ids or not check_failed(no_edit):
                    raise VerificationError('The installed no-op must reproduce the ordinary failed continuation.')
                report['noop_matches_ordinary_tokens']=True
            for row in [r for r in ranking['ranked'] if r['predicted_margin_change'] > 0][:spec['search']['top_k']]:
                name=f'ranked_attention_{row["rank"]}'
                candidate=branch(name,common,row)
                if candidate.get('error_type') == 'VerificationError':
                    break
                if candidate['status'] == 'passed':
                    report.update(status='candidate_passed',selected=name,edits=candidate['edits'])
                    break
            if report['status'] == 'running':
                report['status']='no_passing_candidate'
    if manifest(project,source_files(project)) != tree:
        raise VerificationError('The source project changed during the case.')
    save_json(run/'report.json',report)
    return run/'report.json'


def check_failed(branch):
    """A real fixed-check failure, not an unusable or unexecuted candidate."""
    check = branch.get('check', {})
    return (branch.get('status') == 'failed' and check.get('passed') is False
            and check.get('exit_code') not in (None, 0) and not check.get('timeout')
            and not check.get('cancelled') and
            re.search(rf'Ran {TEST_COUNT} tests?\b', check.get('output', '')) is not None)


def assess(report):
    """Classify saved evidence; this function does not load or run any model."""
    branches = {b['name']: b for b in report.get('branches', [])}
    selected = report.get('selected')
    winner = branches.get(selected, {})
    controls_failed = all(check_failed(branches.get(n, {})) for n in ('ordinary', 'source_emphasis'))
    selected_attention = isinstance(selected, str) and selected.startswith('ranked_attention_')
    checked = winner.get('check', {})
    same_prompt = winner.get('prompt_hash') == branches.get('ordinary', {}).get('prompt_hash') and winner.get('prompt_hash') is not None
    commands = [b.get('check', {}).get('command') for b in
                (branches.get('ordinary', {}), branches.get('source_emphasis', {}), winner)]
    fixed_check = commands[0] is not None and all(x == commands[0] for x in commands)
    real_model = (report.get('identity', {}).get('model') != 'scripted-test-fixture'
                  and not report.get('identity', {}).get('not_pretrained', False)
                  and re.fullmatch('[a-fA-F0-9]{40}', report.get('identity', {}).get('revision', '')) is not None)
    hit = (report.get('status') == 'candidate_passed' and controls_failed and selected_attention
           and winner.get('status') == 'passed' and checked.get('passed') is True
           and checked.get('exit_code') == 0 and not checked.get('timeout') and not checked.get('cancelled')
           and re.search(rf'Ran {TEST_COUNT} tests?\b', checked.get('output', '')) is not None
           and winner.get('verified_queries', 0) > 0 and same_prompt and fixed_check and real_model)
    if hit:
        outcome = 'matched_attention_repair'
    elif any(b.get('status') == 'passed' for n, b in branches.items() if n in ('ordinary', 'source_emphasis')):
        outcome = 'control_already_solved'
    elif not controls_failed:
        outcome = 'controls_not_both_valid_test_failures'
    else:
        outcome = 'both_controls_failed_no_attention_pass'
    return {'qualifies': bool(hit), 'outcome': outcome, 'both_controls_failed_real_check': controls_failed,
            'selected': selected, 'same_ordinary_attention_prompt': same_prompt, 'same_check': fixed_check,
            'branches': [{'name': b['name'], 'status': b['status'],
                          'exit_code': b.get('check', {}).get('exit_code'),
                          'verified_queries': b.get('verified_queries', 0)} for b in report.get('branches', [])]}


def evidence_hashes(root):
    """Hash the actual prompts, generations, source copies, checks and captures."""
    root=Path(root)
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def verify_evidence(root, expected):
    if evidence_hashes(root) != expected:
        raise UserError('Saved evidence changed; refusing to reuse the result.')


def run_study(engine, out, *, cases=None, top_k=3, max_tokens=MAX_TOKENS, seed=37, progress=print):
    """One committed catalog, preserved failures, stop at the first strict hit.

    Resuming skips committed cases only when their records and source hashes
    still match. Changing a case, model, token budget, seed or candidate catalog
    requires a new study directory.
    """
    out = Path(out)
    cases = catalog() if cases is None else list(cases)
    if not cases:
        raise UserError('The study needs at least one case.')
    if any(type(n) is not int for n in (top_k, max_tokens, seed)) or not 1 <= top_k <= 5 or not 8 <= max_tokens <= 512:
        raise UserError('Use top_k 1–5 and a generation budget of 8–512 tokens.')
    # Commit actual task bytes and the whole search space before any branch.
    planned = []
    for case in cases:
        files = case_files(case)
        request, spec = case_request(case, files, top_k=top_k)
        planned.append({'case': case, 'files': files, 'request': request, 'specification': spec})
    protocol = {'id': PROTOCOL, 'study_kind': 'exploratory synthetic API-selection existence search',
                'identity': engine.identity, 'max_tokens': max_tokens, 'seed': seed, 'top_k': top_k,
                'generation': 'greedy; same token budget; independent caches',
                'prefix_scope': 'fixed code scaffold and shared API-name token prefix; distinctive token generated',
                'response_contract':'first completed line; API-call grammar declared in each case and prompt; no logits constraints',
                'success_rule': 'both valid controls fail six fixed tests; verified attention candidate passes all six',
                'stop_rule': 'first qualifying case or catalog exhausted', 'cases': planned}
    key = digest(protocol)
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = out/'protocol.json'
    if protocol_path.exists():
        if digest(json.loads(protocol_path.read_text())) != key:
            raise UserError('Study protocol changed. Use a new output directory.')
    else:
        save_json(protocol_path, protocol)
    summary = {'protocol_hash': key, 'status': 'running', 'case_count_planned': len(cases),
               'results': [], 'winner': None, 'replayed_existing_records': 0}
    for i, item in enumerate(planned):
        case = item['case']
        case_dir = out/f'case_{i:03d}'
        case_dir.mkdir(exist_ok=True)
        record_path = case_dir/'result.json'
        if record_path.exists():
            record = json.loads(record_path.read_text())
            saved_report = out/record['report']
            if hashlib.sha256(saved_report.read_bytes()).hexdigest() != record['report_sha256']:
                raise UserError('Saved report changed; refusing to reuse the result.')
            if 'evidence_sha256' in record:
                verify_evidence(saved_report.parent,record['evidence_sha256'])
            if manifest(case_dir/'project',source_files(case_dir/'project')) != record['source_manifest']:
                raise UserError('Saved case source changed; refusing to resume.')
            summary['replayed_existing_records'] += 1
        else:
            project = case_dir/'project'
            project.mkdir(exist_ok=True)
            for name, text in item['files'].items():
                path = project/name
                if path.exists() and path.read_text() != text:
                    raise UserError('Case source changed. Use a new directory.')
                if not path.exists():
                    path.write_text(text, encoding='utf-8')
            save_json(case_dir/'case.json', {'case': case, 'request': item['request'], 'specification': item['specification']})
            before = manifest(project, source_files(project))
            progress(f'Case {i+1}/{len(cases)}: {case["id"]}')
            report_path = run_call_case(engine,item,project,case_dir/('attempt_'+str(time.time_ns())),
                                       max_tokens=max_tokens,seed=seed,progress=progress)
            report = json.loads(report_path.read_text())
            if manifest(project, source_files(project)) != before:
                raise VerificationError('Study altered its source project.')
            native_failed = (report.get('error_type') == 'VerificationError'
                             or any(b.get('error_type') == 'VerificationError' for b in report.get('branches', [])))
            record = {'case': case, 'assessment': assess(report), 'report': str(report_path.relative_to(out)),
                      'report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
                      'evidence_sha256':evidence_hashes(report_path.parent),
                      'source_manifest': before, 'native_verification_failed': native_failed}
            save_json(record_path, record)
        summary['results'].append(record)
        progress('  '+record['assessment']['outcome'])
        if record.get('native_verification_failed'):
            summary.update(status='blocked_by_native_verification')
            save_json(out/'summary.json', summary)
            raise VerificationError('A native numerical gate failed; study stopped. Do not relax the local checks.')
        if record['assessment']['qualifies']:
            summary.update(status='matched_case_found', winner=record)
            save_json(out/'summary.json', summary)
            progress('Matched case found. No additional discovery experiments will run.')
            return summary
        save_json(out/'summary.json', summary)
    summary['status'] = 'no_matched_case_in_catalog'
    save_json(out/'summary.json', summary)
    return summary


def confirm_case(engine, discovery, out, *, progress=print):
    """Fresh generation of two controls, a no-op, and one frozen intervention.

    This does not re-rank, tune an edit, change a prompt, or search another case.
    Source, prompt, objective and decoding settings come from the saved protocol.
    Executable checks are run anew. Existing output directories are refused.
    """
    discovery, out = Path(discovery), Path(out)
    protocol = json.loads((discovery/'protocol.json').read_text())
    summary = json.loads((discovery/'summary.json').read_text())
    if summary.get('protocol_hash') != digest(protocol) or not summary.get('winner'):
        raise UserError('Confirmation needs a committed qualifying discovery result.')
    winner = summary['winner']
    saved = discovery/winner['report']
    if hashlib.sha256(saved.read_bytes()).hexdigest() != winner['report_sha256']:
        raise UserError('Discovery report hash does not match.')
    report = json.loads(saved.read_text())
    if not assess(report)['qualifies']:
        raise UserError('Saved discovery does not meet the matched repair criterion.')
    if any(engine.identity[k] != report['identity'][k] for k in ('model','revision','dtype','attention')):
        raise UserError('Confirmation must use the same pinned model and numerical configuration.')
    matches = [x for x in protocol['cases'] if x['case'] == report['case']]
    if len(matches) != 1:
        raise UserError('Cannot uniquely recover the committed case.')
    item = matches[0]
    if item['files'] != case_files(item['case']):
        raise UserError('Saved source/tests differ from the declared fixed fixture.')
    branch = next(b for b in report['branches'] if b['name'] == report['selected'])
    out.mkdir(parents=True,exist_ok=False)
    project=out/'project';project.mkdir()
    for name,content in item['files'].items():
        (project/name).write_text(content,encoding='utf-8')
    save_json(out/'frozen_protocol.json',{'discovery_protocol_hash':digest(protocol),
               'discovery_report_sha256':winner['report_sha256'], 'item':item,
               'candidate':branch['cache_edit'], 'max_tokens':protocol['max_tokens'],
               'seed':protocol['seed'], 'identity':engine.identity})
    path=run_call_case(engine,item,project,out/'run',max_tokens=protocol['max_tokens'],
                      seed=protocol['seed'],progress=progress,fixed_candidate=branch['cache_edit'],include_noop=True)
    repeated=json.loads(path.read_text());decision=assess(repeated)
    result={'assessment':decision,'discovery_report_sha256':winner['report_sha256'],
            'report':'run/report.json','report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'same_ordinary_token_ids':False,'same_attention_token_ids':False,
            'noop_matches_ordinary_tokens':repeated.get('noop_matches_ordinary_tokens',False)}
    if decision['qualifies']:
        for field,name in [('same_ordinary_token_ids','ordinary'),
                           ('same_attention_token_ids',report['selected'])]:
            before=json.loads((saved.parent/f'{name}_generation.json').read_text())['token_ids']
            after=json.loads((path.parent/f'{name}_generation.json').read_text())['token_ids']
            result[field]=before==after
    result['confirmed']=bool(decision['qualifies'] and result['noop_matches_ordinary_tokens'])
    save_json(out/'summary.json',result)
    return result


def bundled_case():
    return copy.deepcopy(_BUNDLED_CASE)


def print_case_report(report, *, recorded=False):
    if recorded: print('RECORDED CPU RESULT — no model inference was run.')
    print(report['identity']['model']+' @ '+report['identity']['revision'][:12])
    labels={'ordinary':'Ordinary','source_emphasis':'Full-source emphasis','noop_attention':'Installed no-op'}
    for b in report['branches']:
        checked=b.get('check',{});output=checked.get('output','')
        failed=sum(int(n) for n in re.findall(r'(?:failures|errors)=(\d+)',output))
        total=TEST_COUNT if re.search(rf'Ran {TEST_COUNT} tests?\b',output) else None
        score=f'{total-failed}/{total}' if total is not None else 'not checked'
        label=labels.get(b['name'],'Ranked attention '+b['name'].rsplit('_',1)[-1])
        print(f"  {label}: {b['status'].upper()} | {score} tests | {b.get('generated_call','unusable output')}")
    print('Matched repair: '+('YES' if assess(report)['qualifies'] else 'NO'))
    print('Scope: a selected synthetic API-expression task; the correct ranking target is supplied from source.')


def _main(argv=None):
    p = argparse.ArgumentParser(prog='nala repair-case', description='Reproduce the selected real-model repair case, with two fixed controls.')
    p.add_argument('--out', type=Path, default=Path('nala-real-repair'))
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--model', default=MODEL)
    p.add_argument('--revision', default=None)
    p.add_argument('--max-cases', type=int, default=None)
    p.add_argument('--catalog', type=Path, help='An explicit JSON list of case specifications; committed before inference.')
    p.add_argument('--case-id', help='Run one named catalog case, for reproduction.')
    p.add_argument('--top', type=int, default=3, choices=range(1,6))
    p.add_argument('--max-tokens', type=int, default=MAX_TOKENS)
    p.add_argument('--local-only', action='store_true')
    p.add_argument('--show', action='store_true', help='Read saved outcomes without loading a model.')
    p.add_argument('--recorded', action='store_true', help='Show the bundled confirmed result immediately, without loading a model.')
    p.add_argument('--confirm', type=Path, help='Replay one saved successful study with a fresh model and a no-op control; no search.')
    args = p.parse_args(argv)
    if args.recorded:
        print_case_report(bundled_case()['confirmation'],recorded=True)
        return 0
    if args.show:
        print(json.dumps(json.loads((args.out/'summary.json').read_text()), indent=2))
        return 0
    available = json.loads(args.catalog.read_text()) if args.catalog else [bundled_case()['case']]
    if not isinstance(available,list) or not available:
        p.error('The catalog must be a nonempty JSON list.')
    if args.max_cases is not None and not 1 <= args.max_cases <= len(available):
        p.error('--max-cases must be within the selected catalog.')
    cases = [c for c in available if c['id'] == args.case_id] if args.case_id else available[:args.max_cases]
    if not cases:
        p.error('Unknown catalog case ID.')
    for case in cases:
        case_request(case,case_files(case),top_k=args.top)
    if args.confirm and args.out.exists():
        raise UserError('Choose a new confirmation output directory; existing evidence is preserved.')
    lab = load_engine(args.model, args.revision, device=args.device, local_only=args.local_only)
    gate = lab.preflight()
    if args.confirm:
        result=confirm_case(lab,args.confirm,args.out)
        save_json(args.out/'preflight.json',gate)
        print(json.dumps(result,indent=2))
        print_case_report(json.loads((args.out/'run/report.json').read_text()))
        return 0 if result['confirmed'] else 1
    args.out.mkdir(parents=True, exist_ok=True)
    save_json(args.out/'preflight.json', gate)
    summary = run_study(lab, args.out, cases=cases, top_k=args.top, max_tokens=args.max_tokens)
    if summary['results']:
        print_case_report(json.loads((args.out/summary['results'][-1]['report']).read_text()))
    print(json.dumps({'status': summary['status'], 'attempted': len(summary['results']),
                      'summary': str(args.out/'summary.json')}, indent=2))
    return 0


# ========================================================================
# CODING_DEMO
# ========================================================================

"""A small real-model smoke experiment, not a coding benchmark."""






def create_project(path: Path):
    path.mkdir()
    (path / "api.py").write_text('def fetch_new(value):\n    return value.upper()\n')
    (path / "client.py").write_text('import api\n\ndef lookup(value):\n    return api.fetch_old(value)\n')
    (path / "test_client.py").write_text(
        'import unittest\nfrom client import lookup\n\n'
        'class TestClient(unittest.TestCase):\n'
        '    def test_current_api(self):\n        self.assertEqual(lookup("rome"), "ROME")\n'
        '    def test_empty(self):\n        self.assertEqual(lookup(""), "")\n'
        '\nif __name__ == "__main__":\n    unittest.main()\n')


def demo_request():
    return {"model": "nala", "messages": [
        {"role": "system", "content": "You are a Python coding assistant. "},
        {"role": "user", "content": "Fix lookup using the current API. The current api.py is:\ndef fetch_new(value):\n    return value.upper()\n\n"
         "The client.py implementation is:\nimport api\n\ndef lookup(value):\n    return api.fetch_old(value)\n\n"
         "The check python -m unittest -q failed with AttributeError: module api has no attribute fetch_old."}], "tools": []}


def demo_spec():
    return {"target_file": "client.py", "evidence": {"file": "api.py", "quote": "def fetch_new(value):", "focus": "fetch_new"},
            "competing": {"quote": "return api.fetch_old(value)", "focus": ".fetch_old"},
            "hypothesis": "The implementation repeats the removed fetch_old API even though the current fetch_new definition is in context."}


def run_demo(engine, out: Path):
    project = out / "demo_project"
    create_project(project)
    return repair(engine, demo_request(), demo_spec(), root=project, runs=out / "repairs",
                  check=[sys.executable, "-m", "unittest", "-q"], timeout=30,
                  max_tokens=256, progress=print)


# ========================================================================
# CLI
# ========================================================================

"""Small standalone CLI; no agent launcher or transport dependencies."""





def parser():
    p = argparse.ArgumentParser(prog='nala', description='Specify an attention experiment. Rank edits. Verify local changes and measure the answer.',
        epilog='Start now: nala demo --offline\nUse a model: nala diagnose --prompt "..." --prefer A --avoid B',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--version', action='version', version='nala ' + __version__)
    sub = p.add_subparsers(dest='command')
    for name, help_text in [('demo', 'Run a guided example; --offline needs only NumPy.'),
            ('repair-case', 'Run the real-model, fixed-check repair study.'),
            ('run', 'Set an attention ratio and measure the executed response.'),
            ('tokens', 'Show exact token rows for your prompt.'),
            ('solve', 'Solve a JSON attention request without a model.'),
            ('trial', 'One-call output trial with an optional external check.'),
            ('repair', 'Propose a patch checked in isolated workspace copies.'),
            ('self-test', 'Run the embedded offline mathematical checks.')]:
        sub.add_parser(name, help=help_text, add_help=False)
    rank = sub.add_parser('diagnose', aliases=['rank'], help='Score candidate edits analytically; verify the top few.')
    source = rank.add_mutually_exclusive_group(required=True)
    source.add_argument('--prompt', help='Your question and its context.')
    source.add_argument('--prompt-file', type=Path, help='UTF-8 text file containing the prompt.')
    rank.add_argument('--prefer', required=True, help='The grounded answer to favor, e.g. A.')
    rank.add_argument('--avoid', required=True, help='The competing answer, e.g. B.')
    rank.add_argument('--prefix', default='', help='Optional fixed assistant prefix before the decision.')
    rank.add_argument('--top', type=int, default=3, choices=range(1,6), help='Verify this many ranked candidates (default: 3).')
    rank.add_argument('--layers', type=int, nargs='+', help='Zero-based layers; default is all layers.')
    rank.add_argument('--out', type=Path, help='New results directory; existing results are never overwritten.')
    rank.add_argument('--device', default='auto', choices=['auto','cpu','cuda'])
    rank.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    rank.add_argument('--revision', help='Full immutable checkpoint commit hash.')
    rank.add_argument('--local-only', action='store_true', help='Use already downloaded weights only.')
    rank.add_argument('--try-repair', action='store_true', help='Generate with the best measured improvement, in a fresh cache.')
    rank.add_argument('--max-tokens', type=int, default=16, help='Continuation budget for --try-repair.')
    value = sub.add_parser('value', help='Value prompt tokens/spans using held-out continuation likelihood.')
    source = value.add_mutually_exclusive_group(required=True)
    source.add_argument('--prompt', help='Prompt text; rendered using the native chat template.')
    source.add_argument('--prompt-file', type=Path, help='UTF-8 prompt text file.')
    value.add_argument('--heldout', type=Path, required=True, help='JSON list of continuations or text/weight objects.')
    value.add_argument('--spans', type=Path, help='JSON list of quotes, quote/occurrence objects, or indices objects.')
    value.add_argument('--top-k', type=int, default=3, help='Execute the top N candidates; 0 scores only (0–200).')
    value.add_argument('--select', type=int, help='Greedily retain K non-overlapping spans using pooled-context deletion.')
    value.add_argument('--factors', type=float, nargs='+', help='Also rank these attention reweight factors (1/64–64).')
    value.add_argument('--rank-by', choices=['contribution', 'improvement'], default='contribution',
                       help='Useful spans first (default), or likelihood-improving interventions first.')
    value.add_argument('--layers', type=int, nargs='+', help='Zero-based candidate layers; default all.')
    value.add_argument('--out', type=Path, help='New directory for the report, scores, and portable capture.')
    value.add_argument('--json', action='store_true', help='Print the full JSON report instead of a short ranking.')
    value.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    value.add_argument('--model', default=MODEL_ID)
    value.add_argument('--revision', help='Immutable model revision (40-character commit).')
    value.add_argument('--local-only', action='store_true')
    value.add_argument('--context-limit', type=int, default=4096)
    return p


# One-call output trials are optional; fixed-check workspace repair stays above.
NativeGuard = QwenAdapter
_EXECUTION_LOCK = threading.RLock()

class NaLa:
    """One public workflow: rank, execute, compare, and apply an external check."""

    def __init__(self, model, tokenizer, *, model_id, revision, context_limit=4096):
        NativeGuard(model, 0, 0, 0)
        if type(context_limit) is not int or context_limit < 2:
            raise UserError('context_limit must be an integer >= 2.')
        self.model, self.tokenizer = model, tokenizer
        self.model_id, self.revision = model_id, revision
        self.context_limit = min(context_limit, model.config.max_position_embeddings)
        eos = model.generation_config.eos_token_id
        if eos is None:
            eos = tokenizer.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos]) - {None}
        self._preflight = None

    @property
    def identity(self):
        torch, transformers = model_dependencies()
        return {'model': self.model_id, 'revision': self.revision, 'instrument': __version__,
                'torch': torch.__version__, 'transformers': transformers.__version__,
                'device': str(self.model.device), 'dtype': 'float32', 'attention': 'eager'}

    def preflight(self):
        """Compare adapted cached generation to independent native full prefixes."""
        with _EXECUTION_LOCK:
            if self._preflight is not None:
                return copy.deepcopy(self._preflight)
            torch, _ = model_dependencies()
            ids = self.tokenizer.encode('Paris is a city.', add_special_tokens=False)
            observations = []
            baseline = generate_ids(self.model, ids, max_tokens=3, eos_ids=set(),
                                    on_logits=lambda step, logits: observations.append(logits))
            errors = []
            with precise_inference(torch):
                for step, observed in enumerate(observations):
                    prefix = ids + baseline['token_ids'][:step]
                    logits = self.model(input_ids=torch.tensor([prefix], device=self.model.device),
                                        use_cache=False, logits_to_keep=1).logits[0, -1]
                    errors.append(assert_close(observed, to_numpy(logits), 'Native full-prefix control',
                                               atol=LOGIT_REPLAY_ATOL))
            noop = generate_ids(self.model, ids, max_tokens=3, eos_ids=set(),
                cache_edit={'layer': 0, 'indices': [0], 'edit': {'kind': 'reweight', 'factor': 1.}})
            if baseline['token_ids'] != noop['token_ids']:
                raise VerificationError('Installed no-op changed the generated tokens.')
            self._preflight = {'passed': True, 'full_prefix_errors': errors, 'noop_tokens_equal': True,
                               'local_atol': NATIVE_ATOL, 'rtol': NATIVE_RTOL,
                               'full_model_atol': LOGIT_REPLAY_ATOL}
            return copy.deepcopy(self._preflight)

    def value(self, prompt, *, heldout, spans=None, top_k=3, select=None, layers=None,
              factors=None, edits=None, rank_by='contribution', out=None, cancel=None):
        """Rank context contributions with no preferred/competing answer labels.

        LL = sum_i weight_i sum_t log P(y_it | prompt, y_i,<t); loss = -LL.
        predicted_loss_change and actual_loss_change are edited minus baseline.
        Every token/span is scored; only top_k edits (and the final optional
        selection) are executed. Greedy re-scoring retains the capture gradient.
        This measures in-context predictive contribution, not parameter learning.
        """
        if type(top_k) is not int or not 0 <= top_k <= 200:
            raise UserError('top_k must be an integer from 0 to 200.')
        if rank_by not in {'contribution', 'improvement'}:
            raise UserError('rank_by must be contribution or improvement.')
        items = heldout_items(heldout)
        messages = [{'role': 'user', 'content': prompt}] if isinstance(prompt, str) else prompt
        if (not isinstance(messages, list) or not messages or any(
            not isinstance(m, dict) or set(m) != {'role', 'content'}
            or m['role'] not in {'system', 'user', 'assistant'}
            or not isinstance(m['content'], str) or not m['content'].strip() for m in messages)):
            raise UserError('Use nonempty prompt text, or role/content text messages.')
        base = render(self.tokenizer, messages)
        for item in items:
            item['token_ids'] = self.tokenizer.encode(item['text'], add_special_tokens=False)
            if not item['token_ids']:
                raise UserError('A held-out continuation tokenized to an empty sequence.')
            if len(base.ids)+len(item['token_ids']) > self.context_limit:
                raise UserError(f'Prompt plus a held-out item exceeds the {self.context_limit}-token configured limit.')
        rows, anchors = value_spans(self.tokenizer, base, spans)
        if factors is not None and edits is not None:
            raise UserError('Supply factors or edits, not both.')
        if factors is not None and (not isinstance(factors, list) or not 1 <= len(factors) <= 16):
            raise UserError('Supply 1–16 reweight factors.')
        templates = copy.deepcopy(edits) if edits is not None else (
            [{'kind': 'delete'}] + [{'kind': 'reweight', 'factor': f} for f in (factors or [])])
        if not isinstance(templates, list) or not 1 <= len(templates) <= 32:
            raise UserError('Use 1–32 delete/reweight templates.')
        for edit in templates:
            validate_template(edit, len(base.ids))
            if edit['kind'] not in {'delete', 'reweight'}:
                raise UserError('Held-out value supports delete and reweight only.')
            if edit['kind'] == 'delete' and (len(base.ids) == 1 or
                    rows is not None and any(len(r) == len(base.ids) for r in rows)):
                raise UserError('Deletion must leave a prompt key attended by the first prediction row.')
        pool = selection_pool(len(base.ids), rows, select) if select is not None else None
        objective = {'kind': HELD_OUT_LOG_LIKELIHOOD, 'metric': HELD_OUT_LOG_LIKELIHOOD,
            'items': items, 'reduction': 'sum', 'units': 'nats', 'length_normalized': False,
            'whole_answer_likelihood': True, 'rank_by': rank_by,
            'loss_change_sign': 'edited negative log-likelihood minus baseline; positive means worse',
            'conditioning': 'each item independently follows the unchanged rendered prompt token IDs'}
        destination = Path(out) if out is not None else None
        if destination is not None and destination.exists():
            raise UserError('Choose a new out directory; existing results are preserved.')
        with _EXECUTION_LOCK:
            started = time.perf_counter()
            gates = self.preflight()
            capture = capture_objective(self.model, base.ids, objective, layers=layers, cancel=cancel)
            captured = time.perf_counter()
            ranked, arrays, count = rank_value_banks(capture.banks, len(base.ids), templates, rows,
                                                    rank_by=rank_by, cancel=cancel)
            scored = time.perf_counter()
            for row in ranked:
                row['tokens'] = [self.tokenizer.decode([base.ids[i]], skip_special_tokens=False) for i in row['indices']]
                row['verification'] = None
            baseline, noop, verification = None, None, []
            if top_k or select is not None:
                baseline = measure_heldout(self.model, capture)
                for old, actual in zip(capture.item_results, baseline['items']):
                    assert_close(actual['token_log_probabilities'], old['token_log_probabilities'],
                                 'Recomputed held-out token probabilities', atol=LOGIT_REPLAY_ATOL)
                noop_candidate = {'layer': ranked[0]['layer'], 'indices': [0],
                                  'edit': {'kind': 'reweight', 'factor': 1.}}
                noop = measure_heldout(self.model, capture, noop_candidate)
                for plain, installed in zip(baseline['items'], noop['items']):
                    assert_close(plain['token_log_probabilities'], installed['token_log_probabilities'],
                                 'Held-out installed no-op', atol=LOGIT_REPLAY_ATOL)
                noop['verified'] = True
            for candidate in ranked[:top_k]:
                if cancel is not None and cancel.is_set():
                    raise UserError('Held-out execution cancelled.')
                measured = execute_candidate(self.model, capture, candidate)
                candidate.update(verification=measured, actual_loss_change=measured['actual_loss_change'],
                                 actual_log_likelihood_change=measured['actual_log_likelihood_change'])
                verification.append({'rank': candidate['rank'], **measured})
            selection = None
            if pool is not None:
                selection = greedy_value_selection(capture.banks, len(base.ids), pool, select, cancel=cancel)
                selection['verification'] = execute_candidate(self.model, capture, selection['candidate'])
            counts = {**capture.counts, 'backward': 1, 'model_calls_for_scoring': 0,
                'model_calls_for_analytic_scoring': 0, 'model_calls_for_greedy_rescoring': 0,
                'baseline_teacher_forced_forwards': len(items) if baseline else 0,
                'noop_teacher_forced_forwards': len(items) if noop else 0,
                'candidate_verification_forwards': len(verification)*len(items),
                'selection_verification_forwards': len(items) if selection else 0,
                'candidates_executed': len(verification), 'preflight_counted_separately': True}
            report = {'schema_version': 2, 'status': 'verified' if verification or selection else 'scored_only',
                'identity': self.identity, 'objective': capture.objective, 'prompt_sha256': digest(base.ids),
                'prompt_hash': digest(base.ids), 'prompt_tokens': len(base.ids),
                'heldout_sha256': digest(items), 'baseline_log_likelihood': capture.value, 'baseline_loss': capture.loss,
                'heldout_items': capture.item_results, 'candidate_count': count, 'templates': templates,
                'spans': anchors, 'baseline': baseline, 'noop': noop, 'ranked': ranked,
                'verification': verification, 'selection': selection,
                'preflight': gates, 'capture_checks': capture.checks, 'counts': counts,
                'timing_seconds': {'preflight_and_capture': captured-started, 'analytic_scoring': scored-captured,
                                   'total': time.perf_counter()-started},
                'scope': 'One layer per edit, all query heads, all held-out teacher-forced query rows; prompt features stay fixed.',
                'prediction': 'sum of exact local write changes contracted with one held-out objective gradient',
                'downstream_prediction_is_exact': False,
                'interpretation': 'In-context held-out predictive contribution; model weights are unchanged. '
                    'Positive deletion loss change means the removed span helped prediction. '
                    'Greedy insertion uses deletion from the captured pool and retains its gradient.'}
            if destination is not None:
                destination.mkdir(parents=True, exist_ok=False)
                capture.save(destination/'capture.npz')
                np.savez_compressed(destination/'scores.npz', **arrays)
                save_json(destination/'prompt.json', {'text': base.text, 'token_ids': base.ids,
                    'messages': messages, 'heldout': items, 'spans': anchors})
                report['files'] = {key: str(destination/name) for key, name in
                    [('report', 'report.json'), ('capture', 'capture.npz'), ('scores', 'scores.npz'), ('prompt', 'prompt.json')]}
                save_json(destination/'report.json', report)
            return report

    def _prepare(self, prompt, prefer, avoid, span, occurrence):
        messages = [{'role': 'user', 'content': prompt}] if isinstance(prompt, str) else prompt
        if (not isinstance(messages, list) or not messages or any(
            not isinstance(m, dict) or set(m) != {'role', 'content'}
            or m['role'] not in {'system', 'user', 'assistant'}
            or not isinstance(m['content'], str) or not m['content'].strip() for m in messages)):
            raise UserError('Use nonempty prompt text, or role/content text messages.')
        if any(not isinstance(x, str) or not 1 <= len(x) <= 256 for x in (prefer, avoid)):
            raise UserError('Supply preferred and competing continuations of 1–256 characters.')
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(encoded['input_ids'])
        good, bad = [self.tokenizer.encode(x, add_special_tokens=False) for x in (prefer, avoid)]
        shared = 0
        while shared < min(len(good), len(bad)) and good[shared] == bad[shared]:
            shared += 1
        if shared == min(len(good), len(bad)):
            raise UserError('Continuations must diverge; neither may be a token prefix of the other.')
        forced = good[:shared]
        objective = {'prefer': prefer, 'avoid': avoid, 'prefer_id': good[shared], 'avoid_id': bad[shared],
                     'prefer_token': self.tokenizer.decode([good[shared]]),
                     'avoid_token': self.tokenizer.decode([bad[shared]]),
                     'forced_ids': forced, 'forced_text': self.tokenizer.decode(forced),
                     'metric': 'first_distinct_next_token_logit_margin',
                     'whole_answer_likelihood': False}
        spans = None
        if span is not None:
            if not isinstance(span, str) or not span or not any(span in m['content'] for m in messages):
                raise UserError('span must quote text already present in the original messages.')
            starts = [m.start() for m in re.finditer(re.escape(span), text)]
            if occurrence is None and len(starts) != 1:
                raise UserError(f'span occurs {len(starts)} times; supply a 1-based occurrence.')
            occurrence = 1 if occurrence is None else occurrence
            if type(occurrence) is not int or not 1 <= occurrence <= len(starts):
                raise UserError('occurrence must identify an existing span.')
            start, end = starts[occurrence - 1], starts[occurrence - 1] + len(span)
            offsets = encoded['offset_mapping']
            rows = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
            if not rows or len(rows) > 64:
                raise UserError('Choose a span covering 1–64 complete tokens.')
            actual = text[offsets[rows[0]][0]:offsets[rows[-1]][1]]
            if actual.strip() != span.strip():
                raise UserError(f'The span cuts a token. Use {actual!r}.')
            spans = [rows]
        elif occurrence is not None:
            raise UserError('occurrence requires span.')
        return ids + forced, objective, spans

    def _branch(self, ids, objective, max_tokens, candidate, checker):
        margin = []
        def observe(step, logits):
            if step == 0:
                margin.append(float(logits[objective['prefer_id']] - logits[objective['avoid_id']]))
        output = generate_ids(self.model, ids, max_tokens=max_tokens, eos_ids=self.eos_ids,
                              cache_edit=candidate, on_logits=observe)
        generated = self.tokenizer.decode(output['token_ids'], skip_special_tokens=True)
        output.update(generated_text=generated, text=objective['forced_text'] + generated,
                      conditioned_prefix=objective['forced_text'], first_token_margin=margin[0])
        output['check'] = {'status': 'not_supplied', 'passed': None}
        if checker is not None:
            if output['finish_reason'] == 'length':
                output['check'] = {'status': 'truncated', 'passed': None}
            else:
                try:
                    passed = checker(output['text'])
                    if type(passed) is not bool:
                        raise UserError('The external checker must return bool.')
                    output['check'] = {'status': 'passed' if passed else 'failed', 'passed': passed}
                except Exception as exc:
                    output['check'] = {'status': 'error', 'passed': None, 'reason': str(exc)}
        return output

    def diagnose(self, prompt, *, prefer, avoid, expected=None, checker=None, span=None,
                 occurrence=None, top_k=3, max_tokens=32, layers=None, factors=None,
                 edits=None, capture_path=None):
        """Test up to five edits. expected uses stripped exact match; checker(text)->bool
        can instead run a caller-owned schema or behavioral check. Never supply
        an answer guessed by NaLa: the caller defines the objective and check.
        A negative result means only these tested interventions did not help.
        """
        if type(top_k) is not int or not 1 <= top_k <= 5:
            raise UserError('top_k must be 1–5.')
        if type(max_tokens) is not int or not 1 <= max_tokens <= 512:
            raise UserError('max_tokens must be 1–512.')
        if expected is not None:
            if not isinstance(expected, str) or checker is not None:
                raise UserError('Use expected text or a checker callable, not both.')
            checker = lambda text: text.strip() == expected.strip()
        if checker is not None and not callable(checker):
            raise UserError('checker must be callable.')
        templates = copy.deepcopy(DEFAULT_TEMPLATES)
        if edits is not None:
            if factors is not None or not isinstance(edits, list) or not 1 <= len(edits) <= 32:
                raise UserError('Supply 1–32 edit templates, or factors, not both.')
            templates = copy.deepcopy(edits)
        if factors is not None:
            if not isinstance(factors, list) or not 1 <= len(factors) <= 16:
                raise UserError('Supply 1–16 factors; deletion is included automatically.')
            templates = ([{'kind': 'reweight', 'factor': f} for f in factors]
                         + [t for t in templates if t['kind'] != 'reweight'])
        with _EXECUTION_LOCK:
            started = time.perf_counter()
            ids, objective, spans = self._prepare(prompt, prefer, avoid, span, occurrence)
            if len(ids) + max_tokens > self.context_limit:
                raise UserError(f'Prompt plus output exceeds the {self.context_limit}-token configured limit.')
            if len(ids) == 1 or (spans is not None and len(spans[0]) == len(ids)):
                templates = [t for t in templates if t['kind'] != 'delete']
            for template in templates:
                validate_template(template, len(ids))
            gates = self.preflight()
            capture = capture_margin(self.model, ids, objective['prefer_id'], objective['avoid_id'], layers)
            captured = time.perf_counter()
            ranked, _, count = rank_banks(capture.banks, templates, spans, limit=top_k)
            scored = time.perf_counter()
            if capture_path is not None:
                capture.save(capture_path)
            baseline = self._branch(ids, objective, max_tokens, None, checker)
            assert_close(baseline['first_token_margin'], capture.margin, 'Recomputed baseline margin',
                         atol=LOGIT_REPLAY_ATOL)
            noop_recipe = {'layer': ranked[0]['layer'], 'indices': ranked[0]['indices'],
                           'edit': {'kind': 'reweight', 'factor': 1.}}
            noop = self._branch(ids, objective, max_tokens, noop_recipe, None)
            if baseline['token_ids'] != noop['token_ids']:
                raise VerificationError('No-op changed generated tokens on the actual prompt.')
            assert_close(noop['first_token_margin'], capture.margin, 'Actual-prompt no-op margin',
                         atol=LOGIT_REPLAY_ATOL)
            trials = []
            for candidate in ranked:
                candidate['tokens'] = [self.tokenizer.decode([ids[i]]) for i in candidate['indices']]
                measured = execute_candidate(self.model, capture, candidate)
                row = {**candidate, 'verification': measured, 'output': None}
                if measured['actual_margin_change'] > 0:
                    output = self._branch(ids, objective, max_tokens, candidate, checker)
                    assert_close(output['first_token_margin'] - baseline['first_token_margin'],
                                 measured['actual_margin_change'], 'Repeated candidate margin',
                                 atol=LOGIT_REPLAY_ATOL)
                    row['output'] = output
                trials.append(row)
            helpful = [r for r in trials if r['output'] is not None]
            if checker is not None:
                helpful = [r for r in helpful if r['output']['check']['passed'] is True]
            best = max(helpful, key=lambda r: r['verification']['actual_margin_change'], default=None)
            if checker is not None and baseline['check']['passed'] is True:
                status, best = 'baseline_passed', None
            elif best is not None:
                status = 'check_passed' if checker is not None else 'margin_improved'
            else:
                status = 'no_check_pass' if checker is not None else 'no_verified_margin_improvement'
            summaries = {
                'baseline_passed': 'Ordinary generation already passes the supplied check.',
                'check_passed': 'An edited continuation passes the supplied check; see baseline check status.',
                'margin_improved': 'The supplied decision margin improved. Whole-answer correctness is unchecked.',
                'no_check_pass': 'No executed edited continuation passed the supplied check.',
                'no_verified_margin_improvement': 'None of the tested top candidates improved the supplied margin.'}
            return {'status': status, 'summary': summaries[status], 'identity': self.identity,
                'objective': objective, 'prompt_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                'prompt_tokens': len(ids), 'baseline_margin': capture.margin,
                'candidate_count': count, 'templates': templates, 'span': span,
                'baseline': baseline, 'noop': noop, 'ranked': trials,
                'selected_rank': best['rank'] if best else None,
                'selected_text': best['output']['text'] if best else None,
                'preflight': gates, 'capture_checks': capture.checks,
                'counts': {'capture_prefix_forwards': int(len(ids) > 1), 'capture_query_forwards': 1,
                           'capture_backwards': 1, 'model_calls_for_analytic_scoring': 0,
                           'candidate_verification_forwards': len(trials),
                           'generation_branches': 2 + sum(r['output'] is not None for r in trials)},
                'timing_seconds': {'preflight_and_capture': captured - started,
                                   'analytic_scoring': scored - captured, 'total': time.perf_counter() - started},
                'scope': 'One layer, all its query heads, final prompt query only; later queries are unedited.',
                'interpretation': 'Finite local responses are exact; downstream ranking is approximate. '
                                  'A failed bounded search does not rule out other attention edits. '
                                  'A passing check establishes only what that fixed check covers.'}

def load(model_id=MODEL_ID, revision=None, *, device='auto', local_only=False, context_limit=4096):
    """Load pinned FP32 weights; CUDA loading places them directly on one GPU."""
    torch, _ = model_dependencies()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    if revision is None and model_id == MODEL_ID:
        revision = MODEL_REVISION
    if revision is not None and not re.fullmatch(r'[a-fA-F0-9]{40}', revision):
        raise UserError('revision must be a full immutable 40-character commit hash.')
    config = AutoConfig.from_pretrained(model_id, revision=revision, local_files_only=local_only,
                                        trust_remote_code=False)
    revision = revision or getattr(config, '_commit_hash', None)
    if not revision or not re.fullmatch(r'[a-fA-F0-9]{40}', revision):
        raise UserError('Could not resolve a pinned model revision.')
    if config.model_type != 'qwen2':
        raise UserError('This compact adapter supports Qwen2/Qwen2.5 architecture only.')
    if device not in {'auto', 'cpu', 'cuda'}:
        raise UserError('device must be auto, cpu, or cuda.')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    if device == 'cuda' and not torch.cuda.is_available():
        raise UserError('CUDA was requested but is unavailable.')
    if device == 'cpu':
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    placement = {}
    if device == 'cuda':
        from transformers.utils import is_accelerate_available
        if not is_accelerate_available():
            raise UserError("CUDA loading needs Accelerate: pip install 'accelerate>=1,<2'.")
        # A fixed map keeps every layer on the requested GPU without first
        # materializing the entire FP32 model in host RAM. No automatic offload.
        placement = {'device_map': {'': 'cuda'}}
    tokenizer_source = model_id
    if local_only:
        from transformers.utils.hub import cached_file
        tokenizer_source = str(Path(cached_file(model_id, 'tokenizer_config.json', revision=revision,
                                               local_files_only=True)).parent)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, revision=revision, local_files_only=local_only,
                                              use_fast=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, local_files_only=local_only,
        torch_dtype=torch.float32, attn_implementation='eager', trust_remote_code=False, **placement)
    if device == 'cpu':
        model = model.to(device)
    model.eval()
    return NaLa(model, tokenizer, model_id=model_id, revision=revision, context_limit=context_limit)

def check_api_expression(expression):
    """The six supplied API-case behaviors, evaluated without exec or a shell.

    Only nested unary function names ending in value are admitted. The two API
    functions use the exact string operations in the source fixture. This small
    evaluator is a demo-specific behavioral check, not a general code sandbox.
    """
    cases = [('rOmE', 'ROME'), ('  Maya  ', 'MAYA'), ('\tParis\n', 'PARIS'),
             (' café ', 'CAFÉ'), ('', ''), (' a-b! ', 'A-B!')]
    if not isinstance(expression, str) or len(expression) > 4096:
        return {'valid': False, 'passed': False, 'passed_count': 0, 'total': 6}
    try:
        root = ast.parse(expression.strip(), mode='eval').body
        names, current = [], root
        while isinstance(current, ast.Call):
            if (not isinstance(current.func, ast.Name) or len(current.args) != 1
                    or current.keywords or len(names) >= 16):
                raise ValueError('Expected nested unary API calls.')
            names.append(current.func.id)
            current = current.args[0]
        if not names or not isinstance(current, ast.Name) or current.id != 'value':
            raise ValueError('Expected calls ending in value.')
    except (SyntaxError, ValueError, RecursionError) as exc:
        return {'valid': False, 'passed': False, 'passed_count': 0, 'total': 6, 'reason': str(exc)}
    functions = {'legacy': lambda value: value.strip().lower().swapcase(),
                 'current': lambda value: value.strip().upper().swapcase()}
    results = []
    for value, expected in cases:
        actual, error = value, None
        for name in reversed(names):
            if name not in functions:
                error = f'API has no function {name}'
                break
            actual = functions[name](actual)
        results.append({'input': value, 'expected': expected, 'actual': None if error else actual,
                        'passed': error is None and actual == expected, 'error': error})
    count = sum(r['passed'] for r in results)
    return {'valid': True, 'passed': count == 6, 'passed_count': count, 'total': 6, 'cases': results}

def demo():
    """Recorded outputs + fresh behavioral/algebra replay. No model inference.

    The two 14-element vectors are sufficient to replay the winning edit's
    predicted margin from the original saved layer capture. They are derived
    from that capture, not a new independently measured model result.
    """
    result = copy.deepcopy(RECORDED_CASE)
    projected = result['projected_capture']
    p = np.array(projected['token_probability_by_head'])
    a = np.array(projected['gradient_dot_value_minus_readout'])
    factor = result['selected_edit']['edit']['factor']
    prediction = float(np.sum(((factor - 1) * p / (1 + (factor - 1) * p)) * a))
    assert_close(prediction, result['predicted_margin_change'], 'Recorded finite response replay',
                 atol=1e-9, rtol=1e-10)
    result['replayed_predicted_margin_change'] = prediction
    for branch in result['branches']:
        branch['recomputed_behavior'] = check_api_expression(branch['expression'])
        if branch['recomputed_behavior']['passed_count'] != branch['recorded_tests_passed']:
            raise VerificationError('Recorded expression no longer agrees with its six-case behavior.')
    result['mode'] = 'recorded_outputs_with_fresh_behavior_and_projected_algebra_replay'
    result['fresh_model_inference'] = False
    return result

def show(report):
    """A small terminal summary; the returned dictionary retains the full evidence."""
    if 'branches' in report:
        print('Recorded API repair; no fresh model inference.')
        for row in report['branches']:
            print(f"{row['name']}: {row['expression']} — {row['recomputed_behavior']['passed_count']}/6 checks")
        print('Recorded predicted margin change:', round(report['predicted_margin_change'], 6))
        print('Recorded executed margin change:', round(report['actual_margin_change'], 6))
        print(report['scope'])
    else:
        print(report['summary'])
        print('Baseline:', repr(report['baseline']['text']))
        for row in report['ranked']:
            gain = row['verification']['actual_margin_change']
            text = repr(row['output']['text']) if row['output'] else '(no positive measured gain)'
            print(f"#{row['rank']} layer {row['layer']} {row['tokens']!r}: measured {gain:+.5f}; {text}")
        print(report['interpretation'])


def load_model(*args, **kwargs):
    """Original Engine API, including preflight, encode, rank, and generate."""
    return load_engine(*args, **kwargs)


RECORDED_CASE = _recorded_evidence("RECORDED_CASE")


DEMO_MESSAGES = [{'role': 'system',
  'content': 'You complete one Python API call using the provided source. Return the call on one line.'},
 {'role': 'user',
  'content': 'Repair client.lookup to produce the CURRENT canonical format: trim surrounding whitespace '
             'and convert to uppercase. The configured tests fail because api.removed no longer exists. '
             'Use the current api.py; do not change api.py or the tests.\n'
             '\n'
             'CURRENT api.py (authoritative):\n'
             'def legacy(value):\n'
             '    return value.strip().lower().swapcase()\n'
             '\n'
             'def current(value):\n'
             '    return value.strip().upper().swapcase()\n'
             '\n'
             'HISTORICAL call sites from the retired lower-case format (not authoritative):\n'
             'legacy_example_0: return api.current(value)\n'
             'legacy_example_1: return api.current(value)\n'
             'legacy_example_2: return api.current(value)\n'
             'legacy_example_3: return api.current(value)\n'
             'legacy_example_4: return api.current(value)\n'
             'legacy_example_5: return api.current(value)\n'
             'legacy_example_6: return api.current(value)\n'
             'legacy_example_7: return api.current(value)\n'
             '\n'
             'Current client.py:\n'
             'import api\n'
             '\n'
             '\n'
             'def lookup(value):\n'
             '    return api.removed(value)\n'
             '\n'
             'Complete the missing call in client.py. Output only the function call AFTER "api.", in the '
             'form function_name(value). Output one line only. Do not output JSON, Markdown, '
             'explanations, or multiple alternatives.\n'
             'A nested expression such as first(second(value)) is also allowed. Every called name is '
             'bound to the api module. Output only the expression; the surrounding code supplies the '
             'module qualification.'}]

_BUNDLED_CASE = _recorded_evidence("_BUNDLED_CASE")


def _dense_oracle(bank, edit, rows):
    """Independent full-bank construction, native split-half rotation by pairs."""
    k, v = bank.keys.copy(), bank.values.copy()
    kind = edit['kind']
    if kind == 'position':
        order = np.argsort(-bank.frequencies, kind='stable')
        bands = {'full': order, **dict(zip(('fast', 'middle', 'slow'), np.array_split(order, 3)))}
        for pair in bands[edit['band']]:
            other = pair + k.shape[-1]//2
            c, s = np.cos(edit['delta']*bank.frequencies[pair]), np.sin(edit['delta']*bank.frequencies[pair])
            for j in rows:
                left, right = k[:, j, pair].copy(), k[:, j, other].copy()
                k[:, j, pair], k[:, j, other] = c*left-s*right, s*left+c*right
    if kind in {'key', 'joint'}:
        for j in rows:
            k[:, j] = bank.keys[:, edit['donor']]
    if kind in {'value', 'joint'}:
        for j in rows:
            v[:, j] = bank.values[:, edit['donor']]
    if kind == 'value_scale':
        v[:, rows] *= edit['factor']
    def read(keys, vals, edited=False):
        output = []
        for h, group in enumerate(bank.groups):
            queries = bank.query[h] if bank.query.ndim == 3 else bank.query[h:h+1]
            writes = []
            for r, query in enumerate(queries):
                scores = keys[group] @ query * bank.scale
                scores[~bank.mask[r]] = -np.inf
                if edited and kind == 'delete': scores[rows] = -np.inf
                if edited and kind == 'reweight': scores[rows] += np.log(edit['factor'])
                if not np.isfinite(scores).any():
                    raise UserError('Dense oracle received an empty attended row.')
                weights = np.exp(scores-scores.max()); weights /= weights.sum()
                writes.append(weights @ vals[group])
            output.append(writes if bank.query.ndim == 3 else writes[0])
        return np.array(output)
    baseline, changed = read(bank.keys, bank.values), read(k, v, True)
    return float(np.sum(bank.gradient*(changed-baseline))), changed



def self_test():
    """Offline solver, all-family finite-response and precision gates.

    Full original regression tests were also run separately with imports adapted
    to this one-file layout. A live model must pass its own preflight separately.
    """
    comparisons = 0
    for seed in (401, 1709):
        rng = np.random.default_rng(seed)
        bank = Bank(0, rng.normal(size=(4, 8)), rng.normal(size=(2, 7, 8)),
                    rng.normal(size=(2, 7, 5)), rng.normal(size=(4, 5)),
                    np.array([1., .1, .01, .001]), 8**-.5)
        templates = DEFAULT_TEMPLATES + [{'kind': kind, 'donor': 5} for kind in ('key', 'value', 'joint')]
        scorer = Scorer(bank)
        brute = []
        for t, edit in enumerate(templates):
            singletons = scorer.all_singletons(edit)
            for rows in [[i] for i in range(bank.n)] + [[0, 2, 3]]:
                expected, output = _dense_oracle(bank, edit, rows)
                replay = scorer.local(edit, rows)
                assert_close(replay['output'], output, 'Independent dense output', atol=1e-11, rtol=1e-11)
                assert_close(replay['prediction'], expected, 'Independent finite score', atol=1e-11, rtol=1e-11)
                if len(rows) == 1:
                    assert_close(singletons[rows[0]], expected, 'Vectorized singleton score',
                                 atol=1e-11, rtol=1e-11)
                    brute.append((expected, t, rows[0]))
                comparisons += 1
        top, _, count = rank_banks([bank], templates=templates, limit=5)
        brute.sort(key=lambda r: (-r[0], r[1], r[2]))
        if count != len(brute) or [(r['template_index'], r['indices'][0]) for r in top] != [
                (r[1], r[2]) for r in brute[:5]]:
            raise VerificationError('Ranking differs from the independent dense oracle.')
        assert_close(scorer.all_singletons({'kind': 'reweight', 'factor': 1.}), np.zeros(bank.n),
                     'Exact no-op', atol=1e-12)
        rk, rv = bank.keys + rng.normal(size=bank.keys.shape)*.1, bank.values + rng.normal(size=bank.values.shape)*.1
        deltas = restoration_deltas(bank, rk, rv)
        for j in range(bank.n):
            changed = replace(bank, keys=bank.keys.copy(), values=bank.values.copy())
            changed.keys[:, j], changed.values[:, j] = rk[:, j], rv[:, j]
            dense = _dense_oracle(changed, {'kind': 'reweight', 'factor': 1.}, [0])[1]
            assert_close(deltas[j], dense-scorer.y, 'Independent restoration delta', atol=1e-11, rtol=1e-11)
        restored = restore_cache(bank, rk, rv, budget=3)
        if restored['final_squared_error'] > restored['initial_squared_error'] + 1e-12:
            raise VerificationError('Restoration increased reference-write error.')
    multirow_comparisons = 0
    for seed in (83, 419):
        rng = np.random.default_rng(seed)
        mask = np.arange(7)[None, :] <= np.array([2, 4, 6])[:, None]
        bank = Bank(0, rng.normal(size=(4, 3, 8)), rng.normal(size=(2, 7, 8)),
                    rng.normal(size=(2, 7, 5)), rng.normal(size=(4, 3, 5)),
                    np.array([1., .1, .01, .001]), 8**-.5, mask=mask)
        scorer = Scorer(bank)
        templates = DEFAULT_TEMPLATES + [{'kind': kind, 'donor': 5} for kind in ('key', 'value', 'joint')]
        for edit in templates:
            singles = scorer.all_singletons(edit)
            for rows in [[i] for i in range(bank.n)] + [[0, 2, 5]]:
                expected, output = _dense_oracle(bank, edit, rows)
                replay = scorer.local(edit, rows)
                assert_close(replay['output'], output, 'Multi-row dense output', atol=1e-11, rtol=1e-11)
                assert_close(replay['prediction'], expected, 'Multi-row dense contraction', atol=1e-11, rtol=1e-11)
                if len(rows) == 1:
                    assert_close(singles[rows[0]], expected, 'Multi-row singleton sum', atol=1e-11, rtol=1e-11)
                if np.any(replay['probabilities'][:, ~mask] != 0):
                    raise VerificationError('An edit exposed a causally masked key.')
                multirow_comparisons += 1
        try:
            scorer.local({'kind': 'delete'}, [0, 1, 2])
        except UserError:
            pass
        else:
            raise VerificationError('Deletion emptied a causal query bank without rejection.')
    snap = Snapshot([0., 7.], [[1., 0.], [-1., 0.]], [[1.], [-1.]])
    plan = solve_edit(snap, [{'numerator': 0, 'denominator': 1, 'ratio': 4.}])
    if not plan['can_apply']:
        raise VerificationError('The closed-form solver example failed.')
    assert_close(plan['edit'], [math.log(2.), 0.], 'Independent minimum-norm solution', atol=1e-12)
    assert_close(plan['edited_probabilities'], [.8, .2], 'Requested 4:1 attention', atol=1e-12)
    _, _, error = snap.effective_step()
    if error > 1e-10:
        raise VerificationError('Effective-step readout failed.')
    # A finite RoPE rotation is independently checked through complex exponentials.
    vector = np.array([[1., .3, -.2, .4]])
    moved = rotate_rows(vector, [7.], [1., .01], 'half_split')
    z = (vector[:, :2] + 1j*vector[:, 2:]) * np.exp(7j*np.array([1., .01]))
    assert_close(moved, np.concatenate([z.real, z.imag], axis=-1), 'Finite RoPE rotation', atol=1e-12)
    demo()
    if not assess(bundled_case()['confirmation'])['qualifies']:
        raise VerificationError('The recorded fixed-check case no longer qualifies.')
    return {'passed': True, 'dense_edit_comparisons': comparisons, 'edit_kinds': 7,
            'multirow_dense_edit_comparisons': multirow_comparisons,
            'default_templates': len(DEFAULT_TEMPLATES),
            'other_gates': ['global ranking', 'exact no-op', 'minimum-norm 4:1 solver',
                            'effective-step readout', 'finite RoPE', 'precision restoration',
                            'recorded expressions', 'recorded finite response', 'strict case assessment'],
            'native_inference_tested': False,
            'note': 'Live inference must pass model-specific preflight; callable checks and fixed-check repair are distinct.'}


def main(argv=None):
    """Single entry point for every command family; one error contract for all."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        parser().print_help()
        return 0
    command, rest = argv[0], argv[1:]
    try:
        if command == 'self-test':
            argparse.ArgumentParser(prog='nala self-test',
                                    description='Run offline mathematical gates.').parse_args(rest)
            print(json.dumps(self_test(), indent=2))
        elif command == 'demo' and rest == ['--recorded']:
            show(demo())
        elif command in {'demo', 'run', 'tokens'}:
            return debug_main(argv)
        elif command == 'solve':
            return solve_main(rest)
        elif command == 'repair-case':
            return _main(rest)
        elif command in {'trial', 'repair'}:
            return _live_main(command, rest)
        elif command == 'value':
            return _value_main(argv)
        else:
            return _diagnose_main(argv)
        return 0
    except (UserError, VerificationError, OSError, ValueError) as exc:
        print(f'nala {command}: {exc}', file=sys.stderr)
        return 2


def _live_main(command, rest):
    """The one-call trial API and the fixed-check repair controller."""
    p = argparse.ArgumentParser(prog='nala ' + command)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--local-only', action='store_true')
    if command == 'trial':
        for name in ('prompt', 'prefer', 'avoid'):
            p.add_argument('--' + name, required=True)
        for name in ('expected', 'span', 'out'):
            p.add_argument('--' + name)
        p.add_argument('--top-k', type=int, choices=range(1, 6), default=3)
        p.add_argument('--max-tokens', type=int, default=32)
        args = p.parse_args(rest)
        result = load(device=args.device, local_only=args.local_only).diagnose(
            args.prompt, prefer=args.prefer, avoid=args.avoid, expected=args.expected,
            span=args.span, top_k=args.top_k, max_tokens=args.max_tokens)
        show(result)
        if args.out:
            save_json(Path(args.out), result)
    else:
        for name in ('project', 'runs', 'request', 'spec'):
            p.add_argument('--' + name, type=Path, required=True)
        p.add_argument('--check', required=True, help='One trusted fixed command; no shell operators.')
        args = p.parse_args(rest)
        lab = load_model(device=args.device, local_only=args.local_only)
        lab.preflight()
        print(json.dumps(repair(lab, loads(args.request.read_text()), loads(args.spec.read_text()),
                                root=args.project, runs=args.runs,
                                check=command_argv(args.check), progress=print), indent=2))
    return 0


def _value_main(argv):
    args = parser().parse_args(argv)
    if args.out is not None and args.out.exists():
        raise UserError('Choose a new --out directory; existing results are preserved.')
    items = heldout_items(loads(args.heldout.read_text(encoding='utf-8')))
    spans = loads(args.spans.read_text(encoding='utf-8')) if args.spans is not None else None
    prompt = args.prompt if args.prompt is not None else args.prompt_file.read_text(encoding='utf-8')
    if not prompt.strip() or not 0 <= args.top_k <= 200 or args.select is not None and args.select < 1:
        raise UserError('Use a nonempty prompt, top-k 0–200, and a positive select budget.')
    if spans is not None and (not isinstance(spans, list) or not 1 <= len(spans) <= 64):
        raise UserError('The spans JSON must be a list of 1–64 spans.')
    for factor in args.factors or []:
        validate_template({'kind': 'reweight', 'factor': factor}, 1)
    lab = load(args.model, args.revision, device=args.device, local_only=args.local_only,
               context_limit=args.context_limit)
    out = args.out or Path('runs')/('value-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8])
    result = lab.value(prompt, heldout=items, spans=spans, top_k=args.top_k, select=args.select,
                       layers=args.layers, factors=args.factors, rank_by=args.rank_by, out=out)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        print(f"Held-out loss: {result['baseline_loss']:.6f} nats; {len(items)} independent continuation(s).")
        print(f"Scored {result['candidate_count']} edits from one capture and one backward pass.")
        print(f"Ranking: {args.rank_by}; positive loss change means worse held-out prediction.")
        print('Rank  Layer  Token/span                    Edit             Predicted loss change  Actual loss change')
        for row in result['ranked'][:max(args.top_k, 10)]:
            actual = format(row['actual_loss_change'], '+.6f') if row['verification'] else 'not executed'
            label = repr(''.join(row['tokens']))[:28]
            edit = row['edit']['kind']
            if 'factor' in row['edit']:
                edit += f" x{row['edit']['factor']:g}"
            print(f"{row['rank']:4}  {row['layer']:5}  {label:28}  {edit:16} {row['predicted_loss_change']:+.6f}  {actual}")
        if result['selection'] is not None:
            selection = result['selection']
            print(f"Retained spans {selection['selected_span_indices']} at layer {selection['layer']}.")
        print('Full report: '+result['files']['report'])
    return 0


def _diagnose_main(argv):
    """Analytic ranking with native verification of the top candidates."""
    args = parser().parse_args(argv)
    if args.out is not None and args.out.exists():
        raise UserError('Choose a new --out directory; existing results are preserved.')
    if args.max_tokens < 1:
        raise UserError('--max-tokens must be positive.')
    if args.prefer == args.avoid:
        raise UserError('--prefer and --avoid must name different alternatives.')
    prompt = args.prompt if args.prompt is not None else args.prompt_file.read_text(encoding='utf-8')
    lab = load_model(args.model, args.revision, device=args.device, local_only=args.local_only)
    lab.preflight()
    options = {'prefix': args.prefix}
    if args.layers is not None:
        options['layers'] = args.layers
    diagnosis = lab.diagnose(prompt, prefer=args.prefer, avoid=args.avoid,
                             out=args.out, top_k=args.top, **options)
    diagnosis.show()
    if args.try_repair:
        result = diagnosis.try_repair(max_tokens=args.max_tokens)
        if result['status'] == 'no_verified_improvement':
            print('None of the verified candidates improved this margin. No repair was selected.')
        else:
            print('Baseline continuation: ' + repr(result['baseline']['text']))
            print('Edited continuation:   ' + repr(result['edited']['text']))
            print('Verified margin gain: ' + format(result['actual_margin_change'], '+.6f'))
            print('Repair record: ' + result['report'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
