# NaLa usage reference

Options, input formats, the objective and its signs, and what capture, execution and selection do. For the overview see the [README](../README.md).

## Models, devices and options

The default model is the pinned `Qwen/Qwen2.5-0.5B-Instruct` checkpoint. The first
live run may download it. Use `--local-only` for cached weights. Supported devices
are `cpu`, `cuda`, and `auto` (CUDA if available, otherwise CPU). All model weights
and native attention operations use FP32, eager full causal attention, and fixed
unscaled RoPE. Supported model architectures are Qwen2 and Qwen2.5.

Omit `--spans` to rank every token in the rendered prompt, including chat
template tokens. Held-out continuation tokens are never candidate spans.
`--layers 0 18` restricts the candidate layers; otherwise every layer is captured.
Each candidate edits exactly one layer across all its query heads.

`--top-k N` executes the top N candidates, with `N` between 0 and 200. Zero produces
scores and capture checks only. The full ranking goes in `report.json`; the
terminal shows a short preview. `--json` prints the full report instead.

## Input files

`items.json` is a nonempty list of exact continuation strings or objects:

```json
[
  "The archive code is cedar.",
  {"text": "cedar", "weight": 0.5}
]
```

An object containing only `{"items": [...]}` is also accepted. There may be up
to 64 items. Each item is scored independently after the same rendered prompt;
one item never becomes context for another. Weights must be positive and finite.
Leading spaces are significant. Each continuation is tokenized separately and
appended to the unchanged prompt token IDs. No EOS is appended automatically.

`spans.json` contains up to 64 quoted strings, quote/occurrence objects, or explicit
token-index objects:

```json
[
  "Archive code: cedar.",
  {"quote": "amber", "occurrence": 1},
  {"indices": [4, 5, 6]}
]
```

Quotes must resolve to complete tokens. Repeated quotes need a one-based
occurrence. Explicit indices refer to the rendered prompt, are zero-based, and
must be distinct and in range. Use the saved `prompt.json` or `nala.py tokens` to
inspect the token sequence. A multi-token span is edited jointly; its score is
not the sum of individual token scores.

## Objective and signs

For held-out item `i`, the objective is

```text
LL = sum_i weight_i * sum_t log P(item_i[t] | prompt, item_i[:t])
loss = -LL
loss_change = edited_loss - baseline_loss
```

There is no length normalization. Longer items contribute more token terms;
weights can adjust the caller's intended relative importance. Reported units are
nats. Both predicted and executed changes use exactly the same objective.

Deletion is the default edit family. A positive `predicted_loss_change` means
deleting the span is predicted to make held-out prediction worse: the span was
useful. Default `--rank-by contribution` puts these useful spans first.

Add `--factors 0.5 2` to include attention suppression and boost candidates.
Reweighting multiplies a span's unnormalized attention weights by the factor,
then renormalizes the complete causal bank. Factors range from 1/64 to 64.
Contribution reverses the likelihood change for deletion/suppression and retains
it for boosting. Comparisons across factors reflect the requested intervention
strength; they are not normalized per unit of log odds. `--rank-by improvement`
instead ranks edits by their predicted likelihood increase (loss decrease).

## Capture, execution, and audits

`capture_objective` uses a shared no-grad prefix and a separate, multi-row
teacher-forced forward for each item. It sums item objectives and obtains every
selected layer's gradients in one backward pass. Thus one capture can contain
several model forwards; the report counts them explicitly. Scoring candidates
after capture makes no model calls.

`Bank` accepts `query` with shape `(heads, queries, width)`, matching `gradient`,
and a boolean `(queries, keys)` causal mask. Legacy single-query arrays remain
supported. `Scorer` preserves causal exclusions, computes exact finite local
responses, and sums the gradient contraction over queries. Item contributions
are summed within each candidate layer. Layers are alternative interventions,
not terms in a multi-layer edit.

Execution teacher-forces all continuation tokens, including the first and
last. It edits the prompt span at every corresponding query row. Independent
copies of the shared prefix prevent candidate or item cache leakage. Deletion
masks attention entries; it does not remove input tokens or change positions.

The report follows the existing `ranked`/`verification`/`capture_checks` shape and
adds:

- `baseline_loss`, `baseline_log_likelihood`, and per-item token log probabilities;
- `predicted_loss_change` and `predicted_log_likelihood_change` on every candidate;
- `actual_loss_change`, `actual_log_likelihood_change`, and a `verification`
  record for each executed candidate;
- native probability/output checks and causal-mask audits for each item/layer;
- capture, backward, scoring, execution, and selection counts;
- baseline and installed no-op controls when execution is requested.

With `--out`, the new directory contains `report.json`, `prompt.json`,
`capture.npz`, and `scores.npz`. The score arrays store log-likelihood changes.
Captures use ordinary NumPy arrays and JSON metadata with no pickles.
`load_banks` accepts both the new multi-item format and legacy captures.
Existing output directories are never overwritten.

## Greedy selection and pooled insertion

```sh
python nala.py value --prompt-file examples/prompt.txt \
  --heldout examples/items.json --spans examples/spans.json \
  --select 2 --top-k 3 --out runs/selection-example
```

All candidate context must already be in the prompt. Selection starts with the
pool spans masked and retains K spans greedily. After every retention, group
normalizers and means are updated and every remaining span is re-scored.
The final subset is executed and audited. Selection spans must not overlap and
must leave at least one prompt key outside the pool. Without explicit spans, all
prompt tokens except the final prompt key form the pool; that final key stays as
an attended anchor.

Each captured layer has an independent greedy trajectory. The trajectory with
the best final predicted likelihood chooses the single layer for execution.
The report retains every step's candidate scores for that trajectory, the
selected span indices, protected tokens, and layer summaries.

Selection reuses the same captured gradient; it performs no extra backwards
or model calls to re-score. It is greedy optimization of a local surrogate and
does not promise a globally optimal subset. Insertion means unmasking entries
from this pre-captured pool. It does not model new text, re-tokenization, shifted
positions, or recomputed upstream features of a shorter prompt.

## Low-level API and MCP configuration

`capture_margin` is a wrapper for the margin objective. Low-level
callers can use `capture_objective(model, prompt_ids, {"kind":
"held_out_log_likelihood", "items": [{"token_ids": [...], "weight": 1.0}]})`.

Run `python nala_mcp.py` with the MCP SDK v1 dependencies documented in that file.
The `nala_value` tool accepts the same held-out items and span objects directly
as JSON arrays. It returns the full report. `nala_diagnose` takes a prefer/avoid pair. `NALA_MODEL`, `NALA_REVISION`, `NALA_DEVICE`,
`NALA_LOCAL_ONLY`, and `NALA_CONTEXT_LIMIT` configure the server's model.
