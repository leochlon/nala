# Quick held-out value evaluation

Run from the project directory with the same dependencies as NaLa:

```sh
python eval_value.py --local-only --out runs/value-sanity
```

Use a new output directory each time; omit `--local-only` to allow the initial
model download. The default is the pinned Qwen2.5-0.5B-Instruct checkpoint,
CPU FP32, and layer 18. `--device cuda` and `--layer N` select a different
configuration, which is recorded in the result.

Use `--model`, `--revision` and `--context-limit` to select a larger checkpoint.
For 7B on Colab, use the [self-contained notebook](../notebooks/NaLa_7B_Colab.ipynb)
and its [GPU instructions](../notebooks/README.md). The notebook uses the same
fixtures and thresholds with FP32 eager attention.

## What it checks

The five fixed cases in [value_sanity.json](value_sanity.json) cover changing
held-out targets on the same prompt, current versus legacy API instructions,
combining two facts, and redundant evidence. Every case has four candidate spans
and two independent continuations. Semantic support annotations are used only
for evaluation, never passed to the scorer.

For each case, the evaluator calls the public `NaLa.value` API used by the CLI
and MCP tool. It executes all eight deletion/0.5-reweight candidates, so the
ranking check includes weak and badly ranked edits. It checks predicted versus
teacher-forced actual loss changes with Spearman correlation, sign agreement,
mean absolute error, and top-one contribution regret.

It also retains two spans greedily and compares the executed result with:

- Static top-two selection from the individual deletion ranking.
- Uniform random selection, calculated as the exact mean over all six pairs.
- The best of all six executed pairs, an exhaustive oracle for this small pool.
- The full-context baseline.

The six pairs are measured by a second public API call using the complementary
spans as joint deletion candidates. Subset losses are measured directly; they
are never inferred by adding singleton losses. The independently replayed greedy
subset must match the first call's measurement.

An additional call without `spans` checks that every rendered prompt token is
ranked. Token mode uses `top_k=0` here; its purpose is coverage. This protocol
therefore uses 11 captures/backwards in total, with exactly one backward in each
API call and zero model calls for analytic or greedy re-scoring. The extra
captures are evaluation overhead.

## Gates and units

These smoke thresholds were fixed before the first native run:

| Check | Requirement |
| --- | --- |
| Capture, edit, no-op and coverage checks | All pass |
| Mean Spearman across case/edit-family groups | At least 0.7 |
| Sign agreement | At least 80% over at least 10 material edits |
| Greedy versus uniform random | Mean greedy NLL no worse, within 1e-6 nats per target token |

Sign agreement excludes executed effects smaller than 1e-4 nats per weighted
target token. Undefined constant-array correlations are reported as null and
excluded from the mean; the report includes the number of scored groups.

The feature's objective remains a sum. Only evaluation metrics are divided by
`sum_i weight_i * target_token_count_i` to compare cases of different target
lengths. Candidate order within a case is unchanged by this normalization.
Lower NLL and regret are better; positive gain over a baseline is better.

Exit code 0 means all defined gates passed, 1 means a gate failed, and 2 means
execution failed. All available product reports, captures and subset measurements
are kept in the output directory; `summary.json` contains metrics, fixture hash,
model revision, counters, checks and timings.

## Recorded result

The included [result snapshot](../verification/value_sanity.json) used the
unmodified v1 fixture on the pinned model at layer 18. The run took **14.59 seconds**
including **2.77 seconds** for loading cached weights and **11.82 seconds** for
evaluation on CPU.

| Metric | Observed |
| --- | --- |
| Correctness and defined quality gates | Pass |
| Mean Spearman, 10 groups | 0.96 |
| Sign agreement on material edits | 39/39 |
| Mean absolute prediction error | 0.00104 nats per target token |
| Mean top-one contribution regret | 0 |
| Greedy versus exhaustive pair oracle | Same loss in 5/5 cases |
| Mean greedy improvement over uniform random | 0.03603 nats per target token |
| Greedy improvement over static top-two | 0 |
| Predicted top span matches semantic support | 3/5 |

Both target-swap cases chose the same top span, while their selected pairs
differed. The semantically unexpected top spans were also the top spans by
executed likelihood effect. These observations are retained in the result.

This small evaluation demonstrates agreement with the defined likelihood
interventions. It provides no evidence that greedy selection improves on static
selection in these cases. Semantic relevance is a separate diagnostic and
matched only three of the five top-span annotations.

The oracle is restricted to two of four spans, at one layer, with the original
contextual features and token positions preserved. It measures pooled attention
deletion, not the effect of physically shortening or re-tokenizing the prompt.
It is a quick regression check for using this feature, not a training,
epiplexity, broad retrieval-quality, or all-layer-search benchmark.

The original evaluator regression suite, including three evaluation-metric checks, passed
**35 tests**; the archived console output is in
[value_sanity_tests.txt](../verification/value_sanity_tests.txt).
With the Colab additions, the current local suite passed **44 tests**; see
[colab7b_tests.txt](../verification/colab7b_tests.txt) and the
[local validation record](../verification/colab7b_validation.json).
Actual 7B GPU validation remains the next notebook run.
