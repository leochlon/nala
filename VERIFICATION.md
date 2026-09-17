# Verification of NaLa 0.4.0

Each result below is backed by a file in [verification/](verification/).

## Objective and design

The held-out objective is a weighted sum of teacher-forced token log probabilities; loss is its negative, and every reported change is edited loss minus baseline loss.

Item branches share a detached prefix, and one backward pass obtains gradients for all captured layers and query rows.

Candidate scoring sums query and item contributions within a layer. Layers are alternative interventions, not terms of a multi-layer edit.

Contribution ranking treats a loss increase after deletion as useful context; `rank_by="improvement"` ranks predicted likelihood gains instead.

Greedy retention updates group normalizers and re-scores the remaining spans from the same captured gradient. Pooled insertion unmasks entries of already captured context, with one layer selected for execution.

## Results

| Check | Result | Record |
| --- | --- | --- |
| Offline self-test | Passed; 304 single-row and 304 multi-row dense comparisons over 7 edit kinds; no native inference | [self_test.json](verification/self_test.json) |
| Regression suite, held-out value | 32 passed | [test_results.txt](verification/test_results.txt) |
| Regression suite with evaluator checks | 35 passed | [value_sanity_tests.txt](verification/value_sanity_tests.txt) |
| Regression suite with Colab checks | 44 passed | [colab7b_tests.txt](verification/colab7b_tests.txt) |
| Pretrained run, Qwen2.5-0.5B-Instruct at `7ae557604adf67be50417f59c2c2f167def9a775` | 27 candidates, 2 items, 8 query rows, 1 backward; capture, top-k and selection checks passed; 0 model calls for analytic scoring or greedy re-scoring | [pretrained_smoke.json](verification/pretrained_smoke.json) |
| Five-case evaluation, CPU, cached weights | 14.59 s; mean Spearman 0.96; sign agreement 39/39 material edits; greedy tied the best measured pair and static top-two in 5/5 cases; top span matched semantic support in 3/5 | [value_sanity.json](verification/value_sanity.json) |
| 7B preparation | Metadata-only load: 7,615,616,512 parameters, 28.37 GiB FP32, eager attention, fixed RoPE | [colab7b_validation.json](verification/colab7b_validation.json) |

In the pretrained run the leading deletion had predicted loss change +0.12177 and executed +0.12078; the 2x reweight had predicted −0.09999 and executed −0.09770.

## Not validated

All runs used CPU. CUDA execution, including the 7B notebook on an A100, has not been run.

Scope is FP32 Qwen2/Qwen2.5, one layer per edit, exact local response and approximate downstream prediction.
