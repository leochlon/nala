# Related papers

## Epiplexity Guided Data Selection and Generation for Out-of-Distribution Generalization

The paper asks how to select or generate training data that helps a model generalize to unfamiliar tasks, using epiplexity as a measure of structural information a compute-bounded learner can extract [^epi_intro].

Its practical approximation is the area between the training-loss curve and its final-loss floor: trivial data is quickly exhausted, noise stays unlearnable, and sustained loss reductions indicate learnable structure [^epi_intro].

**EpiSelect** fits cross-domain scaling laws to observed losses during training, estimates the marginal epiplexity gain from another batch in each domain, and turns those estimates into sampling weights [^epi_select].

**EpiGen** trains a learner on a generated batch and rewards the generator for the learner's loss reduction on a reference buffer containing current and earlier generated samples; the generator is optimized with REINFORCE [^epi_generate].

On the token-balanced Common Pile, with a 15-billion-token training budget, EpiSelect's average zero-shot accuracy was 39.4% versus ADO's 37.9% for 124M models, and 43.1% versus 42.5% for 1.3B models [^epi_select].

In the GPT-2 experiment, EpiGen increased the average fine-tuned GLUE score from 0.743 to 0.770; training on samples from a frozen generator reached 0.759 [^epi_generate].

The authors also found that training only on PileCC could outperform ADO on The Pile, motivating their use of a more balanced benchmark [^epi_select].

The evidence is limited to relatively small models and tractable epiplexity proxies, with no proof that optimizing those proxies maximizes true epiplexity; the generation experiment showed essentially no improvement from random initialization [^epi_generate].

## Exact finite attention responses from RoPE derivatives

This paper derives exact changes to a local attention write under finite interventions, retaining the full RoPE rotation and softmax renormalization [^attn_intro].

Adding separate finite-key and baseline-weighted value responses omits an interaction term that measures how a value edit's effect changes when attention is reallocated [^attn_methods].

One baseline backward pass supplies an answer-margin gradient; contracting it with each exact local response predicts a candidate's downstream effect, and native execution measures selected candidates [^attn_methods].

The same calculus supplies a local KL error certificate, minimum-norm query control, cache deletion/restoration criteria, and an algebraic representation of attention as a query-conditioned gradient step [^attn_methods].

Across 92,160 executed positional edits on 768 held-out prompt sets, the reported sign accuracy was 95.36–96.52%, and downstream margin MAE was 73.6–82.5% lower than the positional Jacobian's [^attn_results].

In the 5,120-execution joint-cache study, retaining the key/value interaction reduced margin MAE by more than a factor of nine against separate attribution in every evaluated setting [^attn_results].

The empirical coverage is within the Qwen family, positional edits hold contextual features fixed, and downstream prediction uses a frozen gradient; full intervention-search cost was not measured, and the API repair example was adaptively selected [^attn_results].

## Connection to the implemented feature

The NaLa held-out objective is a weighted sum of teacher-forced continuation log probabilities, `LL = sum_i weight_i * sum_t log P(y_it | prompt, y_i,<t)`, with `loss = -LL` [^implementation].

It sums contributions across query rows and independent held-out items within each candidate layer, using one captured gradient for scoring and greedy re-scoring [^implementation].

A positive deletion loss change indicates that removing the span makes the supplied continuations harder to predict; the default contribution ranking therefore puts helpful spans first [^implementation].

NaLa.value accepts held-out continuations and ranks contributions without preferred/competing answer labels [^implementation].

The implemented score measures in-context predictive contribution with fixed model parameters [^implementation].

EpiGen's learning signal measures the change in reference-buffer loss across actual learner-parameter updates [^epi_generate].

[^epi_intro]: arXiv:2608.11746v1, introduction and §2, pp. 1–3.
[^epi_select]: Same paper, §3 and Table 1, pp. 5–7.
[^epi_generate]: Same paper, §4, Table 2 and discussion, pp. 8–11.
[^attn_intro]: arXiv:2609.14127v1, abstract and §2, pp. 1–3.
[^attn_methods]: Same paper, §§3–4, pp. 4–6.
[^attn_results]: Same paper, §§5–7, pp. 6–9.
[^implementation]: [nala.py](nala.py), `capture_objective`, `rank_value_banks`, `greedy_value_selection` and `NaLa.value`.
