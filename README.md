# NaLa

Measure which parts of a prompt carry an answer, and steer attention when a model ignores them.

Give NaLa a prompt and the answer you want. It ranks every span of the context by how much it carried that answer, then deletes the top spans and re-runs the model to check. On the bundled held-out evaluation the predicted direction matched the measured effect on 39 of 39 edits, with mean rank correlation 0.96 ([results](evals/README.md)).

When a model ignores part of its prompt, NaLa scores candidate attention edits, runs the best one, and tests the output. In the recorded case the top of 109,080 candidates took a model that skipped an API rule in its own prompt from 0/6 checks to 6/6, without retraining or changing the prompt. That is one adaptively selected synthetic task, not a benchmark.

Scoring costs one backward pass however many candidates there are. NaLa is a single MIT-licensed Python file with an MCP server, for Qwen2 and Qwen2.5 checkpoints in FP32.

## Quickstart

```sh
python -m pip install -r requirements.txt
python nala.py value --prompt-file examples/prompt.txt \
  --heldout examples/items.json --spans examples/spans.json \
  --top-k 3 --out runs/value-example
```

The held-out answer here is "The archive code is cedar."

```text
Rank  Layer  Token/span                Edit    Predicted loss change  Actual loss change
   1     21  'Archive code: cedar.\n'  delete  +2.993024              +5.053075
   2     22  'Archive code: cedar.\n'  delete  +1.808492              +2.564666
   3     16  'Archive code: cedar.\n'  delete  +1.545593              +1.835342
```

A positive loss change means the answer got harder to predict without that span. Omit `--spans` to rank every token. Add `--select K` to keep the best K spans. The first run downloads `Qwen/Qwen2.5-0.5B-Instruct` and runs on CPU.

Replay the recorded repair without a model, or search edits for your own prompt:

```sh
python nala.py demo --recorded
python nala.py diagnose --prompt "..." --prefer A --avoid B --try-repair
```

## Python

```python
import nala

prompt = open("examples/prompt.txt").read()
report = nala.load(device="cpu").value(prompt, heldout=["The archive code is cedar."], top_k=2)
```

`python nala_mcp.py` serves `nala_value` and `nala_diagnose` to an agent. Its dependencies are listed in that file.

## How it works

The [paper](https://arxiv.org/abs/2609.14127) derives the exact response of an attention head to a finite edit, through the RoPE rotation and the softmax. It keeps the key/value interaction that separate attribution drops, which cut prediction error by more than 9x in every setting tested. NaLa contracts that response with a frozen baseline gradient, so its downstream prediction is approximate.

## Limits

- One layer per edit.
- Deletion masks attention to a span. It does not remove the text or shift positions.
- 7B needs the [Colab notebook](notebooks/README.md) on a 40 GB A100. That run has not been validated.

## Docs and tests

[Usage reference](docs/USAGE.md) · [Evaluation](evals/README.md) · [Verification records](VERIFICATION.md) · [Related papers](PAPER_SUMMARIES.md)

```sh
python nala.py self-test
python -m pip install pytest 'nbformat>=5,<6'
python -m pytest -q tests
```

With Julie Huang, Maggie Chlon and Gregory Gutin at Hassana Labs.
