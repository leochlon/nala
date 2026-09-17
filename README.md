# NaLa

Measure which parts of a prompt carry the answer, and steer attention when a model ignores them.

Give NaLa a prompt and the answer you want. It ranks every span of the context by how much it carried that answer, from one backward pass, then checks the top spans by deleting them and re-running the model. On the bundled held-out evaluation, the predicted direction matched the measured effect on 39 of 39 edits, with mean rank correlation 0.96 ([results](evals/README.md)).

The same calculus runs the other way. When a model ignores something in its prompt, NaLa scores every candidate attention edit, executes the best, and checks the output. In the recorded case it scored 109,080 candidates from one backward pass; the top edit took a model that skipped an API rule in its own prompt from 0/6 checks to 6/6, with no retraining and no prompt change. That case is one adaptively selected synthetic task, not a benchmark.

One Python file, MIT, with an MCP server. Open-weight Qwen2/Qwen2.5 in FP32.

## Quickstart

```sh
python -m pip install -r requirements.txt
python nala.py value --prompt-file examples/prompt.txt \
  --heldout examples/items.json --spans examples/spans.json \
  --top-k 3 --out runs/value-example
```

The prompt holds three notes; the held-out answer is "The archive code is cedar."

```text
Scored 72 edits from one capture and one backward pass.
Rank  Layer  Token/span                Edit    Predicted loss change  Actual loss change
   1     21  'Archive code: cedar.\n'  delete  +2.993024              +5.053075
   2     22  'Archive code: cedar.\n'  delete  +1.808492              +2.564666
   3     16  'Archive code: cedar.\n'  delete  +1.545593              +1.835342
```

A positive loss change means deleting the span makes the answer harder to predict: the span was carrying it. Omit `--spans` to rank every token. Add `--select K` to keep the best K spans greedily. The first run downloads `Qwen/Qwen2.5-0.5B-Instruct`; it runs on CPU.

See the recorded 0/6 to 6/6 repair without downloading a model:

```sh
python nala.py demo --recorded
```

Search attention edits on your own prompt:

```sh
python nala.py diagnose --prompt "..." --prefer A --avoid B --try-repair
```

## Python

```python
import nala

lab = nala.load(device="cpu")
report = lab.value(
    "Archive code: cedar. Garden color: amber. State the archive code.",
    heldout=["The archive code is cedar."],
    spans=["Archive code: cedar.", "Garden color: amber."],
    top_k=2,
)
```

`python nala_mcp.py` serves the same calls to an agent as `nala_value` and `nala_diagnose`; its dependencies are listed at the top of that file.

## Why it is exact

The [paper](https://arxiv.org/abs/2609.14127) gives the exact response of an attention head to any finite edit, through the RoPE rotation and the softmax, and keeps the key/value interaction that separate attribution drops. In the paper's experiments that term cut prediction error by more than 9x in every setting tested. NaLa contracts the exact local response with one baseline gradient to predict the downstream effect, then measures the executed effect.

## Scope

- Qwen2 and Qwen2.5, unquantized FP32, eager attention, fixed RoPE. One layer per edit.
- The response is exact at the edited head. The downstream prediction uses a frozen gradient, so NaLa executes the top candidates and reports both numbers.
- Deletion masks attention to a span. It does not remove the text or shift positions.
- 7B runs from the [Colab notebook](notebooks/NaLa_7B_Colab.ipynb) on an A100 with 40 GB; that GPU run has not been validated yet ([instructions](notebooks/README.md)).

## More

- [Usage reference](docs/USAGE.md): input formats, objective and signs, capture and audits, greedy selection, MCP configuration.
- [Evaluation](evals/README.md) and [verification records](VERIFICATION.md).
- [Related papers](PAPER_SUMMARIES.md).

```sh
python nala.py self-test
python -m pip install pytest 'nbformat>=5,<6'
python -m pytest -q tests
```

With Julie Huang, Maggie Chlon and Gregory Gutin at Hassana Labs.
