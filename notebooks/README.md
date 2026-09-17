# Run NaLa on 7B in Colab

Upload [NaLa_7B_Colab.ipynb](NaLa_7B_Colab.ipynb) at
[Google Colab](https://colab.research.google.com/), choose **Runtime → Change runtime
type → GPU → A100**, then **Run all**. The notebook contains the source and fixtures;
you do not need to upload the project ZIP or clone a repository.

Use an A100 with 40 GB or more. The pinned Qwen2.5-7B-Instruct checkpoint has
7,615,616,512 parameters, so its FP32 weights occupy about **28.37 GiB**. The
notebook requires another **6 GiB free** as a working allowance for the short
evaluation. It checks actual free VRAM before downloading weights. A 16 GB T4 or
24 GB L4 cannot hold this FP32 model. The first checkpoint download is about
15.2 GB; later runs reuse the cache.

The notebook uses the public checkpoint at revision
`a09a35458c702b33eeacc393d103063234e8bc28`, layer 18, and a 512-token context
limit. It keeps FP32 eager attention and all existing numerical audits. CUDA
loading uses an explicit single-GPU device map through Accelerate so the full
FP32 model is not first materialized in host RAM. It does not quantize or
automatically offload layers to the CPU.

The cells prepare dependencies, run the five-case ranking/selection evaluation,
show the result alongside the recorded 0.5B CPU result, and download a ZIP of the
new reports, captures and logs. The optional custom-prompt cell is disabled by
default. Quality failures and execution errors retain their output for inspection;
they do not prevent the later result-download cell from running.

The first run includes model download/loading time. Runtime and prediction
quality for 7B are measured by the notebook, rather than inferred from the 0.5B
CPU timing. Inspect the semantic-support diagnostic and greedy-versus-static
comparison as well as the numerical gates.

The same evaluation can be run from a GPU terminal:

```sh
python -m pip install -r requirements.txt
python eval_value.py --model Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --device cuda --layer 18 --context-limit 512 --out runs/qwen7b-sanity
```

The terminal command does not perform the notebook's VRAM/disk pre-check.
Choose a new output directory for each run. To use your own prompt, the regular
`nala.py value` command accepts the same `--model`, `--revision`, and `--device`
arguments, with `--layers 18` for the edit layer.

Local validation covered 44 tests, including loader routing, evaluator argument
forwarding, notebook schema/syntax, embedded-source integrity, memory refusal,
and retaining logs after a failed quality gate. A real cached 0.5B CPU run with
the new arguments reproduced the prior metrics. A metadata-only 7B model
confirmed parameter count, FP32 shape, eager attention, disabled sliding windows
and fixed RoPE. 7B GPU execution has not been validated; the downloaded Colab
reports are the next check.

The notebook can be regenerated after source changes with:

```sh
python tools/build_colab.py
```

Its model-size data comes from the pinned
[Qwen model configuration](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/a09a35458c702b33eeacc393d103063234e8bc28/config.json)
and checkpoint metadata; the direct-device loading contract is documented in
[Transformers 4.57.3](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/modeling_utils.py).
