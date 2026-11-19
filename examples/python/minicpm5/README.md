# MiniCPM5 with DSpark

This example runs MiniCPM5 on Apple silicon. Convert the source GPTQ checkpoint
from compressed-tensors `pack-quantized` to MLX affine format once. Generation
and the HTTP server load the converted checkpoint directly. The converter
supports weight-only asymmetric INT4 with group size 128.

Run these commands from the root of the checkout. The tested runtime uses
Python 3.13, MLX 0.32.4 and MLX LM 0.32.2.

```sh
python -m pip install 'mlx==0.32.4' 'mlx-lm==0.32.2'
python -m examples.python.minicpm5.repack \
  /path/to/ct-checkpoint /path/to/new-mlx-checkpoint
python -m examples.python.minicpm5.generate \
  --model /path/to/new-mlx-checkpoint \
  --prompt "Explain why the Moon keeps the same face toward the Earth." \
  --max-tokens 64
```

The converter reads the source weight index. All routed experts for a
projection must be in the same source shard. The converter keeps the INT4
codes, stacks experts in numeric order and stores the affine scales and offsets
in float32. For unsigned code `q` and zero point `z`, MLX reads
`scale * q + (-scale * z)`. The output is a separate checkpoint and includes
the source tokenizer and chat template.

Pass a matching `Qwen3DSparkModel` draft checkpoint with a vanilla Markov
head to enable speculative decoding.

```sh
python -m examples.python.minicpm5.generate \
  --model /path/to/new-mlx-checkpoint \
  --draft-model /path/to/dspark-checkpoint \
  --prompt "Write a Python function to compute a factorial." \
  --max-tokens 64 --json
```

The draft shares the target embedding and output head. The target layer IDs
in the draft config select post-layer residuals. These residuals supply the
draft context K/V. The decoder verifies a fixed number of candidates per
round, bounded by the remaining token budget. It retains only the accepted
prefix and uses the correction or bonus token as the next anchor.

Generation defaults to greedy decoding. To sample, set `--temp` above zero
and optionally set `--top-k`, `--top-p` and `--seed`. Both models apply
temperature, top-k and top-p in that order. The verifier accepts a sampled
proposal with probability `min(1, p/q)`. After a rejection it draws from
the positive part of `p - q`, where `p` and `q` are the target and draft
distributions for that position. Batched verification and single-token
decoding can round near-tied logits differently.

Use `--json` to see the emitted token IDs and DSpark verification counters.

## HTTP server

Start the server with the converted model and its matching draft.

```sh
python -m examples.python.minicpm5.server \
  --model-path /path/to/new-mlx-checkpoint \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path /path/to/dspark-checkpoint \
  --tool-call-parser minicpm5 \
  --host 127.0.0.1 --port 8080
```

The entry point uses the MLX LM HTTP server with a DSpark generator. It loads
the target and draft at startup. Requests to `/v1/chat/completions` can use
`"model": "minicpm5"` and set `stream` for SSE. Send the assistant's
`tool_calls` and the corresponding `role: tool` messages in the next request
to continue a conversation.

```sh
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "minicpm5",
    "messages": [{"role": "user", "content": "What is the current temperature in Beijing? Use the lookup_temperature tool."}],
    "tools": [{"type": "function", "function": {
      "name": "lookup_temperature",
      "description": "Look up the current temperature for a city.",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    }}],
    "chat_template_kwargs": {"enable_thinking": true},
    "temperature": 0,
    "max_tokens": 320
  }'
```

The `minicpm5` parser follows the
[SGLang XML4 argument rules](https://github.com/sgl-project/sglang/pull/49222).
It keeps the first duplicate parameter, skips undeclared parameters when the
schema declares properties, and drops calls that omit required parameters.
In SSE it emits complete calls after their closing tags, including calls with
CDATA split across deltas.

Serving is sequential. Each request gets its own target and draft K/V caches
and uses the draft checkpoint's block size. The log reports DSpark verification
and accepted-token counters for each request. DSpark supports greedy and
temperature, top-k and top-p sampling.
