# MiniCPM5 MoE and DSpark

This example runs a MiniCPM5 MoE language model on Apple silicon. It reads
weight-only asymmetric INT4 checkpoints stored in the compressed-tensors
`pack-quantized` format with 128 weights per group.

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
The example handles one request at a time with standard K/V caches and full
attention in the draft.
