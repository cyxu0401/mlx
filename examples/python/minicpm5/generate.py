"""Run MiniCPM5 on MLX with optional DSpark speculation."""

import argparse
import json
import time
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import generate_step, generation_stream, wired_limit
from mlx_lm.utils import load_model, load_tokenizer

from . import dspark
from .model import Model, ModelArgs


def load_target(path: Path):
    if not path.is_dir():
        raise FileNotFoundError(path)
    return load_model(path, get_model_classes=lambda config: (Model, ModelArgs))


def chat_prompt(tokenizer, prompt, thinking=False):
    return mx.array(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=thinking,
            has_tool_sep=False,
        )
    )


def generate(target, tokenizer, prompt, max_tokens, sampling, draft=None):
    stats = dspark.DraftStats()
    if draft is None:
        steps = generate_step(
            prompt,
            target,
            max_tokens=max_tokens,
            sampler=sampling,
            prefill_step_size=128,
        )
        tokens = ((token, False) for token, _ in steps)
    else:
        steps = dspark.generate_step(
            target,
            draft,
            prompt,
            max_tokens=max_tokens,
            sampling=sampling,
            stats=stats,
            eos_token_ids=tokenizer.eos_token_ids,
        )
        tokens = steps

    emitted = []
    from_draft = 0
    finish_reason = "length"
    start = time.perf_counter()
    with wired_limit(target, [generation_stream]), closing(steps):
        for token, accepted in tokens:
            if token in tokenizer.eos_token_ids:
                finish_reason = "stop"
                break
            emitted.append(token)
            from_draft += accepted

    return {
        "text": tokenizer.decode(emitted),
        "token_ids": emitted,
        "prompt_tokens": prompt.size,
        "finish_reason": finish_reason,
        "elapsed_seconds": time.perf_counter() - start,
        "from_draft": from_draft,
        "dspark": asdict(stats) if draft is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, type=Path, help="Converted MLX checkpoint."
    )
    parser.add_argument(
        "--draft-model", type=Path, help="Matching Qwen3 DSpark checkpoint."
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument(
        "--json", action="store_true", help="Print text, token IDs and counters."
    )
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive.")
    try:
        sampling = dspark.Sampling(args.temp, args.top_k, args.top_p)
    except ValueError as error:
        parser.error(str(error))

    mx.random.seed(args.seed)
    mx.set_cache_limit(256 * 2**20)
    target, config = load_target(args.model)
    tokenizer = load_tokenizer(args.model, eos_token_ids=config["eos_token_id"])
    prompt = chat_prompt(tokenizer, args.prompt, args.thinking)
    draft = dspark.load_draft(args.draft_model, target) if args.draft_model else None
    response = generate(target, tokenizer, prompt, args.max_tokens, sampling, draft)
    print(json.dumps(response, ensure_ascii=False) if args.json else response["text"])


if __name__ == "__main__":
    main()
