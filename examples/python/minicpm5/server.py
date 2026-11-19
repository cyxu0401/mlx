"""Serve MiniCPM5 with DSpark through the MLX LM HTTP server."""

import argparse
import json
import logging
import time
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
from mlx_lm import server
from mlx_lm.generate import (
    TextStateMachine,
    generation_stream,
    make_stop_sequences,
    wired_limit,
)
from mlx_lm.models.cache import LRUPromptCache
from mlx_lm.utils import load_tokenizer

from . import dspark
from .generate import load_target
from .tool_parser import MiniCPM5ToolParser


class ModelProvider(server.ModelProvider):
    def __init__(self, args):
        super().__init__(args)
        self._model_map["minicpm5"] = args.model

    def _load(self, model_path, adapter_path=None, draft_model_path=None):
        if (model_path, adapter_path, draft_model_path) != (
            self.cli_args.model,
            None,
            self.cli_args.draft_model,
        ):
            raise ValueError(
                "This server uses one configured MiniCPM5 and DSpark pair."
            )
        if self.is_distributed:
            raise ValueError("The MiniCPM5 DSpark server runs on one device.")

        model, config = load_target(Path(model_path))
        tokenizer = load_tokenizer(
            Path(model_path), eos_token_ids=config["eos_token_id"]
        )
        draft = dspark.load_draft(Path(draft_model_path), model)
        self.model = model
        self.tokenizer = tokenizer
        self.draft_model = draft
        self.cli_args.num_draft_tokens = draft.args.block_size
        self.model_key = (model_path, adapter_path, draft_model_path)


class ResponseGenerator(server.ResponseGenerator):
    def _serve_single(self, queued_request, _stream):
        queue, request, args = queued_request
        model = self.model_provider.model
        tokenizer = self.model_provider.tokenizer
        draft = self.model_provider.draft_model
        try:
            prompt, _, _, initial_state, _ = self._tokenize(tokenizer, request, args)
            if not prompt:
                raise ValueError("The prompt must contain at least one token.")
            sampling = dspark.Sampling(
                args.sampling.temperature, args.sampling.top_k, args.sampling.top_p
            )
            if args.seed is not None:
                mx.random.seed(args.seed)

            # The template determines the initial reasoning state. The handler
            # parses the XML after reasoning is separated from normal text.
            transitions = {"normal": [(word, "normal") for word in args.stop_words]}
            if tokenizer.has_thinking:
                transitions["reasoning"] = [(tokenizer.think_end, "normal")] + [
                    (word, "reasoning") for word in args.stop_words
                ]
            ctx = server.GenerationContext(
                has_tool_calling=bool(request.tools),
                has_thinking=tokenizer.has_thinking,
                tool_parser=tokenizer.tool_parser,
                text_sm=TextStateMachine(transitions),
                initial_state=initial_state,
                prompt=prompt,
                prompt_cache_count=0,
            )
            stop_matcher = make_stop_sequences(tokenizer, args.stop_words).matcher()
            queue.put(ctx)

            stats = dspark.DraftStats()
            emitted_draft = 0
            detokenizer = tokenizer.detokenizer
            steps = dspark.stream_tokens(
                model,
                draft,
                mx.array(prompt),
                max_tokens=args.max_tokens,
                sampling=sampling,
                stats=stats,
                eos_token_ids=tokenizer.eos_token_ids,
                prefill_step_size=self.cli_args.prefill_step_size,
                prompt_progress_callback=lambda done, total: queue.put((done, total)),
            )
            start = time.perf_counter()
            with wired_limit(model, [generation_stream]), closing(steps):
                for count, (token, from_draft, logits) in enumerate(steps, 1):
                    if ctx._should_stop:
                        break
                    finish = None
                    if stop_matcher.advance(token):
                        finish = "stop"
                    else:
                        detokenizer.add_token(token)
                        if count == args.max_tokens:
                            finish = "length"
                    if finish:
                        detokenizer.finalize()
                    logprob, top_tokens = 0.0, ()
                    if args.logprobs or args.top_logprobs > 0:
                        logits = logits.astype(mx.float32)
                        logprobs = logits - mx.logsumexp(logits)
                        logprob = logprobs[token].item()
                        top_tokens = server._format_top_logprobs(
                            logprobs, args.top_logprobs, tokenizer
                        )
                    queue.put(
                        server.Response(
                            detokenizer.last_segment, token, logprob, finish, top_tokens
                        )
                    )
                    emitted_draft += from_draft
                    if finish:
                        break
            logging.info(
                "DSpark %s",
                json.dumps(
                    dict(
                        asdict(stats),
                        emitted_draft=emitted_draft,
                        prompt_tokens=len(prompt),
                        completion_tokens=count,
                        elapsed_seconds=round(time.perf_counter() - start, 3),
                    )
                ),
            )
            queue.put(None)
        except Exception as error:
            queue.put(error)


class APIHandler(server.APIHandler):
    def validate_model_parameters(self):
        super().validate_model_parameters()
        if self.num_draft_tokens != self.response_generator.cli_args.num_draft_tokens:
            raise ValueError("DSpark uses the draft checkpoint's block size.")
        if self.top_p == 0:
            raise ValueError("DSpark requires top_p greater than zero.")
        unsupported = (
            "min_p",
            "xtc_probability",
            "logit_bias",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
        )
        if names := [name for name in unsupported if getattr(self, name)]:
            raise ValueError(f"DSpark does not support {', '.join(names)}.")

    def handle_completion(self, request, stop_words):
        self._xml4 = MiniCPM5ToolParser(request.tools) if request.tools else None
        self._emitted_tool_call = False
        self._format_tools = server.ToolCallFormatter(
            lambda call, _: call, request.tools, streaming=self.stream
        )
        super().handle_completion(request, stop_words)

    def generate_response(self, text, finish_reason, *args, tool_calls=None, **kwargs):
        if self._xml4 is not None:
            text, calls = self._xml4.feed(text, final=finish_reason is not None)
            tool_calls = self._format_tools(calls)
            self._emitted_tool_call |= bool(tool_calls)
            if finish_reason == "stop" and self._emitted_tool_call:
                finish_reason = "tool_calls"
        return super().generate_response(
            text, finish_reason, *args, tool_calls=tool_calls, **kwargs
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", "--model-path", required=True, type=Path)
    parser.add_argument("--speculative-algorithm", choices=["DSPARK"], default="DSPARK")
    parser.add_argument("--speculative-draft-model-path", required=True, type=Path)
    parser.add_argument("--tool-call-parser", choices=["minicpm5"], default="minicpm5")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--prefill-step-size", type=int, default=128)
    parser.set_defaults(
        adapter_path=None,
        pipeline=False,
        trust_remote_code=False,
        chat_template=None,
        chat_template_args={"has_tool_sep": False},
        allowed_origins=(),
        temp=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
    )
    args = parser.parse_args()
    if min(args.max_tokens, args.prefill_step_size) <= 0:
        parser.error("--max-tokens and --prefill-step-size must be positive.")
    args.model = str(args.model.resolve())
    args.draft_model = str(args.speculative_draft_model_path.resolve())
    logging.basicConfig(level=logging.INFO)
    mx.set_cache_limit(256 * 2**20)

    provider = ModelProvider(args)
    try:
        provider.load_default()
    except (OSError, ValueError) as error:
        parser.error(str(error))
    response_generator = ResponseGenerator(provider, LRUPromptCache(max_size=0))
    try:
        server._run_http_server(
            args.host, args.port, response_generator, handler_class=APIHandler
        )
    finally:
        response_generator.stop_and_join()


if __name__ == "__main__":
    main()
