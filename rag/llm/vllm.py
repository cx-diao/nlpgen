import os

os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_V1_OUTPUT_PROC_CHUNK_SIZE", "8")

import asyncio
import functools
import json
import re
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Dict, Any, Optional

from loguru import logger
from openai_harmony import HarmonyEncodingName, Role, load_harmony_encoding
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput
from vllm.v1.engine.async_llm import AsyncLLM
import torch

from rag.llm.core import LLM
from rag.utils import harmony
from rag.utils.logprob_utils import find_answer_token_position
from rag.utils.system import effective_cpu_count

# ---------------------------------------------------------------------------
# Monkey-patch: skip the expensive per-token tokenizer.decode() that vLLM
# runs *synchronously* inside its output_handler for every logprob entry.
# We never use the decoded_token strings (we decode via Harmony instead),
# so returning None for each token eliminates the bottleneck while keeping
# all logprob values, token IDs, and ranks intact.
# ---------------------------------------------------------------------------
import vllm.transformers_utils.detokenizer_utils as _detok_utils


def _noop_convert_ids_list_to_tokens(_tokenizer, token_ids):
    return [None] * len(token_ids)


_detok_utils.convert_ids_list_to_tokens = _noop_convert_ids_list_to_tokens

import vllm.v1.engine.logprobs as _logprobs_mod

_logprobs_mod.convert_ids_list_to_tokens = _noop_convert_ids_list_to_tokens


def _detect_gpu_config(num_gpus: int) -> Dict[str, int]:
    """Select tensor/data parallelism and max batched tokens based on available GPUs.

    The returned dict contains ``tensor_parallel_size``, ``data_parallel_size``
    and ``max_num_batched_tokens``.  Values are chosen conservatively so that
    the caller can always override individual keys via ``**kwargs``.
    """
    props = torch.cuda.get_device_properties(0)
    gpu_name = props.name.upper()
    total_gb = props.total_memory / (1024**3)

    # vLLM v0.10.2 does not support MXFP4 MoE weights with DP>1 — the fused_moe
    # weight loader produces a tensor shape mismatch when sharding across DP ranks.
    # Use TP=num_gpus, DP=1 until this is fixed upstream.
    dp = 1
    tp = num_gpus

    if "H200" in gpu_name:
        max_bt = 65536
    elif "H100" in gpu_name:
        max_bt = 32768
    elif total_gb > 70:
        max_bt = 16384
    else:
        max_bt = 4096

    if "H200" in gpu_name:
        max_seqs = 500
    elif "H100" in gpu_name:
        max_seqs = 256
    elif total_gb > 70:
        max_seqs = 256
    else:
        max_seqs = 128

    assert tp * dp <= num_gpus, (
        f"Detected config requires {tp * dp} GPUs (TP={tp} × DP={dp}) " f"but only {num_gpus} available"
    )

    config = {
        "tensor_parallel_size": tp,
        "data_parallel_size": dp,
        "max_num_batched_tokens": max_bt,
        "max_num_seqs": max_seqs,
    }
    logger.info(
        "Auto-detected GPU: {} ({:.1f} GB) × {} — using TP={}, DP={}, max_num_batched_tokens={}, max_num_seqs={}",
        props.name,
        total_gb,
        num_gpus,
        tp,
        dp,
        max_bt,
        max_seqs,
    )
    return config


class VLLMModel(LLM):
    def __init__(self, model_name: str, **kwargs: Any) -> None:
        if not model_name.startswith("openai/gpt-oss"):
            raise ValueError(f"VLLMModel only supports openai/gpt-oss models, got {model_name}")

        self.model_name = model_name

        num_gpus = torch.cuda.device_count()
        assert num_gpus > 0, "No CUDA GPUs detected"
        gpu_config = _detect_gpu_config(num_gpus)
        # Explicit kwargs override auto-detected values
        gpu_config.update({k: v for k, v in kwargs.items() if k in gpu_config})

        self._engine_args = AsyncEngineArgs(
            model=model_name,
            trust_remote_code=True,
            # Avoid downloading the repository's redundant 13.8 GB original checkpoint.
            ignore_patterns=["original/*", "original/**/*"],
            tensor_parallel_size=gpu_config["tensor_parallel_size"],
            data_parallel_size=gpu_config["data_parallel_size"],
            max_num_batched_tokens=gpu_config["max_num_batched_tokens"],
            max_num_seqs=gpu_config["max_num_seqs"],
        )
        self._engine: Optional[AsyncLLM] = None
        self._request_counter = 0

        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()
        # Keep defaults close to Ollama's runtime behaviour where possible.
        # - Ollama default num_predict is effectively unbounded (-1), so we
        #   treat missing max_tokens as uncapped for vLLM.
        # - Ollama defaults include top_k=40, top_p=0.9, repeat_penalty=1.1.
        max_tokens_raw = kwargs.get("max_tokens", 16384)  # 8192
        max_tokens: Optional[int] = None if max_tokens_raw is None else int(max_tokens_raw)
        if max_tokens is not None:
            assert max_tokens > 0, "max_tokens must be positive when provided"
        top_k = int(kwargs.get("top_k", 40))
        assert top_k >= 0, "top_k must be non-negative"
        top_p = float(kwargs.get("top_p", 0.9))
        assert 0.0 < top_p <= 1.0, "top_p must be in (0, 1]"
        repetition_penalty = float(kwargs.get("repetition_penalty", 1.1))
        assert repetition_penalty > 0, "repetition_penalty must be positive"
        self.sampling = SamplingParams(
            seed=0,
            logprobs=20,
            max_tokens=max_tokens,
            temperature=0,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_token_ids=self.stop_token_ids,
        )
        self.system_prompt = "Always follow the output format requested by the user."

        cpu_count = effective_cpu_count()
        harmony_workers = min(32, max(4, cpu_count // 2))
        logprob_workers = min(8, max(2, cpu_count // 8))
        self._logprob_pool = ThreadPoolExecutor(max_workers=logprob_workers)
        self._harmony_pool = ProcessPoolExecutor(
            max_workers=harmony_workers,
            mp_context=mp.get_context("spawn"),
            initializer=harmony.init_worker,
            initargs=("HARMONY_GPT_OSS",),
        )
        logger.info(
            "CPU pool sizing: {} CPUs detected → harmony={} processes, logprob={} threads",
            cpu_count,
            harmony_workers,
            logprob_workers,
        )

    # -- Engine lifecycle ---------------------------------------------------

    async def _get_engine(self) -> AsyncLLM:
        if self._engine is None:
            self._engine = AsyncLLM.from_engine_args(self._engine_args)
        return self._engine

    def _next_request_id(self) -> str:
        self._request_counter += 1
        return f"vllm-req-{self._request_counter}"

    def shutdown(self) -> None:
        self._harmony_pool.shutdown(wait=False)
        self._logprob_pool.shutdown(wait=False)
        if self._engine is not None:
            self._engine.shutdown()
            self._engine = None

    # -- Single-request async generation ------------------------------------

    async def _generate_one(self, engine: AsyncLLM, tokens_prompt: TokensPrompt, request_id: str) -> RequestOutput:
        """Submit a single request and drain the async generator to get the final output."""
        final_output: Optional[RequestOutput] = None
        async for output in engine.generate(
            prompt=tokens_prompt,
            sampling_params=self.sampling,
            request_id=request_id,
        ):
            final_output = output
        assert final_output is not None, f"Request {request_id} produced no output"
        return final_output

    # -- Debug helpers ------------------------------------------------------

    def _dump_harmony_debug(self, raw_text: str, output_tokens, *, label: str = "") -> None:
        """Print the raw vLLM text plus the Harmony-parsed messages for one request.

        Re-parses ``output_tokens`` in the main process; this is debug-only work
        and runs at most once per request when ``debug=True``.
        """
        print(f"=== {label}RAW TEXT OUTPUT ({len(output_tokens)} tokens) ===")
        print(raw_text)
        print("=== END RAW TEXT ===\n")

        entries = self.encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)
        print(f"=== {label}PARSED MESSAGES ({len(entries)} messages) ===")
        for message in entries:
            print(json.dumps(message.to_dict()))
        print("=== END PARSED MESSAGES ===\n")

    # -- Logprob extraction helper -------------------------------------------

    def _extract_answer_logprobs(
        self,
        output_tokens,
        final_content: str,
        logprobs,
        logprob_pattern: Optional[re.Pattern],
        *,
        label: str = "",
        debug: bool = False,
    ) -> List[Dict[str, Any]]:
        """Locate the answer token and return its top-k logprobs.

        Runs entirely on the calling thread (designed for ``asyncio.to_thread``).
        """
        if not final_content:
            raise ValueError(f"{label}no 'final' channel content")
        if not logprobs:
            raise ValueError(f"{label}vLLM returned no logprobs")

        answer_pos = find_answer_token_position(
            output_tokens,
            final_content,
            logprob_pattern,
            self.encoding.decode,
        )
        if answer_pos is None:
            raise ValueError(f"{label}could not locate answer token position")
        if answer_pos >= len(logprobs):
            raise ValueError(f"{label}answer_pos={answer_pos} out of range (len(logprobs)={len(logprobs)})")

        if debug:
            print(f"\n=== {label}LOGPROBS AT ANSWER TOKEN (position {answer_pos}) ===")

        result: List[Dict[str, Any]] = []
        timestep_logprobs = logprobs[answer_pos]
        for token_id, logprob_obj in timestep_logprobs.items():
            token_text = self.encoding.decode([token_id])
            result.append(
                {
                    "text": repr(token_text),
                    "token_id": token_id,
                    "logprob": logprob_obj.logprob,
                }
            )
            if debug:
                print(f"{{text: {repr(token_text)}, token_id: {token_id}, logprob: {logprob_obj.logprob:.4f}}}")

        if debug:
            print("=== END LOGPROBS ===\n")
        return result

    # -- Public interface (matches rag.llm.core.LLM ABC) --------------------

    async def async_chat(
        self,
        prompt: str,
        *,
        return_logprobs: bool = False,
        logprob_pattern: Optional[re.Pattern] = None,
        debug: bool = False,
        **kwargs,
    ):
        engine = await self._get_engine()
        loop = asyncio.get_running_loop()

        prefill_ids, _t_inner_tok = await loop.run_in_executor(
            self._harmony_pool, harmony.tokenize, self.system_prompt, prompt
        )
        tokens_prompt = TokensPrompt(prompt_token_ids=prefill_ids)

        request_output = await self._generate_one(engine, tokens_prompt, self._next_request_id())

        gen = request_output.outputs[0]
        output_tokens = gen.token_ids
        logprobs = gen.logprobs

        final_content, _t_inner_parse, _t_total_parse, _n_tokens = await loop.run_in_executor(
            self._harmony_pool, harmony.parse, tuple(output_tokens)
        )

        if debug:
            self._dump_harmony_debug(gen.text, output_tokens)

        response: Dict[str, Any] = {"message": {"content": final_content}}
        if return_logprobs and logprob_pattern is not None:
            response["answer_logprobs"] = loop.run_in_executor(
                self._logprob_pool,
                functools.partial(
                    self._extract_answer_logprobs,
                    output_tokens,
                    final_content,
                    logprobs,
                    logprob_pattern,
                    debug=debug,
                ),
            )
        return response

    async def async_chat_batch(
        self,
        prompt_list: List[str],
        *,
        batch_size: int,
        return_logprobs: bool = False,
        logprob_pattern: Optional[re.Pattern] = None,
        debug: bool = False,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Batch inference using AsyncLLM's truly concurrent request handling.

        All prompts are submitted simultaneously. The engine's continuous-batching
        scheduler manages GPU memory; each request completes independently so fast
        completions are never blocked by slow runaway ones.

        Args:
            prompt_list: List of prompts to process.
            batch_size: Passed through by callers but not used for chunking here;
                the engine's scheduler owns GPU-level batching decisions.
            return_logprobs: Whether to extract answer-token logprobs.
            logprob_pattern: Compiled regex applied to the final-channel text to
                locate the answer token. Must not be *None* when
                *return_logprobs* is True.
            debug: Whether to print debug information.

        Returns:
            List of response dicts ``{"message": {"content": str}, "answer_logprobs": [...]}``.
        """
        engine = await self._get_engine()
        loop = asyncio.get_running_loop()

        tok_futures = [
            loop.run_in_executor(self._harmony_pool, harmony.tokenize, self.system_prompt, p) for p in prompt_list
        ]
        tok_results = await asyncio.gather(*tok_futures)
        tokens_prompts = [TokensPrompt(prompt_token_ids=ids) for ids, _t in tok_results]

        tasks = [self._generate_one(engine, tp, self._next_request_id()) for tp in tokens_prompts]
        request_outputs: List[RequestOutput] = await asyncio.gather(*tasks)

        results: List[Dict[str, Any]] = []
        for idx, request_output in enumerate(request_outputs):
            gen = request_output.outputs[0]
            output_tokens = gen.token_ids
            logprobs_data = gen.logprobs

            final_content, _t_inner_parse, _t_total_parse, _n = await loop.run_in_executor(
                self._harmony_pool, harmony.parse, tuple(output_tokens)
            )

            if debug:
                self._dump_harmony_debug(gen.text, output_tokens, label=f"[item {idx}] ")

            response: Dict[str, Any] = {"message": {"content": final_content}}
            if return_logprobs and logprob_pattern is not None:
                response["answer_logprobs"] = loop.run_in_executor(
                    self._logprob_pool,
                    functools.partial(
                        self._extract_answer_logprobs,
                        output_tokens,
                        final_content,
                        logprobs_data,
                        logprob_pattern,
                        debug=debug,
                        label=f"[item {idx}] ",
                    ),
                )

            results.append(response)

        return results


if __name__ == "__main__":

    async def _smoke_test():
        model = VLLMModel(model_name="openai/gpt-oss-20b")
        try:
            prompt = "What is the weather like in SF?"
            result = await model.async_chat(prompt, debug=True)
            print("\n=== FINAL RESPONSE ===")
            print(result)

            prompts = [
                "What is the capital of France?",
                "What is the capital of Germany?",
                "What is the capital of China?",
                "What is the capital of South Korea?",
                "What is the capital of the United States?",
                "What is the capital of Turkey?",
                "What is the capital of Ireland?",
                "What is the capital of Russia?",
            ]
            results = await model.async_chat_batch(prompts, batch_size=8, debug=True)
            print("\n=== BATCH RESULTS ===")
            print(results)
        finally:
            model.shutdown()

    asyncio.run(_smoke_test())
