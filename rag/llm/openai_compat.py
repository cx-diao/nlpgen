import asyncio
import os
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import AsyncOpenAI

from rag.llm.core import LLM


class OpenAICompatModel(LLM):
    """Chat model served through any OpenAI-compatible Chat Completions endpoint.

    This covers the OpenAI API itself and servers exposing the same interface, such as
    ``vllm serve``, Ollama (``http://localhost:11434/v1``) or hosted model providers.

    Unlike :class:`rag.llm.vllm.VLLMModel`, this backend does not request
    answer-token logprobs, so ``answer_logprobs`` is always ``None``.
    """

    def __init__(
        self,
        model_name: str,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = 0.0,
        reasoning_effort: Optional[str] = None,
        max_concurrency: int = 8,
        max_retries: int = 5,
        timeout: float = 600.0,
    ) -> None:
        """
        Args:
            model_name: Model name as known to the endpoint (e.g. ``gpt-oss:20b`` on Ollama).
            base_url: Endpoint URL. ``None`` uses ``OPENAI_BASE_URL`` if set, otherwise the OpenAI API.
            api_key: API key. ``None`` reads ``OPENAI_API_KEY``; local servers without auth need none.
            temperature: Sampling temperature. ``None`` omits it, for models that reject the parameter.
            reasoning_effort: Optional reasoning effort (``low``/``medium``/``high``) for reasoning models.
            max_concurrency: Maximum number of requests in flight at once.
            max_retries: Retries on rate limits and transient server errors (handled by the OpenAI client).
            timeout: Per-request timeout in seconds.
        """
        assert max_concurrency > 0, "max_concurrency must be positive"
        self.model_name = model_name
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        # Same instruction the vLLM backend sends as the developer message.
        self.system_prompt = "Always follow the output format requested by the user."

        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            if not base_url:
                raise ValueError("Set OPENAI_API_KEY, or pass base_url for a local server that needs no key.")
            api_key = "EMPTY"  # the client requires a value even when the server ignores it

        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=max_retries, timeout=timeout)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        logger.info(
            "OpenAI-compatible backend: model={}, base_url={}, temperature={}, reasoning_effort={}, max_concurrency={}",
            model_name,
            base_url or "OpenAI API",
            temperature,
            reasoning_effort,
            max_concurrency,
        )
        logger.info("Answer-token logprobs are not requested: the *_logprobs output columns will be empty.")

    async def async_chat(self, prompt: str, *, debug: bool = False, **kwargs) -> Dict[str, Any]:
        request: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            request["reasoning_effort"] = self.reasoning_effort

        async with self._semaphore:
            completion = await self._client.chat.completions.create(**request)
        content = completion.choices[0].message.content or ""

        if debug:
            logger.info(f"OpenAICompatModel prompt: {prompt}")
            logger.info(f"OpenAICompatModel output: {content}")
        return {"message": {"content": content}, "answer_logprobs": None}

    async def async_chat_batch(
        self, prompt_list: List[str], *, batch_size: int, debug: bool = False, **kwargs
    ) -> List[Dict[str, Any]]:
        """Send all prompts concurrently (bounded by ``max_concurrency``); ``batch_size`` is unused."""
        return await asyncio.gather(*(self.async_chat(prompt, debug=debug) for prompt in prompt_list))

    def shutdown(self) -> None:
        """Nothing to release explicitly; HTTP connections close with the client."""
