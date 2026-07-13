import asyncio
import re
from typing import Dict, Any, Callable, Optional, List

from loguru import logger


class LLMGenerator:
    """
    RAG Generator that encapsulates the RAG pattern.

    This class handles the common RAG workflow:
    1. Takes a prompt template with placeholders
    2. Uses a prompt builder function to create the final prompt (including RAG functionality)
    3. Generates response using an LLM
    """

    def __init__(
        self,
        prompt_builder: Callable[[Dict[str, Any]], str],
        llm: Any,
        *,
        logprob_pattern: Optional[re.Pattern] = None,
        return_logprobs: bool = False,
        debug: bool = False,
        **kwargs,
    ):
        """
        Initialise the LLMGenerator.

        Args:
            prompt_builder: Function that takes a dictionary of placeholders and returns the complete prompt.
            llm: LLM instance that has an async_chat method.
            logprob_pattern: Optional regex forwarded to the LLM backend so it
                can locate the answer token within the final-channel output.
            return_logprobs: Whether to request logprobs from the LLM backend.
            debug: Whether to enable debug logging.
            **kwargs: Additional arguments passed to prompt_builder.
        """
        self.prompt_builder = prompt_builder
        self.llm = llm
        self.logprob_pattern = logprob_pattern
        self.return_logprobs = return_logprobs
        self.debug = debug
        self.builder_kwargs = kwargs

    async def generate(self, placeholder2value: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a response using RAG.

        Args:
            placeholder2value: Dictionary mapping template placeholders to their values

        Returns:
            Dict containing the generated response text plus optional metadata such as logprobs.
        """
        prompt = await asyncio.to_thread(self.prompt_builder, placeholder2value, **self.builder_kwargs)

        if self.debug:
            logger.info(f"Generated prompt: {prompt}")

        response = await self.llm.async_chat(
            prompt, return_logprobs=self.return_logprobs, logprob_pattern=self.logprob_pattern,
        )

        return {
            "content": response["message"]["content"].strip(),
            "answer_logprobs": response.get("answer_logprobs"),
        }

    async def generate_batch(
        self,
        placeholder2value_list: List[Dict[str, Any]],
        *,
        batch_size: int = 4,
        debug: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for a batch of placeholder dictionaries.

        Args:
            placeholder2value_list: List of dictionaries, each mapping template placeholders to their values

        Returns:
            List of dicts containing generated response text plus optional metadata such as logprobs.
        """
        if self.debug:
            logger.info(f"Processing batch of {len(placeholder2value_list)} requests")

        prompts = [self.prompt_builder(placeholder2value, **self.builder_kwargs) for placeholder2value in placeholder2value_list]
        responses = await self.llm.async_chat_batch(
            prompts,
            batch_size=batch_size,
            debug=debug,
            return_logprobs=self.return_logprobs,
            logprob_pattern=self.logprob_pattern,
        )

        return [
            {
                "content": response["message"]["content"].strip(),
                "answer_logprobs": response.get("answer_logprobs"),
            }
            for response in responses
        ]

    async def __call__(self, placeholder2value: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convenience method to call generate with a placeholder dictionary.

        Args:
            placeholder2value: Dictionary mapping template placeholders to their values

        Returns:
            Dict containing the generated response text plus optional metadata such as logprobs.
        """
        return await self.generate(placeholder2value)
