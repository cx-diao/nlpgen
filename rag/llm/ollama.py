import re
from typing import List, Dict, Any

from ollama import AsyncClient
from tqdm.auto import tqdm
from loguru import logger
from rag.llm.core import LLM


_DUMMY_ANSWER_LOGPROBS = [
    {"text": repr("<dummy>"), "token_id": -1, "logprob": 0.0},
]


class OllamaModel(LLM):
    def __init__(self, model_name: str) -> None:
        # TODO: check ollama status and lauch ollama server if it's not running
        self.model_name = model_name

    async def async_chat(
        self,
        prompt: str,
        clean_thinking: bool = True,
        debug: bool = False,
        **kwargs,
    ):
        message = {"role": "user", "content": prompt}
        return_logprobs = kwargs.get("return_logprobs", False)
        tools = kwargs.get("tools")
        response = await AsyncClient().chat(
            model=self.model_name,
            messages=[message],
            options={"temperature": 0, "seed": 0, "reasoning_effort": "High"},
            tools=tools,
        )
        response = vars(response)  # TODO: support orginal object return
        if clean_thinking:
            cleaned_content = re.sub(r"<think>.*?</think>\n\n?", "", response["message"]["content"], flags=re.DOTALL)
            response["message"]["content"] = cleaned_content
        if debug:
            logger.info(f"OllamaModel async_chat prompt: {prompt}")
            logger.info(f"OllamaModel async_chat raw output: {response['message']['content']}")
        if return_logprobs:
            # Ollama does not expose logprobs, so keep the downstream contract by supplying a stub.
            response["answer_logprobs"] = [dict(entry) for entry in _DUMMY_ANSWER_LOGPROBS]
        return response

    async def async_chat_batch(
        self,
        prompt_list: List[str],
        *,
        batch_size: int,
        debug: bool = False,
        return_logprobs: bool = False,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        '''dummy batch implementation as ollama does not support batch chat'''
        results = []
        for start_idx in tqdm(range(0, len(prompt_list), batch_size), desc="Processing batch requests"):
            batched_prompts = prompt_list[start_idx : start_idx + batch_size]
            responses = [
                await self.async_chat(
                    prompt,
                    debug=debug,
                    return_logprobs=return_logprobs,
                    **kwargs,
                )
                for prompt in batched_prompts
            ]
            results.extend(responses)
        return results

    async def async_conversational_chat(self, messages: List):
        # TODO: maybe incorperate with langchain to manage the session history
        session_chatlog = []
        response = await AsyncClient().chat(model=self.model_name, messages=messages, options={"temperature": 1.0})
        cleaned_content = re.sub(r"<think>.*?</think>\n\n?", "", response["message"]["content"], flags=re.DOTALL)
        return cleaned_content
