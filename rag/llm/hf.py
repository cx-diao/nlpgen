import os
import re
from typing import List

from rag.llm.core import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.distributed import DistributedConfig
from huggingface_hub import login
import torch
import asyncio
from loguru import logger
from tqdm.auto import tqdm

from substantiator.utils.general import PROJ_ROOT


class HFModel(LLM):
    def __init__(self, model_name: str) -> None:

        # login to huggingface
        # if model_name.startswith("openai/"):
        #     token_file = PROJ_ROOT / "huggingface_api_key"
        #     with open(token_file, "r") as f:
        #         token = f.read().strip()
        #     login(token=token)

        # device_map = {
        #     # Enable Expert Parallelism
        #     "distributed_config": DistributedConfig(enable_expert_parallel=1),
        #     # Enable Tensor Parallelism
        #     "tp_plan": "auto",
        # }

        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
        )
        # Set padding token if not already set (required for batching)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            # attn_implementation="kernels-community/vllm-flash-attn3",
            # device_map=device_map,
            device_map="auto",
        )

    def clean_content(self, generated_tokens: str) -> str:
        final_channel_pattern = r'<\|channel\|>final<\|message\|>(.*?)(<\|return\|>|<\|end\|>)'
        match = re.search(final_channel_pattern, generated_tokens, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            logger.warning(f"HFModel clean_content no final channel found, returning full output: {generated_tokens}")
            return generated_tokens

    @torch.inference_mode()
    async def async_chat_gpt_oss(self, prompt: str, *, debug: bool = False, **kwargs):
        if debug:
            logger.info(f"HFModel async_chat prompt: {prompt}")

        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            reasoning_effort=kwargs.get("reasoning_effort", "medium"),
        ).to(self.llm.device)
        outputs = self.llm.generate(**inputs, max_new_tokens=2048, temperature=1.0, top_p=1.0)
        generated_tokens = self.tokenizer.decode(outputs[0])
        cleaned_content = self.clean_content(generated_tokens)

        if debug:
            logger.info(f"HFModel async_chat raw output: {generated_tokens}")
            logger.info(f"HFModel async_chat cleaned output: {cleaned_content}")

        response = {"message": {"content": cleaned_content}}

        return response

    async def async_chat(self, prompt: str, *, debug: bool = False, **kwargs):
        if self.model_name.startswith("openai/gpt-oss"):
            return await self.async_chat_gpt_oss(prompt, debug=debug, **kwargs)
        else:
            raise ValueError(f"HFModel async_chat not supported for model: {self.model_name}")

    @torch.inference_mode()
    async def async_chat_batch_gpt_oss(self, prompts: List[str], *, batch_size: int = 4, debug: bool = False, **kwargs):
        results = []
        for start_idx in tqdm(range(0, len(prompts), batch_size), desc="Processing batch requests"):
            batch_prompts = prompts[start_idx : start_idx + batch_size]
            
            # if debug:
            #     logger.info(f"HFModel async_chat_batch_gpt_oss batch prompts: {batch_prompts}")

            # Create a list of conversations (each conversation is a list of messages)
            batch_messages = [[{"role": "user", "content": prompt}] for prompt in batch_prompts]

            inputs = self.tokenizer.apply_chat_template(
                batch_messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                padding=True,
                reasoning_effort=kwargs.get("reasoning_effort", "medium"),
            ).to(self.llm.device)
            outputs = self.llm.generate(**inputs, max_new_tokens=2048, temperature=1.0, top_p=1.0)
            generated_tokens = self.tokenizer.batch_decode(outputs)
            # TODO: parse logits for Yes/No, and return their list in the results tuple

            for ii in range(len(batch_prompts)):
                generated_tokens_ii = generated_tokens[ii]
                cleaned_content_ii = self.clean_content(generated_tokens_ii)
                if debug:
                    logger.info(f"HFModel async_chat_batch_gpt_oss batch raw output: {generated_tokens_ii}")
                    logger.info(f"HFModel async_chat_batch_gpt_oss batch cleaned content: {cleaned_content_ii}")
                response = {"message": {"content": cleaned_content_ii}}
                results.append(response)

        return results

    async def async_chat_batch(self, prompts: List[str], *, batch_size: int = 4, debug: bool = False, **kwargs):
        if self.model_name.startswith("openai/gpt-oss"):
            return await self.async_chat_batch_gpt_oss(prompts, batch_size=batch_size, debug=debug, **kwargs)
        else:
            raise ValueError(f"HFModel async_chat_batch not supported for model: {self.model_name}")


if __name__ == "__main__":
    model = HFModel(model_name="openai/gpt-oss-20b")
    prompts = ["What is the capital of France?", 
        "What is the capital of Germany?",
        "What is the capital of China?", 
        "What is the capital of South Korea?",
        "What is the capital of the United States?",
        "What is the capital of Turkey?", 
        "What is the capital of Ireland?",
        "What is the capital of Russia?", ]
    results = asyncio.run(model.async_chat_batch(prompts, batch_size=8, debug=True))
    # results = asyncio.run(model.async_chat(prompts[0], debug=True))
    print(results)
