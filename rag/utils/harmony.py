"""Harmony tokenisation / parsing workers for ProcessPoolExecutor.

This module is intentionally lightweight — it only imports openai_harmony so
that worker processes don't pay the cost of importing torch or vLLM, and each
worker gets its own GIL so long-running Rust calls never block the main
process's asyncio event loop.
"""

import time

from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
    ReasoningEffort,
)

_encoding = None


def init_worker(encoding_name: str = "HARMONY_GPT_OSS"):
    """Called once per worker process by ProcessPoolExecutor."""
    global _encoding
    _encoding = load_harmony_encoding(getattr(HarmonyEncodingName, encoding_name))


def tokenize(system_prompt: str, user_prompt: str):
    """Build a Harmony token prompt. Returns (prefill_ids_list, elapsed_s)."""
    t0 = time.monotonic()
    convo = Conversation.from_messages(
        [
            Message.from_role_and_content(
                Role.SYSTEM,
                SystemContent.new().with_reasoning_effort(ReasoningEffort.HIGH),
            ),
            Message.from_role_and_content(
                Role.DEVELOPER,
                DeveloperContent.new().with_instructions(system_prompt),
            ),
            Message.from_role_and_content(Role.USER, user_prompt),
        ]
    )
    prefill_ids = _encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
    elapsed = time.monotonic() - t0
    return list(prefill_ids), elapsed


def parse(output_token_ids):
    """Parse Harmony output tokens. Returns (final_content, inner_parse_s, total_s, n_tokens)."""
    n_tokens = len(output_token_ids)
    t0 = time.monotonic()
    entries = _encoding.parse_messages_from_completion_tokens(output_token_ids, Role.ASSISTANT)
    t_parse = time.monotonic() - t0

    final_content = ""
    for message in entries:
        message_dict = message.to_dict()
        if message_dict["channel"] == "final":
            for content_item in message_dict["content"]:
                if content_item["type"] == "text":
                    final_content = content_item["text"]
                    break
            break

    t_total = time.monotonic() - t0
    return final_content, t_parse, t_total, n_tokens
