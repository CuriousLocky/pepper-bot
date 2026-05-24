from typing import Iterable

import tiktoken

from pepperbot.providers.base import ChatMessage


def get_encoding(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def estimate_text_tokens(text: str, model: str) -> int:
    return len(get_encoding(model).encode(text or ""))


def estimate_message_tokens(messages: Iterable[ChatMessage], model: str) -> int:
    encoding = get_encoding(model)
    total = 0
    for message in messages:
        total += 4 + len(encoding.encode(message.role))
        if isinstance(message.content, str):
            total += len(encoding.encode(message.content))
        else:
            for part in message.content:
                if part.get("type") == "text":
                    total += len(encoding.encode(part.get("text", "")))
                elif part.get("type") == "image_url":
                    total += 85
        if message.tool_calls:
            total += 64 * len(message.tool_calls)
        if message.tool_call_id:
            total += len(encoding.encode(message.tool_call_id))
    return total
