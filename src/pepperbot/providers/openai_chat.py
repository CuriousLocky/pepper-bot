import json
import logging
from typing import Any, Dict, List

from openai import AsyncOpenAI

from pepperbot.config import Config
from pepperbot.providers.base import ChatMessage, ChatProvider, ChatRequest, ChatResult, ToolCall

logger = logging.getLogger(__name__)


def _message_to_dict(message: ChatMessage) -> Dict[str, Any]:
    data: Dict[str, Any] = {"role": message.role, "content": message.content}
    if message.name:
        data["name"] = message.name
    if message.tool_calls:
        data["tool_calls"] = message.tool_calls
    if message.tool_call_id:
        data["tool_call_id"] = message.tool_call_id
    return data


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return str(content)


class OpenAIChatCompletionsProvider(ChatProvider):
    def __init__(self, config: Config, api_key: str | None = None, base_url: str | None = None):
        self.config = config
        self.client = AsyncOpenAI(api_key=api_key or config.chat_api_key(), base_url=base_url or config.chat_api_url())

    async def complete(self, request: ChatRequest) -> ChatResult:
        kwargs: Dict[str, Any] = {
            "model": request.model,
            "messages": [_message_to_dict(message) for message in request.messages],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.allow_tools and self.config.chat_backend.supports_tools and request.tools:
            kwargs["tools"] = request.tools
            kwargs["tool_choice"] = "auto"
        if (
            request.reasoning_effort
            and self.config.chat_backend.supports_reasoning_effort
        ):
            kwargs["reasoning_effort"] = request.reasoning_effort

        response = await self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message
        assistant_message = message.model_dump(exclude_none=True)
        parsed_tool_calls: List[ToolCall] = []

        for tool_call in getattr(message, "tool_calls", None) or []:
            raw_arguments = tool_call.function.arguments or "{}"
            try:
                arguments = json.loads(raw_arguments)
                if not isinstance(arguments, dict):
                    arguments = {"value": arguments}
            except Exception:
                logger.warning("Tool call %s has invalid JSON arguments", tool_call.id)
                arguments = {}
            parsed_tool_calls.append(
                ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=arguments,
                    raw_arguments=raw_arguments,
                )
            )

        return ChatResult(
            text=_content_to_text(getattr(message, "content", None)),
            tool_calls=parsed_tool_calls,
            finish_reason=getattr(choice, "finish_reason", "") or "",
            assistant_message=assistant_message,
            raw=response,
        )
