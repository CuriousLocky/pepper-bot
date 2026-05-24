import logging
from typing import List, Tuple

from pepperbot.config import Config
from pepperbot.context.builder import ContextBuilder
from pepperbot.conversation.sanitizer import ResponseSanitizer
from pepperbot.providers.base import ChatMessage, ChatProvider, ChatRequest
from pepperbot.tools.executor import ToolExecutor, ToolRuntime
from pepperbot.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ChatLoopRunner:
    def __init__(
        self,
        config: Config,
        provider: ChatProvider,
        registry: ToolRegistry,
        executor: ToolExecutor,
        context_builder: ContextBuilder,
    ):
        self.config = config
        self.provider = provider
        self.registry = registry
        self.executor = executor
        self.context_builder = context_builder
        self.sanitizer = ResponseSanitizer([config.bot.name, *config.bot.nicknames])

    async def run(self, messages: List[ChatMessage], runtime: ToolRuntime) -> Tuple[str, List[ChatMessage]]:
        provider_messages = list(messages)
        new_protocol_messages: List[ChatMessage] = []
        tools = self.registry.chat_tools()
        final_text = ""

        for round_index in range(self.config.response.max_tool_rounds):
            result = await self.provider.complete(
                ChatRequest(
                    model=self.config.chat_model(),
                    messages=provider_messages,
                    tools=tools,
                    allow_tools=True,
                    temperature=self.config.model_params.temperature,
                    max_tokens=self.config.context.max_ai_response_token,
                    reasoning_effort=self.config.model_params.reasoning_effort,
                )
            )
            assistant_message = ChatMessage(
                role="assistant",
                content=result.text,
                tool_calls=result.assistant_message.get("tool_calls") if result.assistant_message else None,
            )
            provider_messages.append(assistant_message)
            new_protocol_messages.append(assistant_message)

            if result.text.strip():
                final_text = (final_text + "\n" + result.text).strip() if final_text else result.text.strip()
            if not result.tool_calls:
                break

            tool_results = await self.executor.execute(result.tool_calls, runtime)
            provider_messages.extend(tool_results)
            new_protocol_messages.extend(tool_results)
        else:
            logger.warning("Tool loop reached max_tool_rounds=%s", self.config.response.max_tool_rounds)

        parsed = self.sanitizer.parse(final_text)
        if parsed.retry or not parsed.text.strip():
            final_text = await self._repair_error_response(provider_messages, tools)
            parsed = self.sanitizer.parse(final_text)

        final_text = parsed.text.strip() or self.config.response.fallback_text

        return final_text, new_protocol_messages

    async def _repair_error_response(self, provider_messages: List[ChatMessage], tools: List[dict]) -> str:
        messages = list(provider_messages)
        for _ in range(self.config.response.error_response_retries):
            messages.append(self.context_builder.repair_message())
            result = await self.provider.complete(
                ChatRequest(
                    model=self.config.chat_model(),
                    messages=messages,
                    tools=tools,
                    allow_tools=False,
                    temperature=self.config.model_params.temperature,
                    max_tokens=self.config.context.max_ai_response_token,
                    reasoning_effort=self.config.model_params.reasoning_effort,
                )
            )
            parsed = self.sanitizer.parse(result.text)
            if not parsed.retry and parsed.text.strip():
                return parsed.text.strip()
            messages.append(ChatMessage(role="assistant", content=result.text))
        return ""
