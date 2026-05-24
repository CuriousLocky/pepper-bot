from dataclasses import dataclass
from typing import List

from pepperbot.config import Config
from pepperbot.history.models import ConversationMessage, Thread
from pepperbot.providers.base import ChatMessage, ChatProvider, ChatRequest


class SummaryError(Exception):
    pass


@dataclass
class SummaryResult:
    summary: str
    until_message_id: str


class ThreadSummarizer:
    def __init__(self, config: Config, provider: ChatProvider):
        self.config = config
        self.provider = provider

    async def summarize(self, thread: Thread, messages: List[ConversationMessage]) -> SummaryResult:
        if not messages:
            raise SummaryError("No messages available for summarization")

        transcript = "\n".join(
            f"[{message.id}] {message.role} {message.author.name}: {message.content}"
            for message in messages
        )
        previous = thread.summary.strip() or "No previous summary."
        prompt = (
            "Summarize the following Telegram group chat thread for future context. "
            "Preserve facts, decisions, user preferences, unresolved requests, and tone. "
            "Use the dominant language of the conversation.\n\n"
            f"Previous summary:\n{previous}\n\n"
            f"Messages to summarize:\n{transcript}"
        )
        try:
            result = await self.provider.complete(
                ChatRequest(
                    model=self.config.tool_model.model or self.config.chat_model(),
                    messages=[ChatMessage(role="user", content=prompt)],
                    allow_tools=False,
                    temperature=0.2,
                    max_tokens=1200,
                )
            )
        except Exception as exc:
            raise SummaryError(str(exc)) from exc
        summary = result.text.strip()
        if not summary:
            raise SummaryError("Tool model returned an empty summary")
        return SummaryResult(summary=summary, until_message_id=messages[-1].id)
