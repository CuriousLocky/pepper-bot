import pytest

from pepperbot.config import Config, default_template
from pepperbot.context.builder import ContextBuilder
from pepperbot.conversation.runner import ChatLoopRunner
from pepperbot.history.attachments import AttachmentStore
from pepperbot.history.models import Thread
from pepperbot.providers.base import ChatMessage, ChatResult, ToolCall
from pepperbot.tools.executor import ToolExecutor, ToolRuntime
from pepperbot.tools.registry import ToolRegistry


class FakeProvider:
    def __init__(self):
        self.calls = 0

    async def complete(self, request):
        self.calls += 1
        if self.calls == 1:
            return ChatResult(text="")
        return ChatResult(text="repaired")


class InvalidXmlProvider:
    def __init__(self):
        self.calls = 0

    async def complete(self, request):
        self.calls += 1
        if self.calls == 1:
            return ChatResult(text="<telegram_reply>unfinished")
        return ChatResult(text="<telegram_reply>fixed</telegram_reply>")


class TextToolThenEmptyProvider:
    def __init__(self):
        self.calls = 0
        self.requests = []
        self.raw_tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "randint", "arguments": "{\"a\":1,\"b\":1}"},
            }
        ]

    async def complete(self, request):
        self.calls += 1
        self.requests.append(list(request.messages))
        if self.calls == 1:
            return ChatResult(
                text="<telegram_reply>final</telegram_reply>",
                tool_calls=[ToolCall(id="call_1", name="randint", arguments={"a": 1, "b": 1})],
                assistant_message={"tool_calls": self.raw_tool_calls},
            )
        return ChatResult(text="")


class FakeMemory:
    pass


def make_config() -> Config:
    return Config(
        bot={"token": "t", "name": "Pepper"},
        api={"url": "http://example.com/v1", "key": "k", "model": "m"},
        response={"error_response_retries": 1},
    )


@pytest.mark.asyncio
async def test_empty_response_is_retried_once(tmp_path):
    config = make_config()
    provider = FakeProvider()
    builder = ContextBuilder(config, default_template(), AttachmentStore(str(tmp_path / "attachments")))
    runner = ChatLoopRunner(config, provider, ToolRegistry(config), ToolExecutor(config, FakeMemory()), builder)
    text, _ = await runner.run(
        [ChatMessage(role="user", content="hello")],
        ToolRuntime(chat_id=1, thread=Thread(id="t", chat_id=1, created_at="2026-01-01T00:00:00", updated_at="2026-01-01T00:00:00"), attachment_store=AttachmentStore(str(tmp_path / "attachments"))),
    )
    assert text == "repaired"
    assert provider.calls == 2


@pytest.mark.asyncio
async def test_invalid_xml_response_is_retried_once(tmp_path):
    config = make_config()
    provider = InvalidXmlProvider()
    builder = ContextBuilder(config, default_template(), AttachmentStore(str(tmp_path / "attachments")))
    runner = ChatLoopRunner(config, provider, ToolRegistry(config), ToolExecutor(config, FakeMemory()), builder)
    text, _ = await runner.run(
        [ChatMessage(role="user", content="hello")],
        ToolRuntime(
            chat_id=1,
            thread=Thread(id="t", chat_id=1, created_at="2026-01-01T00:00:00", updated_at="2026-01-01T00:00:00"),
            attachment_store=AttachmentStore(str(tmp_path / "attachments")),
        ),
    )
    assert text == "fixed"
    assert provider.calls == 2


@pytest.mark.asyncio
async def test_text_plus_tool_call_uses_text_as_final_but_persists_tool_call_without_text(tmp_path):
    config = make_config()
    provider = TextToolThenEmptyProvider()
    builder = ContextBuilder(config, default_template(), AttachmentStore(str(tmp_path / "attachments")))
    runner = ChatLoopRunner(config, provider, ToolRegistry(config), ToolExecutor(config, FakeMemory()), builder)

    text, protocol_messages = await runner.run(
        [ChatMessage(role="user", content="hello")],
        ToolRuntime(
            chat_id=1,
            thread=Thread(id="t", chat_id=1, created_at="2026-01-01T00:00:00", updated_at="2026-01-01T00:00:00"),
            attachment_store=AttachmentStore(str(tmp_path / "attachments")),
        ),
    )

    assert text == "final"
    assert provider.calls == 2
    assert provider.requests[1][-2].content == "<telegram_reply>final</telegram_reply>"
    assert provider.requests[1][-2].tool_calls == provider.raw_tool_calls
    assert protocol_messages[0].role == "assistant"
    assert protocol_messages[0].content == ""
    assert protocol_messages[0].tool_calls == provider.raw_tool_calls
    assert protocol_messages[1].role == "tool"
    assert protocol_messages[1].tool_call_id == "call_1"
    assert len(protocol_messages) == 2
