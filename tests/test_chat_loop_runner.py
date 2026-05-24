import pytest

from pepperbot.config import Config, default_template
from pepperbot.context.builder import ContextBuilder
from pepperbot.conversation.runner import ChatLoopRunner
from pepperbot.history.attachments import AttachmentStore
from pepperbot.history.models import Thread
from pepperbot.providers.base import ChatMessage, ChatResult
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
