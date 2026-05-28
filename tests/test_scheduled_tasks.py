from datetime import datetime
from types import SimpleNamespace

import pytest

from pepperbot.config import Config
from pepperbot.history.models import TelegramRef, Thread
from pepperbot.providers.base import ToolCall
from pepperbot.telegram.app import PepperBotApplication
from pepperbot.tools.executor import ToolExecutor, ToolRuntime


class FakeHistory:
    def __init__(self):
        self.thread = None
        self.saved = False

    def create_thread(self, chat_id):
        self.thread = Thread(id="thread", chat_id=chat_id, created_at=datetime.now(), updated_at=datetime.now())
        return self.thread

    def add_message(self, thread, message):
        thread.messages.append(message)

    def save(self):
        self.saved = True


class FakeService:
    class Sanitizer:
        def clean(self, text):
            return text

    sanitizer = Sanitizer()

    async def _memory_context(self, incoming, bot=None, thread=None):
        return {"knowledges": "", "long_term_memory": "", "short_term_memory": "", "known_user_info": ""}

    def _skill_list(self):
        return "No skills available."

    def _append_protocol_messages(self, thread, protocol_messages, tool_runtime):
        pass


class FakeContextBuilder:
    async def build(self, *args, **kwargs):
        return SimpleNamespace(messages=[])


class FakeChatRunner:
    async def run(self, messages, tool_runtime):
        return "scheduled response", []


class FakeDelivery:
    async def send_text(self, bot, chat_id, text, reply_to_message_id=None):
        return [TelegramRef(chat_id=chat_id, message_id=100)]

    async def send_images(self, bot, chat_id, images):
        return []


class FakeReporter:
    async def report(self, *args, **kwargs):
        pass


def make_app():
    app = object.__new__(PepperBotApplication)
    app.config = Config(
        bot={"token": "t", "name": "Pepper"},
        api={"url": "http://example.com/v1", "key": "k", "model": "m"},
    )
    app.history = FakeHistory()
    app.service = FakeService()
    app.context_builder = FakeContextBuilder()
    app.chat_runner = FakeChatRunner()
    app.delivery = FakeDelivery()
    app.reporter = FakeReporter()
    app.attachment_store = SimpleNamespace(save_data_uri=lambda *args, **kwargs: None)
    app.add_blacklist = lambda *args, **kwargs: None
    return app


@pytest.mark.asyncio
async def test_scheduled_task_callback_appends_system_message_and_sends_response():
    app = make_app()
    context = SimpleNamespace(
        bot=SimpleNamespace(id=999),
        job=SimpleNamespace(chat_id=42, data={"title": "Reminder", "content": "Do thing"}),
    )

    await app.execute_task_callback(context)

    assert app.history.saved is True
    assert app.history.thread.messages[0].author.name == "System"
    assert app.history.thread.messages[0].metadata["kind"] == "scheduled_task_trigger"
    assert app.history.thread.messages[-1].content == "scheduled response"


@pytest.mark.asyncio
async def test_set_scheduled_task_rejects_missing_delay():
    executor = ToolExecutor(Config(bot={"token": "t"}, api={"url": "u", "key": "k", "model": "m"}), memory_manager=None)
    runtime = ToolRuntime(
        chat_id=42,
        thread=Thread(id="thread", chat_id=42, created_at=datetime.now(), updated_at=datetime.now()),
        attachment_store=SimpleNamespace(),
        schedule_func=lambda delay, title, content: None,
    )

    result = await executor.execute([ToolCall(id="call", name="set_scheduled_task", arguments={})], runtime)

    assert "Missing required delay" in result[0].content


@pytest.mark.asyncio
async def test_set_scheduled_task_accepts_delay_alias():
    seen = {}

    async def schedule(delay, title, content):
        seen["delay"] = delay
        return "scheduled"

    executor = ToolExecutor(Config(bot={"token": "t"}, api={"url": "u", "key": "k", "model": "m"}), memory_manager=None)
    runtime = ToolRuntime(
        chat_id=42,
        thread=Thread(id="thread", chat_id=42, created_at=datetime.now(), updated_at=datetime.now()),
        attachment_store=SimpleNamespace(),
        schedule_func=schedule,
    )

    result = await executor.execute(
        [ToolCall(id="call", name="set_scheduled_task", arguments={"delay_minutes": "5"})],
        runtime,
    )

    assert result[0].content == "scheduled"
    assert seen["delay"] == 5
