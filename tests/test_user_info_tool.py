from datetime import datetime
from types import SimpleNamespace

import pytest

from pepperbot.config import Config
from pepperbot.history.models import Thread
from pepperbot.providers.base import ToolCall
from pepperbot.tools.executor import ToolExecutor, ToolRuntime
from pepperbot.tools.registry import ToolRegistry


class FakeMemory:
    def __init__(self):
        self.calls = []

    async def update_user_info(self, user_id, name, description, telegram_username=None):
        self.calls.append(
            {
                "user_id": user_id,
                "name": name,
                "description": description,
                "telegram_username": telegram_username,
            }
        )


def make_runtime():
    return ToolRuntime(
        chat_id=42,
        thread=Thread(id="thread", chat_id=42, created_at=datetime.now(), updated_at=datetime.now()),
        attachment_store=SimpleNamespace(),
    )


def test_update_user_info_tool_schema_includes_optional_telegram_username():
    registry = ToolRegistry(Config(bot={"token": "t"}, api={"url": "u", "key": "k", "model": "m"}))
    tool = next(item for item in registry.chat_tools() if item["function"]["name"] == "update_user_info")
    params = tool["function"]["parameters"]

    assert "telegram_username" in params["properties"]
    assert params["properties"]["telegram_username"]["pattern"] == "^@"
    assert "telegram_username" not in params["required"]


@pytest.mark.asyncio
async def test_update_user_info_tool_accepts_at_prefixed_telegram_username():
    memory = FakeMemory()
    executor = ToolExecutor(
        Config(bot={"token": "t"}, api={"url": "u", "key": "k", "model": "m"}),
        memory,
    )

    result = await executor.execute(
        [
            ToolCall(
                id="call",
                name="update_user_info",
                arguments={
                    "user_id": 123,
                    "name": "Alice",
                    "description": "Known tester",
                    "telegram_username": "@alice",
                },
            )
        ],
        make_runtime(),
    )

    assert result[0].content == "User info updated successfully."
    assert memory.calls == [
        {
            "user_id": 123,
            "name": "Alice",
            "description": "Known tester",
            "telegram_username": "@alice",
        }
    ]


@pytest.mark.asyncio
async def test_update_user_info_tool_rejects_telegram_username_without_at_prefix():
    memory = FakeMemory()
    executor = ToolExecutor(
        Config(bot={"token": "t"}, api={"url": "u", "key": "k", "model": "m"}),
        memory,
    )

    result = await executor.execute(
        [
            ToolCall(
                id="call",
                name="update_user_info",
                arguments={
                    "user_id": 123,
                    "name": "Alice",
                    "description": "Known tester",
                    "telegram_username": "alice",
                },
            )
        ],
        make_runtime(),
    )

    assert result[0].content == "Error: telegram_username must start with @."
    assert memory.calls == []
