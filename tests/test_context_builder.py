import pytest

from pepperbot.config import Config, default_template
from pepperbot.context.builder import ContextBuilder
from pepperbot.context.summarizer import SummaryError
from pepperbot.history.attachments import AttachmentStore
from pepperbot.history.models import Actor, ConversationMessage, Thread


def make_config(max_context_window=1000) -> Config:
    return Config(
        bot={"token": "t", "name": "Pepper"},
        api={"url": "http://example.com/v1", "key": "k", "model": "gpt-4o"},
        context={"max_context_window": max_context_window, "summary_trigger_ratio": 0.1, "preserve_recent_messages": 1},
    )


def make_vision_config() -> Config:
    return Config(
        bot={"token": "t", "name": "Pepper"},
        api={"url": "http://example.com/v1", "key": "k", "model": "gpt-4o", "supports_vision": True},
    )


def make_thread() -> Thread:
    thread = Thread(id="t1", chat_id=1, created_at="2026-01-01T00:00:00", updated_at="2026-01-01T00:00:00")
    for idx, text in enumerate(["older " * 200, "recent", "current"]):
        thread.messages.append(
            ConversationMessage(
                id=f"m{idx}",
                role="user",
                author=Actor(id=idx, name=f"User{idx}"),
                content=text,
                created_at="2026-01-01T00:00:00",
            )
        )
    return thread


class FailingSummarizer:
    async def summarize(self, thread, messages):
        raise SummaryError("boom")


@pytest.mark.asyncio
async def test_context_builder_uses_warning_on_summary_failure(tmp_path):
    builder = ContextBuilder(
        make_config(max_context_window=100),
        default_template(),
        AttachmentStore(str(tmp_path / "attachments")),
        FailingSummarizer(),
    )
    result = await builder.build(make_thread(), "m2", "memory")
    rendered = "\n".join(str(message.content) for message in result.messages)
    assert "Older context omitted" in result.warning
    assert 'id="m2"' in rendered
    assert "older older" not in rendered


@pytest.mark.asyncio
async def test_context_builder_pins_referenced_compacted_message(tmp_path):
    thread = make_thread()
    thread.summary = "older summary"
    thread.summary_until_message_id = "m0"
    thread.messages[2].reply_to = "m0"
    builder = ContextBuilder(
        make_config(),
        default_template(),
        AttachmentStore(str(tmp_path / "attachments")),
    )
    result = await builder.build(thread, "m2", "memory")
    rendered = "\n".join(str(message.content) for message in result.messages)
    assert "<pinned_referenced_message>" in rendered
    assert 'id="m0"' in rendered


@pytest.mark.asyncio
async def test_context_builder_attaches_current_images_when_vision_enabled(tmp_path):
    attachment_store = AttachmentStore(str(tmp_path / "attachments"))
    attachment = attachment_store.save_bytes(b"fake-image", "image/png", source="telegram")
    second_attachment = attachment_store.save_bytes(b"fake-image-2", "image/png", source="telegram")
    thread = Thread(id="t1", chat_id=1, created_at="2026-01-01T00:00:00", updated_at="2026-01-01T00:00:00")
    thread.messages.append(
        ConversationMessage(
            id="m0",
            role="user",
            author=Actor(id=1, name="Alice"),
            content="look",
            attachments=[attachment, second_attachment],
            created_at="2026-01-01T00:00:00",
        )
    )
    builder = ContextBuilder(make_vision_config(), default_template(), attachment_store)
    result = await builder.build(thread, "m0", "memory")
    user_message = next(message for message in result.messages if isinstance(message.content, list))
    assert isinstance(user_message.content, list)
    assert user_message.content[1]["type"] == "image_url"
    assert user_message.content[2]["type"] == "image_url"


@pytest.mark.asyncio
async def test_context_builder_exposes_fine_grained_memory_macros(tmp_path):
    builder = ContextBuilder(
        make_config(),
        default_template(),
        AttachmentStore(str(tmp_path / "attachments")),
    )
    result = await builder.build(
        make_thread(),
        "m2",
        memory_sections={
            "knowledges": "knowledge block",
            "long_term_memory": "long block",
            "short_term_memory": "short block",
            "known_user_info": "user block",
        },
        skill_list="- tarot",
    )
    rendered = "\n".join(str(message.content) for message in result.messages)
    assert "knowledge block" in rendered
    assert "long block" in rendered
    assert "short block" in rendered
    assert "user block" in rendered
    assert "- tarot" in rendered
    assert "{{memory_context}}" not in rendered


@pytest.mark.asyncio
async def test_context_builder_preserves_complete_tool_protocol(tmp_path):
    thread = Thread(id="t1", chat_id=1, created_at="2026-01-01T00:00:00", updated_at="2026-01-01T00:00:00")
    thread.messages.extend(
        [
            ConversationMessage(
                id="m0",
                role="assistant",
                author=Actor(name="Pepper", is_bot=True),
                content="",
                created_at="2026-01-01T00:00:00",
                metadata={
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "randint", "arguments": "{\"a\":1,\"b\":2}"},
                        }
                    ]
                },
            ),
            ConversationMessage(
                id="m1",
                role="tool",
                author=Actor(name="Tool", is_bot=True),
                content="2",
                created_at="2026-01-01T00:00:00",
                metadata={"tool_call_id": "call_1", "name": "randint"},
            ),
            ConversationMessage(
                id="m2",
                role="user",
                author=Actor(name="Alice"),
                content="thanks",
                created_at="2026-01-01T00:00:00",
            ),
        ]
    )
    builder = ContextBuilder(make_config(), default_template(), AttachmentStore(str(tmp_path / "attachments")))
    result = await builder.build(thread, "m2", "memory")
    assistant = next(message for message in result.messages if message.role == "assistant" and message.tool_calls)
    tool = next(message for message in result.messages if message.role == "tool")
    assert assistant.tool_calls[0]["id"] == "call_1"
    assert tool.tool_call_id == "call_1"
