import base64
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image
from telegram import ChatMemberMember, PhotoSize, User, UserProfilePhotos

from pepperbot.config import Config
from pepperbot.conversation.service import ConversationService
from pepperbot.history.attachments import AttachmentStore
from pepperbot.history.store import HistoryStore
from pepperbot.providers.base import ToolCall
from pepperbot.telegram.user_info import TelegramUserInfo
from pepperbot.tools.executor import ToolExecutor, ToolRuntime
from pepperbot.tools.registry import ToolRegistry


def setup_lookup(tmp_path, vision=True):
    config = Config(
        bot={"token": "t"}, api={"url": "http://example.com", "key": "k", "model": "m"},
        chat_backend={"supports_vision": vision}, telegram={"image_max_area": 100},
    )
    user = User(123, "Alice", False, username="alice")
    small = PhotoSize("small", "small-unique", 20, 20)
    large = PhotoSize("large", "large-unique", 100, 100)
    raw = BytesIO()
    Image.new("RGB", (20, 20)).save(raw, format="PNG")
    bot = SimpleNamespace(
        get_chat_member=AsyncMock(return_value=ChatMemberMember(user=user)),
        get_chat=AsyncMock(return_value=SimpleNamespace(to_dict=lambda: {"id": 123, "bio": "An accessible bio", "birthday": {"day": 3, "month": 4}})),
        get_user_profile_photos=AsyncMock(side_effect=[
            UserProfilePhotos(total_count=2, photos=[[small, large]]),
            UserProfilePhotos(total_count=2, photos=[[small, large]]),
        ]),
        get_file=AsyncMock(return_value=SimpleNamespace(download_as_bytearray=AsyncMock(return_value=raw.getvalue()))),
    )
    lookup = TelegramUserInfo(config)
    return SimpleNamespace(**locals())


@pytest.mark.asyncio
async def test_fetches_full_profile_and_paginates_photos_resizing_for_vision(tmp_path):
    env = setup_lookup(tmp_path)
    result = await env.lookup.fetch(env.bot, -100, 123, vision=True, user=env.user)
    assert result.data["user"]["username"] == "alice"
    assert result.data["membership"]["status"] == "member"
    assert result.data["profile"]["bio"] == "An accessible bio"
    assert result.data["profile"]["birthday"] == {"day": 3, "month": 4}
    assert len(result.images) == 2
    assert len(result.data["profile_photos"]) == 2
    assert env.bot.get_user_profile_photos.call_args_list[1].kwargs["offset"] == 1
    for image in result.images:
        with Image.open(BytesIO(base64.b64decode(image.split(",", 1)[1]))) as decoded:
            assert decoded.width * decoded.height <= 100
    assert result.content()[1]["type"] == "image_url"


@pytest.mark.asyncio
async def test_text_only_lookup_keeps_photo_metadata_and_lookup_errors(tmp_path):
    env = setup_lookup(tmp_path, vision=False)
    env.bot.get_chat.side_effect = RuntimeError("chat not found")
    result = await env.lookup.fetch(env.bot, -100, 123, vision=False)
    assert result.data["user"]["id"] == 123
    assert result.data["errors"]["profile"] == "chat not found"
    assert len(result.data["profile_photos"]) == 2
    assert isinstance(result.content(), str)
    assert not result.images
    env.bot.get_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_daily_tool_returns_images_and_persists_them_for_history(tmp_path):
    env = setup_lookup(tmp_path)
    attachments = AttachmentStore(str(tmp_path / "attachments"))
    history = HistoryStore(str(tmp_path / "history.json"), attachment_store=attachments)
    thread = history.create_thread(-100)
    runtime = ToolRuntime(chat_id=-100, thread=thread, attachment_store=attachments, bot=env.bot)
    executor = ToolExecutor(env.config, None, env.lookup)
    results = await executor.execute([ToolCall(id="lookup", name="fetch_user_info", arguments={"user_id": 123})], runtime)
    assert len(results[0].content) == 3
    assert len(runtime.tool_attachments["lookup"]) == 2
    conversation = object.__new__(ConversationService)
    conversation.config = env.config
    conversation.history = history
    conversation._append_protocol_messages(thread, results, runtime)
    assert len(thread.messages[-1].attachments) == 2
    assert "An accessible bio" in thread.messages[-1].content
    assert "fetch_user_info" in [tool["function"]["name"] for tool in ToolRegistry(env.config).chat_tools()]


@pytest.mark.asyncio
async def test_daily_tool_validates_id_before_telegram_calls(tmp_path):
    env = setup_lookup(tmp_path)
    executor = ToolExecutor(env.config, None, env.lookup)
    runtime = ToolRuntime(chat_id=-100, thread=None, attachment_store=None, bot=env.bot)
    result = await executor.execute([ToolCall(id="lookup", name="fetch_user_info", arguments={"user_id": -100})], runtime)
    assert "positive Telegram user ID" in result[0].content
    env.bot.get_chat.assert_not_awaited()
