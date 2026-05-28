from datetime import datetime
from io import BytesIO

import pytest
from PIL import Image

from pepperbot.config import Config
from pepperbot.history.attachments import AttachmentStore
from pepperbot.telegram.parser import UpdateParser


class FakeUser:
    def __init__(self, user_id=1, first_name="Alice", username="alice", is_bot=False):
        self.id = user_id
        self.first_name = first_name
        self.username = username
        self.is_bot = is_bot


class FakePhoto:
    def __init__(self, file_id, width=640, height=480):
        self.file_id = file_id
        self.width = width
        self.height = height


class FakeMessage:
    def __init__(self, message_id, caption="", file_id="file", photos=None):
        self.message_id = message_id
        self.chat_id = 42
        self.text = None
        self.caption = caption
        self.photo = photos or [FakePhoto(file_id)]
        self.from_user = FakeUser()
        self.date = datetime(2026, 1, 1)
        self.reply_to_message = None


class FakeFile:
    def __init__(self, data):
        self.data = data

    async def download_as_bytearray(self):
        return bytearray(self.data)


class FakeBot:
    id = 999

    async def get_file(self, file_id):
        return FakeFile(file_id.encode("utf-8"))


class FakeContext:
    bot = FakeBot()


def make_config() -> Config:
    return Config(
        bot={"token": "t", "name": "Pepper", "command": "pepper"},
        api={"url": "http://example.com/v1", "key": "k", "model": "m", "supports_vision": True},
    )


def jpeg_bytes(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height), color="white")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_media_group_parses_as_one_message_with_multiple_attachments(tmp_path):
    parser = UpdateParser(make_config(), AttachmentStore(str(tmp_path / "attachments")))
    incoming = await parser.parse_messages(
        [
            FakeMessage(10, caption="/pepper compare these", file_id="a"),
            FakeMessage(11, file_id="b"),
            FakeMessage(12, file_id="c"),
        ],
        FakeContext(),
        bot_username=None,
    )

    assert incoming is not None
    assert incoming.telegram_message_id == 10
    assert incoming.telegram_username == "@alice"
    assert incoming.text == "compare these"
    assert len(incoming.attachments) == 3
    assert [ref.message_id for ref in incoming.telegram_refs] == [10, 11, 12]


def test_selects_largest_photo_under_max_area(tmp_path):
    config = make_config()
    config.telegram.image_max_area = 500_000
    parser = UpdateParser(config, AttachmentStore(str(tmp_path / "attachments")))
    selected, should_resize = parser._select_photo_size(
        [
            FakePhoto("small", width=320, height=240),
            FakePhoto("medium", width=800, height=600),
            FakePhoto("large", width=1600, height=1200),
        ]
    )
    assert selected.file_id == "medium"
    assert should_resize is False


def test_selects_smallest_and_resizes_when_all_exceed_max_area(tmp_path):
    config = make_config()
    config.telegram.image_max_area = 10_000
    parser = UpdateParser(config, AttachmentStore(str(tmp_path / "attachments")))
    selected, should_resize = parser._select_photo_size(
        [
            FakePhoto("small", width=320, height=240),
            FakePhoto("large", width=1600, height=1200),
        ]
    )
    assert selected.file_id == "small"
    assert should_resize is True

    resized = parser._resize_to_max_area(jpeg_bytes(320, 240), 10_000)
    with Image.open(BytesIO(resized)) as image:
        assert image.size[0] * image.size[1] <= 10_000
