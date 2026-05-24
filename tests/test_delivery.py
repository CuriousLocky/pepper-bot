import pytest

from pepperbot.config import Config
from pepperbot.telegram.delivery import TelegramDelivery


class Sent:
    def __init__(self, message_id):
        self.message_id = message_id


class FakeBot:
    def __init__(self):
        self.calls = []

    async def send_message(self, **kwargs):
        self.calls.append(kwargs)
        return Sent(len(self.calls))


def make_config() -> Config:
    return Config(
        bot={"token": "t", "name": "Pepper"},
        api={"url": "http://example.com/v1", "key": "k", "model": "m"},
        telegram={"chunk_chars": 10, "max_message_chars": 10, "send_retries": 0},
    )


@pytest.mark.asyncio
async def test_text_is_chunked_and_only_first_chunk_replies():
    delivery = TelegramDelivery(make_config())
    bot = FakeBot()
    refs = await delivery.send_text(bot, 123, "hello world again", reply_to_message_id=9)
    assert [ref.message_id for ref in refs] == [1, 2, 3]
    assert bot.calls[0]["reply_to_message_id"] == 9
    assert bot.calls[1]["reply_to_message_id"] is None
    assert bot.calls[2]["reply_to_message_id"] is None
    assert all(len(call["text"]) <= 10 for call in bot.calls)


@pytest.mark.asyncio
async def test_empty_text_uses_fallback():
    delivery = TelegramDelivery(make_config())
    bot = FakeBot()
    await delivery.send_text(bot, 123, "")
    assert "".join(call["text"] for call in bot.calls) == "抱歉，刚才脑袋短路了。"
