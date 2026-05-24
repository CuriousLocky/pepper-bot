import asyncio

import pytest

from pepperbot.telegram.app import PepperBotApplication


class FakeTask:
    def __init__(self, done=False):
        self.cancelled = False
        self._done = done

    def done(self):
        return self._done

    def cancel(self):
        self.cancelled = True


class FakeMessage:
    def __init__(self, message_id, media_group_id="album"):
        self.message_id = message_id
        self.media_group_id = media_group_id


class FakeChat:
    id = 42


class FakeUpdate:
    def __init__(self, message_id):
        self.message = FakeMessage(message_id)
        self.effective_chat = FakeChat()


def make_app_like():
    app = object.__new__(PepperBotApplication)
    app.media_group_buffers = {}
    app.media_group_contexts = {}
    app.media_group_tasks = {}
    return app


def test_media_group_buffer_resets_existing_timer(monkeypatch):
    app = make_app_like()
    created_tasks = []

    def fake_create_task(coro):
        coro.close()
        task = FakeTask()
        created_tasks.append(task)
        return task

    monkeypatch.setattr(asyncio, "create_task", fake_create_task)

    app._buffer_media_group(FakeUpdate(1), object())
    first_task = app.media_group_tasks[(42, "album")]
    app._buffer_media_group(FakeUpdate(2), object())

    assert first_task.cancelled is True
    assert app.media_group_tasks[(42, "album")] is created_tasks[1]
    assert [update.message.message_id for update in app.media_group_buffers[(42, "album")]] == [1, 2]


@pytest.mark.asyncio
async def test_cancelled_flush_does_not_clear_buffer():
    app = make_app_like()

    class Config:
        class Telegram:
            media_group_wait_seconds = 10

        telegram = Telegram()

    app.config = Config()
    key = (42, "album")
    app.media_group_buffers[key] = [FakeUpdate(1)]
    task = asyncio.create_task(app._flush_media_group_after_delay(key))
    await asyncio.sleep(0)
    task.cancel()
    await task

    assert key in app.media_group_buffers
