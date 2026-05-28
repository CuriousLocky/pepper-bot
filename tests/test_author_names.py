from datetime import datetime
from types import SimpleNamespace

import pytest

from pepperbot.config import Config
from pepperbot.conversation.models import IncomingMessage, ReferencedMessage
from pepperbot.conversation.service import ConversationService
from pepperbot.history.models import TelegramRef
from memory import UserInfoEntry


def make_service(user_info=None):
    service = object.__new__(ConversationService)
    service.config = Config(
        bot={"token": "t", "name": "Pepper"},
        api={"url": "http://example.com/v1", "key": "k", "model": "m"},
    )
    service.memory_manager = SimpleNamespace(user_info=user_info or {})
    return service


def test_known_user_name_is_stored_when_message_is_appended():
    service = make_service({123: UserInfoEntry(user_id=123, name="AliceAlias", description="Known")})
    thread = SimpleNamespace(messages=[])
    thread.next_message_id = lambda: f"m{len(thread.messages)}"
    service.history = SimpleNamespace(add_message=lambda t, m: t.messages.append(m))

    service._append_user_message(
        thread,
        IncomingMessage(
            chat_id=1,
            telegram_message_id=10,
            user_id=123,
            user_name="TelegramAlice",
            text="hello",
            created_at=datetime.now(),
            is_command=True,
            is_reply_to_bot=False,
        ),
        reply_to_id=None,
    )

    assert thread.messages[0].author.name == "AliceAlias"
    assert thread.messages[0].metadata["telegram_user_name"] == "TelegramAlice"


def test_unknown_user_name_is_stored_as_stable_user_id():
    service = make_service()
    thread = SimpleNamespace(messages=[])
    thread.next_message_id = lambda: f"m{len(thread.messages)}"
    service.history = SimpleNamespace(add_message=lambda t, m: t.messages.append(m))

    service._append_user_message(
        thread,
        IncomingMessage(
            chat_id=1,
            telegram_message_id=10,
            user_id=123,
            user_name="TelegramAlice",
            text="hello",
            created_at=datetime.now(),
            is_command=True,
            is_reply_to_bot=False,
        ),
        reply_to_id=None,
    )

    assert thread.messages[0].author.name == "user-123"


def test_existing_history_name_is_not_rewritten_after_user_becomes_known():
    service = make_service()
    thread = SimpleNamespace(messages=[])
    thread.next_message_id = lambda: f"m{len(thread.messages)}"
    service.history = SimpleNamespace(add_message=lambda t, m: t.messages.append(m))

    service._append_user_message(
        thread,
        IncomingMessage(
            chat_id=1,
            telegram_message_id=10,
            user_id=123,
            user_name="TelegramAlice",
            text="before",
            created_at=datetime.now(),
            is_command=True,
            is_reply_to_bot=False,
        ),
        reply_to_id=None,
    )
    service.memory_manager.user_info[123] = UserInfoEntry(user_id=123, name="AliceAlias", description="Known")
    service._append_user_message(
        thread,
        IncomingMessage(
            chat_id=1,
            telegram_message_id=11,
            user_id=123,
            user_name="TelegramAlice",
            text="after",
            created_at=datetime.now(),
            is_command=True,
            is_reply_to_bot=False,
        ),
        reply_to_id=None,
    )

    assert thread.messages[0].author.name == "user-123"
    assert thread.messages[1].author.name == "AliceAlias"


def test_referenced_known_user_uses_known_name():
    service = make_service({123: UserInfoEntry(user_id=123, name="AliceAlias", description="Known")})
    reference = ReferencedMessage(
        telegram_ref=TelegramRef(chat_id=1, message_id=9),
        author_id=123,
        author_name="TelegramAlice",
        is_bot=False,
        text="referenced",
        created_at=datetime.now(),
    )

    message = service._message_from_reference(SimpleNamespace(), reference, "m0")

    assert message.author.name == "AliceAlias"


@pytest.mark.asyncio
async def test_known_usernames_are_updated_from_incoming_and_referenced_messages():
    class Memory:
        def __init__(self):
            self.user_info = {
                123: UserInfoEntry(user_id=123, name="AliceAlias", description="Known"),
                456: UserInfoEntry(user_id=456, name="BobAlias", description="Known"),
            }
            self.updates = []

        async def update_user_telegram_username(self, user_id, username):
            self.updates.append((user_id, username))
            return True

    service = make_service()
    service.memory_manager = Memory()
    incoming = IncomingMessage(
        chat_id=1,
        telegram_message_id=10,
        user_id=123,
        user_name="TelegramAlice",
        telegram_username="@alice",
        text="hello",
        created_at=datetime.now(),
        is_command=True,
        is_reply_to_bot=False,
        referenced_message=ReferencedMessage(
            telegram_ref=TelegramRef(chat_id=1, message_id=9),
            author_id=456,
            author_name="TelegramBob",
            author_telegram_username="bob",
            is_bot=False,
            text="referenced",
        ),
    )

    await service._update_known_usernames(incoming)

    assert service.memory_manager.updates == [(123, "@alice"), (456, "bob")]
