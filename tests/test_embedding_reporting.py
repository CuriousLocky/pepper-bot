from datetime import datetime

import pytest

from pepperbot.config import Config
from pepperbot.conversation.models import IncomingMessage
from pepperbot.conversation.service import ConversationService


class EmptyEmbeddingMemory:
    async def get_embeddings(self, inputs):
        return []

    async def get_short_term_str(self, query, query_embeddings=None):
        assert query_embeddings is None
        return "short"

    async def get_long_term_str(self, query, query_embeddings=None):
        assert query_embeddings is None
        return "long"

    def get_all_knowledges_str(self):
        return "knowledge"

    async def get_user_info_str(self, query, current_user_id=None, query_embeddings=None):
        assert query_embeddings is None
        return "user"


class RaisingEmbeddingMemory(EmptyEmbeddingMemory):
    async def get_embeddings(self, inputs):
        raise RuntimeError("embedding down")


class FakeReporter:
    def __init__(self):
        self.reports = []

    async def report(self, bot, title, details, context_preview=""):
        self.reports.append((title, details, context_preview))


def make_service(memory):
    service = object.__new__(ConversationService)
    service.config = Config(
        bot={"token": "t", "name": "Pepper"},
        api={"url": "http://example.com/v1", "key": "k", "model": "m"},
    )
    service.memory_manager = memory
    service.reporter = FakeReporter()
    return service


def make_incoming():
    return IncomingMessage(
        chat_id=42,
        telegram_message_id=7,
        user_id=123,
        user_name="Alice",
        text="hello",
        created_at=datetime.now(),
        is_command=True,
        is_reply_to_bot=False,
    )


@pytest.mark.asyncio
async def test_empty_embedding_result_is_reported_to_admin():
    service = make_service(EmptyEmbeddingMemory())
    sections = await service._memory_context(make_incoming(), bot=object())
    assert sections["known_user_info"] == "user"
    assert service.reporter.reports
    assert service.reporter.reports[0][0] == "Embedding retrieval returned empty result"


@pytest.mark.asyncio
async def test_embedding_exception_is_reported_to_admin():
    service = make_service(RaisingEmbeddingMemory())
    sections = await service._memory_context(make_incoming(), bot=object())
    assert sections["known_user_info"] == "user"
    assert service.reporter.reports
    assert service.reporter.reports[0][0] == "Embedding retrieval failed"
