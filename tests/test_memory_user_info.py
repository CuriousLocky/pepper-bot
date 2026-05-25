import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memory import MemoryManager, MemoryState, UserInfoEntry


class FailingCollection:
    def query(self, **kwargs):
        raise ValueError("vector search unavailable")


class SearchCollection:
    def __init__(self, ids):
        self.ids = ids

    def query(self, **kwargs):
        return {"ids": [self.ids]}


class SyncCollection:
    def __init__(self, existing=None):
        self.existing = existing or {"ids": [], "documents": [], "metadatas": []}
        self.upserts = []

    def get(self, ids=None):
        if ids is None:
            return self.existing
        selected = []
        for index, existing_id in enumerate(self.existing["ids"]):
            if existing_id in ids:
                selected.append(index)
        return {
            "ids": [self.existing["ids"][i] for i in selected],
            "documents": [self.existing["documents"][i] for i in selected],
            "metadatas": [self.existing["metadatas"][i] for i in selected],
        }

    def delete(self, ids):
        pass

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)


class NamedCollection:
    def __init__(self, name):
        self.name = name


class FakeChromaClient:
    def __init__(self, names):
        self.names = names
        self.deleted = []

    def list_collections(self):
        return [NamedCollection(name) for name in self.names]

    def delete_collection(self, name):
        self.deleted.append(name)


def make_manager() -> MemoryManager:
    manager = object.__new__(MemoryManager)
    manager.user_info = {
        123: UserInfoEntry(user_id=123, name="Alice", description="Known tester"),
        456: UserInfoEntry(user_id=456, name="Bob", description="Another user"),
    }
    manager.state = MemoryState(user_lru=[])
    manager.config = SimpleNamespace(
        memory=SimpleNamespace(
            user=SimpleNamespace(selective=True, top_k=5, lru_size=8, relevant_include=3)
        )
    )
    manager.user_collection = FailingCollection()
    manager._save_state = lambda: None
    return manager


def make_sync_manager(existing_user_collection) -> MemoryManager:
    manager = make_manager()
    manager.embedding_signature = manager._embedding_signature("embed-v2")
    manager.user_collection = existing_user_collection
    manager.short_collection = SyncCollection()
    manager.long_collection = SyncCollection()
    manager.short_term_mem = []
    manager.long_term_mem = []
    return manager


@pytest.mark.asyncio
async def test_current_known_user_is_returned_when_vector_search_fails():
    manager = make_manager()
    result = await manager.get_user_info_str("hello", current_user_id=123, query_embeddings=None)
    assert "Alice (123): Known tester" in result


@pytest.mark.asyncio
async def test_empty_query_embeddings_fall_back_without_chroma_empty_embedding_error():
    manager = make_manager()
    result = await manager.get_user_info_str("hello", current_user_id=123, query_embeddings=[])
    assert "Alice (123): Known tester" in result


@pytest.mark.asyncio
async def test_vector_hits_update_lru_then_current_known_user_is_latest():
    manager = make_manager()
    manager.state.user_lru = []
    manager.user_collection = SearchCollection(["456"])

    result = await manager.get_user_info_str("hello", current_user_id=123, query_embeddings=None)

    assert manager.state.user_lru[:2] == [123, 456]
    assert result.splitlines()[0].startswith("- Alice (123):")


@pytest.mark.asyncio
async def test_unknown_sender_is_not_added_to_user_lru():
    manager = make_manager()
    manager.user_collection = SearchCollection(["456"])

    await manager.get_user_info_str("hello", current_user_id=999, query_embeddings=None)

    assert 999 not in manager.state.user_lru
    assert manager.state.user_lru[0] == 456


def test_collection_name_changes_with_embedding_signature():
    manager = make_manager()
    manager.embedding_signature = manager._embedding_signature("embed-v1")
    first = manager._collection_name("users")
    manager.embedding_signature = manager._embedding_signature("embed-v2")
    second = manager._collection_name("users")
    assert first != second


def test_sync_reembeds_user_when_embedding_signature_missing():
    collection = SyncCollection(
        {
            "ids": ["123"],
            "documents": ["Alice: Known tester"],
            "metadatas": [{"user_id": 123}],
        }
    )
    manager = make_sync_manager(collection)
    manager.user_info = {123: manager.user_info[123]}

    manager._sync_memory()

    assert collection.upserts
    assert collection.upserts[0]["ids"] == ["123"]
    assert collection.upserts[0]["metadatas"][0]["embedding_model"] == "embed-v2"


def test_sync_skips_user_when_embedding_signature_matches():
    manager = make_manager()
    signature = manager._embedding_signature("embed-v2")
    collection = SyncCollection(
        {
            "ids": ["123"],
            "documents": ["Alice: Known tester"],
            "metadatas": [{"user_id": 123, **signature}],
        }
    )
    manager = make_sync_manager(collection)
    manager.user_info = {123: manager.user_info[123]}

    manager._sync_memory()

    assert collection.upserts == []


def test_cleanup_stale_vector_collections_deletes_old_managed_names():
    manager = make_manager()
    manager.collection_bases = ("short_term", "long_term", "users")
    manager.embedding_signature = manager._embedding_signature("embed-v2")
    manager.active_collection_names = {base: manager._collection_name(base) for base in manager.collection_bases}
    manager.chroma_client = FakeChromaClient(
        [
            "short_term",
            "long_term_oldhash",
            manager.active_collection_names["users"],
            "unrelated_collection",
        ]
    )

    removed = manager._cleanup_stale_vector_collections()

    assert removed == 2
    assert manager.chroma_client.deleted == ["short_term", "long_term_oldhash"]
