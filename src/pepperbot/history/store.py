import json
import logging
import shutil
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from pepperbot.history.attachments import AttachmentStore
from pepperbot.history.models import Actor, ConversationMessage, HistoryData, TelegramRef, Thread

logger = logging.getLogger(__name__)


class HistoryStore:
    def __init__(
        self,
        storage_path: str = "data/chat-histories.json",
        attachment_store: Optional[AttachmentStore] = None,
        expiration_hours: int = 24,
    ):
        self.storage_path = Path(storage_path)
        self.attachment_store = attachment_store or AttachmentStore()
        self.expiration_hours = expiration_hours
        self.data = HistoryData()
        self.message_map: Dict[Tuple[int, int], Tuple[str, str]] = {}
        self.load()

    def load(self) -> None:
        if not self.storage_path.exists():
            self.data = HistoryData()
            self._rebuild_message_map()
            return
        raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and raw.get("version") == 2:
            self.data = HistoryData.model_validate(raw)
        elif isinstance(raw, dict):
            self._backup_before_migration()
            self.data = self._migrate_legacy(raw)
            self.save()
        else:
            raise ValueError("Unsupported history file format")
        self._rebuild_message_map()

    def _backup_before_migration(self) -> None:
        backup_path = self.storage_path.with_suffix(".json.v1.bak")
        if not backup_path.exists():
            shutil.copyfile(self.storage_path, backup_path)

    def _migrate_legacy(self, raw: Dict[str, object]) -> HistoryData:
        migrated = HistoryData()
        for thread_id, value in raw.items():
            if not isinstance(value, dict):
                continue
            chat_id = int(value.get("chat_id", 0))
            last_updated = self._parse_datetime(value.get("last_updated")) or datetime.now()
            messages = []
            expires_at = last_updated + timedelta(hours=self.expiration_hours)
            for raw_message in value.get("messages", []) or []:
                if not isinstance(raw_message, dict):
                    continue
                legacy_id = raw_message.get("message_id", len(messages))
                message_id = f"m{legacy_id}"
                user_name = raw_message.get("user_name") or "Unknown"
                role = raw_message.get("role") or "user"
                telegram_refs = []
                if raw_message.get("telegram_id") is not None:
                    telegram_refs.append(TelegramRef(chat_id=chat_id, message_id=int(raw_message["telegram_id"])))
                attachments = []
                image_url = raw_message.get("image_url")
                if isinstance(image_url, str) and image_url:
                    try:
                        attachments.append(
                            self.attachment_store.save_data_uri(
                                image_url,
                                source="migration",
                                expires_at=expires_at,
                            )
                        )
                    except Exception as exc:
                        logger.warning("Failed to migrate image attachment: %s", exc)
                reply_to = raw_message.get("reply_to_id")
                messages.append(
                    ConversationMessage(
                        id=message_id,
                        role=role,
                        author=Actor(
                            id=raw_message.get("user_id"),
                            name=str(user_name),
                            is_bot=role == "assistant",
                        ),
                        content=raw_message.get("content") or "",
                        reply_to=f"m{reply_to}" if reply_to is not None else None,
                        telegram_refs=telegram_refs,
                        attachments=attachments,
                        created_at=self._parse_datetime(raw_message.get("timestamp")) or last_updated,
                        metadata={
                            key: raw_message[key]
                            for key in ("tool_calls", "tool_call_id")
                            if raw_message.get(key) is not None
                        },
                    )
                )
            created_at = messages[0].created_at if messages else last_updated
            migrated.threads[thread_id] = Thread(
                id=thread_id,
                chat_id=chat_id,
                state="active",
                created_at=created_at,
                updated_at=last_updated,
                expires_at=expires_at,
                messages=messages,
            )
        return migrated

    def _parse_datetime(self, value: object) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None

    def save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.storage_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(self.data.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp_path.replace(self.storage_path)

    def create_thread(self, chat_id: int, state: str = "active") -> Thread:
        now = datetime.now()
        thread = Thread(
            id=str(uuid.uuid4()),
            chat_id=chat_id,
            state=state,  # type: ignore[arg-type]
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(hours=self.expiration_hours),
        )
        self.data.threads[thread.id] = thread
        return thread

    def get_thread(self, thread_id: str) -> Optional[Thread]:
        return self.data.threads.get(thread_id)

    def find_by_telegram_ref(self, chat_id: int, message_id: int) -> Tuple[Optional[Thread], Optional[str]]:
        result = self.message_map.get((chat_id, message_id))
        if not result:
            return None, None
        thread_id, logical_message_id = result
        return self.data.threads.get(thread_id), logical_message_id

    def add_message(self, thread: Thread, message: ConversationMessage) -> None:
        thread.messages.append(message)
        thread.updated_at = datetime.now()
        thread.expires_at = thread.updated_at + timedelta(hours=self.expiration_hours)
        for existing_message in thread.messages:
            for attachment in existing_message.attachments:
                attachment.expires_at = thread.expires_at
        for ref in message.telegram_refs:
            self.message_map[(ref.chat_id, ref.message_id)] = (thread.id, message.id)

    def touch_thread(self, thread: Thread) -> None:
        thread.updated_at = datetime.now()
        thread.expires_at = thread.updated_at + timedelta(hours=self.expiration_hours)
        for message in thread.messages:
            for attachment in message.attachments:
                attachment.expires_at = thread.expires_at

    def clean_expired(self, now: Optional[datetime] = None) -> int:
        now = now or datetime.now()
        expired = [
            thread_id
            for thread_id, thread in self.data.threads.items()
            if thread.expires_at and thread.expires_at < now
        ]
        for thread_id in expired:
            del self.data.threads[thread_id]
        if expired:
            self._rebuild_message_map()
        live_attachment_ids = []
        for thread in self.data.threads.values():
            for message in thread.messages:
                message.attachments = [
                    attachment
                    for attachment in message.attachments
                    if not attachment.expires_at or attachment.expires_at >= now
                ]
                live_attachment_ids.extend(attachment.id for attachment in message.attachments)
        self.attachment_store.cleanup(now, live_attachment_ids)
        return len(expired)

    def live_attachment_ids(self) -> Iterable[str]:
        for thread in self.data.threads.values():
            for message in thread.messages:
                for attachment in message.attachments:
                    yield attachment.id

    def _rebuild_message_map(self) -> None:
        self.message_map = {}
        for thread in self.data.threads.values():
            for message in thread.messages:
                for ref in message.telegram_refs:
                    self.message_map[(ref.chat_id, ref.message_id)] = (thread.id, message.id)
