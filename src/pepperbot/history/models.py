from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Actor(BaseModel):
    id: Optional[int] = None
    name: str = "Unknown"
    is_bot: bool = False


class TelegramRef(BaseModel):
    chat_id: int
    message_id: int


class AttachmentRef(BaseModel):
    id: str
    kind: Literal["image"] = "image"
    path: Optional[str] = None
    mime_type: str = "application/octet-stream"
    source: Literal["telegram", "generated", "migration", "external"] = "telegram"
    url: Optional[str] = None
    created_at: datetime
    expires_at: Optional[datetime] = None


class ConversationMessage(BaseModel):
    id: str
    role: Literal["user", "assistant", "system", "tool"]
    author: Actor
    content: str = ""
    reply_to: Optional[str] = None
    telegram_refs: List[TelegramRef] = Field(default_factory=list)
    attachments: List[AttachmentRef] = Field(default_factory=list)
    created_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Thread(BaseModel):
    id: str
    chat_id: int
    state: Literal["active", "awaiting_expiration_reply"] = "active"
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    summary: str = ""
    summary_until_message_id: Optional[str] = None
    messages: List[ConversationMessage] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def next_message_id(self) -> str:
        max_id = -1
        for message in self.messages:
            if message.id.startswith("m") and message.id[1:].isdigit():
                max_id = max(max_id, int(message.id[1:]))
        return f"m{max_id + 1}"

    def get_message(self, message_id: str) -> Optional[ConversationMessage]:
        return next((message for message in self.messages if message.id == message_id), None)


class HistoryData(BaseModel):
    version: int = 2
    threads: Dict[str, Thread] = Field(default_factory=dict)
