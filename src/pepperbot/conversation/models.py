from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from pepperbot.history.models import AttachmentRef, TelegramRef


@dataclass
class ReferencedMessage:
    telegram_ref: TelegramRef
    author_id: Optional[int]
    author_name: str
    is_bot: bool
    text: str
    author_telegram_username: Optional[str] = None
    attachments: List[AttachmentRef] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class IncomingMessage:
    chat_id: int
    telegram_message_id: int
    user_id: int
    user_name: str
    text: str
    created_at: datetime
    is_command: bool
    is_reply_to_bot: bool
    telegram_username: Optional[str] = None
    referenced_message: Optional[ReferencedMessage] = None
    attachments: List[AttachmentRef] = field(default_factory=list)
    telegram_refs: List[TelegramRef] = field(default_factory=list)


@dataclass
class ConversationResult:
    handled: bool
    text: str = ""
    images: List[str] = field(default_factory=list)
    reason: str = ""
