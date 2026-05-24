import logging
import re
from datetime import datetime
from io import BytesIO
from math import sqrt
from typing import List, Optional

from PIL import Image
from telegram import Message, Update
from telegram.ext import ContextTypes

from pepperbot.config import Config
from pepperbot.conversation.models import IncomingMessage, ReferencedMessage
from pepperbot.history.attachments import AttachmentStore
from pepperbot.history.models import TelegramRef

logger = logging.getLogger(__name__)


class UpdateParser:
    def __init__(self, config: Config, attachment_store: AttachmentStore):
        self.config = config
        self.attachment_store = attachment_store

    async def parse(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        bot_username: Optional[str],
    ) -> Optional[IncomingMessage]:
        if not update.message:
            return None
        return await self.parse_messages([update.message], context, bot_username)

    async def parse_messages(
        self,
        messages: List[Message],
        context: ContextTypes.DEFAULT_TYPE,
        bot_username: Optional[str],
    ) -> Optional[IncomingMessage]:
        messages = sorted([message for message in messages if message], key=lambda item: item.message_id)
        if not messages:
            return None

        chat_id = messages[0].chat_id
        if self.config.bot.chat_whitelist and chat_id not in self.config.bot.chat_whitelist:
            return None

        user = next((message.from_user for message in messages if message.from_user), None)
        if not user:
            return None

        raw_texts = [message.text or message.caption or "" for message in messages]
        has_photo = any(message.photo for message in messages)
        if not any(raw_texts) and not has_photo:
            return None

        is_reply_to_bot = any(self._is_reply_to_bot(message, context) for message in messages)
        is_command = any(self._is_command(text, bot_username) for text in raw_texts)
        if any(text.startswith("/") for text in raw_texts if text) and not is_command:
            return None
        if not is_command and not is_reply_to_bot:
            return None

        canonical = self._canonical_message(messages, raw_texts, bot_username)
        text_parts = []
        for text in raw_texts:
            text = text.strip()
            if not text:
                continue
            if self._is_command(text, bot_username):
                text = re.sub(rf"^/{re.escape(self.config.bot.command)}(?:@\w+)?\s*", "", text, count=1).strip()
            if text:
                text_parts.append(text)

        attachments = []
        if self.config.chat_supports_vision():
            for message in messages:
                if message.photo:
                    attachments.extend(await self._download_photos(message, context, source="telegram"))

        reference = None
        referenced_message = canonical.reply_to_message or next(
            (message.reply_to_message for message in messages if message.reply_to_message),
            None,
        )
        if referenced_message:
            reference = await self._referenced_message(referenced_message, context)

        return IncomingMessage(
            chat_id=chat_id,
            telegram_message_id=canonical.message_id,
            user_id=user.id,
            user_name=user.first_name or user.username or "Unknown",
            text="\n".join(text_parts).strip(),
            created_at=canonical.date or datetime.now(),
            is_command=is_command,
            is_reply_to_bot=is_reply_to_bot,
            referenced_message=reference,
            attachments=attachments,
            telegram_refs=[TelegramRef(chat_id=chat_id, message_id=message.message_id) for message in messages],
        )

    def _canonical_message(
        self,
        messages: List[Message],
        raw_texts: List[str],
        bot_username: Optional[str],
    ) -> Message:
        for message, text in zip(messages, raw_texts):
            if self._is_command(text, bot_username):
                return message
        for message, text in zip(messages, raw_texts):
            if text:
                return message
        return messages[0]

    def _is_reply_to_bot(self, message: Message, context: ContextTypes.DEFAULT_TYPE) -> bool:
        return bool(
            message.reply_to_message
            and message.reply_to_message.from_user
            and message.reply_to_message.from_user.id == context.bot.id
        )

    def _is_command(self, text: str, bot_username: Optional[str]) -> bool:
        pattern = rf"^/{re.escape(self.config.bot.command)}(?:@(\w+))?(?:\s|$)"
        match = re.search(pattern, text or "")
        if not match:
            return False
        target = match.group(1)
        return not target or bool(bot_username and target.lower() == bot_username.lower())

    async def _referenced_message(
        self,
        message: Message,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> ReferencedMessage:
        attachments = []
        if self.config.chat_supports_vision() and message.photo:
            attachments.extend(await self._download_photos(message, context, source="telegram"))
        author = message.from_user
        return ReferencedMessage(
            telegram_ref=TelegramRef(chat_id=message.chat_id, message_id=message.message_id),
            author_id=author.id if author else None,
            author_name=(author.first_name or author.username or "Unknown") if author else "Unknown",
            is_bot=bool(author and author.is_bot),
            text=(message.text or message.caption or "").strip(),
            attachments=attachments,
            created_at=message.date or datetime.now(),
        )

    async def _download_photos(
        self,
        message: Message,
        context: ContextTypes.DEFAULT_TYPE,
        source: str,
    ):
        # Telegram exposes one photo as multiple PhotoSize variants. A media group
        # is aggregated by the caller into one logical IncomingMessage with multiple
        # attachments.
        try:
            photo, should_resize = self._select_photo_size(message.photo)
            telegram_file = await context.bot.get_file(photo.file_id)
            data = await telegram_file.download_as_bytearray()
            image_bytes = bytes(data)
            if should_resize:
                image_bytes = self._resize_to_max_area(image_bytes, self.config.telegram.image_max_area)
            return [self.attachment_store.save_bytes(image_bytes, "image/jpeg", source=source)]
        except Exception:
            logger.exception("Failed to download Telegram photo")
            return []

    def _select_photo_size(self, photos):
        max_area = self.config.telegram.image_max_area
        sorted_photos = sorted(photos, key=lambda item: self._photo_area(item))
        if max_area <= 0:
            return sorted_photos[-1], False
        qualified = [photo for photo in sorted_photos if self._photo_area(photo) <= max_area]
        if qualified:
            return qualified[-1], False
        return sorted_photos[0], True

    def _photo_area(self, photo) -> int:
        return int(getattr(photo, "width", 0) or 0) * int(getattr(photo, "height", 0) or 0)

    def _resize_to_max_area(self, image_bytes: bytes, max_area: int) -> bytes:
        if max_area <= 0:
            return image_bytes
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                width, height = image.size
                area = width * height
                if area <= max_area:
                    return image_bytes
                scale = sqrt(max_area / area)
                new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
                resized = image.resize(new_size, Image.Resampling.LANCZOS)
                if resized.mode not in ("RGB", "L"):
                    resized = resized.convert("RGB")
                output = BytesIO()
                resized.save(output, format="JPEG", quality=90)
                return output.getvalue()
        except Exception:
            logger.warning("Failed to resize Telegram photo; using original bytes", exc_info=True)
            return image_bytes
