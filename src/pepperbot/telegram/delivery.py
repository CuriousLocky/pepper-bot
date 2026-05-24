import asyncio
import base64
import logging
from pathlib import Path
from typing import Iterable, List, Optional

from telegram.error import TelegramError

from pepperbot.config import Config
from pepperbot.history.attachments import DATA_URI_RE
from pepperbot.history.models import TelegramRef

logger = logging.getLogger(__name__)


class TelegramDelivery:
    def __init__(self, config: Config):
        self.config = config

    def chunks(self, text: str) -> List[str]:
        text = text or self.config.response.fallback_text
        limit = min(self.config.telegram.chunk_chars, self.config.telegram.max_message_chars)
        if len(text) <= limit:
            return [text]
        chunks: List[str] = []
        remaining = text
        while remaining:
            if len(remaining) <= limit:
                chunks.append(remaining)
                break
            split_at = remaining.rfind("\n", 0, limit)
            if split_at < max(1, limit // 2):
                split_at = remaining.rfind(" ", 0, limit)
            if split_at < max(1, limit // 2):
                split_at = limit
            chunks.append(remaining[:split_at].rstrip())
            remaining = remaining[split_at:].lstrip()
        return chunks

    async def send_text(
        self,
        bot,
        chat_id: int,
        text: str,
        reply_to_message_id: Optional[int] = None,
    ) -> List[TelegramRef]:
        refs: List[TelegramRef] = []
        for index, chunk in enumerate(self.chunks(text)):
            reply_to = reply_to_message_id if index == 0 else None
            sent = await self._retry(
                bot.send_message,
                chat_id=chat_id,
                text=chunk,
                reply_to_message_id=reply_to,
            )
            refs.append(TelegramRef(chat_id=chat_id, message_id=sent.message_id))
        return refs

    async def send_images(self, bot, chat_id: int, images: Iterable[str]) -> List[TelegramRef]:
        refs: List[TelegramRef] = []
        for image in images:
            photo = self._photo_payload(image)
            sent = await self._retry(bot.send_photo, chat_id=chat_id, photo=photo)
            refs.append(TelegramRef(chat_id=chat_id, message_id=sent.message_id))
        return refs

    async def _retry(self, func, **kwargs):
        attempts = max(1, self.config.telegram.send_retries + 1)
        last_error: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                return await func(**kwargs)
            except TelegramError as exc:
                last_error = exc
                logger.warning("Telegram send failed on attempt %s/%s: %s", attempt + 1, attempts, exc)
                if attempt + 1 < attempts:
                    await asyncio.sleep(min(2**attempt, 5))
        assert last_error is not None
        raise last_error

    def _photo_payload(self, image: str):
        match = DATA_URI_RE.match(image)
        if match:
            return base64.b64decode(match.group("data"))
        path = Path(image)
        if path.exists():
            return path.read_bytes()
        return image


class AdminReporter:
    def __init__(self, config: Config, delivery: TelegramDelivery):
        self.config = config
        self.delivery = delivery

    async def report(self, bot, title: str, details: str, context_preview: str = "") -> None:
        if not self.config.admin.report_major_failures:
            return
        chat_ids = self.config.admin.report_chat_ids or self.config.admin.ids
        if not chat_ids:
            return
        text = f"[PepperBot major failure]\n{title}\n\n{details}"
        if self.config.admin.include_context_preview and context_preview:
            preview = context_preview[: self.config.admin.context_preview_chars]
            text += f"\n\nContext preview:\n{preview}"
        for chat_id in chat_ids:
            try:
                await self.delivery.send_text(bot, chat_id, text)
            except Exception:
                logger.exception("Failed to send admin failure report to %s", chat_id)
