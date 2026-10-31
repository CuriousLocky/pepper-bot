import base64
import json
from dataclasses import dataclass, field
from io import BytesIO
from math import sqrt
from typing import Any, Dict, List

from PIL import Image

from pepperbot.config import Config
from pepperbot.providers.base import Content


@dataclass
class UserProfile:
    data: Dict[str, Any]
    images: List[str] = field(default_factory=list)

    def content(self) -> Content:
        text = json.dumps(self.data, ensure_ascii=False, default=str)
        if not self.images:
            return text
        return [
            {"type": "text", "text": text},
            *[{"type": "image_url", "image_url": {"url": image}} for image in self.images],
        ]


class TelegramUserInfo:
    """Best-effort access to Telegram's available profile fields, not invented attributes."""

    def __init__(self, config: Config, memory_manager: Any = None):
        self.config = config
        self.memory_manager = memory_manager

    async def fetch(self, bot, chat_id: int, user_id: int, *, vision: bool, user=None) -> UserProfile:
        data: Dict[str, Any] = {"user_id": user_id, "chat_id": chat_id, "errors": {}}
        images: List[str] = []
        if user is not None:
            data["user"] = user.to_dict()
        try:
            member = await bot.get_chat_member(chat_id=chat_id, user_id=user_id)
            data["membership"] = member.to_dict()
            # Keep optional fields present in the original join update as well.
            data["user"] = {**data.get("user", {}), **member.user.to_dict()}
        except Exception as exc:
            data["errors"]["membership"] = str(exc)
        try:
            # Includes bio, birthday, personal_chat and business fields when exposed
            # by Telegram. This can be inaccessible before a user interacts with us.
            chat = await bot.get_chat(chat_id=user_id)
            data["profile"] = chat.to_dict()
        except Exception as exc:
            data["errors"]["profile"] = str(exc)

        known = getattr(self.memory_manager, "user_info", {}).get(user_id)
        if known is not None:
            data["known_user_info"] = known.model_dump(mode="json")

        data["profile_photos"] = []
        offset = 0
        try:
            while True:
                page = await bot.get_user_profile_photos(user_id=user_id, offset=offset, limit=100)
                data["profile_photo_count"] = page.total_count
                if not page.photos:
                    break
                for sizes in page.photos:
                    data["profile_photos"].append([size.to_dict() for size in sizes])
                    if vision and sizes:
                        try:
                            images.append(await self.download_photo(bot, sizes))
                        except Exception as exc:
                            data["errors"][f"photo_{len(data['profile_photos'])}"] = str(exc)
                offset += len(page.photos)
                if offset >= page.total_count:
                    break
        except Exception as exc:
            data["errors"]["profile_photos"] = str(exc)
        return UserProfile(data, images)

    async def download_photo(self, bot, sizes) -> str:
        max_area = self.config.telegram.image_max_area
        ordered = sorted(sizes, key=lambda size: size.width * size.height)
        qualified = [size for size in ordered if size.width * size.height <= max_area]
        photo = (qualified[-1] if qualified else ordered[0]) if max_area > 0 else ordered[-1]
        file = await bot.get_file(photo.file_id)
        raw = bytes(await file.download_as_bytearray())
        with Image.open(BytesIO(raw)) as image:
            area = image.width * image.height
            if max_area > 0 and area > max_area:
                scale = sqrt(max_area / area)
                image = image.resize(
                    (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                    Image.Resampling.LANCZOS,
                )
            output = BytesIO()
            image.convert("RGB").save(output, format="JPEG", quality=90)
        return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")
