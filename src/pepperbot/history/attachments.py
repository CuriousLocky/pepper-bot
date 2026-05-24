import base64
import mimetypes
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Set

from pepperbot.history.models import AttachmentRef


DATA_URI_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<data>.*)$", re.DOTALL)


def _extension_for_mime(mime_type: str) -> str:
    if mime_type == "image/jpeg":
        return ".jpg"
    if mime_type == "image/png":
        return ".png"
    return mimetypes.guess_extension(mime_type) or ".bin"


class AttachmentStore:
    def __init__(self, root_path: str = "data/attachments"):
        self.root_path = Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)

    def save_bytes(
        self,
        data: bytes,
        mime_type: str,
        source: str,
        expires_at: Optional[datetime] = None,
    ) -> AttachmentRef:
        attachment_id = str(uuid.uuid4())
        path = self.root_path / f"{attachment_id}{_extension_for_mime(mime_type)}"
        path.write_bytes(data)
        return AttachmentRef(
            id=attachment_id,
            kind="image",
            path=str(path),
            mime_type=mime_type,
            source=source,  # type: ignore[arg-type]
            created_at=datetime.now(),
            expires_at=expires_at,
        )

    def save_data_uri(
        self,
        data_uri: str,
        source: str,
        expires_at: Optional[datetime] = None,
    ) -> AttachmentRef:
        match = DATA_URI_RE.match(data_uri)
        if not match:
            return AttachmentRef(
                id=str(uuid.uuid4()),
                kind="image",
                mime_type="application/octet-stream",
                source="external",
                url=data_uri,
                created_at=datetime.now(),
                expires_at=expires_at,
            )
        data = base64.b64decode(match.group("data"))
        return self.save_bytes(data, match.group("mime"), source, expires_at)

    def read_data_uri(self, attachment: AttachmentRef) -> Optional[str]:
        if attachment.url:
            return attachment.url
        if not attachment.path:
            return None
        path = Path(attachment.path)
        if not path.exists():
            return None
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{attachment.mime_type};base64,{encoded}"

    def cleanup(self, now: datetime, live_attachment_ids: Iterable[str]) -> int:
        live_ids: Set[str] = set(live_attachment_ids)
        removed = 0
        for path in self.root_path.iterdir():
            if not path.is_file():
                continue
            attachment_id = path.stem
            if attachment_id in live_ids:
                continue
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
        return removed
