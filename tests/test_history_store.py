import base64
import json
from pathlib import Path

from pepperbot.history.attachments import AttachmentStore
from pepperbot.history.store import HistoryStore


def test_legacy_history_migrates_and_externalizes_base64_image(tmp_path: Path):
    image_data = base64.b64encode(b"fake-image").decode("ascii")
    legacy = {
        "thread-1": {
            "chat_id": 42,
            "last_updated": "2026-01-01T00:00:00",
            "messages": [
                {
                    "role": "user",
                    "content": "hello",
                    "message_id": 0,
                    "telegram_id": 10,
                    "user_id": 7,
                    "user_name": "Alice",
                    "timestamp": "2026-01-01T00:00:00",
                    "image_url": f"data:image/png;base64,{image_data}",
                }
            ],
        }
    }
    path = tmp_path / "chat-histories.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    store = HistoryStore(str(path), AttachmentStore(str(tmp_path / "attachments")), expiration_hours=1)

    assert store.data.version == 2
    thread = store.data.threads["thread-1"]
    assert thread.messages[0].attachments
    assert Path(thread.messages[0].attachments[0].path).exists()
    found_thread, message_id = store.find_by_telegram_ref(42, 10)
    assert found_thread.id == "thread-1"
    assert message_id == "m0"
    assert (tmp_path / "chat-histories.json.v1.bak").exists()
