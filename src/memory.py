import json
import re
import yaml
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from pydantic import BaseModel

class MemoryEvent(BaseModel):
    content: str
    timestamp: datetime

class UserInfoEntry(BaseModel):
    user_id: int
    name: str
    description: str

class MemoryManager:
    def __init__(self, 
                 short_term_path: str = "data/short-term.json", 
                 long_term_path: str = "data/long-term.txt",
                 user_info_path: str = "data/known-users.yaml"):
        self.short_term_path = Path(short_term_path)
        self.long_term_path = Path(long_term_path)
        self.user_info_path = Path(user_info_path)
        
        self.short_term_mem: List[MemoryEvent] = self._load_short_term()
        self.long_term_mem: str = self._load_long_term()
        self.user_info: Dict[int, UserInfoEntry] = self._load_user_info()

    def _load_short_term(self) -> List[MemoryEvent]:
        if self.short_term_path.exists():
            try:
                data = json.loads(self.short_term_path.read_text(encoding="utf-8"))
                return [MemoryEvent(content=e["content"], timestamp=datetime.fromisoformat(e["timestamp"])) for e in data]
            except Exception:
                return []
        return []

    def _save_short_term(self):
        self.short_term_path.parent.mkdir(parents=True, exist_ok=True)
        data = [{"content": e.content, "timestamp": e.timestamp.isoformat()} for e in self.short_term_mem]
        self.short_term_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load_long_term(self) -> str:
        if self.long_term_path.exists():
            return self.long_term_path.read_text(encoding="utf-8")
        return ""

    def _save_long_term(self):
        self.long_term_path.parent.mkdir(parents=True, exist_ok=True)
        self.long_term_path.write_text(self.long_term_mem, encoding="utf-8")

    def _load_user_info(self) -> Dict[int, UserInfoEntry]:
        if self.user_info_path.exists():
            try:
                data = yaml.safe_load(self.user_info_path.read_text(encoding="utf-8")) or {}
                return {int(uid): UserInfoEntry(**info) for uid, info in data.items()}
            except Exception:
                return {}
        return {}

    def _save_user_info(self):
        self.user_info_path.parent.mkdir(parents=True, exist_ok=True)
        data = {str(uid): entry.model_dump() for uid, entry in self.user_info.items()}
        self.user_info_path.write_text(yaml.dump(data, allow_unicode=True), encoding="utf-8")
        
    def _clean_short_term_content(self, content: str) -> str:
        # Clean redundant whitespace and timestamp patterns if present
        # Matches patterns like "[YYYY-MM-DD HH:MM]" or similar at the start of the content
        pattern = r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\]\s*"
        cleaned_content = re.sub(pattern, "", content).strip()
        return cleaned_content

    async def add_short_term_event(self, content: str):
        content = self._clean_short_term_content(content)
        self.short_term_mem.append(MemoryEvent(content=content, timestamp=datetime.now()))
        await asyncio.to_thread(self._save_short_term)

    async def update_user_info(self, user_id: int, name: str, description: str):
        # remove redundant newlines in description
        description = re.sub(r'\n+', ' ', description).strip()
        self.user_info[user_id] = UserInfoEntry(user_id=user_id, name=name, description=description)
        await asyncio.to_thread(self._save_user_info)

    def get_short_term_str(self) -> str:
        if not self.short_term_mem:
            return "No short-term memories."
        # Use a short timestamp format: [YYYY-MM-DD HH:MM]
        return "\n".join([f"- [{e.timestamp.strftime('%Y-%m-%d %H:%M')}] {e.content}" for e in self.short_term_mem])

    def get_long_term_str(self) -> str:
        return self.long_term_mem if self.long_term_mem else "No long-term memories."

    def get_user_info_str(self) -> str:
        if not self.user_info:
            return "No known user information."
        return "\n".join([f"- {info.name} ({info.user_id}): {info.description}" for info in self.user_info.values()])

    def check_expirations(self, expiration_days: int):
        now = datetime.now()
        expired_indices = []
        for i, event in enumerate(self.short_term_mem):
            if now - event.timestamp > timedelta(days=expiration_days):
                expired_indices.append(i)
        
        # Returns events that need consolidation
        expired_events = [self.short_term_mem[i] for i in expired_indices]
        # Removal logic will be handled after AI consolidation to be safe, 
        # or we can remove them and let the caller handle consolidation.
        # For now, let's just return them.
        return expired_events

    async def remove_short_term_events(self, events: List[MemoryEvent]):
        self.short_term_mem = [e for e in self.short_term_mem if e not in events]
        await asyncio.to_thread(self._save_short_term)

    async def append_long_term(self, content: str):
        if self.long_term_mem:
            self.long_term_mem += "\n" + content
        else:
            self.long_term_mem = content
        await asyncio.to_thread(self._save_long_term)
