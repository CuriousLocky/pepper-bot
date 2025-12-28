import json
import uuid
import tiktoken
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class Message(BaseModel):
    role: str # 'user', 'assistant', 'system'
    content: str
    message_id: int # Internal sequential ID (0, 1, 2...)
    telegram_id: Optional[int] = None # Original Telegram Message ID
    user_id: Optional[int] = None
    user_name: str
    reply_to_id: Optional[int] = None # Internal ID of the message replied to
    timestamp: datetime
    image_url: Optional[str] = None # For multimodal support

class ChatHistory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chat_id: int
    messages: List[Message]
    last_updated: datetime

    def get_internal_id(self, telegram_id: int) -> Optional[int]:
        for msg in self.messages:
            if msg.telegram_id == telegram_id:
                return msg.message_id
        return None

    def get_next_message_id(self) -> int:
        if not self.messages:
            return 1
        return max(m.message_id for m in self.messages) + 1

    def format_for_llm(self, known_users: Dict[int, str] = {}, token_limit: int = 8000, model: str = "gpt-4o") -> List[Dict[str, Any]]:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        formatted_messages = []
        current_tokens = 0
        
        # Iterate backwards to keep most recent messages
        for msg in reversed(self.messages):
            content = msg.content
            # Reply to ID is already internal
            reply_str = f" (reply to msg {msg.reply_to_id})" if msg.reply_to_id is not None else ""
            
            if msg.role == "user" and msg.user_id:
                if msg.user_id in known_users:
                    display_name = known_users[msg.user_id]
                else:
                    display_name = f"unknown-user {msg.user_id}"
            else:
                display_name = msg.user_name

            display_content = f"[msg {msg.message_id}] {display_name}{reply_str}: {content}"
            
            # Estimate tokens for this message
            # OpenAI overhead: ~4 tokens per message + tokens in content + role + name
            # Image tokens: rough estimate 85 (low detail) if image_url exists
            msg_tokens = 4 
            msg_tokens += len(encoding.encode(display_content))
            msg_tokens += len(encoding.encode(msg.role))
            if msg.image_url:
                msg_tokens += 85
            
            if current_tokens + msg_tokens > token_limit:
                break
            
            current_tokens += msg_tokens
            
            # OpenAI format
            msg_obj = {"role": msg.role, "content": display_content}
            if msg.image_url:
                msg_obj["content"] = [
                    {"type": "text", "text": display_content},
                    {"type": "image_url", "image_url": {"url": msg.image_url}}
                ]
            formatted_messages.append(msg_obj)
            
        return formatted_messages[::-1]

class HistoryManager:
    def __init__(self, storage_path: str = "data/chat-histories.json"):
        self.storage_path = Path(storage_path)
        self.histories: Dict[str, ChatHistory] = {} # Key: thread_id
        self.message_map: Dict[int, str] = {} # Key: telegram_message_id, Value: thread_id
        self._load()

    def _load(self):
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text(encoding="utf-8"))
                # Handle migration logic is complex, skipping strict migration for prototype.
                # Assuming data structure matches.
                
                # Basic check if data is in new structure (dict of threads)
                # If loading fails, start empty.
                self.histories = {k: ChatHistory(**v) for k, v in data.items()}
                self._rebuild_message_map()

            except Exception as e:
                # print(f"Error loading history: {e}")
                self.histories = {}
        else:
            self.histories = {}

    def _rebuild_message_map(self):
        self.message_map = {}
        for thread_id, hist in self.histories.items():
            for msg in hist.messages:
                if msg.telegram_id:
                    self.message_map[msg.telegram_id] = thread_id

    def _save(self):
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v.model_dump(mode='json') for k, v in self.histories.items()}
        self.storage_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def create_thread(self, chat_id: int) -> ChatHistory:
        hist = ChatHistory(chat_id=chat_id, messages=[], last_updated=datetime.now())
        self.histories[hist.id] = hist
        self._save()
        return hist

    def get_thread(self, thread_id: str) -> Optional[ChatHistory]:
        return self.histories.get(thread_id)

    def get_thread_id_by_message_id(self, telegram_message_id: int) -> Optional[str]:
        return self.message_map.get(telegram_message_id)

    def add_message(self, thread_id: str, message: Message):
        if thread_id in self.histories:
            history = self.histories[thread_id]
            history.messages.append(message)
            history.last_updated = datetime.now()
            
            # Update map using telegram_id
            if message.telegram_id:
                self.message_map[message.telegram_id] = thread_id
            
            self._save()

    def clean_expired(self, hours: int):
        now = datetime.now()
        expired_ids = [tid for tid, h in self.histories.items() if now - h.last_updated > timedelta(hours=hours)]
        for tid in expired_ids:
            # Remove messages from map
            for msg in self.histories[tid].messages:
                if msg.telegram_id and msg.telegram_id in self.message_map:
                    del self.message_map[msg.telegram_id]
            del self.histories[tid]
        
        if expired_ids:
            self._save()
