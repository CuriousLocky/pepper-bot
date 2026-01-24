import json
import re
import yaml
import asyncio
import logging
import chromadb
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel
from openai import AsyncOpenAI, OpenAI

logger = logging.getLogger(__name__)

class MemoryEvent(BaseModel):
    content: str
    timestamp: datetime
    id: Optional[str] = None
    image: Optional[str] = None # Base64 data URI

class UserInfoEntry(BaseModel):
    user_id: int
    name: str
    description: str

class MemoryState(BaseModel):
    short_term_lru: List[str] = [] # List of IDs
    long_term_lru: List[str] = []  # List of IDs
    user_lru: List[int] = []       # List of user_ids

class ChromaEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, api_key: str, api_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=api_url)
        self.model = model

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        results = []
        for item in input:
            response = self.client.embeddings.create(
                input=[],
                model=self.model,
                extra_body={"messages": [{"role": "user", "content": [{"type": "text", "text": item}]}]}
            )
            results.append(response.data[0].embedding)
        return results
        

class MemoryManager:
    def __init__(self, config: Any, 
                 short_term_path: str = "data/short-term.json", 
                 long_term_path: str = "data/long-term.json",
                 user_info_path: str = "data/known-users.yaml",
                 state_path: str = "data/memory-state.json"):
        self.config = config
        self.short_term_path = Path(short_term_path)
        self.long_term_path = Path(long_term_path)
        self.user_info_path = Path(user_info_path)
        self.state_path = Path(state_path)
        
        # Initialize ChromaDB
        self.db_path = Path(config.memory.db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(self.db_path))
        
        # Setup Embedding Function
        api_key = config.memory.api_key or config.api.key
        api_url = config.memory.api_url or config.api.url
        self.embedding_fn = ChromaEmbeddingFunction(
            api_key=api_key,
            api_url=api_url,
            model=config.memory.embedding_model
        )
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=api_url)
        
        # Collections
        self.short_collection = self.chroma_client.get_or_create_collection(
            name="short_term", 
            embedding_function=self.embedding_fn
        )
        self.long_collection = self.chroma_client.get_or_create_collection(
            name="long_term", 
            embedding_function=self.embedding_fn
        )
        self.user_collection = self.chroma_client.get_or_create_collection(
            name="users", 
            embedding_function=self.embedding_fn
        )
        
        self.short_term_mem: List[MemoryEvent] = self._load_short_term()
        self.long_term_mem: List[MemoryEvent] = self._load_long_term()
        self.user_info: Dict[int, UserInfoEntry] = self._load_user_info()
        self.state: MemoryState = self._load_state()

        # Sync memory to ensure vector store matches files (handling offline edits)
        self._sync_memory()

    def _sync_memory(self):
        """Syncs file-based memory with ChromaDB, verifying IDs and updating embeddings if content changed."""
        logger.info("Syncing memory to vector store...")
        
        # Sync Users
        # 1. Identify valid IDs from file
        valid_uids = set(str(uid) for uid in self.user_info.keys())
        
        # 2. Get all IDs currently in DB
        # Note: getting all IDs might be heavy if millions, but acceptable for this scale
        all_db_data = self.user_collection.get() 
        all_db_ids = set(all_db_data['ids']) if all_db_data['ids'] else set()
        
        # 3. Delete IDs in DB that are not in file
        to_delete = list(all_db_ids - valid_uids)
        if to_delete:
            logger.info(f"Removing {len(to_delete)} stale user entries from Chroma...")
            self.user_collection.delete(ids=to_delete)
            # Also clean LRU
            self.state.user_lru = [uid for uid in self.state.user_lru if str(uid) not in to_delete]
            
        # 4. Upsert missing or changed
        # We can optimize by only fetching existing for valid_uids
        existing = self.user_collection.get(ids=list(valid_uids))
        existing_map = {id: doc for id, doc in zip(existing['ids'], existing['documents'])} if existing['ids'] else {}
        
        to_upsert_ids = []
        to_upsert_docs = []
        to_upsert_metas = []

        for uid, info in self.user_info.items():
            sid = str(uid)
            doc_content = f"{info.name}: {info.description}"
            
            # Check if needs update (missing or content changed)
            if sid not in existing_map or existing_map[sid] != doc_content:
                to_upsert_ids.append(sid)
                to_upsert_docs.append(doc_content)
                to_upsert_metas.append({"user_id": uid})
        
        if to_upsert_ids:
            logger.info(f"Syncing {len(to_upsert_ids)} user entries to Chroma...")
            self.user_collection.upsert(
                ids=to_upsert_ids,
                documents=to_upsert_docs,
                metadatas=to_upsert_metas
            )

        # Sync Short Term
        self._sync_collection(self.short_term_mem, self.short_collection, "short-term")
        
        # Sync Long Term
        self._sync_collection(self.long_term_mem, self.long_collection, "long-term")

    def _sync_collection(self, memory_list: List[MemoryEvent], collection, name: str):
        # 1. Identify valid IDs from file
        valid_ids = set(e.id for e in memory_list)
        
        # 2. Get all IDs currently in DB
        all_db_data = collection.get()
        all_db_ids = set(all_db_data['ids']) if all_db_data['ids'] else set()
        
        # 3. Delete IDs in DB that are not in file
        to_delete = list(all_db_ids - valid_ids)
        if to_delete:
            logger.info(f"Removing {len(to_delete)} stale {name} events from Chroma...")
            collection.delete(ids=to_delete)
            # Update LRU if needed
            if name == "short-term":
                 self.state.short_term_lru = [rid for rid in self.state.short_term_lru if rid not in to_delete]
            elif name == "long-term":
                 self.state.long_term_lru = [rid for rid in self.state.long_term_lru if rid not in to_delete]

        if not memory_list:
            return

        ids = [e.id for e in memory_list]
        existing = collection.get(ids=ids)
        existing_map = {id: doc for id, doc in zip(existing['ids'], existing['documents'])} if existing['ids'] else {}
        
        to_upsert_ids = []
        to_upsert_docs = []
        to_upsert_metas = []

        for event in memory_list:
            # Check if needs update (missing in DB or content mismatch)
            # This handles cases where ID exists but content was manually edited in the JSON file
            if event.id not in existing_map or existing_map[event.id] != event.content:
                to_upsert_ids.append(event.id)
                to_upsert_docs.append(event.content)
                to_upsert_metas.append({"timestamp": event.timestamp.isoformat()})
        
        if to_upsert_ids:
            logger.info(f"Syncing {len(to_upsert_ids)} {name} events to Chroma...")
            collection.upsert(
                ids=to_upsert_ids,
                documents=to_upsert_docs,
                metadatas=to_upsert_metas
            )

    def _load_short_term(self) -> List[MemoryEvent]:
        if not self.short_term_path.exists():
            return []
        try:
            data = json.loads(self.short_term_path.read_text(encoding="utf-8"))
            events = data.get("events", [])
            mem_events = []
            for e in events:
                mem_events.append(MemoryEvent(
                    content=e["content"], 
                    timestamp=datetime.fromisoformat(e["timestamp"]),
                    id=e.get("id") or f"st_{int(datetime.fromisoformat(e['timestamp']).timestamp())}_{hash(e['content']) % 10000}"
                ))
            return sorted(mem_events, key=lambda x: x.timestamp)
        except Exception:
            return []

    def _save_short_term(self):
        self.short_term_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 2,
            "events": [{"content": e.content, "timestamp": e.timestamp.isoformat(), "id": e.id} for e in self.short_term_mem]
        }
        self.short_term_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load_long_term(self) -> List[MemoryEvent]:
        # Handle migration from .txt to .json
        old_path = self.long_term_path.with_suffix(".txt")
        if old_path.exists() and not self.long_term_path.exists():
            content = old_path.read_text(encoding="utf-8")
            lines = [l.strip() for l in content.split("\n") if l.strip()]
            mem_events = []
            base_time = datetime.now(timezone.utc) - timedelta(days=365)
            for i, line in enumerate(lines):
                # Preserve order with synthetic timestamps
                ts = base_time + timedelta(minutes=i)
                mem_events.append(MemoryEvent(
                    content=line, 
                    timestamp=ts,
                    id=f"lt_{int(ts.timestamp())}_{i}"
                ))
            # Save the new json
            self.long_term_mem = mem_events
            self._save_long_term()
            return mem_events

        if not self.long_term_path.exists():
            return []
        try:
            data = json.loads(self.long_term_path.read_text(encoding="utf-8"))
            events = data.get("events", [])
            return [MemoryEvent(
                content=e["content"], 
                timestamp=datetime.fromisoformat(e["timestamp"]),
                id=e.get("id")
            ) for e in events]
        except Exception:
            return []

    def _save_long_term(self):
        self.long_term_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 2,
            "events": [{"content": e.content, "timestamp": e.timestamp.isoformat(), "id": e.id} for e in self.long_term_mem]
        }
        self.long_term_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

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

    def _load_state(self) -> MemoryState:
        if self.state_path.exists():
            try:
                return MemoryState.model_validate_json(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                return MemoryState()
        return MemoryState()

    def _save_state(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(self.state.model_dump_json(indent=2), encoding="utf-8")

    async def add_short_term_event(self, content: str):
        event_id = f"st_{int(datetime.now(timezone.utc).timestamp())}_{hash(content) % 10000}"
        event = MemoryEvent(content=content, timestamp=datetime.now(timezone.utc), id=event_id)
        self.short_term_mem.append(event)
        
        # Add to Chroma
        self.short_collection.add(
            ids=[event_id],
            documents=[content],
            metadatas=[{"timestamp": event.timestamp.isoformat()}]
        )
        
        await asyncio.to_thread(self._save_short_term)

    async def append_long_term(self, content: str):
        event_id = f"lt_{int(datetime.now(timezone.utc).timestamp())}_{hash(content) % 10000}"
        event = MemoryEvent(content=content, timestamp=datetime.now(timezone.utc), id=event_id)
        self.long_term_mem.append(event)
        
        # Add to Chroma
        self.long_collection.add(
            ids=[event_id],
            documents=[content],
            metadatas=[{"timestamp": event.timestamp.isoformat()}]
        )
        
        await asyncio.to_thread(self._save_long_term)

    async def update_user_info(self, user_id: int, name: str, description: str):
        description = re.sub(r'\n+', ' ', description).strip()
        entry = UserInfoEntry(user_id=user_id, name=name, description=description)
        self.user_info[user_id] = entry
        
        # Update Chroma
        self.user_collection.upsert(
            ids=[str(user_id)],
            documents=[f"{name}: {description}"],
            metadatas=[{"user_id": user_id}]
        )
        
        # LRU update logic for user is handled in get_user_info_str usually, 
        # but here it's an explicit update, so maybe move it to top?
        self._update_lru(self.state.user_lru, user_id, self.config.memory.user.lru_size)
        
        await asyncio.to_thread(self._save_user_info)
        await asyncio.to_thread(self._save_state)

    def _update_lru(self, lru_list: List[Any], item: Any, max_size: int):
        if item in lru_list:
            lru_list.remove(item)
        lru_list.insert(0, item)
        while len(lru_list) > max_size:
            lru_list.pop()

    async def get_embeddings(self, texts: List[Union[str, List[Dict]]]) -> List[List[float]]:
        """Fetch embeddings for a list of strings or multimodal inputs using the async client."""
        if not texts:
            return []
        
        processed_inputs = []
        for item in texts:
            if isinstance(item, list):
                text_part = ""
                image_part = ""
                for part in item:
                    if part.get("type") == "text":
                        text_part += part.get("text", "")
                    elif part.get("type") == "image_url":
                        image_part = part.get("image_url", {})
                
                if image_part:
                    logger.info("image part length: {}".format(len(image_part.get("url", ""))))
                    processed_inputs.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": image_part},
                                {"type": "text", "text": text_part}
                            ]
                        }
                    )
                    # processed_inputs.append([image_part.get("url")])
                else:
                    # Fallback if no image found in list (shouldn't happen for multimodal intent)
                    processed_inputs.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text_part}
                            ]
                        }
                    )
            else:
                # String input
                processed_inputs.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": item}
                        ]
                    }
                )

        try:
            # logger.info(f"extra_body being sent for embeddings: {"messages": processed_inputs}")
            response = await self.async_client.embeddings.create(
                input=[],
                model=self.config.memory.embedding_model,
                extra_body={"messages": processed_inputs}
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Error fetching embeddings: {e}")
            return []

    async def get_short_term_str(self, query: str, query_embeddings: Optional[List[List[float]]] = None) -> str:
        if not self.short_term_mem:
            return "No short-term memories."

        # 1. Always include recent events
        now = datetime.now(timezone.utc)
        always_include = [e for e in self.short_term_mem if now - e.timestamp < timedelta(hours=self.config.memory.short.always_include_hours)]
        always_ids = {e.id for e in always_include}

        # 2. Search for relevant older events
        relevant_ids = []
        if self.config.memory.short.selective and (query or query_embeddings):
            results = self.short_collection.query(
                query_texts=[query] if query_embeddings is None else None,
                query_embeddings=query_embeddings,
                n_results=self.config.memory.short.top_k
            )
            if results["ids"] and results["ids"][0]:
                # IDs are strings
                search_ids = results["ids"][0]
                # Update LRU with these (reversed so most relevant is last added to top)
                for sid in reversed(search_ids):
                    self._update_lru(self.state.short_term_lru, sid, self.config.memory.short.relevant_size)
                relevant_ids = self.state.short_term_lru
        
        # Combine. Keep order of IDs in LRU but filter those that are in always_include or don't exist
        # Wait, the prompt should include all always_include PLUS relevant from LRU up to relevant_size
        # Actually, "Relevant short-term events forms a 'lru' cache... The cache size is configurable in config.yaml."
        # This suggests the cache ITSELF is what is included.
        
        # Build set of IDs to include
        include_ids = set(always_ids)
        added_from_lru = 0
        for rid in self.state.short_term_lru:
            if rid not in always_ids:
                # Check if it actually exists in memory (it might have been deleted)
                if any(e.id == rid for e in self.short_term_mem):
                    include_ids.add(rid)
                    added_from_lru += 1
                    if added_from_lru >= self.config.memory.short.relevant_size:
                        break

        # Get the actual events and sort by time
        to_prompt = [e for e in self.short_term_mem if e.id in include_ids]
        to_prompt.sort(key=lambda x: x.timestamp)

        await asyncio.to_thread(self._save_state)
        
        if not to_prompt:
            return "No relevant short-term memories."
        return "\n".join([f"- [{e.timestamp.strftime('%Y-%m-%d %H:%M')}] {e.content}" for e in to_prompt])

    async def get_long_term_str(self, query: str, query_embeddings: Optional[List[List[float]]] = None) -> str:
        if not self.long_term_mem:
            return "No long-term memories."

        if self.config.memory.long.selective and (query or query_embeddings):
            results = self.long_collection.query(
                query_texts=[query] if query_embeddings is None else None,
                query_embeddings=query_embeddings,
                n_results=self.config.memory.long.top_k
            )
            if results["ids"] and results["ids"][0]:
                search_ids = results["ids"][0]
                for sid in reversed(search_ids):
                    self._update_lru(self.state.long_term_lru, sid, self.config.memory.long.relevant_size)
        
        # Include from LRU
        include_ids = set(self.state.long_term_lru[:self.config.memory.long.relevant_size])
        
        to_prompt = [e for e in self.long_term_mem if e.id in include_ids]
        # Sort by original order (timestamp here)
        to_prompt.sort(key=lambda x: x.timestamp)

        await asyncio.to_thread(self._save_state)

        if not to_prompt:
            return "No relevant long-term memories."
        return "\n".join([f"- {e.content}" for e in to_prompt])

    def get_all_long_term_str(self) -> str:
        """Returns string representation of ALL long-term memories."""
        if not self.long_term_mem:
            return "No long-term memories."
        # No LRU updates here as this is for system maintenance
        sorted_mem = sorted(self.long_term_mem, key=lambda x: x.timestamp)
        return "\n".join([f"- {e.content}" for e in sorted_mem])

    async def get_user_info_str(self, query: str, current_user_id: Optional[int] = None, query_embeddings: Optional[List[List[float]]] = None) -> str:
        if not self.user_info:
            return "No known user information."

        # 1. Update LRU with current sender
        relevant_from_search = []
        if self.config.memory.user.selective and (query or query_embeddings):
            results = self.user_collection.query(
                query_texts=[query] if query_embeddings is None else None,
                query_embeddings=query_embeddings,
                n_results=self.config.memory.user.top_k
            )
            if results["ids"] and results["ids"][0]:
                search_ids = [int(sid) for sid in results["ids"][0]]
                # Most relevant 2 to cache top
                end = 2 if len(search_ids) >=2 else len(search_ids)
                for sid in reversed(search_ids[:end]):
                    self._update_lru(self.state.user_lru, search_ids[0], self.config.memory.user.lru_size)
                relevant_from_search = search_ids
                
        if current_user_id:
            self._update_lru(self.state.user_lru, current_user_id, self.config.memory.user.lru_size)          

        # Build final list
        # "The prompt of this round should include information of: [A, B, C, D, E] + [F]"
        # Where [A, B, C, D, E] is LRU cache, and F is a relevant one NOT in LRU. 
        
        prompt_user_ids = list(self.state.user_lru) # already capped by lru_size
        
        # Add other relevant ones from search if not in prompt_user_ids
        for rid in relevant_from_search:
            if rid not in prompt_user_ids:
                if len(prompt_user_ids) >= self.config.memory.user.lru_size + self.config.memory.user.relevant_include:
                    break
                prompt_user_ids.append(rid)

        to_prompt = []
        for uid in prompt_user_ids:
            if uid in self.user_info:
                info = self.user_info[uid]
                to_prompt.append(f"- {info.name} ({info.user_id}): {info.description}")

        await asyncio.to_thread(self._save_state)

        if not to_prompt:
            return "No relevant user information."
        return "\n".join(to_prompt)

    def get_all_user_info_str(self) -> str:
        """Returns string representation of ALL user info."""
        if not self.user_info:
            return "No known user information."
        return "\n".join([f"- {info.name} ({info.user_id}): {info.description}" for info in self.user_info.values()])

    def check_expirations(self, expiration_days: int) -> List[MemoryEvent]:
        now = datetime.now(timezone.utc)
        expired_events = []
        for event in self.short_term_mem:
            if now - event.timestamp > timedelta(days=expiration_days):
                expired_events.append(event)
        return expired_events

    async def remove_short_term_events(self, events: List[MemoryEvent]):
        event_ids = {e.id for e in events}
        self.short_term_mem = [e for e in self.short_term_mem if e.id not in event_ids]
        
        # Remove from Chroma
        if event_ids:
            self.short_collection.delete(ids=list(event_ids))
            # Also clean LRU
            self.state.short_term_lru = [rid for rid in self.state.short_term_lru if rid not in event_ids]
        
        await asyncio.to_thread(self._save_short_term)
        await asyncio.to_thread(self._save_state)