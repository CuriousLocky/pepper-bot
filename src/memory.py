import json
import re
import yaml
import asyncio
import logging
import chromadb
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Any, Union, Tuple
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
                 knowledges_path: str = "data/knowledges.json",
                 user_info_path: str = "data/known-users.yaml",
                 state_path: str = "data/memory-state.json"):
        self.config = config
        self.short_term_path = Path(short_term_path)
        self.long_term_path = Path(long_term_path)
        self.knowledges_path = Path(knowledges_path)
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
        
        # Setup Tool Model Client for migration and classification
        tool_api_key = config.tool_model.api_key or config.api.key
        tool_api_url = config.tool_model.api_url or config.api.url
        self.tool_client = AsyncOpenAI(api_key=tool_api_key, base_url=tool_api_url)
        
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
        self.knowledges_mem: List[MemoryEvent] = self._load_knowledges()
        self.user_info: Dict[int, UserInfoEntry] = self._load_user_info()
        self.state: MemoryState = self._load_state()

        # Sync memory to ensure vector store matches files (handling offline edits)
        self._sync_memory()

        # Check and perform version migration if needed
        self._check_and_migrate_sync()

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
            "version": 3,
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
            "version": 3,
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

    def _load_knowledges(self) -> List[MemoryEvent]:
        if not self.knowledges_path.exists():
            return []
        try:
            data = json.loads(self.knowledges_path.read_text(encoding="utf-8"))
            events = data.get("events", [])
            return [MemoryEvent(
                content=e["content"],
                timestamp=datetime.fromisoformat(e["timestamp"]),
                id=e.get("id")
            ) for e in events]
        except Exception:
            return []

    def _save_knowledges(self):
        self.knowledges_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 3,
            "events": [{"content": e.content, "timestamp": e.timestamp.isoformat(), "id": e.id} for e in self.knowledges_mem]
        }
        self.knowledges_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _get_file_version(self, file_path: Path) -> int:
        """Get version from a JSON file."""
        if not file_path.exists():
            return 0
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            return data.get("version", 0)
        except:
            return 0

    def _check_and_migrate_sync(self):
        """Check and perform version migrations if needed (synchronous wrapper)."""
        long_term_version = self._get_file_version(self.long_term_path)
        
        if long_term_version == 2:
            asyncio.run(self._migrate_to_v3())

    async def _check_and_migrate(self):
        """Check and perform version migrations if needed."""
        long_term_version = self._get_file_version(self.long_term_path)
        
        if long_term_version == 2:
            await self._migrate_to_v3()

    async def _migrate_to_v3(self):
        """Migrate from version 2 to 3: Separate knowledges from events."""
        logger.info("Starting migration to version 3: Separating knowledges from events...")
        
        if not self.long_term_mem:
            logger.info("No long-term memories to migrate.")
            return
        
        total = len(self.long_term_mem)
        batch_size = 10
        parallel_batches = 5
        
        # Create all batches with their original indices
        batches = []
        for i in range(0, total, batch_size):
            batch = self.long_term_mem[i:i + batch_size]
            batches.append((i, batch))  # (original_start_index, batch)
        
        logger.info(f"Created {len(batches)} batches to process in parallel groups of {parallel_batches}")
        
        # Process batches in parallel groups
        all_results = []
        for i in range(0, len(batches), parallel_batches):
            batch_group = batches[i:i + parallel_batches]
            
            # Process each batch in the group in parallel (5 requests at a time)
            tasks = [self._process_batch_with_index(start_idx, batch) for start_idx, batch in batch_group]
            group_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect results (handle exceptions)
            for result in group_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch processing failed with exception: {result}")
                    all_results.append(None)
                else:
                    all_results.append(result)
        
        # Sort results by original batch index to maintain order
        # This ensures that even though batches are processed in parallel,
        # the final list maintains the original document order
        all_results.sort(key=lambda x: x[0] if x else 0)
        
        # Combine results in order
        knowledge_entries = []
        event_entries = []
        total_knowledges = 0
        total_events = 0
        
        for result in all_results:
            if result is None:
                continue
            
            start_idx, knowl_indices, evt_indices, batch_knowledges, batch_events = result
            batch_num = (start_idx // batch_size) + 1
            
            knowledge_entries.extend(batch_knowledges)
            event_entries.extend(batch_events)
            total_knowledges += len(batch_knowledges)
            total_events += len(batch_events)
            
            logger.info(f"Batch {batch_num} processed: {len(batch_knowledges)} knowledges, {len(batch_events)} events")
        
        logger.info(f"Migration complete: {total_knowledges} knowledges, {total_events} events")
        
        # Update memory lists
        self.knowledges_mem = knowledge_entries
        self.long_term_mem = event_entries
        
        # Save all files with version 3
        await asyncio.to_thread(self._save_long_term)
        await asyncio.to_thread(self._save_short_term)
        await asyncio.to_thread(self._save_knowledges)
        
        logger.info("Migration to version 3 complete.")

    async def _process_batch_with_index(self, start_idx: int, batch: List[MemoryEvent]) -> Tuple[int, List[int], List[int], List[MemoryEvent], List[MemoryEvent]]:
        """Process a single batch and return results with its original index."""
        batch_indices = list(range(len(batch)))
        batch_num = (start_idx // 10) + 1
        
        try:
            knowledge_indices, event_indices = await self._classify_batch(batch)
            
            # Validate that all indices are covered exactly once
            all_indices = set(knowledge_indices) | set(event_indices)
            expected_indices = set(batch_indices)
            
            if all_indices != expected_indices:
                logger.warning(f"Batch {batch_num}: Index mismatch. Expected {expected_indices}, got {all_indices}. Attempting to fill gaps...")
                
                # Fill missing indices as events
                missing_indices = expected_indices - all_indices
                event_indices.extend(missing_indices)
                
            # Check for duplicates
            knowledge_set = set(knowledge_indices)
            event_set = set(event_indices)
            duplicates = (knowledge_set & event_set)
            
            if duplicates:
                logger.warning(f"Batch {batch_num}: Duplicate indices {duplicates}. Moving duplicates to events.")
                knowledge_indices = [idx for idx in knowledge_indices if idx not in duplicates]
                event_indices.extend(list(duplicates))
            
            # Sort indices to maintain order
            knowledge_indices.sort()
            event_indices.sort()
            
            # Assign entries based on classification
            batch_knowledges = []
            batch_events = []
            
            for idx in knowledge_indices:
                if 0 <= idx < len(batch):
                    batch_knowledges.append(batch[idx])
            
            for idx in event_indices:
                if 0 <= idx < len(batch):
                    batch_events.append(batch[idx])
            
            return (start_idx, knowledge_indices, event_indices, batch_knowledges, batch_events)
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_num}: {e}, defaulting all to events")
            # Default all entries in batch to events
            return (start_idx, [], list(range(len(batch))), [], batch.copy())

    async def _classify_batch(self, batch: List[MemoryEvent]) -> Tuple[List[int], List[int]]:
        """Classify a batch of entries using tool calling."""
        from openai.types.chat import ChatCompletionMessageParam
        
        entries_text = ""
        for idx, event in enumerate(batch):
            entries_text += f"\n[{idx}] {event.content}"
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "classify_knowledge",
                    "description": "Classify entries as knowledge. Provide a list of indices that should be classified as knowledge.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "indices": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "List of indices (0-9) corresponding to knowledge entries"
                            }
                        },
                        "required": ["indices"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "classify_event",
                    "description": "Classify entries as events. Provide a list of indices that should be classified as events.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "indices": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "List of indices (0-9) corresponding to event entries"
                            }
                        },
                        "required": ["indices"]
                    }
                }
            }
        ]
        
        prompt = f"""Classify the following memory entries:
{entries_text}

Classification criteria:
- KNOWLEDGE: Plain facts about the bot itself, rules the bot should follow, methodologies, procedures, or persistent information that defines how the bot operates. These are general truths or instructions regardless of time. They should not be related to a specific action a user took.
- EVENT: Specific occurrences describing what someone did, personal interactions, temporary situations, or time-dependent information.

Use the provided tools to classify each entry. Each entry should be classified exactly once as either knowledge or event."""
        
        knowledge_indices: List[int] = []
        event_indices: List[int] = []
        
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                messages: List[ChatCompletionMessageParam] = [
                    {"role": "user", "content": prompt}
                ]
                
                response = await self.tool_client.chat.completions.create(
                    model=self.config.tool_model.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.1
                )
                
                response_message = response.choices[0].message
                
                if response_message.tool_calls:
                    for tool_call in response_message.tool_calls:
                        function_name = tool_call.function.name
                        import json
                        function_args = json.loads(tool_call.function.arguments)
                        
                        if function_name == "classify_knowledge":
                            indices = function_args.get("indices", [])
                            knowledge_indices.extend(indices)
                        elif function_name == "classify_event":
                            indices = function_args.get("indices", [])
                            event_indices.extend(indices)
                else:
                    logger.warning("No tool calls in response, defaulting all to events")
                    event_indices = list(range(len(batch)))
                
                break
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}, retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {max_retries} attempts failed: {e}, defaulting all to events")
                    event_indices = list(range(len(batch)))
                    break
        
        return knowledge_indices, event_indices

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

    async def append_knowledge(self, content: str):
        """Append a knowledge to the knowledges store."""
        event_id = f"kn_{int(datetime.now(timezone.utc).timestamp())}_{hash(content) % 10000}"
        event = MemoryEvent(content=content, timestamp=datetime.now(timezone.utc), id=event_id)
        self.knowledges_mem.append(event)
        
        await asyncio.to_thread(self._save_knowledges)

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

    def get_latest_long_term_str(self, limit: int) -> str:
        """Returns string representation of the latest N long-term memories."""
        if not self.long_term_mem:
            return "No long-term memories."
        # Sort by timestamp descending (newest first) and take limit
        sorted_mem = sorted(self.long_term_mem, key=lambda x: x.timestamp, reverse=True)
        latest_mem = sorted_mem[:limit]
        # Return in chronological order (oldest first)
        latest_mem_sorted = sorted(latest_mem, key=lambda x: x.timestamp)
        return "\n".join([f"- {e.content}" for e in latest_mem_sorted])

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

    def get_all_knowledges_str(self) -> str:
        """Returns string representation of ALL knowledges."""
        if not self.knowledges_mem:
            return "No knowledge stored."
        sorted_mem = sorted(self.knowledges_mem, key=lambda x: x.timestamp)
        return "\n".join([f"- {e.content}" for e in sorted_mem])