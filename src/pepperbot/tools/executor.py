import json
import logging
import os
from dataclasses import dataclass, field
from random import randint
from typing import Any, Awaitable, Callable, Dict, List, Optional

from get_url_content import get_url_content
from image_gen import generate_image
from websearch import web_search

from pepperbot.config import Config
from pepperbot.history.attachments import AttachmentStore
from pepperbot.history.models import Thread
from pepperbot.providers.base import ChatMessage, ToolCall

logger = logging.getLogger(__name__)


@dataclass
class ToolRuntime:
    chat_id: int
    thread: Thread
    attachment_store: AttachmentStore
    schedule_func: Optional[Callable[[int, str, str], Awaitable[str]]] = None
    list_func: Optional[Callable[[], Awaitable[str]]] = None
    block_user_func: Optional[Callable[[int, int], Awaitable[str]]] = None
    generated_images: List[str] = field(default_factory=list)
    generated_image_context: Dict[str, str] = field(default_factory=dict)


class ToolExecutor:
    def __init__(self, config: Config, memory_manager: Any):
        self.config = config
        self.memory_manager = memory_manager

    async def execute(self, tool_calls: List[ToolCall], runtime: ToolRuntime) -> List[ChatMessage]:
        results: List[ChatMessage] = []
        for tool_call in tool_calls:
            try:
                output = await self._execute_one(tool_call, runtime)
            except Exception as exc:
                logger.exception("Tool execution failed: %s", tool_call.name)
                output = f"Error executing tool {tool_call.name}: {exc}"
            results.append(
                ChatMessage(
                    role="tool",
                    content=output,
                    tool_call_id=tool_call.id,
                )
            )
        return results

    async def _execute_one(self, tool_call: ToolCall, runtime: ToolRuntime) -> str:
        name = tool_call.name
        args = tool_call.arguments
        if name == "generate_image":
            return await self._generate_image(args, tool_call.id, runtime)
        if name == "add_short_term_memory":
            await self.memory_manager.add_short_term_event(str(args.get("content", "")))
            return "Short-term memory added successfully."
        if name == "web_search":
            return json.dumps(web_search(str(args.get("query", "")), self.config.search), ensure_ascii=False)
        if name == "get_url_content":
            return get_url_content(str(args.get("url", "")))
        if name == "set_scheduled_task":
            if not runtime.schedule_func:
                return "Error: Scheduling context not available."
            return await runtime.schedule_func(
                int(args.get("time_in_minute", 0)),
                str(args.get("title", "Untitled")),
                str(args.get("content", "")),
            )
        if name == "get_scheduled_task_list":
            if not runtime.list_func:
                return "Error: Scheduling context not available."
            return await runtime.list_func()
        if name == "block_user":
            if not runtime.block_user_func:
                return "Error: Blacklist context not available."
            return await runtime.block_user_func(
                int(args.get("user_id", 0)), int(args.get("duration_minutes", 0))
            )
        if name == "randint":
            return str(randint(int(args.get("a", 0)), int(args.get("b", 0))))
        if name == "fetch_skill":
            return await self._fetch_skill(str(args.get("skill_name", "")))
        return f"Unknown tool: {name}"

    async def _generate_image(self, args: Dict[str, Any], tool_call_id: str, runtime: ToolRuntime) -> str:
        msg_id = args.get("msg_id")
        image_input = None
        if msg_id:
            normalized_msg_id = str(msg_id)
            if normalized_msg_id.isdigit():
                normalized_msg_id = f"m{normalized_msg_id}"
            target = runtime.thread.get_message(normalized_msg_id)
            if not target:
                return f"Error: Message {msg_id} not found."
            image_attachment = next((a for a in target.attachments if a.kind == "image"), None)
            if not image_attachment:
                return f"Error: Message {msg_id} does not contain an image."
            image_input = runtime.attachment_store.read_data_uri(image_attachment)
            if not image_input:
                return f"Error: Image attachment for message {msg_id} is unavailable."

        success, full_res_base64, resized_base64, text_content = await generate_image(
            str(args.get("description", "")),
            self.config,
            image_base64=image_input,
        )
        if not success or not full_res_base64:
            return f"Failed to generate image. Model output:\n{text_content}"
        runtime.generated_images.append(full_res_base64)
        if resized_base64:
            runtime.generated_image_context[tool_call_id] = resized_base64
        output = "The image is successfully generated and will attach to the next message."
        if text_content:
            output += f"\n\n{text_content}"
        return output

    async def _fetch_skill(self, skill_name: str) -> str:
        if not self.config.skills.enabled:
            return "Skills feature is disabled."
        skill_path = os.path.join(self.config.skills.root_path, f"{skill_name}.md")
        if not os.path.isfile(skill_path):
            return f"Skill '{skill_name}' not found."
        with open(skill_path, "r", encoding="utf-8") as f:
            return f.read()
