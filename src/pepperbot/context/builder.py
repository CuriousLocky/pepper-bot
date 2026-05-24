from datetime import datetime, timezone
from typing import Dict, List, Optional

from pepperbot.config import Config, TemplateConfig
from pepperbot.context.summarizer import SummaryError, ThreadSummarizer
from pepperbot.context.template import TemplateRenderer
from pepperbot.context.token_budget import estimate_message_tokens
from pepperbot.history.attachments import AttachmentStore
from pepperbot.history.models import ConversationMessage, Thread
from pepperbot.providers.base import ChatMessage


class ContextBuildResult:
    def __init__(self, messages: List[ChatMessage], warning: str = ""):
        self.messages = messages
        self.warning = warning


class ContextBuilder:
    def __init__(
        self,
        config: Config,
        template: TemplateConfig,
        attachment_store: AttachmentStore,
        summarizer: Optional[ThreadSummarizer] = None,
    ):
        self.config = config
        self.renderer = TemplateRenderer(template)
        self.attachment_store = attachment_store
        self.summarizer = summarizer

    async def build(
        self,
        thread: Thread,
        current_message_id: str,
        memory_context: str = "",
        known_users_context: str = "",
        skill_list: str = "",
        memory_sections: Optional[Dict[str, str]] = None,
    ) -> ContextBuildResult:
        warning = ""
        sections = memory_sections or self._memory_sections_from_legacy(memory_context, known_users_context)
        variables = self._variables(
            thread,
            current_message_id,
            sections,
            skill_list,
            warning="",
        )
        messages = self._render_template_messages(thread, current_message_id, variables)
        token_count = estimate_message_tokens(messages, self.config.chat_model())
        trigger = int(self.config.context.max_context_window * self.config.context.summary_trigger_ratio)

        if token_count > trigger and self.summarizer:
            try:
                to_summarize = self._messages_to_summarize(thread, current_message_id)
                if to_summarize:
                    result = await self.summarizer.summarize(thread, to_summarize)
                    thread.summary = result.summary
                    thread.summary_until_message_id = result.until_message_id
                    variables = self._variables(
                        thread,
                        current_message_id,
                        sections,
                        skill_list,
                        warning="",
                    )
                    messages = self._render_template_messages(thread, current_message_id, variables)
            except SummaryError as exc:
                warning = f"Older context omitted. Reason: summarization failed: {exc}"
                variables = self._variables(
                    thread,
                    current_message_id,
                    sections,
                    skill_list,
                    warning=warning,
                    force_recent_only=True,
                )
                messages = self._render_template_messages(
                    thread,
                    current_message_id,
                    variables,
                    force_recent_only=True,
                )

        return ContextBuildResult(messages=messages, warning=warning)

    def repair_message(self, variables: Optional[Dict[str, str]] = None) -> ChatMessage:
        return self.renderer.render_repair_message(variables or {})

    def _messages_to_summarize(self, thread: Thread, current_message_id: str) -> List[ConversationMessage]:
        recent_count = self.config.context.preserve_recent_messages
        candidates = [message for message in thread.messages if message.id != current_message_id]
        if thread.summary_until_message_id:
            seen = False
            filtered = []
            for message in candidates:
                if seen:
                    filtered.append(message)
                elif message.id == thread.summary_until_message_id:
                    seen = True
            candidates = filtered
        if len(candidates) <= recent_count:
            return []
        return candidates[: -recent_count]

    def _variables(
        self,
        thread: Thread,
        current_message_id: str,
        memory_sections: Dict[str, str],
        skill_list: str,
        warning: str,
        force_recent_only: bool = False,
    ) -> Dict[str, str]:
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        return {
            "date_time": now_utc,
            "bot_name": self.config.bot.name,
            "chat_id": str(thread.chat_id),
            "thread_id": thread.id,
            "skill_list": skill_list or "No skills available.",
            "knowledges": memory_sections.get("knowledges", ""),
            "long_term_memory": memory_sections.get("long_term_memory", ""),
            "short_term_memory": memory_sections.get("short_term_memory", ""),
            "known_user_info": memory_sections.get("known_user_info", ""),
            "thread_summary": self._render_thread_summary(thread, warning),
        }

    def _render_template_messages(
        self,
        thread: Thread,
        current_message_id: str,
        variables: Dict[str, str],
        force_recent_only: bool = False,
    ) -> List[ChatMessage]:
        rendered: List[ChatMessage] = []
        for entry in self.renderer.template.messages:
            if entry.type == "thread_context":
                rendered.extend(self._thread_context_messages(thread, current_message_id, variables, force_recent_only))
                continue
            message = self.renderer.render_template_message(entry, variables)
            if message is not None:
                rendered.append(message)
        return rendered

    def _memory_sections_from_legacy(self, memory_context: str, known_users_context: str) -> Dict[str, str]:
        return {
            "knowledges": "",
            "long_term_memory": "",
            "short_term_memory": memory_context.strip(),
            "known_user_info": known_users_context.strip(),
        }

    def _render_thread_summary(self, thread: Thread, warning: str) -> str:
        if warning and thread.summary:
            return f"{thread.summary}\n\n{warning}"
        return warning or thread.summary

    def _thread_context_messages(
        self,
        thread: Thread,
        current_message_id: str,
        variables: Dict[str, str],
        force_recent_only: bool,
    ) -> List[ChatMessage]:
        rendered: List[ChatMessage] = []
        summary = variables.get("thread_summary", "").strip()
        if summary:
            rendered.append(ChatMessage(role="user", content=f"<summary>\n{self._escape(summary)}\n</summary>"))

        pinned = self._pinned_referenced_message(thread, current_message_id, force_recent_only)
        if pinned:
            rendered.append(
                ChatMessage(
                    role="user",
                    content="\n".join([
                        "<pinned_referenced_message>",
                        self._render_message(pinned),
                        "</pinned_referenced_message>",
                    ]),
                )
            )

        rendered.extend(self._expand_message_sequence(self._visible_messages(thread, current_message_id, force_recent_only)))
        current = thread.get_message(current_message_id)
        if current:
            rendered.extend(self._expand_message_sequence([current]))
        return rendered

    def _pinned_referenced_message(
        self,
        thread: Thread,
        current_message_id: str,
        force_recent_only: bool,
    ) -> Optional[ConversationMessage]:
        current = thread.get_message(current_message_id)
        messages = self._visible_messages(thread, current_message_id, force_recent_only)
        if current and current.reply_to:
            visible_ids = {message.id for message in messages}
            referenced = thread.get_message(current.reply_to)
            if referenced and referenced.id not in visible_ids:
                return referenced
        return None

    def _expand_message_sequence(self, messages: List[ConversationMessage]) -> List[ChatMessage]:
        rendered: List[ChatMessage] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            tool_calls = message.metadata.get("tool_calls") if message.role == "assistant" else None
            if tool_calls:
                tool_messages = []
                cursor = index + 1
                while cursor < len(messages) and messages[cursor].role == "tool":
                    tool_messages.append(messages[cursor])
                    cursor += 1
                if self._tool_group_complete(tool_calls, tool_messages):
                    rendered.append(ChatMessage(role="assistant", content=message.content or "", tool_calls=tool_calls))
                    rendered.extend(self._tool_message_to_chat(tool_message) for tool_message in tool_messages)
                    index = cursor
                    continue
            if message.role == "tool":
                rendered.append(ChatMessage(role="user", content=self._render_message(message)))
            else:
                rendered.append(self._normal_message_to_chat(message))
            index += 1
        return rendered

    def _tool_group_complete(self, tool_calls, tool_messages: List[ConversationMessage]) -> bool:
        required_ids = {
            tool_call.get("id")
            for tool_call in tool_calls
            if isinstance(tool_call, dict) and tool_call.get("id")
        }
        result_ids = {
            message.metadata.get("tool_call_id")
            for message in tool_messages
            if message.metadata.get("tool_call_id")
        }
        return bool(required_ids) and required_ids.issubset(result_ids)

    def _normal_message_to_chat(self, message: ConversationMessage) -> ChatMessage:
        return ChatMessage(role=message.role, content=self._wrapped_content(message, allow_image_payloads=message.role == "user"))

    def _tool_message_to_chat(self, message: ConversationMessage) -> ChatMessage:
        return ChatMessage(
            role="tool",
            content=self._tool_content(message),
            tool_call_id=message.metadata.get("tool_call_id"),
        )

    def _wrapped_content(self, message: ConversationMessage, allow_image_payloads: bool) -> str | List[Dict[str, object]]:
        text = self._render_message(message)
        if not allow_image_payloads or not self.config.chat_supports_vision():
            return text
        image_parts = self._image_parts(message)
        if not image_parts:
            return text
        return [{"type": "text", "text": text}, *image_parts]

    def _tool_content(self, message: ConversationMessage) -> str | List[Dict[str, object]]:
        text = message.content or ""
        if message.attachments:
            text += "\n" + "\n".join(
                f"[attachment {attachment.id}: {attachment.kind}]" for attachment in message.attachments
            )
        if not self.config.chat_supports_vision():
            return text
        image_parts = self._image_parts(message)
        if not image_parts:
            return text
        return [{"type": "text", "text": text}, *image_parts]

    def _image_parts(self, message: ConversationMessage) -> List[Dict[str, object]]:
        parts: List[Dict[str, object]] = []
        for attachment in message.attachments:
            if attachment.kind != "image":
                continue
            data_uri = self.attachment_store.read_data_uri(attachment)
            if data_uri:
                parts.append({"type": "image_url", "image_url": {"url": data_uri}})
        return parts

    def _visible_messages(
        self,
        thread: Thread,
        current_message_id: str,
        force_recent_only: bool,
    ) -> List[ConversationMessage]:
        messages = [message for message in thread.messages if message.id != current_message_id]
        if thread.summary_until_message_id:
            found = False
            visible = []
            for message in messages:
                if found:
                    visible.append(message)
                elif message.id == thread.summary_until_message_id:
                    found = True
            messages = visible
        if force_recent_only:
            messages = messages[-self.config.context.preserve_recent_messages :]
        return messages

    def _render_message(self, message: ConversationMessage) -> str:
        attrs = [
            f'id="{self._escape_attr(message.id)}"',
            f'role="{self._escape_attr(message.role)}"',
            f'author="{self._escape_attr(message.author.name)}"',
        ]
        if message.reply_to:
            attrs.append(f'reply_to="{self._escape_attr(message.reply_to)}"')
        if message.attachments:
            attrs.append(f'attachments="{len(message.attachments)}"')
        content = self._escape(message.content or "")
        if message.attachments:
            content += "\n" + "\n".join(
                f"[attachment {attachment.id}: {attachment.kind}]" for attachment in message.attachments
            )
        return f"<message {' '.join(attrs)}>\n{content}\n</message>"

    def _escape(self, text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    def _escape_attr(self, text: str) -> str:
        return self._escape(text).replace('"', "&quot;")
