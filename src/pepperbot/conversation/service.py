import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Optional

from pepperbot.config import Config
from pepperbot.context.builder import ContextBuilder
from pepperbot.conversation.models import ConversationResult, IncomingMessage, ReferencedMessage
from pepperbot.conversation.runner import ChatLoopRunner
from pepperbot.conversation.sanitizer import ResponseSanitizer
from pepperbot.history.models import Actor, ConversationMessage, TelegramRef, Thread
from pepperbot.history.store import HistoryStore
from pepperbot.providers.base import ChatMessage
from pepperbot.telegram.delivery import AdminReporter, TelegramDelivery
from pepperbot.tools.executor import ToolRuntime

logger = logging.getLogger(__name__)


@dataclass
class ConversationRuntime:
    bot: Any
    schedule_func: Optional[Callable[[int, str, str], Awaitable[str]]] = None
    list_func: Optional[Callable[[], Awaitable[str]]] = None
    block_user_func: Optional[Callable[[int, int], Awaitable[str]]] = None
    send_typing_func: Optional[Callable[[], Awaitable[None]]] = None


class ConversationService:
    def __init__(
        self,
        config: Config,
        history: HistoryStore,
        memory_manager: Any,
        context_builder: ContextBuilder,
        chat_runner: ChatLoopRunner,
        delivery: TelegramDelivery,
        reporter: AdminReporter,
    ):
        self.config = config
        self.history = history
        self.memory_manager = memory_manager
        self.context_builder = context_builder
        self.chat_runner = chat_runner
        self.delivery = delivery
        self.reporter = reporter
        self.sanitizer = ResponseSanitizer([config.bot.name, *config.bot.nicknames])
        self._locks: Dict[str, asyncio.Lock] = {}

    async def handle_incoming(
        self,
        incoming: IncomingMessage,
        runtime: ConversationRuntime,
    ) -> ConversationResult:
        if incoming.is_command and incoming.is_reply_to_bot:
            return ConversationResult(handled=False, reason="command_reply_to_bot_ignored")
        if not incoming.is_command and not incoming.is_reply_to_bot:
            return ConversationResult(handled=False, reason="not_activated")

        if self.config.bot.chat_whitelist and incoming.chat_id not in self.config.bot.chat_whitelist:
            return ConversationResult(handled=False, reason="chat_not_whitelisted")

        thread, reply_to_id, should_call_ai = await self._resolve_thread(incoming, runtime)
        if not thread:
            return ConversationResult(handled=False, reason="thread_not_resolved")
        if not should_call_ai:
            return ConversationResult(handled=True, reason="expiration_notice_sent")

        lock = self._locks.setdefault(thread.id, asyncio.Lock())
        async with lock:
            if runtime.send_typing_func:
                try:
                    await runtime.send_typing_func()
                except Exception:
                    logger.debug("Failed to send typing action", exc_info=True)

            current_message = self._append_user_message(thread, incoming, reply_to_id)
            self.history.save()
            try:
                memory_sections = await self._memory_context(incoming, runtime.bot, thread)
                context = await self.context_builder.build(
                    thread,
                    current_message.id,
                    memory_sections=memory_sections,
                    skill_list=self._skill_list(),
                )
                if context.warning:
                    await self.reporter.report(
                        runtime.bot,
                        "Context summarization failed",
                        context.warning,
                        context_preview=self._thread_preview(thread),
                    )
                tool_runtime = ToolRuntime(
                    chat_id=incoming.chat_id,
                    thread=thread,
                    attachment_store=self.history.attachment_store,
                    schedule_func=runtime.schedule_func,
                    list_func=runtime.list_func,
                    block_user_func=runtime.block_user_func,
                )
                response_text, protocol_messages = await self.chat_runner.run(context.messages, tool_runtime)
                response_text = self.sanitizer.clean(response_text) or self.config.response.fallback_text
            except Exception as exc:
                logger.exception("Conversation processing failed")
                await self.reporter.report(
                    runtime.bot,
                    "Conversation processing failed",
                    repr(exc),
                    context_preview=self._thread_preview(thread),
                )
                response_text = self.config.response.fallback_text
                protocol_messages = []
                tool_runtime = ToolRuntime(
                    chat_id=incoming.chat_id,
                    thread=thread,
                    attachment_store=self.history.attachment_store,
                )

            telegram_refs = []
            generated_attachments = []
            if tool_runtime.generated_images:
                try:
                    image_refs = await self.delivery.send_images(runtime.bot, incoming.chat_id, tool_runtime.generated_images)
                    telegram_refs.extend(image_refs)
                    for image in tool_runtime.generated_images:
                        generated_attachments.append(
                            self.history.attachment_store.save_data_uri(
                                image,
                                source="generated",
                                expires_at=thread.expires_at,
                            )
                        )
                except Exception as exc:
                    logger.exception("Generated image delivery failed")
                    await self.reporter.report(
                        runtime.bot,
                        "Generated image delivery failed",
                        repr(exc),
                        context_preview=self._thread_preview(thread),
                    )

            try:
                text_refs = await self.delivery.send_text(
                    runtime.bot,
                    incoming.chat_id,
                    response_text,
                    reply_to_message_id=incoming.telegram_message_id,
                )
                telegram_refs.extend(text_refs)
            except Exception as exc:
                await self.reporter.report(
                    runtime.bot,
                    "Telegram text delivery failed",
                    repr(exc),
                    context_preview=response_text,
                )
                raise

            self._append_protocol_messages(thread, protocol_messages, tool_runtime)
            assistant_message = ConversationMessage(
                id=thread.next_message_id(),
                role="assistant",
                author=Actor(id=getattr(runtime.bot, "id", None), name=self.config.bot.name, is_bot=True),
                content=response_text,
                reply_to=current_message.id,
                telegram_refs=telegram_refs,
                attachments=generated_attachments,
                created_at=datetime.now(),
                metadata={"protocol_message_count": len(protocol_messages)},
            )
            self.history.add_message(thread, assistant_message)
            self.history.save()
            return ConversationResult(
                handled=True,
                text=response_text,
                images=tool_runtime.generated_images,
            )

    def _append_protocol_messages(
        self,
        thread: Thread,
        protocol_messages: list[ChatMessage],
        tool_runtime: ToolRuntime,
    ) -> None:
        for protocol_message in protocol_messages:
            if protocol_message.role == "assistant" and protocol_message.tool_calls:
                self.history.add_message(
                    thread,
                    ConversationMessage(
                        id=thread.next_message_id(),
                        role="assistant",
                        author=Actor(name=self.config.bot.name, is_bot=True),
                        content=self._chat_content_text(protocol_message.content),
                        created_at=datetime.now(),
                        metadata={"tool_calls": protocol_message.tool_calls},
                    ),
                )
            elif protocol_message.role == "tool":
                attachments = []
                if protocol_message.tool_call_id in tool_runtime.generated_image_context:
                    try:
                        attachments.append(
                            self.history.attachment_store.save_data_uri(
                                tool_runtime.generated_image_context[protocol_message.tool_call_id],
                                source="generated",
                                expires_at=thread.expires_at,
                            )
                        )
                    except Exception:
                        logger.warning("Failed to persist generated tool image context", exc_info=True)
                self.history.add_message(
                    thread,
                    ConversationMessage(
                        id=thread.next_message_id(),
                        role="tool",
                        author=Actor(name="Tool", is_bot=True),
                        content=self._chat_content_text(protocol_message.content),
                        attachments=attachments,
                        created_at=datetime.now(),
                        metadata={
                            "tool_call_id": protocol_message.tool_call_id,
                            "name": protocol_message.name,
                        },
                    ),
                )

    def _chat_content_text(self, content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return str(content or "")

    async def _resolve_thread(
        self,
        incoming: IncomingMessage,
        runtime: ConversationRuntime,
    ) -> tuple[Optional[Thread], Optional[str], bool]:
        if incoming.is_command:
            thread = self.history.create_thread(incoming.chat_id)
            reply_to_id = None
            if incoming.referenced_message and not incoming.referenced_message.is_bot:
                referenced = self._message_from_reference(thread, incoming.referenced_message, "m0")
                self.history.add_message(thread, referenced)
                reply_to_id = referenced.id
            return thread, reply_to_id, True

        if incoming.is_reply_to_bot and incoming.referenced_message:
            ref = incoming.referenced_message.telegram_ref
            thread, logical_reply_id = self.history.find_by_telegram_ref(ref.chat_id, ref.message_id)
            if thread:
                if thread.state == "awaiting_expiration_reply":
                    notice_id = thread.metadata.get("expiration_notice_message_id")
                    if logical_reply_id == notice_id:
                        thread.state = "active"
                        return thread, logical_reply_id, True
                    return thread, logical_reply_id, False
                return thread, logical_reply_id, True
            return await self._start_expired_thread(incoming, runtime)
        return None, None, False

    async def _start_expired_thread(
        self,
        incoming: IncomingMessage,
        runtime: ConversationRuntime,
    ) -> tuple[Thread, Optional[str], bool]:
        thread = self.history.create_thread(incoming.chat_id, state="awaiting_expiration_reply")
        reply_to_id = None
        if incoming.referenced_message:
            referenced = self._message_from_reference(thread, incoming.referenced_message, "m0")
            self.history.add_message(thread, referenced)
            reply_to_id = referenced.id
        self._append_user_message(thread, incoming, reply_to_id)
        notice_refs = await self.delivery.send_text(
            runtime.bot,
            incoming.chat_id,
            self.config.telegram.expired_thread_notice,
            reply_to_message_id=incoming.telegram_message_id,
        )
        notice = ConversationMessage(
            id=thread.next_message_id(),
            role="assistant",
            author=Actor(id=getattr(runtime.bot, "id", None), name=self.config.bot.name, is_bot=True),
            content=self.config.telegram.expired_thread_notice,
            reply_to=thread.messages[-1].id if thread.messages else None,
            telegram_refs=notice_refs,
            created_at=datetime.now(),
            metadata={"kind": "expiration_notice"},
        )
        self.history.add_message(thread, notice)
        thread.metadata["expiration_notice_message_id"] = notice.id
        self.history.save()
        return thread, notice.id, False

    def _append_user_message(
        self,
        thread: Thread,
        incoming: IncomingMessage,
        reply_to_id: Optional[str],
    ) -> ConversationMessage:
        message = ConversationMessage(
            id=thread.next_message_id(),
            role="user",
            author=self._actor_for_user(incoming.user_id, incoming.user_name, is_bot=False),
            content=incoming.text or ("[Image]" if incoming.attachments else "Hello!"),
            reply_to=reply_to_id,
            telegram_refs=incoming.telegram_refs or [TelegramRef(chat_id=incoming.chat_id, message_id=incoming.telegram_message_id)],
            attachments=incoming.attachments,
            created_at=incoming.created_at,
            metadata={"telegram_user_name": incoming.user_name},
        )
        self.history.add_message(thread, message)
        return message

    def _message_from_reference(
        self,
        thread: Thread,
        reference: ReferencedMessage,
        message_id: str,
    ) -> ConversationMessage:
        return ConversationMessage(
            id=message_id,
            role="assistant" if reference.is_bot else "user",
            author=self._actor_for_user(reference.author_id, reference.author_name, is_bot=reference.is_bot),
            content=reference.text or ("[Image]" if reference.attachments else ""),
            telegram_refs=[reference.telegram_ref],
            attachments=reference.attachments,
            created_at=reference.created_at,
            metadata={"telegram_user_name": reference.author_name},
        )

    def _actor_for_user(self, user_id: Optional[int], telegram_name: str, is_bot: bool) -> Actor:
        if is_bot:
            return Actor(id=user_id, name=self.config.bot.name, is_bot=True)
        if user_id is not None and user_id in self.memory_manager.user_info:
            return Actor(id=user_id, name=self.memory_manager.user_info[user_id].name, is_bot=False)
        if user_id is not None:
            return Actor(id=user_id, name=f"user-{user_id}", is_bot=False)
        return Actor(id=None, name="user-unknown", is_bot=False)

    async def _memory_context(
        self,
        incoming: IncomingMessage,
        bot: Any = None,
        thread: Optional[Thread] = None,
    ) -> Dict[str, str]:
        query_text = incoming.text or ""
        query_embeddings = None
        try:
            query_input: Any = query_text
            if incoming.attachments and self.config.embedding_backend.supports_multimodal:
                parts = [{"type": "text", "text": query_text}]
                for attachment in incoming.attachments:
                    data_uri = self.history.attachment_store.read_data_uri(attachment)
                    if data_uri:
                        parts.append({"type": "image_url", "image_url": {"url": data_uri}})
                query_input = parts
            should_embed = bool(query_text or incoming.attachments)
            query_embeddings = await self.memory_manager.get_embeddings([query_input]) if should_embed else None
            if should_embed and not query_embeddings:
                details = (
                    "Embedding endpoint returned an empty result. Falling back to text-only memory retrieval.\n"
                    f"chat_id={incoming.chat_id}, user_id={incoming.user_id}, text_preview={query_text[:300]!r}"
                )
                logger.warning(details)
                if bot is not None:
                    await self.reporter.report(
                        bot,
                        "Embedding retrieval returned empty result",
                        details,
                        context_preview=self._thread_preview(thread) if thread else "",
                    )
                query_embeddings = None
        except Exception as exc:
            details = (
                "Failed to fetch query embedding; falling back to text-only memory retrieval.\n"
                f"error={exc!r}\nchat_id={incoming.chat_id}, user_id={incoming.user_id}, text_preview={query_text[:300]!r}"
            )
            logger.warning(details, exc_info=True)
            if bot is not None:
                await self.reporter.report(
                    bot,
                    "Embedding retrieval failed",
                    details,
                    context_preview=self._thread_preview(thread) if thread else "",
                )
            query_embeddings = None
        short_mem = await self.memory_manager.get_short_term_str(query_text, query_embeddings=query_embeddings)
        long_mem = await self.memory_manager.get_long_term_str(query_text, query_embeddings=query_embeddings)
        knowledges = self.memory_manager.get_all_knowledges_str()
        user_info = await self.memory_manager.get_user_info_str(
            query_text,
            incoming.user_id,
            query_embeddings=query_embeddings,
        )
        return {
            "knowledges": knowledges,
            "long_term_memory": long_mem,
            "short_term_memory": short_mem,
            "known_user_info": user_info,
        }

    def _skill_list(self) -> str:
        if not self.config.skills.enabled:
            return "Skills feature is disabled."
        root = self.config.skills.root_path
        if not os.path.isdir(root):
            return "No skills available."
        skill_names = [os.path.splitext(name)[0] for name in os.listdir(root) if name.endswith(".md")]
        return "\n".join(f"- {name}" for name in sorted(skill_names)) or "No skills available."

    def _thread_preview(self, thread: Thread) -> str:
        return "\n".join(
            f"{message.id} {message.role} {message.author.name}: {message.content[:200]}"
            for message in thread.messages[-20:]
        )
