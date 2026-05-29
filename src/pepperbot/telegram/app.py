import asyncio
import logging
from datetime import datetime
from random import randint
from typing import Dict, List, Set, Tuple

import requests
from telegram import Update
from telegram.ext import Application, ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from memory import KnownUserInfoSafetyError, MemoryManager

from pepperbot.config import Config, load_config, load_template
from pepperbot.context.builder import ContextBuilder
from pepperbot.context.summarizer import ThreadSummarizer
from pepperbot.conversation.runner import ChatLoopRunner
from pepperbot.conversation.service import ConversationRuntime, ConversationService
from pepperbot.history.attachments import AttachmentStore
from pepperbot.history.models import Actor, ConversationMessage
from pepperbot.history.store import HistoryStore
from pepperbot.providers.factory import create_chat_provider
from pepperbot.providers.openai_chat import OpenAIChatCompletionsProvider
from pepperbot.telegram.delivery import AdminReporter, TelegramDelivery
from pepperbot.telegram.parser import UpdateParser
from pepperbot.tools.executor import ToolExecutor
from pepperbot.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class PepperBotApplication:
    def __init__(self, config_path: str = "config/config.yaml", template_path: str = "config/template.json"):
        self.config: Config = load_config(config_path)
        self.template = load_template(template_path)
        self.attachment_store = AttachmentStore(self.config.attachments.root_path)
        self.history = HistoryStore(
            "data/chat-histories.json",
            attachment_store=self.attachment_store,
            expiration_hours=self.config.context.history_expiration_hours,
        )
        try:
            self.memory = MemoryManager(
                config=self.config,
                short_term_path="data/short-term.json",
                long_term_path="data/long-term.json",
                knowledges_path="data/knowledges.json",
                user_info_path="data/known-users.yaml",
            )
        except KnownUserInfoSafetyError as exc:
            self._send_startup_admin_alert("Known user info safety guard triggered", str(exc))
            raise
        self.chat_provider = create_chat_provider(self.config)
        tool_provider = OpenAIChatCompletionsProvider(
            self.config,
            api_key=self.config.tool_model.api_key or self.config.chat_api_key(),
            base_url=self.config.tool_model.api_url or self.config.chat_api_url(),
        )
        self.summarizer = ThreadSummarizer(self.config, tool_provider)
        self.context_builder = ContextBuilder(self.config, self.template, self.attachment_store, self.summarizer)
        self.registry = ToolRegistry(self.config)
        self.executor = ToolExecutor(self.config, self.memory)
        self.delivery = TelegramDelivery(self.config)
        self.reporter = AdminReporter(self.config, self.delivery)
        self.chat_runner = ChatLoopRunner(
            self.config,
            self.chat_provider,
            self.registry,
            self.executor,
            self.context_builder,
        )
        self.service = ConversationService(
            self.config,
            self.history,
            self.memory,
            self.context_builder,
            self.chat_runner,
            self.delivery,
            self.reporter,
        )
        self.parser = UpdateParser(self.config, self.attachment_store)
        self.bot_username = None
        self.blacklist: Set[int] = set()
        self.media_group_buffers: Dict[Tuple[int, str], List[Update]] = {}
        self.media_group_contexts: Dict[Tuple[int, str], ContextTypes.DEFAULT_TYPE] = {}
        self.media_group_tasks: Dict[Tuple[int, str], asyncio.Task] = {}

    def _send_startup_admin_alert(self, title: str, details: str) -> None:
        if not self.config.admin.report_major_failures:
            return
        chat_ids = self.config.admin.report_chat_ids or self.config.admin.ids
        if not chat_ids:
            return
        text = f"[PepperBot major failure]\n{title}\n\n{details}"
        for chat_id in chat_ids:
            try:
                response = requests.post(
                    f"https://api.telegram.org/bot{self.config.bot.token}/sendMessage",
                    data={"chat_id": chat_id, "text": text},
                    timeout=10,
                )
                response.raise_for_status()
            except Exception:
                logger.exception("Failed to send startup admin alert to %s", chat_id)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.message:
            await update.message.reply_text(
                f"Hello! I'm {self.config.bot.name}. Mention me or reply to my messages to chat!"
            )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.message:
            await update.message.reply_text(
                f"I am {self.config.bot.name}. Use /{self.config.bot.command} to wake me up."
            )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.message and update.message.media_group_id:
            self._buffer_media_group(update, context)
            return

        incoming = await self.parser.parse(update, context, self.bot_username)
        if not incoming:
            return
        await self._handle_incoming(incoming, context)

    def _buffer_media_group(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.effective_chat or not update.message.media_group_id:
            return
        key = (update.effective_chat.id, update.message.media_group_id)
        self.media_group_buffers.setdefault(key, []).append(update)
        self.media_group_contexts[key] = context
        existing_task = self.media_group_tasks.get(key)
        if existing_task and not existing_task.done():
            existing_task.cancel()
        self.media_group_tasks[key] = asyncio.create_task(self._flush_media_group_after_delay(key))

    async def _flush_media_group_after_delay(self, key: Tuple[int, str]) -> None:
        try:
            await asyncio.sleep(self.config.telegram.media_group_wait_seconds)
        except asyncio.CancelledError:
            return
        updates = self.media_group_buffers.pop(key, [])
        context = self.media_group_contexts.pop(key, None)
        self.media_group_tasks.pop(key, None)
        try:
            if not updates or context is None:
                return
            messages = [update.message for update in updates if update.message]
            incoming = await self.parser.parse_messages(messages, context, self.bot_username)
            if not incoming:
                return
            await self._handle_incoming(incoming, context)
        except Exception as exc:
            logger.exception("Failed to process media group %s", key)
            if context is not None:
                await self.reporter.report(context.bot, "Media group processing failed", repr(exc))

    async def _handle_incoming(self, incoming, context: ContextTypes.DEFAULT_TYPE):
        if self.config.black_list.enable and incoming.user_id in self.blacklist:
            blocked = self.config.black_list.blocked_messages[
                randint(0, len(self.config.black_list.blocked_messages) - 1)
            ]
            await self.delivery.send_text(context.bot, incoming.chat_id, blocked, incoming.telegram_message_id)
            return

        runtime = ConversationRuntime(
            bot=context.bot,
            schedule_func=lambda delay, title, content: self.schedule_task(context, delay, title, content, incoming.chat_id),
            list_func=lambda: self.get_scheduled_tasks(context),
            block_user_func=self.add_blacklist,
            send_typing_func=lambda: context.bot.send_chat_action(chat_id=incoming.chat_id, action="typing"),
        )
        await self.service.handle_incoming(incoming, runtime)

    async def schedule_task(self, context, delay_minutes: int, title: str, content: str, chat_id: int) -> str:
        if delay_minutes < 0:
            return "Error: Delay cannot be negative."
        if delay_minutes > 1440:
            return "Error: Delay cannot exceed 1440 minutes (24 hours)."
        if not context.job_queue:
            return "Error: Job queue is not available."
        context.job_queue.run_once(
            self.execute_task_callback,
            delay_minutes * 60,
            chat_id=chat_id,
            data={"title": title, "content": content},
        )
        return f"Task '{title}' scheduled in {delay_minutes} minutes."

    async def get_scheduled_tasks(self, context) -> str:
        if not context.job_queue:
            return "No scheduled tasks."
        tasks = []
        for job in context.job_queue.jobs():
            if not self._is_scheduled_task_job(job) or not job.next_t:
                continue
            remaining = job.next_t - datetime.now(job.next_t.tzinfo) if job.next_t.tzinfo else job.next_t - datetime.utcnow()
            tasks.append(f"- {job.data.get('title', 'Untitled')} (in {int(remaining.total_seconds() / 60)} min)")
        return "Scheduled Tasks:\n" + "\n".join(tasks) if tasks else "No scheduled tasks."

    def _is_scheduled_task_job(self, job) -> bool:
        callback = getattr(job, "callback", None)
        return (
            getattr(callback, "__self__", None) is self
            and getattr(callback, "__func__", None) is self.execute_task_callback.__func__
        )

    async def execute_task_callback(self, context: ContextTypes.DEFAULT_TYPE):
        job = context.job
        chat_id = job.chat_id
        data = job.data or {}
        title = data.get("title", "Untitled")
        content = data.get("content", "")
        text = f"Scheduled Task Triggered:\nTitle: {title}\nContent: {content}"
        try:
            if chat_id is None:
                raise ValueError("Scheduled task job has no chat_id")
            # Scheduled tasks are not Telegram replies, so this path sends directly to the chat.
            thread = self.history.create_thread(chat_id)
            message = ConversationMessage(
                id=thread.next_message_id(),
                role="user",
                author=Actor(name="System", is_bot=True),
                content=text,
                created_at=datetime.now(),
                metadata={"kind": "scheduled_task_trigger", "title": title},
            )
            self.history.add_message(thread, message)
            self.history.save()
            memory_sections = await self.service._memory_context(
                type("ScheduledIncoming", (), {"text": text, "attachments": [], "user_id": None, "chat_id": chat_id})(),
                context.bot,
                thread,
            )
            context_result = await self.context_builder.build(
                thread,
                message.id,
                memory_sections=memory_sections,
                skill_list=self.service._skill_list(),
            )
            tool_runtime = self._tool_runtime_for_task(context, chat_id, thread)
            response_text, protocol_messages = await self.chat_runner.run(context_result.messages, tool_runtime)
            telegram_refs = []
            generated_attachments = []
            if tool_runtime.generated_images:
                telegram_refs.extend(await self.delivery.send_images(context.bot, chat_id, tool_runtime.generated_images))
                for image in tool_runtime.generated_images:
                    generated_attachments.append(
                        self.attachment_store.save_data_uri(image, source="generated", expires_at=thread.expires_at)
                    )
            clean_text = self.service.sanitizer.clean(response_text) or self.config.response.fallback_text
            telegram_refs.extend(await self.delivery.send_text(context.bot, chat_id, clean_text))
            self.service._append_protocol_messages(thread, protocol_messages, tool_runtime)
            self.history.add_message(
                thread,
                ConversationMessage(
                    id=thread.next_message_id(),
                    role="assistant",
                    author=Actor(id=getattr(context.bot, "id", None), name=self.config.bot.name, is_bot=True),
                    content=clean_text,
                    reply_to=message.id,
                    telegram_refs=telegram_refs,
                    attachments=generated_attachments,
                    created_at=datetime.now(),
                ),
            )
            self.history.save()
        except Exception as exc:
            logger.exception("Scheduled task failed")
            await self.reporter.report(context.bot, "Scheduled task failed", repr(exc), context_preview=text)
            if chat_id is not None:
                await self.delivery.send_text(context.bot, chat_id, self.config.response.fallback_text)

    def _tool_runtime_for_task(self, context, chat_id: int, thread):
        from pepperbot.tools.executor import ToolRuntime

        return ToolRuntime(
            chat_id=chat_id,
            thread=thread,
            attachment_store=self.attachment_store,
            schedule_func=lambda delay, title, content: self.schedule_task(context, delay, title, content, chat_id),
            list_func=lambda: self.get_scheduled_tasks(context),
            block_user_func=self.add_blacklist,
        )

    async def add_blacklist(self, user_id: int, duration_minutes: int) -> str:
        if not self.config.black_list.enable:
            return "Blacklist feature is disabled."
        if user_id in self.config.admin.ids:
            return "Error: Cannot block an admin user."
        if duration_minutes > self.config.black_list.max_minute:
            return f"Error: Maximum block duration is {self.config.black_list.max_minute} minutes."
        if duration_minutes <= 0:
            return "Error: Duration must be a positive integer."
        if user_id in self.blacklist:
            return "User is already blacklisted."
        self.blacklist.add(user_id)
        asyncio.get_event_loop().call_later(duration_minutes * 60, self.remove_blacklist, user_id)
        return f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] User {user_id} has been blacklisted for {duration_minutes} minutes."

    def remove_blacklist(self, user_id: int):
        self.blacklist.discard(user_id)

    async def scheduled_maintenance(self, context: ContextTypes.DEFAULT_TYPE):
        self.history.clean_expired()
        expired_events = self.memory.check_expirations(self.config.memory.short.expiration_days)
        if expired_events:
            # Preserve the existing memory consolidation behavior for now.
            from llm import LLMClient

            await LLMClient(self.config, self.memory).consolidate_memory(expired_events, "")
            await self.memory.remove_short_term_events(expired_events)
        self.history.save()

    async def shutdown(self, application: Application):
        self.history.save()
        self.memory._save_short_term()
        self.memory._save_long_term()
        self.memory._save_knowledges()
        try:
            self.memory._save_user_info()
        except KnownUserInfoSafetyError as exc:
            await self.reporter.report(application.bot, "Known user info safety guard triggered", str(exc))
            raise
        self.memory._save_state()

    async def post_init(self, application: Application):
        bot_info = await application.bot.get_me()
        self.bot_username = bot_info.username
        logger.info("Bot ID is %s, username is %s", bot_info.id, bot_info.username)
        await self.refresh_known_user_telegram_usernames(application.bot)

    async def refresh_known_user_telegram_usernames(self, bot) -> None:
        group_chat_ids = self._startup_username_refresh_chat_ids()
        if not group_chat_ids:
            logger.info("Skipping Telegram username startup refresh; no negative group chat IDs configured or in history")
            return

        updated = 0
        refreshed = 0
        unresolved = 0
        refreshed_user_ids: Set[int] = set()
        for chat_id in group_chat_ids:
            try:
                administrators = await bot.get_chat_administrators(chat_id=chat_id)
            except Exception as exc:
                logger.debug("Could not fetch Telegram administrators from chat %s: %s", chat_id, exc, exc_info=True)
                continue
            for administrator in administrators:
                user = getattr(administrator, "user", None)
                user_id = getattr(user, "id", None)
                if user is None or user_id not in self.memory.user_info:
                    continue
                if user_id in refreshed_user_ids:
                    continue
                refreshed_user_ids.add(user_id)
                refreshed += 1
                try:
                    if await self.memory.update_user_telegram_username(user_id, getattr(user, "username", None)):
                        updated += 1
                except Exception:
                    logger.warning("Failed to store Telegram username for known user %s", user_id, exc_info=True)

        for user_id in list(self.memory.user_info.keys()):
            if user_id in refreshed_user_ids:
                continue
            found = False
            for chat_id in group_chat_ids:
                try:
                    member = await bot.get_chat_member(chat_id=chat_id, user_id=user_id)
                except Exception as exc:
                    logger.debug(
                        "Could not refresh Telegram username for known user %s from chat %s: %s",
                        user_id,
                        chat_id,
                        exc,
                        exc_info=True,
                    )
                    continue
                user = getattr(member, "user", None)
                if user is None:
                    logger.debug("Telegram get_chat_member returned no user for known user %s in chat %s", user_id, chat_id)
                    continue
                found = True
                try:
                    if await self.memory.update_user_telegram_username(user_id, getattr(user, "username", None)):
                        updated += 1
                except Exception:
                    logger.warning("Failed to store Telegram username for known user %s", user_id, exc_info=True)
                break
            if found:
                refreshed += 1
            else:
                unresolved += 1
        logger.info(
            "Telegram username startup refresh checked %s group chats; refreshed=%s updated=%s unresolved=%s",
            len(group_chat_ids),
            refreshed,
            updated,
            unresolved,
        )

    def _startup_username_refresh_chat_ids(self) -> List[int]:
        chat_ids = {chat_id for chat_id in self.config.bot.chat_whitelist if chat_id < 0}
        for thread in getattr(getattr(self.history, "data", None), "threads", {}).values():
            chat_id = getattr(thread, "chat_id", None)
            if isinstance(chat_id, int) and chat_id < 0:
                chat_ids.add(chat_id)
        return sorted(chat_ids)

    def run(self):
        application = (
            ApplicationBuilder()
            .token(self.config.bot.token)
            .post_shutdown(self.shutdown)
            .post_init(self.post_init)
            .connect_timeout(30.0)
            .read_timeout(30.0)
            .write_timeout(30.0)
            .build()
        )
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler(self.config.bot.command, self.handle_message))
        application.add_handler(MessageHandler(filters.TEXT | filters.PHOTO, self.handle_message))
        if application.job_queue:
            application.job_queue.run_repeating(self.scheduled_maintenance, interval=3600, first=10)
        logger.info("Bot started polling...")
        application.run_polling(drop_pending_updates=False)
