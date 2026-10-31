import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict

from telegram import User

from pepperbot.config import Config
from pepperbot.conversation.models import IncomingMessage
from pepperbot.conversation.service import ConversationRuntime, ConversationService
from pepperbot.history.models import Actor, ConversationMessage, Thread
from pepperbot.moderation.models import CaseOutcome, CaseStatus, Classification, ReviewDecision, SpamCase, SpamCaseStore, parse_decision, utc_now
from pepperbot.providers.base import ChatMessage, ChatProvider, ChatRequest
from pepperbot.telegram.user_info import TelegramUserInfo

logger = logging.getLogger(__name__)


class SpamModeration:
    def __init__(
        self, config: Config, conversation: ConversationService, tool_provider: ChatProvider,
        user_info: TelegramUserInfo, runtime_factory: Callable[[Any, int], ConversationRuntime],
    ):
        self.config = config
        self.settings = config.spam_detection
        self.conversation = conversation
        self.history = conversation.history
        self.tool_provider = tool_provider
        self.user_info = user_info
        self.runtime_factory = runtime_factory
        self.store = SpamCaseStore(self.settings.state_path)
        self._locks: Dict[str, asyncio.Lock] = {}
        # Fail startup on a missing prompt instead of silently skipping moderation.
        for path in (
            self.settings.classifier_prompt_path, self.settings.challenge_prompt_path,
            self.settings.final_warning_prompt_path, self.settings.review_prompt_path, self.settings.outcome_prompt_path,
        ):
            Path(path).read_text(encoding="utf-8")

    def allowed_chat(self, chat_id: int) -> bool:
        return chat_id < 0 and (not self.config.bot.chat_whitelist or chat_id in self.config.bot.chat_whitelist)

    async def _bot_is_admin(self, bot, chat_id: int) -> bool:
        member = await bot.get_chat_member(chat_id=chat_id, user_id=bot.id)
        return member.status == "administrator"

    def _stop_without_admin(self, case: SpamCase, queue) -> None:
        # Stop silently: moderation must not generate group messages without admin
        # status. Preserve an already-completed action if only its announcement was pending.
        if case.outcome is None:
            case.outcome = "cancelled"
            case.outcome_reason = "Pepper is no longer an administrator in this chat."
        self._finish(case, case.outcome)
        if queue is not None:
            for job in queue.get_jobs_by_name(f"spam-review:{case.id}"):
                job.schedule_removal()
        logger.info("Stopped spam case %s: Pepper is not an admin in %s", case.id, case.chat_id)

    async def restore(self, application) -> None:
        if application.job_queue is None:
            raise RuntimeError("Spam detection requires the Telegram job queue")
        admin_chats: Dict[int, bool] = {}
        for case in self.store.cases.values():
            if not case.active or not self.allowed_chat(case.chat_id):
                continue
            try:
                if case.chat_id not in admin_chats:
                    admin_chats[case.chat_id] = await self._bot_is_admin(application.bot, case.chat_id)
            except Exception:
                logger.warning("Could not verify bot admin status while restoring spam case %s", case.id, exc_info=True)
                self._schedule(application.job_queue, case, retry=True)
                continue
            if not admin_chats[case.chat_id]:
                self._stop_without_admin(case, application.job_queue)
                continue
            if case.thread_id:
                thread = self.history.get_thread(case.thread_id)
                if thread:
                    thread.metadata["moderation_pending"] = True
            # Give polling a chance to ingest queued replies after downtime before
            # reviewing an overdue case as "no response".
            self._schedule(application.job_queue, case, minimum_delay=30)

    def _schedule(self, queue, case: SpamCase, retry: bool = False, minimum_delay: float = 0) -> None:
        if queue is None:
            raise RuntimeError("Spam detection requires the Telegram job queue")
        name = f"spam-review:{case.id}"
        for job in queue.get_jobs_by_name(name):
            job.schedule_removal()
        delay = 60.0 if retry else 0.0
        deadline = case.final_deadline if case.status == "final_pending" else case.deadline
        if not retry and case.status in {"pending", "final_pending"} and deadline:
            delay = max(0.0, (deadline - utc_now()).total_seconds())
        queue.run_once(self.process_case, max(minimum_delay, delay), data=case.id, chat_id=case.chat_id, name=name)

    async def member_joined(self, user, chat_id: int, joined_at: datetime, join_info: dict, context) -> None:
        if not self.allowed_chat(chat_id) or user.id == context.bot.id or user.id in self.config.admin.ids:
            return
        if joined_at.tzinfo is None:
            joined_at = joined_at.replace(tzinfo=timezone.utc)
        # A service message and a chat_member update can describe the same join.
        for existing in self.store.cases.values():
            if existing.chat_id == chat_id and existing.user_id == user.id:
                if existing.active or abs((existing.joined_at - joined_at).total_seconds()) < 5:
                    return
        try:
            if not await self._bot_is_admin(context.bot, chat_id):
                return
        except Exception:
            # Unknown permissions must not start profile lookups, LLM calls or jobs.
            logger.warning("Skipping spam detection: could not verify bot admin status in %s", chat_id, exc_info=True)
            return
        case = SpamCase(
            id=str(uuid.uuid4()), chat_id=chat_id, user_id=user.id,
            user=user.to_dict(), joined_at=joined_at, join_info=join_info,
        )
        self.store.cases[case.id] = case
        self.store.save()
        self._schedule(context.job_queue, case)

    async def observe(self, message) -> None:
        if not message or not message.from_user or not self.allowed_chat(message.chat_id):
            return
        # Join/leave and other service messages are not answers to a challenge.
        if not (message.text or message.caption or message.effective_attachment):
            return
        case = self.store.active_for(message.chat_id, message.from_user.id)
        if not case:
            return
        observation = {"message_id": message.message_id, "observed_at": utc_now().isoformat(), "message": message.to_dict()}
        for index, previous in enumerate(case.observations):
            if previous["message_id"] == message.message_id:
                case.observations[index] = observation
                break
        else:
            case.observations.append(observation)
        self.store.save()

    async def process_case(self, context) -> None:
        case = self.store.cases.get(context.job.data)
        if not case or not case.active or not self.allowed_chat(case.chat_id):
            return
        async with self._locks.setdefault(case.id, asyncio.Lock()):
            try:
                # Recheck queued/restored work, since the bot may have been demoted
                # after the join event or between verification deadlines.
                if not await self._bot_is_admin(context.bot, case.chat_id):
                    self._stop_without_admin(case, context.job_queue)
                    return
                if case.status == "screening":
                    await self._classify(case, context)
                if case.status == "challenging":
                    await self._challenge(case, context)
                if case.status == "pending":
                    if case.deadline is None:
                        raise RuntimeError("Moderation case has no review deadline")
                    if utc_now() < case.deadline:
                        self._schedule(context.job_queue, case)
                        return
                    await self._warn(case, context)
                if case.status == "final_pending":
                    if case.final_deadline is None or case.warned_at is None:
                        raise RuntimeError("Moderation case has no delivered final warning or final deadline")
                    if utc_now() < case.final_deadline:
                        self._schedule(context.job_queue, case)
                        return
                    await self._review(case, context)
                if case.status == "ban_pending":
                    await self._ban(case, context.bot)
                if case.status == "announcing":
                    await self._announce(case, context)
                case.last_error = ""
                self.store.save()
            except Exception as exc:
                logger.exception("Spam case %s failed", case.id)
                case.last_error = str(exc)
                self.store.save()
                await self.conversation.reporter.report(
                    context.bot, "Spam moderation failed",
                    f"case={case.id}, chat={case.chat_id}, user={case.user_id}, phase={case.status}\n{exc!r}",
                )
                if case.active:
                    self._schedule(context.job_queue, case, retry=True)

    async def _classify(self, case: SpamCase, context) -> None:
        user = User.de_json(case.user, context.bot)
        profile = await self.user_info.fetch(
            context.bot, case.chat_id, case.user_id, vision=self.config.tool_model.supports_vision, user=user,
        )
        profile.data["join"] = {"joined_at": case.joined_at.isoformat(), **case.join_info}
        case.profile = profile.data
        status = profile.data.get("membership", {}).get("status")
        if status in {"administrator", "creator"}:
            self._finish(case, "cleared")
            return
        result = await self.tool_provider.complete(ChatRequest(
            model=self.config.tool_model.model or self.config.chat_model(),
            messages=[
                ChatMessage(role="system", content=Path(self.settings.classifier_prompt_path).read_text(encoding="utf-8")),
                ChatMessage(role="user", content=profile.content()),
            ],
            allow_tools=False, temperature=0, max_tokens=1000,
        ))
        classification = parse_decision(result.text, Classification)
        case.classification_reason = classification.reason
        if not classification.suspected:
            self._finish(case, "cleared")
            return

        thread = self.history.create_thread(case.chat_id)
        thread.metadata.update({"moderation_case_id": case.id, "moderation_pending": True})
        self.history.add_message(thread, ConversationMessage(
            id=thread.next_message_id(), role="user", author=Actor(name="Telegram profile", is_bot=True),
            content=json.dumps(profile.data, ensure_ascii=False, default=str), created_at=datetime.now(),
            attachments=[self.history.attachment_store.save_data_uri(
                image, source="telegram", expires_at=thread.expires_at,
            ) for image in profile.images],
            metadata={"kind": "moderation_profile"},
        ))
        case.thread_id = thread.id
        case.status = "challenging"
        self.history.save()
        self.store.save()

    def _thread(self, case: SpamCase) -> Thread:
        thread = self.history.get_thread(case.thread_id)
        if thread is None:
            raise RuntimeError(f"Moderation thread {case.thread_id} is missing; cannot review without its evidence")
        return thread

    async def _challenge(self, case: SpamCase, context) -> None:
        thread = self._thread(case)
        replies = [message for message in thread.messages if message.metadata.get("kind") == "moderation_challenge_reply"]
        if not replies:
            evidence = json.dumps({
                "target_user_id": case.user_id,
                "classification_reason": case.classification_reason,
                "review_delay_minutes": self.settings.review_delay_minutes,
            }, ensure_ascii=False)
            await self.conversation.system_turn(
                thread, Path(self.settings.challenge_prompt_path).read_text(encoding="utf-8"), evidence,
                self.runtime_factory(context, case.chat_id), user_id=case.user_id,
                mention_user=User.de_json(case.user, context.bot),
            )
            replies = [message for message in thread.messages if message.metadata.get("kind") == "moderation_challenge_reply"]
        if not replies:
            raise RuntimeError("The clarification message was not delivered")
        case.challenged_at = replies[-1].created_at.astimezone(timezone.utc)
        case.deadline = case.challenged_at + timedelta(minutes=self.settings.review_delay_minutes)
        case.status = "pending"
        self.store.save()

    async def _review(self, case: SpamCase, context) -> None:
        thread = self._thread(case)
        async with self.conversation.thread_lock(thread.id):
            prompt = Path(self.settings.review_prompt_path).read_text(encoding="utf-8")
            instruction_message = ConversationMessage(
                id=thread.next_message_id(), role="system", author=Actor(name="System", is_bot=True),
                content=prompt, created_at=datetime.now(), metadata={"kind": "moderation_review"},
            )
            self.history.add_message(thread, instruction_message)
            after_challenge = [item for item in case.observations
                               if datetime.fromisoformat(item["observed_at"]) >= case.challenged_at]
            evidence = ConversationMessage(
                id=thread.next_message_id(), role="user", author=Actor(name="Moderation evidence", is_bot=True),
                content=json.dumps({
                    "target_user_id": case.user_id, "chat_id": case.chat_id,
                    "profile": case.profile, "classification_reason": case.classification_reason,
                    "challenged_at": case.challenged_at.isoformat(), "deadline": case.deadline.isoformat(),
                    "warned_at": case.warned_at.isoformat(), "final_deadline": case.final_deadline.isoformat(),
                    "target_messages_after_challenge": len(after_challenge),
                    "target_messages_after_final_warning": sum(
                        datetime.fromisoformat(item["observed_at"]) >= case.warned_at for item in after_challenge
                    ),
                    "observed_target_messages": case.observations,
                }, ensure_ascii=False, default=str),
                attachments=[attachment for message in thread.messages
                             if message.metadata.get("kind") == "moderation_profile"
                             for attachment in message.attachments],
                created_at=datetime.now(), metadata={"kind": "moderation_review_evidence"},
            )
            instruction_message.metadata["instruction_for"] = evidence.id
            self.history.add_message(thread, evidence)
            self.history.save()
            incoming = IncomingMessage(
                chat_id=case.chat_id, telegram_message_id=0, user_id=case.user_id, user_name="",
                text=evidence.content, created_at=datetime.now(), is_command=False, is_reply_to_bot=False,
            )
            sections = await self.conversation._memory_context(incoming, context.bot, thread)
            built = await self.conversation.context_builder.build(
                thread, evidence.id, memory_sections=sections, skill_list=self.conversation._skill_list(),
            )
            # Keep the persona/thread, but override the template's Telegram-output
            # contract for this internal decision. No tools run during adjudication.
            result = await self.conversation.chat_runner.provider.complete(ChatRequest(
                model=self.config.chat_model(),
                messages=[*built.messages, ChatMessage(role="system", content=prompt)],
                allow_tools=False, temperature=0, max_tokens=1000,
                reasoning_effort=self.config.model_params.reasoning_effort,
            ))
            decision = parse_decision(result.text, ReviewDecision)
            self.history.add_message(thread, ConversationMessage(
                id=thread.next_message_id(), role="assistant", author=Actor(name=self.config.bot.name, is_bot=True),
                content=decision.model_dump_json(), created_at=datetime.now(),
                metadata={"kind": "moderation_decision"},
            ))
            self.history.save()
            case.decision = decision
            if decision.is_spam:
                case.status = "ban_pending"
                self.store.save()
            else:
                self._queue_announcement(case, "cleared", decision.reason)

    async def _ban(self, case: SpamCase, bot) -> None:
        if case.user_id in self.config.admin.ids or case.user_id == bot.id:
            self._queue_announcement(case, "cancelled", "The target is a configured admin or Pepper itself.")
            return
        member = await bot.get_chat_member(chat_id=case.chat_id, user_id=case.user_id)
        if member.status in {"administrator", "creator"}:
            self._queue_announcement(case, "cancelled", "The member is now a group administrator or owner.")
            return
        me = await bot.get_chat_member(chat_id=case.chat_id, user_id=bot.id)
        if me.status != "administrator" or not me.can_restrict_members:
            raise RuntimeError("Pepper requires the Ban/Restrict Users administrator right")
        if not await bot.ban_chat_member(chat_id=case.chat_id, user_id=case.user_id, revoke_messages=True):
            raise RuntimeError("Telegram did not confirm the ban")
        self._queue_announcement(case, "banned", case.decision.reason)
        logger.info("Banned spam member %s in %s: %s", case.user_id, case.chat_id, case.decision.reason)

    async def _warn(self, case: SpamCase, context) -> None:
        thread = self._thread(case)
        kind = "moderation_final_warning"
        replies = [message for message in thread.messages if message.metadata.get("kind") == f"{kind}_reply"]
        if not replies:
            evidence = json.dumps({
                "target_user_id": case.user_id,
                "classification_reason": case.classification_reason,
                "initial_deadline": case.deadline.isoformat(),
                "final_review_delay_minutes": self.settings.final_review_delay_minutes,
                "observed_target_messages": case.observations,
            }, ensure_ascii=False, default=str)
            await self.conversation.system_turn(
                thread, Path(self.settings.final_warning_prompt_path).read_text(encoding="utf-8"), evidence,
                self.runtime_factory(context, case.chat_id), user_id=case.user_id,
                mention_user=User.de_json(case.user, context.bot), kind=kind, allow_tools=False,
            )
            replies = [message for message in thread.messages if message.metadata.get("kind") == f"{kind}_reply"]
        if not replies:
            raise RuntimeError("The final warning was not delivered")
        case.warned_at = replies[-1].created_at.astimezone(timezone.utc)
        case.final_deadline = case.warned_at + timedelta(minutes=self.settings.final_review_delay_minutes)
        case.status = "final_pending"
        case.last_error = ""
        self.store.save()

    def _queue_announcement(self, case: SpamCase, outcome: CaseOutcome, reason: str) -> None:
        # Checkpoint the completed action before generating/sending its announcement.
        # Delivery retries and restarts must not repeat the review or moderation action.
        case.outcome = outcome
        case.outcome_reason = reason
        case.status = "announcing"
        self.store.save()

    async def _announce(self, case: SpamCase, context) -> None:
        if case.outcome is None:
            raise RuntimeError("Moderation case has no completed outcome to announce")
        thread = self._thread(case)
        kind = "moderation_outcome"
        if not any(message.metadata.get("kind") == f"{kind}_reply" for message in thread.messages):
            evidence = json.dumps({
                "target_user_id": case.user_id,
                "target_name": User.de_json(case.user, context.bot).full_name,
                "outcome": case.outcome,
                "reason": case.outcome_reason,
                "account_banned": case.outcome == "banned",
                "messages_revoked": case.outcome == "banned",
            }, ensure_ascii=False)
            await self.conversation.system_turn(
                thread, Path(self.settings.outcome_prompt_path).read_text(encoding="utf-8"), evidence,
                self.runtime_factory(context, case.chat_id), user_id=case.user_id,
                kind=kind, allow_tools=False,
            )
        self._finish(case, case.outcome)

    def _finish(self, case: SpamCase, status: CaseStatus) -> None:
        case.status = status
        if case.thread_id:
            thread = self.history.get_thread(case.thread_id)
            if thread:
                thread.metadata["moderation_pending"] = False
                self.history.touch_thread(thread)
                self.history.save()
        self.store.save()
