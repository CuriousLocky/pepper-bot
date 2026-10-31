import json
from datetime import datetime, timedelta
from itertools import count
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram import Chat, Message, User

from pepperbot.config import Config, TemplateConfig
from pepperbot.context.builder import ContextBuilder
from pepperbot.conversation.models import IncomingMessage, ReferencedMessage
from pepperbot.conversation.runner import ChatLoopRunner
from pepperbot.conversation.service import ConversationRuntime, ConversationService
from pepperbot.history.attachments import AttachmentStore
from pepperbot.history.models import TelegramRef
from pepperbot.history.store import HistoryStore
from pepperbot.moderation.models import SpamCaseStore, utc_now
from pepperbot.moderation.service import SpamModeration
from pepperbot.providers.base import ChatResult
from pepperbot.telegram.delivery import TelegramDelivery
from pepperbot.telegram.user_info import UserProfile
from pepperbot.tools.executor import ToolExecutor
from pepperbot.tools.registry import ToolRegistry


class Queue:
    def __init__(self):
        self.jobs = []

    def get_jobs_by_name(self, name):
        return [job for job in self.jobs if job.name == name]

    def run_once(self, callback, delay, **kwargs):
        job = SimpleNamespace(callback=callback, delay=delay, **kwargs)
        job.schedule_removal = lambda: self.jobs.remove(job)
        self.jobs.append(job)


class Provider:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return ChatResult(text=self.responses.pop(0))


def setup_moderation(tmp_path, classification=None, review=None, vision=True):
    config = Config(
        bot={"token": "t", "name": "Pepper", "chat_whitelist": [-100]},
        api={"url": "http://example.com/v1", "key": "k", "model": "chat-model"},
        chat_backend={"supports_vision": True},
        tool_model={"model": "classifier-model", "supports_vision": vision},
        spam_detection={"enabled": True, "review_delay_minutes": 2, "state_path": str(tmp_path / "cases.json")},
    )
    user = User(id=123, first_name="🦊 Alice", is_bot=False)
    bot = SimpleNamespace(id=999)
    bot.get_chat_member = AsyncMock(side_effect=lambda chat_id, user_id: SimpleNamespace(
        status="administrator" if user_id == bot.id else "member", can_restrict_members=True,
    ))
    bot.ban_chat_member = AsyncMock(return_value=True)
    message_ids = count(40)
    bot.send_message = AsyncMock(side_effect=lambda **kwargs: SimpleNamespace(message_id=next(message_ids)))
    user_info = SimpleNamespace(fetch=AsyncMock(return_value=UserProfile(
        {"user": user.to_dict(), "profile": {"bio": "advertising"}},
        ["data:image/jpeg;base64,aW1hZ2U="] if vision else [],
    )))
    attachments = AttachmentStore(str(tmp_path / "attachments"))
    history = HistoryStore(str(tmp_path / "history.json"), attachment_store=attachments)
    memory = SimpleNamespace(
        user_info={}, get_embeddings=AsyncMock(return_value=[[0.1]]),
        get_short_term_str=AsyncMock(return_value=""), get_long_term_str=AsyncMock(return_value=""),
        get_user_info_str=AsyncMock(return_value=""), get_all_knowledges_str=lambda: "",
    )
    template = TemplateConfig(messages=[
        {"id": "persona", "role": "system", "content": "You are Pepper. TEST PERSONA."},
        {"id": "thread", "type": "thread_context"},
        {"id": "contract", "role": "user", "content": "Reply in telegram_reply XML."},
    ])
    builder = ContextBuilder(config, template, attachments)
    chat = Provider(
        "Please reply and explain why you joined within 2 minutes.",
        "Final warning: please reply within 1 more minute before the final spam review.",
        review or '{"is_spam":true,"reason":"Advertising and no explanation"}',
        "Verification complete.",
    )
    tool = Provider(classification or '{"suspected":true,"reason":"Advertising profile"}')
    delivery = TelegramDelivery(config)
    runner = ChatLoopRunner(config, chat, ToolRegistry(config), ToolExecutor(config, memory, user_info), builder)
    reporter = SimpleNamespace(report=AsyncMock())
    conversation = ConversationService(config, history, memory, builder, runner, delivery, reporter)
    factory = lambda context, chat_id: ConversationRuntime(bot=context.bot)
    moderation = SpamModeration(config, conversation, tool, user_info, factory)
    queue = Queue()
    context = SimpleNamespace(bot=bot, job_queue=queue)
    return SimpleNamespace(**locals())


async def start_case(env):
    await env.moderation.member_joined(env.user, -100, utc_now(), {"invite_link": "test"}, env.context)
    case = next(iter(env.moderation.store.cases.values()))
    env.context.job = SimpleNamespace(data=case.id)
    await env.moderation.process_case(env.context)
    return case


async def reach_final_deadline(env, case, moderation=None):
    case.deadline = utc_now() - timedelta(seconds=1)
    await (moderation or env.moderation).process_case(env.context)
    assert case.status == "final_pending"
    case.final_deadline = utc_now() - timedelta(seconds=1)


@pytest.mark.asyncio
async def test_full_flow_isolated_classifier_then_persona_then_ban(tmp_path):
    env = setup_moderation(tmp_path)
    case = await start_case(env)
    assert case.status == "pending"
    assert (case.deadline - case.challenged_at).total_seconds() == 120
    assert 115 <= env.queue.jobs[-1].delay <= 120
    env.user_info.fetch.assert_awaited_once()
    assert env.user_info.fetch.call_args.kwargs["vision"] is True
    request = env.tool.requests[0]
    assert request.model == "classifier-model"
    assert request.allow_tools is False
    assert len(request.messages) == 2
    assert "TEST PERSONA" not in str(request.messages)
    assert request.messages[1].content[1]["type"] == "image_url"
    assert "invite_link" in request.messages[1].content[0]["text"]
    assert "TEST PERSONA" in str(env.chat.requests[0].messages)
    sent = env.bot.send_message.call_args.kwargs
    assert sent["entities"][0].user.id == env.user.id
    assert sent["entities"][0].length == len(env.user.full_name.encode("utf-16-le")) // 2
    assert sent["text"].startswith(env.user.full_name)
    thread, _ = env.history.find_by_telegram_ref(-100, 40)
    assert thread.id == case.thread_id
    assert not env.bot.ban_chat_member.called

    case.deadline = utc_now() - timedelta(seconds=1)
    await env.moderation.process_case(env.context)
    assert case.status == "final_pending"
    assert (case.final_deadline - case.warned_at).total_seconds() == 60
    assert 55 <= env.queue.jobs[-1].delay <= 60
    assert env.bot.send_message.await_count == 2
    assert "Final warning" in env.bot.send_message.call_args.kwargs["text"]
    assert env.bot.send_message.call_args.kwargs["entities"][0].user.id == 123
    assert any(message.metadata.get("kind") == "moderation_final_warning" and message.role == "system" for message in thread.messages)
    env.bot.ban_chat_member.assert_not_awaited()
    # An early callback must neither review nor send a second warning.
    await env.moderation.process_case(env.context)
    assert env.bot.send_message.await_count == 2
    assert len(env.chat.requests) == 2

    case.final_deadline = utc_now() - timedelta(seconds=1)
    await env.moderation.process_case(env.context)
    assert case.status == "banned"
    env.bot.ban_chat_member.assert_awaited_once_with(chat_id=-100, user_id=123, revoke_messages=True)
    assert len(env.history.data.threads) == 1
    assert any(message.metadata.get("kind") == "moderation_review" and message.role == "system" for message in thread.messages)
    assert env.chat.requests[-2].model == "chat-model"
    assert env.chat.requests[-2].allow_tools is False
    assert env.chat.requests[-2].messages[-1].role == "system"
    assert env.chat.requests[-1].allow_tools is False
    assert "account_banned" in str(env.chat.requests[-1].messages)
    assert env.bot.send_message.await_count == 3
    assert env.bot.send_message.call_args.kwargs["text"] == "Verification complete."
    outcome = next(message for message in thread.messages if message.metadata.get("kind") == "moderation_outcome_evidence")
    assert json.loads(outcome.content)["outcome"] == "banned"
    assert not thread.metadata["moderation_pending"]


@pytest.mark.asyncio
async def test_clear_classification_never_activates_pepper(tmp_path):
    env = setup_moderation(tmp_path, classification='{"suspected":false,"reason":"Ordinary account"}', vision=False)
    case = await start_case(env)
    assert case.status == "cleared"
    assert env.user_info.fetch.call_args.kwargs["vision"] is False
    assert isinstance(env.tool.requests[0].messages[1].content, str)
    assert not env.chat.requests
    assert not env.history.data.threads
    env.bot.send_message.assert_not_awaited()
    env.bot.ban_chat_member.assert_not_awaited()


@pytest.mark.asyncio
async def test_observes_non_reply_and_edited_messages_from_target_only(tmp_path):
    env = setup_moderation(tmp_path, review='{"is_spam":false,"reason":"Credible explanation"}')
    case = await start_case(env)
    target = Message(message_id=80, date=utc_now(), chat=Chat(-100, "supergroup"), from_user=env.user, text="I am here to chat")
    await env.moderation.observe(target)
    edited = Message(message_id=80, date=target.date, edit_date=utc_now(), chat=target.chat, from_user=env.user, text="A friend invited me to chat")
    await env.moderation.observe(edited)
    await env.moderation.observe(Message(message_id=81, date=utc_now(), chat=target.chat, from_user=User(456, "Other", False), text="He is fine"))
    assert len(case.observations) == 1
    await reach_final_deadline(env, case)
    await env.moderation.process_case(env.context)
    evidence = next(message for message in env.history.get_thread(case.thread_id).messages if message.metadata.get("kind") == "moderation_review_evidence")
    payload = json.loads(evidence.content)
    assert payload["target_messages_after_challenge"] == 1
    assert payload["observed_target_messages"][0]["message"]["text"] == "A friend invited me to chat"
    assert case.status == "cleared"
    assert env.bot.send_message.await_count == 3
    outcome = next(message for message in env.history.get_thread(case.thread_id).messages if message.metadata.get("kind") == "moderation_outcome_evidence")
    assert json.loads(outcome.content)["account_banned"] is False
    assert json.loads(outcome.content)["messages_revoked"] is False
    env.bot.ban_chat_member.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["screening", "final_pending"])
async def test_invalid_decisions_retry_without_ban(tmp_path, phase):
    env = setup_moderation(
        tmp_path,
        classification='{"suspected":"true","reason":"bad boolean"}' if phase == "screening" else None,
        review='{"is_spam":"false","reason":"bad boolean"}',
    )
    case = await start_case(env)
    if phase == "final_pending":
        await reach_final_deadline(env, case)
        await env.moderation.process_case(env.context)
    assert case.status == phase
    assert case.last_error
    assert env.queue.jobs[-1].delay == 60
    env.bot.ban_chat_member.assert_not_awaited()
    env.reporter.report.assert_awaited()


@pytest.mark.asyncio
async def test_restart_restores_deadline_and_protects_pending_thread(tmp_path):
    env = setup_moderation(tmp_path)
    case = await start_case(env)
    thread = env.history.get_thread(case.thread_id)
    thread.expires_at = datetime.now() - timedelta(hours=1)
    thread.messages[0].attachments[0].expires_at = thread.expires_at
    assert env.history.clean_expired() == 0
    assert env.attachments.read_data_uri(thread.messages[0].attachments[0])
    restored = SpamModeration(env.config, env.conversation, env.tool, env.user_info, env.factory)
    queue = Queue()
    await restored.restore(SimpleNamespace(bot=env.bot, job_queue=queue))
    saved_case = restored.store.cases[case.id]
    assert saved_case.deadline == case.deadline
    assert queue.jobs[0].data == case.id
    assert 115 <= queue.jobs[0].delay <= 120
    assert env.bot.send_message.await_count == 1
    await reach_final_deadline(env, saved_case, moderation=restored)
    await restored.process_case(SimpleNamespace(bot=env.bot, job_queue=queue, job=SimpleNamespace(data=case.id)))
    assert saved_case.status == "banned"


@pytest.mark.asyncio
async def test_duplicate_joins_and_chat_whitelist(tmp_path):
    env = setup_moderation(tmp_path)
    case = await start_case(env)
    await env.moderation.member_joined(env.user, -100, case.joined_at, {}, env.context)
    await env.moderation.member_joined(env.user, -200, utc_now(), {}, env.context)
    await env.moderation.member_joined(env.user, 100, utc_now(), {}, env.context)
    assert len(env.moderation.store.cases) == 1
    assert env.bot.send_message.await_count == 1


@pytest.mark.asyncio
async def test_failed_challenge_has_no_deadline_and_retries(tmp_path):
    env = setup_moderation(tmp_path)
    env.bot.send_message.side_effect = RuntimeError("Telegram unavailable")
    case = await start_case(env)
    assert case.status == "challenging"
    assert case.deadline is None
    assert env.queue.jobs[-1].delay == 60
    assert SpamCaseStore(env.config.spam_detection.state_path).cases[case.id].status == "challenging"
    env.bot.ban_chat_member.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_ban_permission_keeps_decision_for_retry(tmp_path):
    env = setup_moderation(tmp_path)
    case = await start_case(env)
    env.bot.get_chat_member.side_effect = lambda chat_id, user_id: SimpleNamespace(
        status="administrator" if user_id == env.bot.id else "member", can_restrict_members=False,
    )
    await reach_final_deadline(env, case)
    await env.moderation.process_case(env.context)
    assert case.status == "ban_pending"
    assert case.decision.is_spam
    assert env.bot.send_message.await_count == 2  # No outcome claim before cleanup succeeds.
    env.bot.ban_chat_member.assert_not_awaited()
    count = len(env.chat.requests)
    env.bot.get_chat_member.side_effect = lambda chat_id, user_id: SimpleNamespace(
        status="administrator" if user_id == env.bot.id else "member", can_restrict_members=True,
    )
    await env.moderation.process_case(env.context)
    assert case.status == "banned"
    assert len(env.chat.requests) == count + 1  # Only the outcome announcement, not another decision.


@pytest.mark.asyncio
async def test_member_promoted_during_review_is_not_banned(tmp_path):
    env = setup_moderation(tmp_path)
    case = await start_case(env)
    env.bot.get_chat_member.side_effect = lambda **kwargs: SimpleNamespace(status="administrator", can_restrict_members=True)
    await reach_final_deadline(env, case)
    await env.moderation.process_case(env.context)
    assert case.status == "cancelled"
    assert case.outcome == "cancelled"
    assert env.bot.send_message.await_count == 3
    env.bot.ban_chat_member.assert_not_awaited()


@pytest.mark.asyncio
async def test_reply_continues_verification_thread_without_replaying_internal_instructions(tmp_path):
    env = setup_moderation(tmp_path, review='{"is_spam":false,"reason":"Legitimate member"}')
    case = await start_case(env)
    env.chat.responses.insert(0, "Thanks for explaining!")
    incoming = IncomingMessage(
        chat_id=-100, telegram_message_id=80, user_id=123, user_name="Alice",
        text="My friend invited me here", created_at=utc_now(), is_command=False, is_reply_to_bot=True,
        referenced_message=ReferencedMessage(
            telegram_ref=TelegramRef(chat_id=-100, message_id=40), author_id=999,
            author_name="Pepper", is_bot=True, text="Please explain",
        ),
    )
    await env.conversation.handle_incoming(incoming, ConversationRuntime(bot=env.bot))
    assert len(env.history.data.threads) == 1
    assert "My friend invited me here" in str(env.chat.requests[-1].messages)
    assert "A new group member has been flagged" not in str(env.chat.requests[-1].messages)
    await reach_final_deadline(env, case)
    await env.moderation.process_case(env.context)
    thread = env.history.get_thread(case.thread_id)
    assert case.status == "cleared"
    # A subsequent normal conversation must not inherit the JSON-only review instruction.
    built = await env.builder.build(thread, thread.messages[-1].id)
    assert "Override the usual Telegram-reply format" not in str(built.messages)


@pytest.mark.asyncio
async def test_overdue_restart_allows_time_for_queued_replies(tmp_path):
    env = setup_moderation(tmp_path)
    case = await start_case(env)
    case.deadline = utc_now() - timedelta(minutes=5)
    env.moderation.store.save()
    restored = SpamModeration(env.config, env.conversation, env.tool, env.user_info, env.factory)
    queue = Queue()
    await restored.restore(SimpleNamespace(bot=env.bot, job_queue=queue))
    assert queue.jobs[0].delay == 30
    env.bot.ban_chat_member.assert_not_awaited()


@pytest.mark.asyncio
async def test_grace_period_response_is_considered_at_final_review(tmp_path):
    env = setup_moderation(tmp_path, review='{"is_spam":false,"reason":"Explained during grace period"}')
    case = await start_case(env)
    env.config.spam_detection.final_review_delay_minutes = 3
    case.deadline = utc_now() - timedelta(seconds=1)
    await env.moderation.process_case(env.context)
    assert (case.final_deadline - case.warned_at).total_seconds() == 180
    assert 175 <= env.queue.jobs[-1].delay <= 180
    # The member's response can be an ordinary message after the final warning.
    await env.moderation.observe(Message(
        message_id=80, date=utc_now(), chat=Chat(-100, "supergroup"), from_user=env.user,
        text="Sorry, just saw this! A friend invited me to discuss games.",
    ))
    case.final_deadline = utc_now() - timedelta(seconds=1)
    await env.moderation.process_case(env.context)
    evidence = next(message for message in env.history.get_thread(case.thread_id).messages
                    if message.metadata.get("kind") == "moderation_review_evidence")
    assert json.loads(evidence.content)["target_messages_after_final_warning"] == 1
    assert "just saw this" in str(env.chat.requests[-2].messages)
    assert case.status == "cleared"
    env.bot.ban_chat_member.assert_not_awaited()
    assert env.bot.send_message.await_count == 3


@pytest.mark.asyncio
async def test_warning_failure_does_not_start_final_deadline(tmp_path):
    env = setup_moderation(tmp_path)
    case = await start_case(env)
    case.deadline = utc_now() - timedelta(minutes=5)
    env.bot.send_message.side_effect = RuntimeError("Telegram unavailable")
    await env.moderation.process_case(env.context)
    assert case.status == "pending"
    assert case.warned_at is None
    assert case.final_deadline is None
    env.bot.ban_chat_member.assert_not_awaited()
    assert env.queue.jobs[-1].delay == 60

    env.chat.responses.insert(0, "Final warning: one more minute to respond.")
    env.bot.send_message.side_effect = lambda **kwargs: SimpleNamespace(message_id=41)
    await env.moderation.process_case(env.context)
    assert case.status == "final_pending"
    assert 55 <= (case.final_deadline - utc_now()).total_seconds() <= 60
    env.bot.ban_chat_member.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_checkpoint", [False, True])
async def test_restart_preserves_final_warning_and_deadline_without_repeating_warning(tmp_path, missing_checkpoint):
    env = setup_moderation(tmp_path)
    case = await start_case(env)
    case.deadline = utc_now() - timedelta(seconds=1)
    await env.moderation.process_case(env.context)
    warned_at, final_deadline = case.warned_at, case.final_deadline
    if missing_checkpoint:
        # Simulate a restart between saving the delivered warning in history and
        # checkpointing the next moderation phase.
        case.status = "pending"
        case.warned_at = case.final_deadline = None
        env.moderation.store.save()
    restored = SpamModeration(env.config, env.conversation, env.tool, env.user_info, env.factory)
    queue = Queue()
    await restored.restore(SimpleNamespace(bot=env.bot, job_queue=queue))
    recovered = restored.store.cases[case.id]
    await restored.process_case(env.context)
    assert recovered.status == "final_pending"
    assert recovered.warned_at == warned_at
    assert recovered.final_deadline == final_deadline
    assert env.bot.send_message.await_count == 2
    assert len(env.chat.requests) == 2
    recovered.final_deadline = utc_now() - timedelta(seconds=1)
    await restored.process_case(env.context)
    assert recovered.status == "banned"
    assert env.bot.send_message.await_count == 3


@pytest.mark.asyncio
async def test_cleanup_precedes_announcement_and_delivery_retry_does_not_repeat_cleanup(tmp_path):
    env = setup_moderation(tmp_path)
    case = await start_case(env)
    await reach_final_deadline(env, case)
    events = []

    async def ban(**kwargs):
        events.append("ban")
        return True

    async def failed_send(**kwargs):
        events.append("announcement")
        raise RuntimeError("Telegram unavailable")

    env.bot.ban_chat_member.side_effect = ban
    env.bot.send_message.side_effect = failed_send
    await env.moderation.process_case(env.context)
    assert events == ["ban", "announcement"]
    assert case.status == "announcing"
    assert case.outcome == "banned"
    assert env.history.get_thread(case.thread_id).metadata["moderation_pending"]
    assert SpamCaseStore(env.config.spam_detection.state_path).cases[case.id].outcome == "banned"
    requests_before = len(env.chat.requests)

    restored = SpamModeration(env.config, env.conversation, env.tool, env.user_info, env.factory)
    env.chat.responses.append("The spam account has been removed and its messages cleaned up.")
    env.bot.send_message.side_effect = lambda **kwargs: SimpleNamespace(message_id=42)
    await restored.process_case(env.context)
    assert restored.store.cases[case.id].status == "banned"
    env.bot.ban_chat_member.assert_awaited_once()
    assert len(env.chat.requests) == requests_before + 1
    assert "account_banned" in str(env.chat.requests[-1].messages)
    assert not env.history.get_thread(case.thread_id).metadata["moderation_pending"]


@pytest.mark.asyncio
async def test_delivered_announcement_is_not_repeated_after_restart(tmp_path):
    env = setup_moderation(tmp_path)
    case = await start_case(env)
    await reach_final_deadline(env, case)
    await env.moderation.process_case(env.context)
    # History already contains the successful announcement; case checkpoint lagged.
    case.status = "announcing"
    env.moderation.store.save()
    restored = SpamModeration(env.config, env.conversation, env.tool, env.user_info, env.factory)
    await restored.process_case(env.context)
    assert restored.store.cases[case.id].status == "banned"
    assert env.bot.send_message.await_count == 3
    env.bot.ban_chat_member.assert_awaited_once()
    assert len(env.chat.requests) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["member", "restricted", "left", "kicked"])
async def test_non_admin_join_never_starts_spam_detection(tmp_path, status):
    env = setup_moderation(tmp_path)
    env.bot.get_chat_member.side_effect = lambda **kwargs: SimpleNamespace(status=status)
    await env.moderation.member_joined(env.user, -100, utc_now(), {}, env.context)
    env.bot.get_chat_member.assert_awaited_once_with(chat_id=-100, user_id=env.bot.id)
    assert not env.moderation.store.cases
    assert not env.queue.jobs
    assert not env.history.data.threads
    assert not env.moderation.store.path.exists()
    env.user_info.fetch.assert_not_awaited()
    assert not env.tool.requests
    assert not env.chat.requests
    env.bot.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_admin_promotion_allows_future_joins_without_stale_permission_cache(tmp_path):
    env = setup_moderation(tmp_path)
    env.bot.get_chat_member.side_effect = lambda **kwargs: SimpleNamespace(status="member")
    await env.moderation.member_joined(env.user, -100, utc_now(), {}, env.context)
    assert not env.moderation.store.cases
    env.bot.get_chat_member.side_effect = lambda **kwargs: SimpleNamespace(status="administrator")
    case = await start_case(env)
    assert case.status == "pending"
    assert len(env.tool.requests) == 1
    assert env.bot.send_message.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["screening", "pending", "final_pending", "announcing"])
async def test_demotion_stops_queued_work_without_more_llm_calls_or_messages(tmp_path, phase):
    env = setup_moderation(tmp_path)
    await env.moderation.member_joined(env.user, -100, utc_now(), {}, env.context)
    case = next(iter(env.moderation.store.cases.values()))
    env.context.job = SimpleNamespace(data=case.id)
    if phase != "screening":
        await env.moderation.process_case(env.context)
    if phase == "final_pending":
        await reach_final_deadline(env, case)
    if phase == "announcing":
        # An action already completed; only its announcement was pending.
        case.status = "announcing"
        case.outcome = "banned"
        case.outcome_reason = "Spam confirmed and removed"
        env.moderation.store.save()
    calls_before = (len(env.tool.requests), len(env.chat.requests), env.user_info.fetch.await_count, env.bot.send_message.await_count)
    env.bot.get_chat_member.side_effect = lambda **kwargs: SimpleNamespace(status="member")
    await env.moderation.process_case(env.context)
    assert not case.active
    assert case.status == ("banned" if phase == "announcing" else "cancelled")
    assert not env.queue.jobs
    assert calls_before == (len(env.tool.requests), len(env.chat.requests), env.user_info.fetch.await_count, env.bot.send_message.await_count)
    env.bot.ban_chat_member.assert_not_awaited()
    if case.thread_id:
        assert not env.history.get_thread(case.thread_id).metadata["moderation_pending"]


@pytest.mark.asyncio
async def test_restart_skips_cases_in_chats_where_bot_is_not_admin(tmp_path):
    env = setup_moderation(tmp_path)
    case = await start_case(env)
    env.bot.get_chat_member.side_effect = lambda **kwargs: SimpleNamespace(status="member")
    restored = SpamModeration(env.config, env.conversation, env.tool, env.user_info, env.factory)
    queue = Queue()
    await restored.restore(SimpleNamespace(bot=env.bot, job_queue=queue))
    assert restored.store.cases[case.id].status == "cancelled"
    assert not queue.jobs
    assert not env.history.get_thread(case.thread_id).metadata["moderation_pending"]
    assert len(env.tool.requests) == 1
    assert len(env.chat.requests) == 1
    assert env.bot.send_message.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("entry_point", ["join", "queued", "restore"])
async def test_unknown_admin_status_never_triggers_classifier(tmp_path, entry_point):
    env = setup_moderation(tmp_path)
    if entry_point != "join":
        await env.moderation.member_joined(env.user, -100, utc_now(), {}, env.context)
        case = next(iter(env.moderation.store.cases.values()))
        env.context.job = SimpleNamespace(data=case.id)
    env.bot.get_chat_member.side_effect = RuntimeError("Telegram unavailable")
    if entry_point == "join":
        await env.moderation.member_joined(env.user, -100, utc_now(), {}, env.context)
        assert not env.moderation.store.cases
        assert not env.queue.jobs
    elif entry_point == "queued":
        await env.moderation.process_case(env.context)
        assert case.status == "screening"
        assert env.queue.jobs[-1].delay == 60
    else:
        restored = SpamModeration(env.config, env.conversation, env.tool, env.user_info, env.factory)
        queue = Queue()
        await restored.restore(SimpleNamespace(bot=env.bot, job_queue=queue))
        assert restored.store.cases[case.id].status == "screening"
        assert queue.jobs[-1].delay == 60
    env.user_info.fetch.assert_not_awaited()
    assert not env.tool.requests
    assert not env.chat.requests
    env.bot.send_message.assert_not_awaited()
