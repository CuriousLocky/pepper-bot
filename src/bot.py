import logging
import asyncio
import base64
import io
import re
from datetime import datetime
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters, Application

from config import load_config, load_system_prompt
from memory import MemoryManager
from history import HistoryManager, Message
from llm import LLMClient

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class PepperBot:
    def __init__(self):
        self.config = load_config()
        self.system_prompt_template = load_system_prompt()
        self.memory = MemoryManager(
            short_term_path="data/short-term.json",
            long_term_path="data/long-term.txt",
            user_info_path="data/known-users.yaml"
        )
        self.history = HistoryManager("data/chat-histories.json")
        self.llm = LLMClient(self.config, self.memory)

    async def _get_image_base64(self, message, context: ContextTypes.DEFAULT_TYPE) -> str | None:
        """Helper to download image from message and convert to base64."""
        if not message.photo:
            return None
        
        try:
            # Get largest photo
            photo = message.photo[-1]
            logger.info(f"Downloading image file_id: {photo.file_id}...")
            file = await context.bot.get_file(photo.file_id)
            logger.info("file got, downloading as bytearray...")
            # Download to memory
            byte_array = await file.download_as_bytearray()
            
            # Encode to base64
            b64_data = base64.b64encode(byte_array).decode('utf-8')
            logger.info("Image successfully encoded to base64.")
            return f"data:image/jpeg;base64,{b64_data}"
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Hello! I'm Pepper. Mention me or reply to my messages to chat!")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("I am Pepper. I can chat with you and remember things. Use /pepper to wake me up!")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message:
            return

        # Extract text content (message text or caption)
        msg_text = update.message.text or update.message.caption or ""
        
        # If no text/caption and no photo, ignore
        if not msg_text and not update.message.photo:
            return

        chat_id = update.effective_chat.id
        user = update.effective_user
        
        # Check whitelist
        if self.config.bot.chat_whitelist and chat_id not in self.config.bot.chat_whitelist:
            return

        is_reply_to_bot = False
        if update.message.reply_to_message and update.message.reply_to_message.from_user.id == context.bot.id:
            is_reply_to_bot = True

        is_command = msg_text.startswith('/pepper')
        
        # Activation conditions: /pepper command OR reply to bot
        if not (is_command or is_reply_to_bot):
            return
        
        logger.info(f"Processing message from {chat_id} (User: {user.first_name})...")

        # Identify Thread
        thread_id = None
        hist = None
        
        if is_reply_to_bot:
            # Try to find existing thread
            reply_to_telegram_id = update.message.reply_to_message.message_id
            thread_id = self.history.get_thread_id_by_message_id(reply_to_telegram_id)
            
            if thread_id:
                hist = self.history.get_thread(thread_id)
            else:
                # Thread not found or expired
                # If the user explicitly used the command, we can start a new thread.
                if is_command:
                    hist = self.history.create_thread(chat_id)
                    thread_id = hist.id
                else:
                    logger.info(f"Thread not found for reply to message {reply_to_telegram_id}")
                    return
        elif is_command:
            # New command activation -> New thread
            hist = self.history.create_thread(chat_id)
            thread_id = hist.id

        # Prepare message content (strip command if present)
        text = msg_text
        if is_command:
            # Remove /pepper and optional @botname, plus leading whitespace
            text = re.sub(r'^/pepper(?:@\w+)?\s*', '', text, count=1)

        # Check for image
        image_url = None
        if self.config.api.supports_vision and update.message.photo:
            logger.info("Processing attached image...")
            image_url = await self._get_image_base64(update.message, context)
            logger.info("Image processed.")
            if not text:
                text = "[Image]"
        elif not text:
             # No text (after stripping) and no image -> Generic greeting
             text = "Hello!"

        referenced_msg = update.message.reply_to_message
        
        # If activated by /pepper and replying to someone else (NOT bot), we ingest that message.
        if is_command and referenced_msg and not is_reply_to_bot:
             # Add referenced message as Msg 0
             ref_user_name = referenced_msg.from_user.first_name
             
             # Check for image in referenced message
             ref_image_url = None
             if self.config.api.supports_vision and referenced_msg.photo:
                 logger.info("Processing referenced message image...")
                 ref_image_url = await self._get_image_base64(referenced_msg, context)
                 logger.info("Referenced image processed.")

             self.history.add_message(thread_id, Message(
                 role="user",
                 content=referenced_msg.text or (referenced_msg.caption or "[Image/Media]"),
                 message_id=0, # Msg 0 per requirements
                 telegram_id=referenced_msg.message_id,
                 user_id=referenced_msg.from_user.id,
                 user_name=ref_user_name,
                 timestamp=referenced_msg.date,
                 image_url=ref_image_url
             ))

        # Add current user message
        msg_telegram_id = update.message.message_id
        
        # Determine internal ID
        internal_msg_id = hist.get_next_message_id()

        # Determine reply target (internal ID)
        internal_reply_to_id = None
        if referenced_msg:
            if is_command and not is_reply_to_bot:
                 # If /pepper reply to user, we added ref as 0
                 internal_reply_to_id = 0
            elif is_reply_to_bot:
                 # Reply to bot in existing thread
                 ref_telegram_id = referenced_msg.message_id
                 internal_reply_to_id = hist.get_internal_id(ref_telegram_id)
        
        self.history.add_message(thread_id, Message(
            role="user",
            content=text,
            message_id=internal_msg_id,
            telegram_id=msg_telegram_id,
            user_id=user.id,
            user_name=user.first_name,
            reply_to_id=internal_reply_to_id,
            timestamp=update.message.date,
            image_url=image_url
        ))

        # Send typing action
        logger.info("Sending chat action (typing)...")
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        logger.info("Chat action sent.")

        # Get response
        # Get known users map for formatting
        known_users_map = {uid: entry.name for uid, entry in self.memory.user_info.items()}
        messages_payload = hist.format_for_llm(
            known_users_map, 
            token_limit=self.config.context.max_context_window,
            model=self.config.api.model
        )

        logger.info(f"Sending request to LLM (Payload messages: {len(messages_payload)})...")
        response_text = await self.llm.get_response(messages_payload, self.system_prompt_template)
        logger.info("Received response from LLM.")

        # Send response
        logger.info("Sending response to Telegram...")
        sent_msg = await context.bot.send_message(
            chat_id=chat_id, 
            text=response_text, 
            reply_to_message_id=msg_telegram_id
        )
        logger.info(f"Response sent. Message ID: {sent_msg.message_id}")

        # Add assistant response to history
        assistant_internal_id = hist.get_next_message_id()
        self.history.add_message(thread_id, Message(
            role="assistant",
            content=response_text,
            message_id=assistant_internal_id,
            telegram_id=sent_msg.message_id,
            user_id=context.bot.id,
            user_name="Pepper", # Or config bot name
            reply_to_id=internal_msg_id,
            timestamp=datetime.now()
        ))

    async def scheduled_maintenance(self, context: ContextTypes.DEFAULT_TYPE):
        # 1. Clean expired chat histories
        self.history.clean_expired(self.config.context.history_expiration_hours)
        
        # 2. Check expired short-term memories
        expired_events = self.memory.check_expirations(self.config.tools.short_term_mem_expiration_days)
        if expired_events:
            logger.info(f"Consolidating {len(expired_events)} memory events...")
            await self.llm.consolidate_memory(expired_events, self.system_prompt_template)
            # Remove them after consolidation
            self.memory.remove_short_term_events(expired_events)
            logger.info("Memory consolidation complete.")

    async def shutdown(self, application: Application):
        logger.info("Gracefully shutting down...")
        # Ensure all data is saved
        self.history.save()
        self.memory._save_short_term()
        self.memory._save_long_term()
        self.memory._save_user_info()
        logger.info("Shutdown complete.")

    async def post_init(self, application: Application):
        bot_info = await application.bot.get_me()
        logger.info(f"Post-init: Bot ID is {bot_info.id}, Username is {bot_info.username}")

    def run(self):
        application = ApplicationBuilder().token(self.config.bot.token).post_shutdown(self.shutdown).post_init(self.post_init).connect_timeout(30.0).read_timeout(30.0).write_timeout(30.0).build()
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("help", self.help_command))
        # Handle /pepper command
        application.add_handler(CommandHandler("pepper", self.handle_message))
        # Handle replies (messages that don't start with /pepper but reply to bot are handled in handle_message logic)
        # We need a message handler that catches text messages AND photos. 
        # We allow commands here to ensure captioned images with /pepper (which might be skipped by CommandHandler) are caught.
        application.add_handler(MessageHandler(filters.TEXT | filters.PHOTO, self.handle_message))

        # Job queue for maintenance (every hour)
        if application.job_queue:
            application.job_queue.run_repeating(self.scheduled_maintenance, interval=3600, first=10)

        logger.info("Bot started polling...")
        application.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    bot = PepperBot()
    bot.run()
