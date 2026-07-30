import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field


class BotConfig(BaseModel):
    token: str
    chat_whitelist: List[int] = Field(default_factory=list)
    command: str = "pepper"
    name: str = "Pepper"
    nicknames: List[str] = Field(default_factory=list)


class AdminConfig(BaseModel):
    ids: List[int] = Field(default_factory=list)
    report_chat_ids: List[int] = Field(default_factory=list)
    report_major_failures: bool = True
    include_context_preview: bool = True
    context_preview_chars: int = 2000


class ApiConfig(BaseModel):
    url: str
    key: str
    model: str
    supports_vision: bool = False


class ChatBackendConfig(BaseModel):
    provider: Literal["openai_chat", "openai_responses"] = "openai_chat"
    api_url: str = ""
    api_key: str = ""
    model: str = ""
    supports_tools: bool = True
    supports_vision: Optional[bool] = None
    supports_reasoning_effort: bool = True


class EmbeddingBackendConfig(BaseModel):
    api_url: str = ""
    api_key: str = ""
    model: str = "text-embedding-3-small"
    request_format: Literal["standard_input", "vllm_chat_messages", "siliconflow_vl"] = "vllm_chat_messages"
    supports_multimodal: bool = True


class ModelParams(BaseModel):
    temperature: float = 0.7
    reasoning_effort: Optional[str] = None


class ResponseConfig(BaseModel):
    fallback_text: str = "抱歉，刚才脑袋短路了。"
    error_response_retries: int = 1
    max_tool_rounds: int = 8


class TelegramConfig(BaseModel):
    max_message_chars: int = 4096
    chunk_chars: int = 3900
    send_retries: int = 2
    media_group_wait_seconds: float = 1.0
    image_max_area: int = 1048576
    expired_thread_notice: str = "这段对话已经过期啦。回复这条消息的话，我会带着刚才的引用重新开始。"


class ContextConfig(BaseModel):
    max_context_window: int = 10000
    max_ai_response_token: int = 1000
    history_expiration_hours: int = 24
    summary_trigger_ratio: float = 0.8
    preserve_recent_messages: int = 20


class SearchConfig(BaseModel):
    provider: str = "duckduckgo"
    max_results: int = 5
    google_api_key: Optional[str] = None
    google_cse_id: Optional[str] = None
    tavily_api_key: Optional[str] = None


class ImageGenerationConfig(BaseModel):
    enabled: bool = False
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    model: str = "dall-e-3"
    resolution_scale: float = 0.5


class ToolModelConfig(BaseModel):
    api_url: str = ""
    api_key: str = ""
    model: str = "gpt-4o-mini"


class MemoryShortConfig(BaseModel):
    selective: bool = True
    top_k: int = 20
    always_include_hours: int = 24
    relevant_size: int = 40
    expiration_days: int = 5


class MemoryLongConfig(BaseModel):
    selective: bool = True
    top_k: int = 30
    relevant_size: int = 50
    max_entries: int = 50


class MemoryUserConfig(BaseModel):
    selective: bool = True
    top_k: int = 5
    lru_size: int = 8
    relevant_include: int = 5


class MemoryConfig(BaseModel):
    db_path: str = "data/chroma_db"
    short: MemoryShortConfig = Field(default_factory=MemoryShortConfig)
    long: MemoryLongConfig = Field(default_factory=MemoryLongConfig)
    user: MemoryUserConfig = Field(default_factory=MemoryUserConfig)


class BlackListConfig(BaseModel):
    enable: bool = False
    max_minute: int = 30
    blocked_messages: List[str] = Field(default_factory=lambda: ["You are blocked by this bot."])


class SkillsConfig(BaseModel):
    enabled: bool = True
    root_path: str = "skills"


class AttachmentConfig(BaseModel):
    root_path: str = "data/attachments"


class Config(BaseModel):
    bot: BotConfig
    api: ApiConfig
    model_params: ModelParams = Field(default_factory=ModelParams)
    context: ContextConfig = Field(default_factory=ContextConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    image_generation: ImageGenerationConfig = Field(default_factory=ImageGenerationConfig)
    tool_model: ToolModelConfig = Field(default_factory=ToolModelConfig)
    black_list: BlackListConfig = Field(default_factory=BlackListConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    admin: AdminConfig = Field(default_factory=AdminConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    response: ResponseConfig = Field(default_factory=ResponseConfig)
    chat_backend: ChatBackendConfig = Field(default_factory=ChatBackendConfig)
    embedding_backend: EmbeddingBackendConfig = Field(default_factory=EmbeddingBackendConfig)
    attachments: AttachmentConfig = Field(default_factory=AttachmentConfig)

    def chat_api_url(self) -> str:
        return self.chat_backend.api_url or self.api.url

    def chat_api_key(self) -> str:
        return self.chat_backend.api_key or self.api.key

    def chat_model(self) -> str:
        return self.chat_backend.model or self.api.model

    def chat_supports_vision(self) -> bool:
        if self.chat_backend.supports_vision is not None:
            return self.chat_backend.supports_vision
        return self.api.supports_vision

    def embedding_api_url(self) -> str:
        return self.embedding_backend.api_url or self.api.url

    def embedding_api_key(self) -> str:
        return self.embedding_backend.api_key or self.api.key

    def embedding_model(self) -> str:
        return self.embedding_backend.model


class TemplateMessage(BaseModel):
    id: str
    type: Literal["text", "thread_context"] = "text"
    role: Optional[str] = None
    content: str = ""
    content_file: Optional[str] = None
    omit_if_empty: bool = False


class TemplateConfig(BaseModel):
    version: int = 1
    messages: List[TemplateMessage]
    empty_response_repair_message: TemplateMessage = Field(
        default_factory=lambda: TemplateMessage(
            id="empty_response_repair",
            type="text",
            role="user",
            content="The previous assistant response had no usable Telegram text or used invalid XML. Produce the final Telegram message body now.",
        )
    )


def _set_nested(data: Dict[str, Any], keys: List[str], value: Any) -> None:
    current = data
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def _parse_env_value(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except Exception:
        return value


def _apply_env_overrides(data: Dict[str, Any]) -> None:
    aliases = {
        "PEPPER_BOT_TOKEN": ["bot", "token"],
        "PEPPER_API_KEY": ["api", "key"],
        "PEPPER_API_URL": ["api", "url"],
        "PEPPER_API_MODEL": ["api", "model"],
    }
    for name, path in aliases.items():
        if name in os.environ:
            _set_nested(data, path, _parse_env_value(os.environ[name]))

    prefix = "PEPPER__"
    for name, raw_value in os.environ.items():
        if not name.startswith(prefix):
            continue
        path = [part.lower() for part in name[len(prefix) :].split("__") if part]
        if path:
            _set_nested(data, path, _parse_env_value(raw_value))


def _normalize_legacy_config(data: Dict[str, Any]) -> None:
    black_list = data.get("black_list")
    if isinstance(black_list, dict):
        legacy_admin = black_list.get("admin")
        if legacy_admin and not (isinstance(data.get("admin"), dict) and data["admin"].get("ids")):
            data.setdefault("admin", {})["ids"] = legacy_admin
        if "max_minutes" in black_list and "max_minute" not in black_list:
            black_list["max_minute"] = black_list["max_minutes"]
        black_list.pop("admin", None)
    memory = data.get("memory")
    if isinstance(memory, dict):
        embedding_backend = data.setdefault("embedding_backend", {})
        if "api_url" in memory and not embedding_backend.get("api_url"):
            embedding_backend["api_url"] = memory.pop("api_url")
        if "api_key" in memory and not embedding_backend.get("api_key"):
            embedding_backend["api_key"] = memory.pop("api_key")
        if "embedding_model" in memory and not embedding_backend.get("model"):
            embedding_backend["model"] = memory.pop("embedding_model")
    embedding_backend = data.get("embedding_backend")
    if isinstance(embedding_backend, dict):
        legacy_request_format = embedding_backend.get("request_format")
        if legacy_request_format == "chat_messages":
            embedding_backend["request_format"] = "vllm_chat_messages"
        elif legacy_request_format == "openai_vl":
            embedding_backend["request_format"] = "siliconflow_vl"
        embedding_backend.pop("provider", None)
    response = data.get("response")
    if isinstance(response, dict):
        if "empty_response_retries" in response and "error_response_retries" not in response:
            response["error_response_retries"] = response["empty_response_retries"]
        response.pop("empty_response_retries", None)


def load_config(config_path: str = "config/config.yaml") -> Config:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    _normalize_legacy_config(data)
    _apply_env_overrides(data)
    return Config(**data)


def load_system_prompt(prompt_path: str = "config/system_prompt.txt") -> str:
    return Path(prompt_path).read_text(encoding="utf-8")


def default_template() -> TemplateConfig:
    return TemplateConfig(
        messages=[
            TemplateMessage(
                id="persona",
                type="text",
                role="system",
                content_file="config/system_prompt.txt",
            ),
            TemplateMessage(
                id="thread",
                type="thread_context",
            ),
            TemplateMessage(
                id="runtime_memory",
                type="text",
                role="user",
                content=(
                    "Current UTC time: {{date_time}}\n"
                    "Bot name: {{bot_name}}\n\n"
                    "Available skills:\n{{skill_list}}\n\n"
                    "Knowledges:\n{{knowledges}}\n\n"
                    "Long-term memory:\n{{long_term_memory}}\n\n"
                    "Known user info:\n{{known_user_info}}\n\n"
                    "Short-term memory:\n{{short_term_memory}}"
                ),
                omit_if_empty=True,
            ),
            TemplateMessage(
                id="response_contract",
                type="text",
                role="user",
                content=(
                    "Reply with the Telegram message body only. Preferred format:\n"
                    "<telegram_reply>message body</telegram_reply>\n"
                    "Do not include message IDs, author labels, or extra XML outside the reply."
                ),
            ),
        ]
    )


def load_template(template_path: str = "config/template.json") -> TemplateConfig:
    path = Path(template_path)
    if not path.exists():
        return default_template()
    import json

    return TemplateConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
