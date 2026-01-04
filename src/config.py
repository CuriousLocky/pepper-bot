import yaml
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional

class BotConfig(BaseModel):
    token: str
    chat_whitelist: List[int] = []
    command: str = "pepper"
    name: str = "Pepper"
    nicknames: List[str] = []

class ApiConfig(BaseModel):
    url: str
    key: str
    model: str
    supports_vision: bool = False

class ModelParams(BaseModel):
    temperature: float = 0.7
    reasoning_effort: str = "medium"

class ContextConfig(BaseModel):
    max_context_window: int = 10000
    max_ai_response_token: int = 1000
    history_expiration_hours: int = 24

class ToolSettings(BaseModel):
    memory_capacity: int = 100
    short_term_mem_expiration_days: int = 1

class SearchConfig(BaseModel):
    provider: str = "duckduckgo" # "duckduckgo" or "google"
    google_api_key: Optional[str] = None
    google_cse_id: Optional[str] = None

class Config(BaseModel):
    bot: BotConfig
    api: ApiConfig
    model_params: ModelParams
    context: ContextConfig
    tools: ToolSettings
    search: SearchConfig = Field(default_factory=SearchConfig)

def load_config(config_path: str = "config/config.yaml") -> Config:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(**data)

def load_system_prompt(prompt_path: str = "config/system_prompt.txt") -> str:
    return Path(prompt_path).read_text(encoding="utf-8")
