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
    reasoning_effort: Optional[str] = None

class ContextConfig(BaseModel):
    max_context_window: int = 10000
    max_ai_response_token: int = 1000
    history_expiration_hours: int = 24

class SearchConfig(BaseModel):
    provider: str = "duckduckgo" # "duckduckgo" or "google"
    google_api_key: Optional[str] = None
    google_cse_id: Optional[str] = None

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
    api_url: str = ""
    api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    db_path: str = "data/chroma_db"
    short: MemoryShortConfig = Field(default_factory=MemoryShortConfig)
    long: MemoryLongConfig = Field(default_factory=MemoryLongConfig)
    user: MemoryUserConfig = Field(default_factory=MemoryUserConfig)

class BlackListConfig(BaseModel):
    enable: bool = False
    admin: List[int] = []
    max_minute: int = 30
    blocked_messages: List[str] = ["You are blocked by this bot."]
    
class SkillsConfig(BaseModel):
    enabled: bool = True
    root_path: str = "skills"

class Config(BaseModel):
    bot: BotConfig
    api: ApiConfig
    model_params: ModelParams
    context: ContextConfig
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    image_generation: ImageGenerationConfig = Field(default_factory=ImageGenerationConfig)
    tool_model: ToolModelConfig = Field(default_factory=ToolModelConfig)
    black_list: BlackListConfig = Field(default_factory=BlackListConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)

def load_config(config_path: str = "config/config.yaml") -> Config:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(**data)

def load_system_prompt(prompt_path: str = "config/system_prompt.txt") -> str:
    return Path(prompt_path).read_text(encoding="utf-8")
