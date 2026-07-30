from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


Content = str | List[Dict[str, Any]]


@dataclass
class ChatMessage:
    role: str
    content: Content
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]
    raw_arguments: str = ""


@dataclass
class ChatRequest:
    model: str
    messages: List[ChatMessage]
    tools: List[Dict[str, Any]] = field(default_factory=list)
    allow_tools: bool = True
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None


@dataclass
class ChatResult:
    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    assistant_message: Optional[Dict[str, Any]] = None
    raw: Any = None


@dataclass
class EmbeddingInput:
    text: str
    images: List[str] = field(default_factory=list)


@dataclass
class EmbeddingRequest:
    inputs: List[EmbeddingInput]
    model: str
    request_format: Literal["standard_input", "vllm_chat_messages", "siliconflow_vl"] = "vllm_chat_messages"
    supports_multimodal: bool = True


@dataclass
class EmbeddingResult:
    embeddings: List[List[float]]


class ChatProvider:
    async def complete(self, request: ChatRequest) -> ChatResult:
        raise NotImplementedError


class EmbeddingProvider:
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        raise NotImplementedError
