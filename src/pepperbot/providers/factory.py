from pepperbot.config import Config
from pepperbot.providers.base import ChatProvider, EmbeddingProvider
from pepperbot.providers.openai_chat import OpenAIChatCompletionsProvider
from pepperbot.providers.openai_embeddings import OpenAIEmbeddingsProvider


def create_chat_provider(config: Config) -> ChatProvider:
    if config.chat_backend.provider == "openai_responses":
        raise NotImplementedError("OpenAI Responses provider is planned but not implemented in this version.")
    return OpenAIChatCompletionsProvider(config)


def create_embedding_provider(config: Config) -> EmbeddingProvider:
    return OpenAIEmbeddingsProvider(api_key=config.embedding_api_key(), base_url=config.embedding_api_url())
