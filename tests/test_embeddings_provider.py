import pytest

from pepperbot.providers.base import EmbeddingInput, EmbeddingRequest
from pepperbot.providers.vllm_embeddings import VLLMEmbeddingsProvider


class FakeEmbeddingData:
    embedding = [1.0, 2.0]


class FakeEmbeddingResponse:
    data = [FakeEmbeddingData()]


class FakeEmbeddings:
    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return FakeEmbeddingResponse()


class FakeClient:
    def __init__(self):
        self.embeddings = FakeEmbeddings()


@pytest.mark.asyncio
async def test_standard_input_embeddings_omit_images():
    provider = VLLMEmbeddingsProvider(api_key="k", base_url="http://example.com/v1")
    provider.client = FakeClient()
    result = await provider.embed(
        EmbeddingRequest(
            inputs=[EmbeddingInput(text="hello", images=["data:image/png;base64,abc"])],
            model="embed",
            request_format="standard_input",
        )
    )
    assert result.embeddings == [[1.0, 2.0]]
    assert provider.client.embeddings.kwargs["input"] == ["hello"]
    assert "extra_body" not in provider.client.embeddings.kwargs


@pytest.mark.asyncio
async def test_chat_message_embeddings_include_images():
    provider = VLLMEmbeddingsProvider(api_key="k", base_url="http://example.com/v1")
    provider.client = FakeClient()
    await provider.embed(
        EmbeddingRequest(
            inputs=[EmbeddingInput(text="hello", images=["data:image/png;base64,abc"])],
            model="embed",
            request_format="chat_messages",
            supports_multimodal=True,
        )
    )
    kwargs = provider.client.embeddings.kwargs
    assert kwargs["input"] == []
    assert kwargs["extra_body"]["messages"][0]["content"][0]["type"] == "image_url"
