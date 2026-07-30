import pytest

from pepperbot.providers.base import EmbeddingInput, EmbeddingRequest
from pepperbot.providers.openai_embeddings import OpenAIEmbeddingsProvider


class FakeEmbeddingData:
    def __init__(self, embedding=None):
        self.embedding = embedding if embedding is not None else [1.0, 2.0]


class FakeEmbeddingResponse:
    def __init__(self, data=None):
        self.data = data if data is not None else [FakeEmbeddingData()]


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeEmbeddingResponse([FakeEmbeddingData([float(len(self.calls))])])


class FakeClient:
    def __init__(self):
        self.embeddings = FakeEmbeddings()


@pytest.mark.asyncio
async def test_standard_input_embeddings_omit_images():
    provider = OpenAIEmbeddingsProvider(api_key="k", base_url="http://example.com/v1")
    provider.client = FakeClient()
    result = await provider.embed(
        EmbeddingRequest(
            inputs=[EmbeddingInput(text="hello", images=["data:image/png;base64,abc"])],
            model="embed",
            request_format="standard_input",
        )
    )
    assert result.embeddings == [[1.0]]
    assert provider.client.embeddings.calls[0]["input"] == ["hello"]
    assert "extra_body" not in provider.client.embeddings.calls[0]


@pytest.mark.asyncio
async def test_vllm_chat_messages_embeddings_include_images():
    provider = OpenAIEmbeddingsProvider(api_key="k", base_url="http://example.com/v1")
    provider.client = FakeClient()
    await provider.embed(
        EmbeddingRequest(
            inputs=[EmbeddingInput(text="hello", images=["data:image/png;base64,abc"])],
            model="embed",
            request_format="vllm_chat_messages",
            supports_multimodal=True,
        )
    )
    kwargs = provider.client.embeddings.calls[0]
    assert kwargs["input"] == []
    assert kwargs["extra_body"]["messages"][0]["content"][0]["type"] == "image_url"


@pytest.mark.asyncio
async def test_siliconflow_vl_combines_text_and_image_into_one_object():
    provider = OpenAIEmbeddingsProvider(api_key="k", base_url="http://example.com/v1")
    provider.client = FakeClient()
    result = await provider.embed(
        EmbeddingRequest(
            inputs=[EmbeddingInput(text="hello", images=["data:image/png;base64,abc"])],
            model="embed",
            request_format="siliconflow_vl",
            supports_multimodal=True,
        )
    )
    assert result.embeddings == [[1.0]]
    sent_input = provider.client.embeddings.calls[0]["input"]
    assert sent_input == {"text": "hello", "image": "data:image/png;base64,abc"}
    assert "extra_body" not in provider.client.embeddings.calls[0]


@pytest.mark.asyncio
async def test_siliconflow_vl_text_only_sends_text_object():
    provider = OpenAIEmbeddingsProvider(api_key="k", base_url="http://example.com/v1")
    provider.client = FakeClient()
    await provider.embed(
        EmbeddingRequest(
            inputs=[EmbeddingInput(text="hello")],
            model="embed",
            request_format="siliconflow_vl",
            supports_multimodal=True,
        )
    )
    sent_input = provider.client.embeddings.calls[0]["input"]
    assert sent_input == {"text": "hello"}


@pytest.mark.asyncio
async def test_siliconflow_vl_multiple_inputs_one_embedding_each():
    provider = OpenAIEmbeddingsProvider(api_key="k", base_url="http://example.com/v1")
    provider.client = FakeClient()
    result = await provider.embed(
        EmbeddingRequest(
            inputs=[
                EmbeddingInput(text="first", images=["data:image/png;base64,abc"]),
                EmbeddingInput(text="second"),
            ],
            model="embed",
            request_format="siliconflow_vl",
            supports_multimodal=True,
        )
    )
    assert len(result.embeddings) == 2
    assert provider.client.embeddings.calls[0]["input"] == {"text": "first", "image": "data:image/png;base64,abc"}
    assert provider.client.embeddings.calls[1]["input"] == {"text": "second"}


@pytest.mark.asyncio
async def test_siliconflow_vl_omits_image_when_multimodal_disabled():
    provider = OpenAIEmbeddingsProvider(api_key="k", base_url="http://example.com/v1")
    provider.client = FakeClient()
    await provider.embed(
        EmbeddingRequest(
            inputs=[EmbeddingInput(text="hello", images=["data:image/png;base64,abc"])],
            model="embed",
            request_format="siliconflow_vl",
            supports_multimodal=False,
        )
    )
    sent_input = provider.client.embeddings.calls[0]["input"]
    assert sent_input == {"text": "hello"}
