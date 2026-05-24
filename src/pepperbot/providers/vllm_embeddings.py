import logging
from typing import Any, Dict, List

from openai import AsyncOpenAI

from pepperbot.providers.base import EmbeddingProvider, EmbeddingRequest, EmbeddingResult

logger = logging.getLogger(__name__)


class VLLMEmbeddingsProvider(EmbeddingProvider):
    def __init__(self, api_key: str, base_url: str):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        if request.request_format == "standard_input":
            inputs: List[str] = []
            for item in request.inputs:
                if item.images:
                    logger.warning(
                        "Embedding request contains images but standard_input format only supports text; images are omitted."
                    )
                inputs.append(item.text)
            response = await self.client.embeddings.create(input=inputs, model=request.model)
            return EmbeddingResult(embeddings=[data.embedding for data in response.data])

        messages: List[Dict[str, Any]] = []
        for item in request.inputs:
            content: List[Dict[str, Any]] = []
            if request.supports_multimodal:
                for image in item.images:
                    content.append({"type": "image_url", "image_url": {"url": image}})
            elif item.images:
                logger.warning(
                    "Embedding request contains images but multimodal embeddings are disabled; images are omitted."
                )
            content.append({"type": "text", "text": item.text})
            messages.append({"role": "user", "content": content})

        response = await self.client.embeddings.create(
            input=[],
            model=request.model,
            extra_body={"messages": messages},
        )
        return EmbeddingResult(embeddings=[data.embedding for data in response.data])
