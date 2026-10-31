import asyncio
import base64
import logging
from io import BytesIO
from typing import Any

import requests
from openai import AsyncOpenAI
from PIL import Image

from pepperbot.config import Config

logger = logging.getLogger(__name__)


def _get(value: Any, key: str, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    result = getattr(value, key, None)
    if result is None:
        result = (getattr(value, "model_extra", None) or {}).get(key, default)
    return result


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(_get(part, "text", "")) for part in content if _get(part, "type") == "text")
    return ""


def _image_source(item: Any) -> bytes | str:
    encoded = _get(item, "b64_json")
    if encoded:
        return base64.b64decode(encoded, validate=True)
    url = _get(item, "url")
    if not url:
        image_url = _get(item, "image_url")
        url = image_url if isinstance(image_url, str) else _get(image_url, "url")
    if not isinstance(url, str) or not url:
        raise ValueError("Image response contains neither a URL nor b64_json")
    return url


def _prepare_image(source: bytes | str, scale: float, timeout: float) -> tuple[str, str]:
    if isinstance(source, bytes):
        raw = source
    elif source.startswith("data:"):
        header, separator, encoded = source.partition(",")
        if not separator or ";base64" not in header:
            raise ValueError("Image response contains an invalid base64 data URI")
        raw = base64.b64decode(encoded, validate=True)
    else:
        # Signed provider URLs are downloaded without sending the API credential.
        with requests.get(source, timeout=timeout) as response:
            response.raise_for_status()
            raw = response.content

    with Image.open(BytesIO(raw)) as image:
        image.load()
        mime_type = Image.MIME.get(image.format, "image/png")
        full_res = f"data:{mime_type};base64," + base64.b64encode(raw).decode("ascii")
        size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        resized = image.resize(size, Image.Resampling.LANCZOS)
        if resized.mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16"}:
            resized = resized.convert("RGB")
        output = BytesIO()
        resized.save(output, format="PNG")
    preview = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")
    return full_res, preview


async def generate_image(
    prompt: str, config: Config, image_base64: str | None = None,
) -> tuple[bool, str | None, str | None, str]:
    """Return (success, full-resolution data URI, history-preview data URI, text)."""
    settings = config.image_generation
    if not settings.enabled:
        return False, None, None, "Image generation is disabled."
    if image_base64 is not None and not settings.accepts_image_input():
        return False, None, None, (
            "This image backend is configured for text-to-image only and cannot use a reference image. "
            "Use an image-editing model with image_generation.supports_image_input enabled."
        )

    try:
        async with AsyncOpenAI(
            api_key=settings.api_key or config.api.key,
            base_url=settings.api_url or config.api.url,
            timeout=settings.timeout_seconds,
        ) as client:
            params = dict(settings.request_params)
            if settings.api_format == "images_generations":
                if image_base64 is not None:
                    params[settings.image_input_field] = image_base64
                response = await client.images.generate(model=settings.model, prompt=prompt, extra_body=params)
                # SiliconFlow returns images[], often also data[]. OpenAI-compatible
                # providers may return data[].b64_json instead of a URL.
                images = _get(response, "data") or _get(response, "images")
                if not images:
                    raise ValueError("Images API returned no images in data or images")
                source = _image_source(images[0])
                text = _text(_get(images[0], "revised_prompt"))
            else:
                content: Any = prompt
                if image_base64 is not None:
                    content = [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_base64}},
                    ]
                params.setdefault("temperature", 1.0)
                response = await client.chat.completions.create(
                    model=settings.model, messages=[{"role": "user", "content": content}], extra_body=params,
                )
                choices = _get(response, "choices")
                if not choices:
                    raise ValueError("Chat image backend returned no choices")
                message = _get(choices[0], "message")
                text = _text(_get(message, "content"))
                images = _get(message, "images")
                if not images:
                    return False, None, None, text or "Image URL not found in chat response."
                source = _image_source(images[0])

        full_res, preview = await asyncio.to_thread(
            _prepare_image, source, settings.resolution_scale, settings.timeout_seconds,
        )
        return True, full_res, preview, text
    except Exception as exc:
        logger.exception("Image generation failed (%s)", settings.api_format)
        return False, None, None, str(exc)
