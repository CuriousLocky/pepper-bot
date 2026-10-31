import base64
import json
from io import BytesIO

import httpx
import pytest
import requests
from openai import AsyncOpenAI as SDKClient
from PIL import Image
from pydantic import ValidationError

import image_gen
from config import Config as LegacyConfig
from pepperbot.config import Config, ImageGenerationConfig
from pepperbot.tools.registry import ToolRegistry


def image_bytes(format="PNG", size=(8, 6)):
    output = BytesIO()
    Image.new("RGB", size, "red").save(output, format=format)
    return output.getvalue()


def data_uri(raw):
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def dimensions(uri):
    with Image.open(BytesIO(base64.b64decode(uri.split(",", 1)[1]))) as image:
        return image.size


def make_config(config_class=Config, **image_settings):
    return config_class(
        bot={"token": "test-token"},
        api={"url": "https://default.example/v1", "key": "default-key", "model": "chat-model"},
        model_params={}, context={},
        image_generation={"enabled": True, "model": "Tongyi-MAI/Z-Image", **image_settings},
    )


def mock_api(monkeypatch, response, status=200):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json=response)

    def client(**kwargs):
        return SDKClient(
            **kwargs, max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

    monkeypatch.setattr(image_gen, "AsyncOpenAI", client)
    return calls


def mock_download(monkeypatch, raw, status=200):
    calls = []

    def download(url, **kwargs):
        calls.append((url, kwargs))
        response = requests.Response()
        response.status_code = status
        response.url = url
        response._content = raw
        response._content_consumed = True
        return response

    monkeypatch.setattr(image_gen.requests, "get", download)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize("config_class", [Config, LegacyConfig])
async def test_images_api_request_options_and_base64_response(monkeypatch, config_class):
    raw = image_bytes()
    calls = mock_api(monkeypatch, {"created": 123, "data": [{"b64_json": base64.b64encode(raw).decode(), "revised_prompt": "A red mug"}]})
    params = {"image_size": "512x512", "seed": 42, "negative_prompt": "text", "num_inference_steps": 20}
    config = make_config(
        config_class, api_format="images_generations", api_url="https://images.example/v1",
        api_key="image-key", request_params=params, timeout_seconds=45,
    )
    success, full, preview, text = await image_gen.generate_image("A mug", config)
    assert success
    assert dimensions(full) == (8, 6)
    assert dimensions(preview) == (4, 3)
    assert text == "A red mug"
    assert str(calls[0].url) == "https://images.example/v1/images/generations"
    assert calls[0].headers["Authorization"] == "Bearer image-key"
    assert calls[0].extensions["timeout"]["read"] == 45
    assert json.loads(calls[0].content) == {"model": "Tongyi-MAI/Z-Image", "prompt": "A mug", **params}
    assert config.image_generation.request_params == params


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["data", "images"])
async def test_provider_url_response_even_when_base64_was_requested(monkeypatch, field):
    raw = image_bytes(format="JPEG")
    url = "https://cdn.example/image.jpg?signature=test"
    calls = mock_api(monkeypatch, {"created": 123, field: [{"url": url}], "seed": 42})
    downloads = mock_download(monkeypatch, raw)
    config = make_config(api_format="images_generations", request_params={"size": "768x512", "response_format": "b64_json"})
    success, full, preview, text = await image_gen.generate_image("A mug", config)
    assert success
    assert full.startswith("data:image/jpeg;base64,")
    assert dimensions(preview) == (4, 3)
    assert text == ""
    assert str(calls[0].url) == "https://default.example/v1/images/generations"
    assert calls[0].headers["Authorization"] == "Bearer default-key"
    assert downloads == [(url, {"timeout": 180})]
    assert json.loads(calls[0].content)["response_format"] == "b64_json"


@pytest.mark.asyncio
@pytest.mark.parametrize("image_field", ["url", "image_url"])
async def test_legacy_gemini_chat_generation_and_reference_images(monkeypatch, image_field):
    uri = data_uri(image_bytes())
    source = {image_field: uri if image_field == "url" else {"url": uri}}
    calls = mock_api(monkeypatch, {
        "id": "chat-image", "choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": "Here is your image.", "images": [source],
        }}],
    })
    config = make_config(model="gemini-3-image", request_params={"temperature": 0.8, "modalities": ["text", "image"]})
    success, full, preview, text = await image_gen.generate_image("Make it blue", config, image_base64=uri)
    assert success
    assert text == "Here is your image."
    assert dimensions(full) == (8, 6)
    assert dimensions(preview) == (4, 3)
    assert calls[0].url.path == "/v1/chat/completions"
    body = json.loads(calls[0].content)
    assert body["temperature"] == 0.8
    assert body["modalities"] == ["text", "image"]
    assert body["messages"][0]["content"] == [
        {"type": "text", "text": "Make it blue"},
        {"type": "image_url", "image_url": {"url": uri}},
    ]


@pytest.mark.asyncio
async def test_chat_default_text_prompt_and_temperature_remain_compatible(monkeypatch):
    calls = mock_api(monkeypatch, {"choices": [{"message": {
        "role": "assistant", "content": "", "images": [{"url": data_uri(image_bytes())}],
    }}]})
    success, *_ = await image_gen.generate_image("A mug", make_config())
    assert success
    body = json.loads(calls[0].content)
    assert body["temperature"] == 1.0
    assert body["messages"] == [{"role": "user", "content": "A mug"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["image", "input_image"])
async def test_images_api_editing_sends_reference_only_when_enabled(monkeypatch, field):
    uri = data_uri(image_bytes())
    calls = mock_api(monkeypatch, {"images": [{"url": uri}]})
    config = make_config(
        api_format="images_generations", model="Qwen/Qwen-Image-Edit",
        supports_image_input=True, image_input_field=field, request_params={"image_size": "1328x1328"},
    )
    success, *_ = await image_gen.generate_image("Make it blue", config, image_base64=uri)
    assert success
    assert json.loads(calls[0].content) == {
        "model": "Qwen/Qwen-Image-Edit", "prompt": "Make it blue",
        field: uri, "image_size": "1328x1328",
    }
    assert config.image_generation.request_params == {"image_size": "1328x1328"}
    tool = next(tool["function"] for tool in ToolRegistry(config).chat_tools() if tool["function"]["name"] == "generate_image")
    assert "msg_id" in tool["parameters"]["properties"]


@pytest.mark.asyncio
@pytest.mark.parametrize("settings", [
    {"enabled": False},
    {"api_format": "images_generations"},
    {"api_format": "chat_completions", "supports_image_input": False},
])
async def test_disabled_generation_or_unsupported_reference_makes_no_request(monkeypatch, settings):
    calls = mock_api(monkeypatch, {})
    config = make_config(**settings)
    success, full, preview, text = await image_gen.generate_image("Edit this", config, image_base64=data_uri(image_bytes()))
    assert not success
    assert full is None and preview is None
    assert "disabled" in text or "text-to-image only" in text
    assert not calls
    if config.image_generation.enabled:
        tool = next(tool["function"] for tool in ToolRegistry(config).chat_tools() if tool["function"]["name"] == "generate_image")
        assert "msg_id" not in tool["parameters"]["properties"]


@pytest.mark.asyncio
@pytest.mark.parametrize("body,status,error", [
    ({"data": []}, 200, "no images"),
    ({"data": [{}]}, 200, "neither a URL nor b64_json"),
    ({"images": [{"b64_json": "@@@"}]}, 200, "base64"),
    ({"images": [{"url": "data:image/png,not-base64"}]}, 200, "invalid base64 data URI"),
    ({"error": {"message": "upstream image request failed with status 404"}}, 404, "upstream image request failed"),
])
async def test_images_api_errors_are_returned_to_tool(monkeypatch, body, status, error):
    mock_api(monkeypatch, body, status)
    success, full, preview, text = await image_gen.generate_image("A mug", make_config(api_format="images_generations"))
    assert not success
    assert full is None and preview is None
    assert error in text


@pytest.mark.asyncio
async def test_expired_provider_image_url_is_reported(monkeypatch):
    mock_api(monkeypatch, {"data": [{"url": "https://cdn.example/expired.png"}]})
    mock_download(monkeypatch, b"expired", status=403)
    result = await image_gen.generate_image("A mug", make_config(api_format="images_generations"))
    assert result[0] is False
    assert "403" in result[3]


@pytest.mark.asyncio
async def test_tiny_history_scale_never_produces_zero_sized_image(monkeypatch):
    mock_api(monkeypatch, {"images": [{"url": data_uri(image_bytes(size=(1, 1)))}]})
    success, full, preview, _ = await image_gen.generate_image("Pixel", make_config(api_format="images_generations", resolution_scale=0.01))
    assert success
    assert dimensions(full) == dimensions(preview) == (1, 1)


@pytest.mark.parametrize("settings", [
    {"api_format": "unknown"}, {"resolution_scale": 0}, {"timeout_seconds": 0},
    {"request_params": {"model": "other"}}, {"request_params": {"prompt": "fixed"}},
    {"request_params": {"image": "ignored-reference"}}, {"request_params": {"stream": True}},
    {"request_params": {"batch_size": 2}},
])
def test_invalid_image_configuration_is_rejected(settings):
    with pytest.raises(ValidationError):
        ImageGenerationConfig(**settings)
