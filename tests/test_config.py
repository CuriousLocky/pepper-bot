from pepperbot.config import _normalize_legacy_config


def test_legacy_blacklist_admin_moves_to_admin_ids():
    data = {
        "black_list": {"admin": [1], "max_minutes": 5},
        "memory": {
            "api_url": "http://embedding/v1",
            "api_key": "embed-key",
            "embedding_model": "embed-model",
        },
        "response": {"empty_response_retries": 2},
    }
    _normalize_legacy_config(data)
    assert data["admin"]["ids"] == [1]
    assert "admin" not in data["black_list"]
    assert data["black_list"]["max_minute"] == 5
    assert data["embedding_backend"]["api_url"] == "http://embedding/v1"
    assert data["embedding_backend"]["api_key"] == "embed-key"
    assert data["embedding_backend"]["model"] == "embed-model"
    assert "api_url" not in data["memory"]
    assert data["response"]["error_response_retries"] == 2
    assert "empty_response_retries" not in data["response"]


def test_legacy_embedding_request_formats_are_renamed():
    data = {
        "embedding_backend": {
            "provider": "vllm",
            "request_format": "chat_messages",
        }
    }
    _normalize_legacy_config(data)
    assert data["embedding_backend"]["request_format"] == "vllm_chat_messages"
    assert "provider" not in data["embedding_backend"]

    data = {"embedding_backend": {"provider": "openai_embedding", "request_format": "openai_vl"}}
    _normalize_legacy_config(data)
    assert data["embedding_backend"]["request_format"] == "siliconflow_vl"
    assert "provider" not in data["embedding_backend"]


def test_canonical_embedding_request_formats_are_preserved():
    for fmt in ("standard_input", "vllm_chat_messages", "siliconflow_vl"):
        data = {"embedding_backend": {"request_format": fmt}}
        _normalize_legacy_config(data)
        assert data["embedding_backend"]["request_format"] == fmt
