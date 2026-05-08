"""End-to-end API tests for Project 8."""

from __future__ import annotations

import json


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["backend"] == "echo"
    assert body["registered_models"] >= 1


def test_root_advertises_openai_compatible_path(client):
    body = client.get("/").json()
    assert body["openai_compatible"] == "/v1"


def test_list_models_returns_registered_entries(client):
    body = client.get("/v1/models").json()
    assert body["object"] == "list"
    ids = [model["id"] for model in body["data"]]
    assert "phi-3-mini-offline" in ids
    assert "mistral-7b-offline" in ids


def test_get_model_returns_quantization_info(client):
    body = client.get("/v1/models/phi-3-mini-offline").json()
    assert body["quantization"]
    assert body["context_window"] >= 1024


def test_get_unknown_model_returns_404(client):
    response = client.get("/v1/models/missing")
    assert response.status_code == 404


def test_chat_completion_returns_openai_shape(client):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "phi-3-mini-offline",
            "messages": [{"role": "user", "content": "Explain LLM quantization."}],
            "max_tokens": 16,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "phi-3-mini-offline"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["finish_reason"] in {"stop", "length"}
    usage = body["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_chat_completion_unknown_model_returns_404(client):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "ghost-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
        },
    )
    assert response.status_code == 404


def test_chat_completion_streaming_emits_sse(client):
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "phi-3-mini-offline",
            "messages": [{"role": "user", "content": "Explain continuous batching."}],
            "max_tokens": 8,
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    assert "data: [DONE]" in body
    data_lines = [line for line in body.splitlines() if line.startswith("data: ") and "[DONE]" not in line]
    assert data_lines, "expected SSE data lines"
    parsed = [json.loads(line[len("data: ") :]) for line in data_lines]
    assert any(chunk["choices"][0]["delta"].get("role") == "assistant" for chunk in parsed)
    assert any(chunk["choices"][0]["delta"].get("content") for chunk in parsed)


def test_text_completion_returns_usage(client):
    response = client.post(
        "/v1/completions",
        json={
            "model": "phi-3-mini-offline",
            "prompt": "Describe paged attention briefly.",
            "max_tokens": 12,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"]
    assert body["usage"]["prompt_tokens"] > 0


def test_usage_endpoint_aggregates_by_key_and_model(client):
    client.post(
        "/v1/completions",
        json={"model": "phi-3-mini-offline", "prompt": "hello", "max_tokens": 4},
        headers={"Authorization": "Bearer key-A"},
    )
    client.post(
        "/v1/completions",
        json={"model": "mistral-7b-offline", "prompt": "hello again", "max_tokens": 4},
        headers={"Authorization": "Bearer key-B"},
    )
    body = client.get("/usage").json()
    assert body["total_requests"] == 2
    assert "key-A" in body["by_api_key"]
    assert "key-B" in body["by_api_key"]
    assert "phi-3-mini-offline" in body["by_model"]
    assert "mistral-7b-offline" in body["by_model"]
    assert body["total_cost_usd"] >= 0.0


def test_metrics_endpoint_returns_prometheus_text(client):
    client.post(
        "/v1/completions",
        json={"model": "phi-3-mini-offline", "prompt": "warm up metrics", "max_tokens": 4},
    )
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    text = response.text
    assert "llm_requests_total" in text
    assert "llm_tokens_in_total" in text
    assert "llm_uptime_seconds" in text


def test_validation_rejects_empty_messages(client):
    response = client.post(
        "/v1/chat/completions",
        json={"model": "phi-3-mini-offline", "messages": [], "max_tokens": 4},
    )
    assert response.status_code == 422
