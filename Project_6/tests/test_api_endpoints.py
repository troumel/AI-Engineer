"""Integration tests for the Project 6 agent API."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import dependencies
from app.main import app
from app.services.agent_service import AgentService


@pytest.fixture(autouse=True)
def initialize_test_services(tmp_path: Path):
    dependencies._agent_service = AgentService(
        conversations_file=str(tmp_path / "conversations.json"),
        sqlite_database=str(tmp_path / "agent.db"),
        planner_backend="rule_based",
        max_iterations=4,
    )
    yield
    dependencies._agent_service = None


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_returns_api_info(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_lists_default_tools(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["planner_backend"] == "rule_based"
    assert payload["tool_count"] == 5


def test_tools_endpoint_lists_all_tools(client):
    response = client.get("/tools")
    assert response.status_code == 200
    payload = response.json()
    names = {tool["name"] for tool in payload["tools"]}
    assert names == {
        "calculator",
        "get_weather",
        "web_search",
        "sql_query",
        "current_datetime",
    }


def test_tool_invoke_calculator(client):
    response = client.post(
        "/tools/invoke",
        json={"tool_name": "calculator", "arguments": {"expression": "2 + 3 * 4"}},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["result"]["value"] == 14


def test_tool_invoke_unknown_returns_404(client):
    response = client.post(
        "/tools/invoke",
        json={"tool_name": "nonexistent", "arguments": {}},
    )
    assert response.status_code == 404


def test_tool_invoke_validation_error_returns_failure(client):
    response = client.post(
        "/tools/invoke",
        json={"tool_name": "calculator", "arguments": {}},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is False
    assert "expression" in (payload["error"] or "")


def test_chat_uses_calculator_for_arithmetic(client):
    response = client.post("/chat", json={"message": "What is 12 * 11?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["finished"] is True
    assert any(step["tool_name"] == "calculator" for step in payload["steps"])
    assert "132" in payload["answer"]


def test_chat_uses_weather_tool(client):
    response = client.post(
        "/chat",
        json={"message": "What's the weather in London today?"},
    )
    assert response.status_code == 200
    payload = response.json()
    tool_names = [step["tool_name"] for step in payload["steps"] if step["tool_name"]]
    assert "get_weather" in tool_names
    assert "London" in payload["answer"]


def test_chat_uses_sql_tool_for_database_question(client):
    response = client.post(
        "/chat",
        json={"message": "Show me all products in the database."},
    )
    assert response.status_code == 200
    payload = response.json()
    tool_names = [step["tool_name"] for step in payload["steps"] if step["tool_name"]]
    assert "sql_query" in tool_names
    assert "Mechanical Keyboard" in payload["answer"]


def test_chat_continues_existing_conversation(client):
    first = client.post("/chat", json={"message": "What is 2 + 2?"}).json()
    conversation_id = first["conversation_id"]

    second = client.post(
        "/chat",
        json={"message": "What is the weather in Berlin?", "conversation_id": conversation_id},
    ).json()
    assert second["conversation_id"] == conversation_id

    detail = client.get(f"/conversations/{conversation_id}").json()
    assert detail["message_count"] == 4
    assert detail["messages"][0]["role"] == "user"
    assert detail["messages"][1]["role"] == "assistant"


def test_conversations_endpoint_lists_history(client):
    client.post("/chat", json={"message": "What time is it?"})
    response = client.get("/conversations")
    assert response.status_code == 200
    assert response.json()["count"] == 1


def test_unknown_conversation_returns_404(client):
    response = client.get("/conversations/missing")
    assert response.status_code == 404
