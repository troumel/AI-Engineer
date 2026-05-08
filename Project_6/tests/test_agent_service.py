"""Unit tests for tools, planner, and agent service."""

from pathlib import Path

import pytest

from app.services.agent_service import AgentService
from app.services.planner import RuleBasedPlanner, ScratchpadEntry
from app.services.tools import ToolError, build_default_registry


@pytest.fixture
def database_path(tmp_path: Path) -> str:
    return str(tmp_path / "agent.db")


@pytest.fixture
def service(tmp_path: Path) -> AgentService:
    return AgentService(
        conversations_file=str(tmp_path / "conversations.json"),
        sqlite_database=str(tmp_path / "agent.db"),
        planner_backend="rule_based",
        max_iterations=4,
    )


def test_calculator_rejects_unsupported_chars(database_path: str):
    registry = build_default_registry(database_path)
    with pytest.raises(ToolError):
        registry.invoke("calculator", {"expression": "__import__('os').system('echo hi')"})


def test_calculator_evaluates_expression(database_path: str):
    registry = build_default_registry(database_path)
    result = registry.invoke("calculator", {"expression": "(3 + 4) * 2"})
    assert result["value"] == 14


def test_sql_tool_rejects_non_select(database_path: str):
    registry = build_default_registry(database_path)
    with pytest.raises(ToolError):
        registry.invoke("sql_query", {"query": "DROP TABLE products"})


def test_sql_tool_returns_seeded_rows(database_path: str):
    registry = build_default_registry(database_path)
    result = registry.invoke(
        "sql_query",
        {"query": "SELECT name FROM products WHERE category = 'furniture' ORDER BY name"},
    )
    names = [row["name"] for row in result["rows"]]
    assert names == ["Office Chair", "Standing Desk"]


def test_weather_tool_supports_fahrenheit(database_path: str):
    registry = build_default_registry(database_path)
    result = registry.invoke("get_weather", {"city": "London", "unit": "fahrenheit"})
    assert result["unit"] == "fahrenheit"
    assert result["temperature"] == 57.2


def test_planner_chooses_calculator_for_math():
    planner = RuleBasedPlanner()
    action = planner.decide(
        message="What is 7 * 6?",
        history=[],
        tools=_fake_tool_set(),
        scratchpad=[],
    )
    assert action.tool_name == "calculator"
    assert action.arguments["expression"].replace(" ", "") == "7*6"


def test_planner_finalizes_after_observation():
    planner = RuleBasedPlanner()
    scratchpad = [
        ScratchpadEntry(
            tool_name="calculator",
            arguments={"expression": "1 + 1"},
            observation={"expression": "1 + 1", "value": 2},
        )
    ]
    action = planner.decide(
        message="What is 1 + 1?",
        history=[],
        tools=_fake_tool_set(),
        scratchpad=scratchpad,
    )
    assert action.is_final is True
    assert "2" in action.final_answer


def test_agent_records_steps(service: AgentService):
    response = service.chat(message="What is 5 + 5?", conversation_id=None)
    assert response.finished is True
    assert response.iterations >= 2  # one tool call + one final
    assert response.answer.endswith(".") or response.answer.endswith("0.")
    assert "10" in response.answer


def test_agent_persists_conversations_across_instances(tmp_path: Path):
    args = dict(
        conversations_file=str(tmp_path / "conversations.json"),
        sqlite_database=str(tmp_path / "agent.db"),
        planner_backend="rule_based",
        max_iterations=4,
    )
    first = AgentService(**args)
    response = first.chat(message="What is 1 + 1?", conversation_id=None)

    second = AgentService(**args)
    detail = second.get_conversation(response.conversation_id)
    assert detail.message_count == 2


def _fake_tool_set():
    """Minimal tool list used by planner tests."""

    class _StubTool:
        def __init__(self, name: str) -> None:
            self.name = name

    return [
        _StubTool("calculator"),
        _StubTool("get_weather"),
        _StubTool("web_search"),
        _StubTool("sql_query"),
        _StubTool("current_datetime"),
    ]
