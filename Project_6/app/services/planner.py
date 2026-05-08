"""Agent planner abstractions with a deterministic offline rule-based default.

The agent loop follows a ReAct-style pattern:

    while not done and iterations < max_iterations:
        action = planner.decide(message, history, tools, scratchpad)
        if action.is_final: break
        observation = registry.invoke(action.tool, action.arguments)
        scratchpad.append((action, observation))

The `RuleBasedPlanner` inspects the user's message + observation history and
chooses tools using simple keyword heuristics. This keeps the project fully
testable without any LLM API. A real deployment can swap in
`OpenAIFunctionCallingPlanner` (or similar) without touching the agent loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.services.tools import Tool, ToolRegistry


@dataclass
class PlannerAction:
    """One decision returned by the planner."""

    thought: str
    is_final: bool = False
    final_answer: str = ""
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScratchpadEntry:
    """Captures one tool invocation and its observation."""

    tool_name: str
    arguments: dict[str, Any]
    observation: Any
    error: str | None = None


class Planner(Protocol):
    """Decide the next action in the agent loop."""

    name: str

    def decide(
        self,
        message: str,
        history: list[dict],
        tools: list[Tool],
        scratchpad: list[ScratchpadEntry],
    ) -> PlannerAction:
        ...


class RuleBasedPlanner:
    """Deterministic, offline planner using keyword heuristics.

    The planner is intentionally simple: it looks at the user's latest
    message, then chooses a tool based on the words it contains. After each
    tool runs once, the planner stops calling that tool again (avoiding
    infinite loops) and finally synthesises a textual answer from the
    accumulated observations.
    """

    name = "rule_based"

    _CALC_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?(?:\s*[+\-*/%]\s*[0-9]+(?:\.[0-9]+)?)+)")
    _CITY_RE = re.compile(r"weather.*?in\s+([A-Za-z][A-Za-z\s]{1,40})", re.IGNORECASE)

    def decide(
        self,
        message: str,
        history: list[dict],
        tools: list[Tool],
        scratchpad: list[ScratchpadEntry],
    ) -> PlannerAction:
        used_tools = {entry.tool_name for entry in scratchpad}
        available_tools = {tool.name for tool in tools}
        message_lower = message.lower()

        # 1. Calculator: detect arithmetic expressions.
        if "calculator" in available_tools and "calculator" not in used_tools:
            calc_match = self._CALC_RE.search(message)
            arithmetic_keywords = any(
                token in message_lower
                for token in ("calculate", "compute", "what is", "what's", "evaluate", "sum", "product")
            )
            if calc_match and (arithmetic_keywords or len(message.split()) <= 6):
                expression = calc_match.group(1)
                return PlannerAction(
                    thought=f"User asked an arithmetic question; using calculator on '{expression}'.",
                    tool_name="calculator",
                    arguments={"expression": expression},
                )

        # 2. Weather lookup.
        if (
            "get_weather" in available_tools
            and "get_weather" not in used_tools
            and "weather" in message_lower
        ):
            city_match = self._CITY_RE.search(message)
            city = city_match.group(1).strip() if city_match else self._extract_proper_noun(message)
            if city:
                unit = "fahrenheit" if "fahrenheit" in message_lower or "°f" in message_lower else "celsius"
                return PlannerAction(
                    thought=f"User asked about weather; fetching forecast for '{city}'.",
                    tool_name="get_weather",
                    arguments={"city": city, "unit": unit},
                )

        # 3. SQL queries against the seed database.
        if (
            "sql_query" in available_tools
            and "sql_query" not in used_tools
            and any(keyword in message_lower for keyword in ("product", "products", "customer", "stock", "inventory", "database", "sql"))
        ):
            query = self._build_sql_query(message_lower)
            if query:
                return PlannerAction(
                    thought=f"User asked a database question; running SQL query: {query}.",
                    tool_name="sql_query",
                    arguments={"query": query},
                )

        # 4. Current datetime.
        if (
            "current_datetime" in available_tools
            and "current_datetime" not in used_tools
            and any(keyword in message_lower for keyword in ("date", "time", "today", "now"))
        ):
            return PlannerAction(
                thought="User asked about the current time; calling current_datetime tool.",
                tool_name="current_datetime",
                arguments={"timezone": "UTC"},
            )

        # 5. Web search fallback for informational queries.
        has_successful_observation = any(entry.error is None for entry in scratchpad)
        if (
            "web_search" in available_tools
            and "web_search" not in used_tools
            and not has_successful_observation
            and any(
                keyword in message_lower
                for keyword in ("search", "find", "look up", "lookup", "tell me about", "explain")
            )
        ):
            query = self._extract_search_query(message)
            return PlannerAction(
                thought=f"Falling back to web_search for query '{query}'.",
                tool_name="web_search",
                arguments={"query": query, "max_results": 3},
            )

        # 6. Done — synthesise an answer from the scratchpad.
        return PlannerAction(
            thought="Sufficient information gathered; producing final answer.",
            is_final=True,
            final_answer=self._compose_final_answer(message, scratchpad),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_proper_noun(self, message: str) -> str | None:
        for token in message.split():
            cleaned = token.strip(".,!?")
            if cleaned and cleaned[0].isupper() and cleaned.lower() not in {"i", "what", "when", "where", "why"}:
                return cleaned
        return None

    def _extract_search_query(self, message: str) -> str:
        cleaned = re.sub(
            r"(please|can you|could you|tell me about|search for|search|find|look up|lookup|what is|what's|explain)",
            "",
            message,
            flags=re.IGNORECASE,
        )
        return cleaned.strip(" .?!") or message

    def _build_sql_query(self, message_lower: str) -> str | None:
        if "customer" in message_lower:
            return "SELECT id, name, country FROM customers ORDER BY id"
        if "stock" in message_lower or "inventory" in message_lower:
            return "SELECT name, stock FROM products ORDER BY stock DESC"
        if "product" in message_lower or "products" in message_lower:
            return "SELECT id, name, category, price FROM products ORDER BY price"
        return None

    def _compose_final_answer(self, message: str, scratchpad: list[ScratchpadEntry]) -> str:
        if not scratchpad:
            return (
                "I do not have a tool that fits that request, but here is what I understood: "
                f"'{message.strip()}'."
            )

        parts: list[str] = []
        for entry in scratchpad:
            if entry.error is not None:
                parts.append(f"Tool '{entry.tool_name}' failed: {entry.error}.")
                continue
            parts.append(self._summarize(entry))
        return " ".join(parts)

    def _summarize(self, entry: ScratchpadEntry) -> str:
        observation = entry.observation
        if entry.tool_name == "calculator" and isinstance(observation, dict):
            return f"The result of {observation['expression']} is {observation['value']}."
        if entry.tool_name == "get_weather" and isinstance(observation, dict):
            return (
                f"Weather in {observation['city']}: {observation['condition']}, "
                f"{observation['temperature']}°{'F' if observation['unit'] == 'fahrenheit' else 'C'}."
            )
        if entry.tool_name == "current_datetime" and isinstance(observation, dict):
            return f"Current UTC time is {observation['iso']}."
        if entry.tool_name == "web_search" and isinstance(observation, dict):
            results = observation.get("results", [])
            if not results:
                return f"No search results for '{observation.get('query', '')}'."
            top = results[0]
            return f"Top search result: {top['title']} — {top['snippet']}"
        if entry.tool_name == "sql_query" and isinstance(observation, dict):
            rows = observation.get("rows", [])
            return f"SQL query returned {observation.get('row_count', 0)} rows: {rows}."
        return f"Tool '{entry.tool_name}' returned: {observation}."


def build_planner(backend: str) -> Planner:
    """Construct a planner for the requested backend.

    Only the rule-based planner is wired in by default. Selecting an LLM
    backend silently falls back to it if the optional client / API key are
    missing, so tests stay deterministic without external dependencies.
    """
    normalized = (backend or "rule_based").lower()
    if normalized in {"rule_based", "rules", "offline", "default"}:
        return RuleBasedPlanner()

    if normalized in {"openai", "anthropic"}:  # pragma: no cover - upgrade path
        try:
            from app.services.llm_planners import build_llm_planner

            return build_llm_planner(normalized)
        except Exception:
            return RuleBasedPlanner()

    raise ValueError(f"Unsupported planner backend '{backend}'.")


__all__ = [
    "Planner",
    "PlannerAction",
    "RuleBasedPlanner",
    "ScratchpadEntry",
    "build_planner",
]
