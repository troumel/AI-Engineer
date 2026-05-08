"""Tool registry: defines callable tools the agent can use.

Each tool exposes a JSON-schema-style description (name, description,
parameters) plus a Python callable. This mirrors how OpenAI / Anthropic
function-calling APIs describe tools, so swapping the local rule-based
planner for a real LLM later only requires translating these descriptions
into the provider's tool schema format.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


class ToolError(Exception):
    """Raised when a tool fails during execution."""


class Tool:
    """Wraps a callable with metadata describing its parameters."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: list[dict[str, Any]],
        handler: Callable[[dict[str, Any]], Any],
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler

    def execute(self, arguments: dict[str, Any]) -> Any:
        self._validate(arguments)
        return self.handler(arguments)

    def _validate(self, arguments: dict[str, Any]) -> None:
        for parameter in self.parameters:
            if parameter.get("required", True) and parameter["name"] not in arguments:
                raise ToolError(
                    f"Tool '{self.name}' missing required argument '{parameter['name']}'."
                )


class ToolRegistry:
    """Holds registered tools and dispatches invocations."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered.")
        self._tools[tool.name] = tool

    def list_tools(self) -> list[Tool]:
        return sorted(self._tools.values(), key=lambda t: t.name)

    def get(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"Tool '{name}' is not registered.")
        return tool

    def invoke(self, name: str, arguments: dict[str, Any]) -> Any:
        return self.get(name).execute(arguments)

    def __len__(self) -> int:
        return len(self._tools)


# ---------------------------------------------------------------------------
# Built-in tool implementations
# ---------------------------------------------------------------------------


_ALLOWED_CALC_CHARS = re.compile(r"^[0-9+\-*/().,\s%]+$")


def _calculator_handler(arguments: dict[str, Any]) -> dict[str, Any]:
    expression = str(arguments.get("expression", "")).strip()
    if not expression:
        raise ToolError("Expression must be a non-empty string.")
    if not _ALLOWED_CALC_CHARS.match(expression):
        raise ToolError(
            "Expression contains unsupported characters. "
            "Only digits, whitespace, and the operators + - * / ( ) % are allowed."
        )
    try:
        # Evaluated against an empty namespace with no builtins to avoid
        # arbitrary code execution; the regex above further restricts the input.
        value = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
    except ZeroDivisionError as exc:
        raise ToolError(f"Math error: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ToolError(f"Could not evaluate expression: {exc}") from exc
    return {"expression": expression, "value": value}


def _build_weather_handler() -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a deterministic mock weather handler.

    Real deployments would call OpenWeatherMap / WeatherAPI / etc. The mock
    derives a stable forecast from the city name so tests are deterministic
    and offline.
    """

    presets = {
        "london": ("cloudy", 14),
        "paris": ("sunny", 22),
        "berlin": ("rainy", 11),
        "new york": ("partly cloudy", 18),
        "tokyo": ("clear", 25),
        "sydney": ("windy", 20),
    }

    def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        city = str(arguments.get("city", "")).strip()
        if not city:
            raise ToolError("City must be provided.")
        unit = str(arguments.get("unit", "celsius")).lower()
        if unit not in {"celsius", "fahrenheit"}:
            raise ToolError("Unit must be 'celsius' or 'fahrenheit'.")

        condition, temp_c = presets.get(
            city.lower(),
            ("partly cloudy", 17 + (hash(city.lower()) % 10)),
        )
        if unit == "fahrenheit":
            temperature = round(temp_c * 9 / 5 + 32, 1)
        else:
            temperature = float(temp_c)
        return {
            "city": city,
            "condition": condition,
            "temperature": temperature,
            "unit": unit,
        }

    return handler


def _build_search_handler() -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Tiny in-memory web-search style index."""

    documents = [
        {
            "title": "FastAPI Tutorial",
            "url": "https://example.com/fastapi-tutorial",
            "snippet": "FastAPI is a modern, fast web framework for building APIs with Python.",
        },
        {
            "title": "Function Calling with LLMs",
            "url": "https://example.com/function-calling",
            "snippet": "Function calling lets large language models invoke tools to retrieve "
            "structured data and perform actions.",
        },
        {
            "title": "ReAct: Reasoning and Acting",
            "url": "https://example.com/react",
            "snippet": "The ReAct pattern interleaves reasoning steps with tool actions to "
            "ground language model outputs.",
        },
        {
            "title": "Vector Databases Overview",
            "url": "https://example.com/vector-databases",
            "snippet": "Vector databases store embeddings for similarity search and retrieval-"
            "augmented generation.",
        },
    ]

    def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query", "")).strip()
        if not query:
            raise ToolError("Query must be provided.")
        max_results = int(arguments.get("max_results", 3))
        max_results = max(1, min(max_results, 10))

        tokens = {token.lower() for token in re.findall(r"[a-zA-Z0-9]+", query)}
        scored: list[tuple[int, dict[str, Any]]] = []
        for document in documents:
            text = f"{document['title']} {document['snippet']}".lower()
            score = sum(1 for token in tokens if token in text)
            if score > 0:
                scored.append((score, document))
        scored.sort(key=lambda item: item[0], reverse=True)
        results = [document for _, document in scored[:max_results]]
        return {"query": query, "count": len(results), "results": results}

    return handler


def _build_sql_handler(database_path: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """SQLite query tool restricted to SELECT statements on a seed database."""

    db_path = Path(database_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _seed_database(db_path)

    def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query", "")).strip().rstrip(";")
        if not query:
            raise ToolError("SQL query must be provided.")
        if not query.lower().startswith("select"):
            raise ToolError("Only SELECT statements are allowed.")
        if ";" in query:
            raise ToolError("Multiple statements are not allowed.")

        with sqlite3.connect(db_path) as connection:
            connection.row_factory = sqlite3.Row
            try:
                cursor = connection.execute(query)
            except sqlite3.Error as exc:
                raise ToolError(f"SQL error: {exc}") from exc
            rows = [dict(row) for row in cursor.fetchall()]
        return {"query": query, "row_count": len(rows), "rows": rows}

    return handler


def _seed_database(db_path: Path) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                price REAL NOT NULL,
                stock INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                country TEXT NOT NULL
            );
            """
        )
        cursor = connection.execute("SELECT COUNT(*) FROM products")
        if cursor.fetchone()[0] == 0:
            connection.executemany(
                "INSERT INTO products(id, name, category, price, stock) VALUES(?,?,?,?,?)",
                [
                    (1, "Mechanical Keyboard", "electronics", 129.99, 42),
                    (2, "Wireless Mouse", "electronics", 39.5, 128),
                    (3, "Standing Desk", "furniture", 449.0, 15),
                    (4, "Office Chair", "furniture", 299.0, 22),
                    (5, "Notebook", "stationery", 4.5, 250),
                ],
            )
            connection.executemany(
                "INSERT INTO customers(id, name, country) VALUES(?,?,?)",
                [
                    (1, "Alice", "UK"),
                    (2, "Bob", "US"),
                    (3, "Carla", "DE"),
                ],
            )


def _datetime_handler(arguments: dict[str, Any]) -> dict[str, Any]:
    timezone_name = str(arguments.get("timezone", "UTC")).upper()
    if timezone_name != "UTC":
        raise ToolError("Only the 'UTC' timezone is supported in offline mode.")
    now = datetime.now(timezone.utc)
    return {
        "timezone": "UTC",
        "iso": now.isoformat(),
        "epoch_seconds": int(now.timestamp()),
    }


def build_default_registry(database_path: str) -> ToolRegistry:
    """Register the default tool set used by the agent."""
    registry = ToolRegistry()

    registry.register(
        Tool(
            name="calculator",
            description="Evaluate a basic arithmetic expression.",
            parameters=[
                {
                    "name": "expression",
                    "type": "string",
                    "description": "Arithmetic expression using digits and the operators + - * / ( ) %.",
                    "required": True,
                }
            ],
            handler=_calculator_handler,
        )
    )
    registry.register(
        Tool(
            name="get_weather",
            description="Return mock current weather for a city.",
            parameters=[
                {
                    "name": "city",
                    "type": "string",
                    "description": "City name, e.g. 'London'.",
                    "required": True,
                },
                {
                    "name": "unit",
                    "type": "string",
                    "description": "Temperature unit: 'celsius' or 'fahrenheit'.",
                    "required": False,
                },
            ],
            handler=_build_weather_handler(),
        )
    )
    registry.register(
        Tool(
            name="web_search",
            description="Search a small in-memory document index by keyword.",
            parameters=[
                {
                    "name": "query",
                    "type": "string",
                    "description": "Free-text search query.",
                    "required": True,
                },
                {
                    "name": "max_results",
                    "type": "integer",
                    "description": "Maximum number of results to return (1-10).",
                    "required": False,
                },
            ],
            handler=_build_search_handler(),
        )
    )
    registry.register(
        Tool(
            name="sql_query",
            description="Run a SELECT query against the seed SQLite database (tables: products, customers).",
            parameters=[
                {
                    "name": "query",
                    "type": "string",
                    "description": "A single SELECT statement (no semicolons).",
                    "required": True,
                }
            ],
            handler=_build_sql_handler(database_path),
        )
    )
    registry.register(
        Tool(
            name="current_datetime",
            description="Return the current UTC date and time.",
            parameters=[
                {
                    "name": "timezone",
                    "type": "string",
                    "description": "Timezone name. Only 'UTC' is supported in offline mode.",
                    "required": False,
                }
            ],
            handler=_datetime_handler,
        )
    )

    return registry


__all__ = [
    "Tool",
    "ToolError",
    "ToolRegistry",
    "build_default_registry",
]
