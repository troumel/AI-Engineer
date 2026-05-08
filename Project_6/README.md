# Project 6 — AI Agent with Tool Use (Function Calling)

A FastAPI service that exposes an AI agent capable of multi-step reasoning
over a registry of tools (calculator, weather lookup, web search, SQL
queries, current datetime). Implements a **ReAct**-style loop:

```
while not done and iterations < max_iterations:
    action = planner.decide(message, history, tools, scratchpad)
    if action.is_final: break
    observation = registry.invoke(action.tool, action.arguments)
    scratchpad.append((action, observation))
```

## Highlights

- **Tool registry** with JSON-schema-style parameter descriptions
- **Five built-in tools**: `calculator`, `get_weather`, `web_search`,
  `sql_query` (SELECT-only on a seeded SQLite DB), `current_datetime`
- **Pluggable planner** abstraction with a deterministic offline
  `RuleBasedPlanner` default; OpenAI / Anthropic function-calling slots are
  reserved
- **Persistent conversation memory** stored as JSON
- **Direct tool invocation** endpoint for debugging
- **Iteration cap** prevents runaway agent loops
- **Sandboxed calculator** (regex-restricted, no builtins) and
  **read-only SQL** (rejects non-SELECT statements and multi-statement input)

## Project structure

```
Project_6/
├── app/
│   ├── config.py
│   ├── dependencies.py
│   ├── main.py
│   ├── models/schemas.py
│   ├── routers/{health,tools,chat}.py
│   └── services/
│       ├── agent_service.py        # orchestration
│       ├── conversation_memory.py  # JSON-backed history
│       ├── planner.py              # rule-based planner + abstraction
│       └── tools.py                # tool registry + built-in tools
├── data/                           # runtime conversation + sqlite store
├── docker/{Dockerfile,docker-compose.yml}
├── requirements.txt
└── tests/
    ├── test_api_endpoints.py
    └── test_agent_service.py
```

## Running locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8005
```

Open http://localhost:8005/docs for the interactive Swagger UI.

## Running tests

```powershell
pytest tests -q
```

## Endpoints

| Method | Path                              | Description                              |
| ------ | --------------------------------- | ---------------------------------------- |
| GET    | `/`                               | API metadata                             |
| GET    | `/health`                         | Service + planner status                 |
| GET    | `/tools`                          | List registered tools                    |
| POST   | `/tools/invoke`                   | Invoke a tool directly (debug)           |
| POST   | `/chat`                           | Send a message; agent runs ReAct loop    |
| GET    | `/conversations`                  | List stored conversations                |
| GET    | `/conversations/{conversation_id}`| Full transcript for a conversation       |

## Example chat

```bash
curl -X POST http://localhost:8005/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is 12 * 11?"}'
```

The response includes the user-facing `answer`, the `iterations` count, and a
`steps[]` array recording every thought, tool invocation, and observation.

## Upgrading to OpenAI / Anthropic function calling

1. Uncomment `openai` or `anthropic` in `requirements.txt`.
2. Set `PLANNER_BACKEND=openai` (or `anthropic`) and provide the API key.
3. Implement `app/services/llm_planners.py` with a `build_llm_planner(name)`
   factory returning an object with a `decide(...)` method that translates
   the existing tool list into the provider's function-calling schema and
   parses the model's response into a `PlannerAction`. The factory in
   `planner.build_planner` already routes to it and silently falls back to
   the rule-based planner if the import fails.

## Notes for production

- Replace the JSON conversation store with PostgreSQL or Redis.
- Add authentication and per-user rate limiting at the API gateway layer.
- Add tracing via LangSmith / LangFuse / OpenTelemetry.
- Replace the in-memory document index with a real search backend.
