# Project 6 — Tutor Walkthrough: AI Agent with Tool Use

> A guided tour of [Project_6](../Project_6/) for a junior AI engineer. This is the project where the AI **stops being a chat completion and starts being an agent** — a system that reasons in a loop, decides which external tools to call, observes their results, and iterates. The pattern you build here (ReAct) underpins ChatGPT plugins, Claude Tools, LangChain agents, OpenAI function calling, and every "AI that does things" you'll meet professionally.

---

## 1. What this project actually does

You expose a `POST /chat` endpoint. The user sends a message. The agent then runs a **reasoning loop**:

```
while not done and iterations < max_iterations:
    action = planner.decide(message, history, tools, scratchpad)
    if action.is_final:
        return action.final_answer
    observation = tool_registry.invoke(action.tool_name, action.arguments)
    scratchpad.append((action, observation))
```

Five built-in tools are available: `calculator`, `get_weather`, `web_search`, `sql_query` (SELECT-only on a seeded SQLite DB), and `current_datetime`. Conversations persist to a JSON file. The default planner is a deterministic **rule-based** one — no LLM API key required — but the abstraction is shaped so you can swap in OpenAI/Anthropic function calling without touching the loop.

Auxiliary endpoints:

- `GET /tools` — catalogue with JSON-schema-style parameter descriptions
- `POST /tools/invoke` — call a tool directly (debugging)
- `GET /conversations` and `GET /conversations/{id}` — transcript history
- `GET /health` — planner backend + tool/conversation counts

---

## 2. Why this project matters — agents are the next abstraction

You've already built systems that *answer* (Project 2 sentiment, Project 3 RAG, Project 5 captioning). Each is a single forward pass: input → model → output. **Agents are different**: the model is no longer the whole system, it's just the *decision-maker* inside a loop. The system can:

- **Take actions in the real world** (run SQL, call APIs, write files) — not just produce text.
- **Recover from errors** mid-task by observing failures and choosing a new path.
- **Decompose problems** ("first look up the user's timezone, then convert this UTC timestamp").
- **Decide when it's done** ("I have enough information; produce the final answer").

This is the first project where you build *control flow over an LLM*, not just *use* an LLM. That mental shift — model as decision oracle, not as answer producer — is the most important thing to internalise.

---

## 3. The ReAct pattern, explained

**ReAct** = **Re**ason + **Act**. It's a 2022 paper that crystallised the loop:

1. **Thought** — the model writes a brief reasoning step ("I need to compute 12 × 11; the calculator tool fits").
2. **Action** — the model picks a tool and arguments.
3. **Observation** — the tool runs; its result is appended to the **scratchpad**.
4. Repeat. Eventually the model emits a "final answer" instead of an action.

The scratchpad ([`list[ScratchpadEntry]`](../Project_6/app/services/planner.py#L30-L37)) is the agent's working memory for *this turn*. It's distinct from the conversation history (the user/assistant transcript across turns). Conflating them is a common bug:

| | Lives in | Survives turn? | Used for |
|---|---|---|---|
| **Scratchpad** | RAM during the loop | No | Prevent repeated tool calls; compose final answer |
| **Conversation history** | `data/conversations.json` | Yes | Long-term context across turns |

The agent loop in [`AgentService.chat`](../Project_6/app/services/agent_service.py#L73-L138) keeps these strictly separate.

---

## 4. Architecture in one picture

```
                  POST /chat  {message, conversation_id?}
                           │
                           ▼
                ┌─────────────────────┐
                │    AgentService     │
                │      .chat()        │
                └─────────────────────┘
                  │       │       │
        ┌─────────┘       │       └────────┐
        ▼                 ▼                ▼
 ┌────────────┐   ┌──────────────┐  ┌──────────────┐
 │  Planner   │   │ ToolRegistry │  │ Conversation │
 │ (rule_based│   │  (5 tools)   │  │   Memory     │
 │  or LLM)   │   │              │  │ (JSON file)  │
 └────────────┘   └──────────────┘  └──────────────┘
        │                 │
        │  decide()       │  invoke()
        │                 │
        └────► loop ◄─────┘
              up to max_iterations
              accumulating in scratchpad
```

Three collaborators behind one service. Same pattern as previous projects, but the **planner is the new thing** — that's where the AI lives.

---

## 5. Concept-by-concept walkthrough

### 5.1 Tools as data, not code

[`Tool`](../Project_6/app/services/tools.py#L23-L48) bundles four things:

```python
Tool(
    name="calculator",
    description="Evaluate a basic arithmetic expression.",
    parameters=[{"name": "expression", "type": "string", "description": "...", "required": True}],
    handler=_calculator_handler,
)
```

The first three fields (name, description, parameters) **describe** the tool in a way that's intentionally JSON-schema-shaped. That's not aesthetic — it's because OpenAI's `function_call` and Anthropic's `tool_use` APIs both expect this exact shape. When you upgrade to a real LLM planner, translating our tool list into the provider's schema is a 10-line function. **The description fields are read by the LLM at runtime**, so write them as if instructing a colleague: clear, complete, with example inputs.

The fourth field, `handler`, is the Python callable. The planner never sees it. The registry calls it after the planner decides which tool to use. This separation — *description for the model, handler for the runtime* — is the core idea behind every function-calling system.

[`ToolRegistry`](../Project_6/app/services/tools.py#L51-L77) is a dict + dispatcher with two methods worth noting:

- `invoke(name, arguments)` — looks up the tool and calls `Tool.execute(arguments)`, which **first validates** required arguments before calling the handler. This catches "model forgot to pass a required arg" at the registry boundary, returning a structured `ToolError` that the agent can show to the planner on the next iteration.
- The registry validates **structurally** (required keys present), not semantically (calculator expression is well-formed). Semantic validation lives inside each handler. That layering — registry catches bad calls, handler catches bad inputs — is worth keeping when you build your own.

### 5.2 The five built-in tools — and the safety lessons in each

Each tool teaches a security/design lesson worth more than the tool itself.

**`calculator`** ([handler](../Project_6/app/services/tools.py#L86-L104)). Uses Python's `eval`. **`eval` is a known footgun** — by default it gives the input full access to your runtime. Two defences are layered here:

```python
_ALLOWED_CALC_CHARS = re.compile(r"^[0-9+\-*/().,\s%]+$")
...
value = eval(expression, {"__builtins__": {}}, {})
```

The regex rejects anything that isn't a digit, operator, or whitespace — so `__import__` literally cannot appear. Then `eval` is called with `__builtins__` blanked out, so even if the regex were relaxed, the input couldn't reach `open`, `exec`, or anything dangerous. **Defence in depth.** Either layer alone might be enough; both together are robust.

> **Production-grade lesson:** if a junior engineer ever ships `eval(user_input)` without these guards, that's a CVE. Memorise the pattern.

**`get_weather`** ([handler](../Project_6/app/services/tools.py#L107-L143)). A deterministic mock. Real production would call OpenWeatherMap. The interesting bit is the fallback for unknown cities:

```python
condition, temp_c = presets.get(city.lower(), ("partly cloudy", 17 + (hash(city.lower()) % 10)))
```

A **deterministic** fake (same city → same forecast) so tests pass on every run. When you swap in the real API, you keep this exact contract — `dict[str, Any]` with `city`, `condition`, `temperature`, `unit`. The planner code doesn't change.

**`web_search`** ([handler](../Project_6/app/services/tools.py#L146-L191)). A toy in-memory keyword index over four hardcoded documents. Useless for production, perfect for showing the *shape* — query in, ranked results out. Real swaps: Tavily, Brave Search, Bing Web Search, or an internal Elasticsearch.

**`sql_query`** ([handler](../Project_6/app/services/tools.py#L194-L222)). The most subtly dangerous tool, with three layered protections:

```python
if not query.lower().startswith("select"):
    raise ToolError("Only SELECT statements are allowed.")
if ";" in query:
    raise ToolError("Multiple statements are not allowed.")
```

1. **SELECT-only** — agent cannot `DROP`, `UPDATE`, `INSERT`, or `DELETE`.
2. **No semicolons** — prevents stacked-statement injection (`SELECT 1; DROP TABLE products;`).
3. **No prepared statements** — wait, that's a *missing* defence. The agent's query goes through unparameterised. SQLite won't honour string-interpolated queries from external data, but a real database tool should accept parameters separately, never as part of the query string.

**The deeper lesson:** *every* tool that gives an agent access to the outside world is a privilege boundary. Every guard rail you add is a control. "Allow LLM to write arbitrary SQL against prod" is a classic incident waiting to happen. Read-only, table-restricted, schema-pinned, time-limited queries are the production answer.

**`current_datetime`** ([handler](../Project_6/app/services/tools.py#L240-L249)). Trivial, but exists because LLMs have a knowledge cutoff and famously hallucinate dates. Always give an agent a `now()` tool.

### 5.3 The `Planner` Protocol — where the AI lives

[`Planner`](../Project_6/app/services/planner.py#L48-L60) is a `Protocol`:

```python
class Planner(Protocol):
    name: str
    def decide(self, message, history, tools, scratchpad) -> PlannerAction: ...
```

A `PlannerAction` is either a "use this tool" or a "we're done, here's the final answer":

```python
@dataclass
class PlannerAction:
    thought: str
    is_final: bool = False
    final_answer: str = ""
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
```

This dataclass is **the universal interchange format** between the loop and any planner. Whether the planner is a regex-based heuristic or GPT-4o, it returns the same shape. That's why the upgrade path to a real LLM is purely additive — implement a new class that returns `PlannerAction`s.

### 5.4 The rule-based planner: structural fidelity, zero intelligence

[`RuleBasedPlanner`](../Project_6/app/services/planner.py#L63-L208) is keyword-driven dispatch:

- "calculate", "compute", "what is", or a short message containing an arithmetic regex match → `calculator`
- contains "weather" + a proper-noun city → `get_weather`
- "product"/"customer"/"stock"/"inventory" → `sql_query` with a hand-built SELECT
- "date"/"time"/"today"/"now" → `current_datetime`
- "search"/"find"/"explain"/"tell me about" + nothing else has worked → `web_search`
- else → final answer composed by [`_compose_final_answer`](../Project_6/app/services/planner.py#L186-L196)

It's not smart. It's not meant to be. **Its job is to exercise the loop.** The two things it does well are worth borrowing for any real planner:

1. **Termination guarantee.** Each tool name is added to `used_tools` after first use; the planner refuses to call the same tool twice. Combined with the `max_iterations` cap, the loop *cannot* run forever.
2. **Final-answer synthesis** ([`_compose_final_answer`](../Project_6/app/services/planner.py#L186-L208)). When the planner has nothing more to do, it stitches the scratchpad observations into a human-readable answer. Every entry has a per-tool summary (`_summarize`) — calculator → "the result of X is Y", weather → "Weather in X: condition, temp", etc. **The final answer is grounded in the observations**, not synthesised by an LLM. Even when you upgrade the planner to an LLM, keep this layer for the offline path.

> **Junior-engineer trap:** you'll see `RuleBasedPlanner` produce nonsense answers for "what's the meaning of life?" and conclude the project doesn't work. That's correct behaviour for a heuristic without a real LLM. The point is the *machinery*; quality of reasoning is the planner's job.

### 5.5 The agent loop, line by line

[`AgentService.chat`](../Project_6/app/services/agent_service.py#L73-L138):

```python
conversation_id = conversation_id or self.memory.create_conversation()
history = self.memory.get_messages(conversation_id)
self.memory.append_message(conversation_id, role="user", content=message)
```

History is captured **before** the new user message is appended. Why? Because the planner's `decide` takes `history` as the *prior context*, and the new message is passed separately. Conflating them would cause the planner to treat the question as a previous-turn artefact — subtle but real.

```python
for iteration in range(1, self.max_iterations + 1):
    action = self.planner.decide(message=message, history=history, tools=tools, scratchpad=scratchpad)
    if action.is_final:
        steps.append(AgentStep(iteration=iteration, thought=action.thought))
        finished = True
        final_answer = action.final_answer
        break
```

The loop is bounded by `max_iterations` (default 6). If the planner asks for too many tool calls, the loop exits with a graceful "I reached the max" message. **This bound is non-negotiable.** Real LLM planners can hallucinate infinite tool sequences ("now I'll search again, now I'll search again"). Without the cap, your API will hang — or burn money.

```python
try:
    observation = self.registry.invoke(tool_name, action.arguments)
    scratchpad.append(ScratchpadEntry(...observation=observation))
    steps.append(AgentStep(...observation=observation))
except ToolError as exc:
    scratchpad.append(ScratchpadEntry(...error=str(exc)))
    steps.append(AgentStep(...error=str(exc)))
```

**Tool failures don't crash the loop.** They become observations the planner can react to. This is critical for robustness — a real LLM agent often recovers from bad arguments by reading the error and trying again.

```python
self.memory.append_message(conversation_id, role="assistant", content=final_answer)
return ChatResponse(
    conversation_id=conversation_id,
    answer=final_answer,
    iterations=len(steps),
    steps=steps,
    finished=finished,
)
```

The full step trace is returned in the response. **This is observability built in.** A user sees not just "the answer is 132" but the entire reasoning path: thoughts, tools called, arguments passed, observations received. For agents this is non-optional. When (not if) the agent does something weird, the trace is your only debugging surface.

### 5.6 Conversation memory: persistent across turns

[`ConversationMemory`](../Project_6/app/services/conversation_memory.py) is the same JSON-as-DB pattern from Project 4 (NER registry) and Project 5 (image metadata):

- Single JSON file, rewritten on each mutation, under a `Lock`.
- Loaded on startup; survives restart.
- Read methods (`get_messages`, `get`, `list_conversations`) return *copies*, not live references — so callers can't mutate persisted state by accident.

The `conversation_id` is generated server-side on first message if the client doesn't supply one. The first response carries it back so the client can keep the same conversation across subsequent calls.

> **.NET parallel:** Equivalent to a `ConversationsRepository` over EF Core / SQLite. For real production, swap to Postgres or Redis — same interface, different backend.

### 5.7 The factory + fallback pattern (third time this project)

[`build_planner`](../Project_6/app/services/planner.py#L211-L228):

```python
if normalized in {"rule_based", "rules", "offline", "default"}:
    return RuleBasedPlanner()
if normalized in {"openai", "anthropic"}:
    try:
        from app.services.llm_planners import build_llm_planner
        return build_llm_planner(normalized)
    except Exception:
        return RuleBasedPlanner()
```

Same shape as Projects 3 (cache backends), 4 (model registry), 5 (vision providers). **Consistency matters.** When every "swappable backend" follows the same factory + try/import + fallback structure, junior engineers spending five minutes in one file know how to navigate every other one.

### 5.8 Direct tool invocation: the debug back door

`POST /tools/invoke` lets you call a tool **without** running the planner. Why expose this?

- Smoke-test a new tool's contract independently of agent reasoning.
- Reproduce a tool failure from a chat trace — copy the exact arguments, replay them, see the error.
- Build a "manual mode" UI where humans drive the tools instead of the agent.

This kind of bypass endpoint is invaluable. Any time you build a system with an opinionated control loop, expose a low-level interface alongside it. Diagnostic ergonomics matter.

### 5.9 Async + threads + JSON-on-disk

Same pattern as previous projects:

- `await asyncio.to_thread(service.chat, ...)` — the agent loop is synchronous and may run multiple tool calls; we don't want to block the event loop.
- All state mutations (conversations, scratchpad, registry) happen on the worker thread.
- The conversation file is rewritten under a `Lock`, which is held briefly per operation.

**Failure mode under multiple workers:** if you run `uvicorn --workers 4`, each worker has its own in-memory `ConversationMemory` cache and its own `Lock`. Two simultaneous chat calls hitting different workers can interleave file writes and lose data. The fix is the same as everywhere else: persist state to a real DB / Redis. The single-file JSON is correct only for a single process.

---

## 6. Error mapping summary

| Where | Trigger | Status |
|---|---|---|
| `chat` | `KeyError` (conversation_id not found) | 404 |
| `get_conversation` | `KeyError` | 404 |
| `tools.invoke_tool` | Unknown tool name | 404 |
| `tools.invoke_tool` | Tool error (bad arguments, runtime failure) | 200 with `success=false`, `error="..."` |
| Anywhere | Unhandled exception | 500 |

Note the deliberate choice for `invoke_tool`: tool *failures* return 200 with a structured error payload. That's because tool errors are **expected** in agent flows (the planner is supposed to read them and retry); they are not server errors. Distinguishing transport errors (404, 500) from domain errors (200 with `success=false`) is a real-world API design skill.

---

## 7. Self-quiz

1. What is the ReAct pattern? Name its four steps and what role each plays.
2. What is the difference between the **scratchpad** and the **conversation history**? Why must they be kept separate?
3. Why is the calculator tool defended with **both** a regex *and* `__builtins__={}`? Either alone might suffice — what attack does the second layer stop?
4. Why does `sql_query` reject semicolons even though SQLite would accept them?
5. Why does the agent loop catch `ToolError` and continue instead of failing the request?
6. The `RuleBasedPlanner` refuses to call the same tool twice in one turn. What problem does that prevent?
7. Why is `max_iterations` non-optional, and what happens if a real LLM planner asks for an 11th tool call?
8. Why does `POST /chat` return the full `steps[]` trace, not just the answer?
9. What is the upgrade path from `RuleBasedPlanner` to OpenAI function calling, in concrete code-change terms? What stays the same?
10. Why does `POST /tools/invoke` return 200 with `success=false` for tool errors, instead of 400 or 500?

---

## 8. Hands-on next steps

- **Implement an OpenAI function-calling planner.** Translate the existing tool list into the [function schema format](https://platform.openai.com/docs/guides/function-calling), call `client.chat.completions.create(..., tools=...)`, parse `tool_calls` from the response into `PlannerAction`. Loop wraps unchanged.
- **Add a `python_repl` tool.** Make it sandboxed (no `__builtins__`, restricted modules). Watch how often the agent prefers it over the calculator.
- **Add tool-call telemetry.** Count invocations per tool, average iterations per chat. Expose at `GET /metrics`. Real agents need observability into "which tools is the model leaning on?".
- **Add per-tool rate limits.** `web_search` should be rate-limited at the tool layer, not the API layer — the agent might make 5 calls in one chat. Token-bucket per tool.
- **Add a `confirm_destructive_action` tool.** For tools that mutate state (eventually, if you add `sql_update`), require an explicit user confirmation in a follow-up turn. This is the standard pattern for safe agent autonomy.
- **Plug in OpenTelemetry tracing.** One span per iteration, child spans for `planner.decide` and `registry.invoke`. Now you can see `chat` traces in Jaeger or LangFuse with full timing breakdowns.

---

## 9. .NET parallels

| Concept here | .NET equivalent |
|---|---|
| `Tool` + `ToolRegistry` | `ICommandHandler` + a Mediator-style dispatcher |
| `Planner` Protocol | `IAgentPlanner` interface |
| `build_planner` factory | DI registration with `IConfiguration`-based switch |
| `ScratchpadEntry` | A per-request `AgentContext` object |
| `ConversationMemory` | EF Core repository over SQLite/Postgres |
| `AgentService.chat` loop | An application service / command handler |
| `Depends(get_agent_service)` | Constructor-injected singleton |
| `lifespan` startup | `Program.cs` startup configuration |
| `ChatResponse.steps[]` | Returning a domain log alongside the result |

---

## 10. File cheat-sheet

| File | Purpose | Key idea |
|---|---|---|
| [app/config.py](../Project_6/app/config.py) | Settings | `planner_backend`, `max_agent_iterations`, conversation/SQLite paths |
| [app/main.py](../Project_6/app/main.py) | App wiring | Lifespan + 3 routers |
| [app/dependencies.py](../Project_6/app/dependencies.py) | DI | One singleton `AgentService` |
| [app/models/schemas.py](../Project_6/app/models/schemas.py) | Pydantic | `ChatRequest/Response`, `AgentStep`, `ToolInfo`, `ConversationDetail` |
| [app/services/tools.py](../Project_6/app/services/tools.py) | Tool registry | `Tool`, `ToolRegistry`, 5 built-in handlers, sandboxing |
| [app/services/planner.py](../Project_6/app/services/planner.py) | Planner abstraction | Protocol, rule-based default, `PlannerAction`, factory |
| [app/services/conversation_memory.py](../Project_6/app/services/conversation_memory.py) | Persistence | JSON file + `Lock`, copies on read |
| [app/services/agent_service.py](../Project_6/app/services/agent_service.py) | Orchestration | The ReAct loop, error handling, step recording |
| [app/routers/chat.py](../Project_6/app/routers/chat.py) | Chat + history | `POST /chat`, `GET /conversations*` |
| [app/routers/tools.py](../Project_6/app/routers/tools.py) | Tools | `GET /tools`, `POST /tools/invoke` |
| [app/routers/health.py](../Project_6/app/routers/health.py) | Health | Reports planner backend, tool count, conversation count |

---

## 11. The single most important takeaway

> **An agent is a control loop, not a model.**
>
> The model decides "what to do next" given the current state. The runtime owns the loop, the tool registry, the scratchpad, the iteration cap, the error recovery, the observability, and — most importantly — the **boundaries** that make the agent safe to deploy.
>
> Every tool you give an agent is a privilege. Every tool needs validation, sandboxing, audit, and a kill-switch. The most dangerous mistake in agent engineering is to assume the model "knows what it's doing" and remove the guard rails. It doesn't. It guesses. Your job is to make the guesses safe.
>
> Build the loop correctly once. Then any LLM — today's GPT, tomorrow's open-source 70B, the next decade's whatever — slots in as a planner. The architecture is the asset. The model is the commodity.
