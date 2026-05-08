"""High-level agent orchestration."""

from __future__ import annotations

from typing import Any

from app.models.schemas import (
    AgentStep,
    ChatResponse,
    ConversationDetail,
    ConversationInfo,
    ConversationListResponse,
    HealthCheckResponse,
    MessageRecord,
    ToolInfo,
    ToolInvocationResponse,
    ToolParameter,
    ToolsResponse,
)
from app.services.conversation_memory import ConversationMemory
from app.services.planner import (
    Planner,
    ScratchpadEntry,
    build_planner,
)
from app.services.tools import Tool, ToolError, ToolRegistry, build_default_registry


class AgentService:
    """Coordinate the planner, tool registry, and conversation memory."""

    def __init__(
        self,
        conversations_file: str,
        sqlite_database: str,
        planner_backend: str,
        max_iterations: int,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1.")
        self.max_iterations = max_iterations
        self.registry: ToolRegistry = build_default_registry(database_path=sqlite_database)
        self.memory = ConversationMemory(storage_path=conversations_file)
        self.planner: Planner = build_planner(planner_backend)

    # ------------------------------------------------------------------
    # Tool surface
    # ------------------------------------------------------------------

    def list_tools(self) -> ToolsResponse:
        infos = [self._tool_to_info(tool) for tool in self.registry.list_tools()]
        return ToolsResponse(count=len(infos), tools=infos)

    def invoke_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolInvocationResponse:
        try:
            result = self.registry.invoke(tool_name, arguments)
        except ToolError as exc:
            return ToolInvocationResponse(
                tool_name=tool_name,
                success=False,
                error=str(exc),
            )
        return ToolInvocationResponse(tool_name=tool_name, success=True, result=result)

    # ------------------------------------------------------------------
    # Chat surface
    # ------------------------------------------------------------------

    def chat(self, message: str, conversation_id: str | None) -> ChatResponse:
        conversation_id = conversation_id or self.memory.create_conversation()
        history = self.memory.get_messages(conversation_id)
        self.memory.append_message(conversation_id, role="user", content=message)

        steps: list[AgentStep] = []
        scratchpad: list[ScratchpadEntry] = []
        finished = False
        final_answer = ""
        tools = self.registry.list_tools()

        for iteration in range(1, self.max_iterations + 1):
            action = self.planner.decide(
                message=message,
                history=history,
                tools=tools,
                scratchpad=scratchpad,
            )

            if action.is_final:
                steps.append(
                    AgentStep(
                        iteration=iteration,
                        thought=action.thought,
                    )
                )
                finished = True
                final_answer = action.final_answer
                break

            tool_name = action.tool_name or ""
            try:
                observation = self.registry.invoke(tool_name, action.arguments)
                scratchpad.append(
                    ScratchpadEntry(
                        tool_name=tool_name,
                        arguments=action.arguments,
                        observation=observation,
                    )
                )
                steps.append(
                    AgentStep(
                        iteration=iteration,
                        thought=action.thought,
                        tool_name=tool_name,
                        tool_arguments=action.arguments,
                        observation=observation,
                    )
                )
            except ToolError as exc:
                scratchpad.append(
                    ScratchpadEntry(
                        tool_name=tool_name,
                        arguments=action.arguments,
                        observation=None,
                        error=str(exc),
                    )
                )
                steps.append(
                    AgentStep(
                        iteration=iteration,
                        thought=action.thought,
                        tool_name=tool_name,
                        tool_arguments=action.arguments,
                        error=str(exc),
                    )
                )

        if not finished:
            final_answer = (
                "I reached the maximum number of reasoning steps before producing a final answer."
            )

        self.memory.append_message(conversation_id, role="assistant", content=final_answer)
        return ChatResponse(
            conversation_id=conversation_id,
            answer=final_answer,
            iterations=len(steps),
            steps=steps,
            finished=finished,
        )

    # ------------------------------------------------------------------
    # Conversation surface
    # ------------------------------------------------------------------

    def list_conversations(self) -> ConversationListResponse:
        items = [ConversationInfo(**entry) for entry in self.memory.list_conversations()]
        return ConversationListResponse(count=len(items), conversations=items)

    def get_conversation(self, conversation_id: str) -> ConversationDetail:
        record = self.memory.get(conversation_id)
        return ConversationDetail(
            conversation_id=record["conversation_id"],
            created_at=record["created_at"],
            updated_at=record["updated_at"],
            message_count=len(record["messages"]),
            messages=[MessageRecord(**message) for message in record["messages"]],
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def get_health(self) -> HealthCheckResponse:
        return HealthCheckResponse(
            status="healthy",
            planner_backend=self.planner.name,
            tool_count=len(self.registry),
            conversation_count=len(self.memory),
            max_iterations=self.max_iterations,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tool_to_info(self, tool: Tool) -> ToolInfo:
        return ToolInfo(
            name=tool.name,
            description=tool.description,
            parameters=[
                ToolParameter(
                    name=parameter["name"],
                    type=parameter["type"],
                    description=parameter["description"],
                    required=bool(parameter.get("required", True)),
                )
                for parameter in tool.parameters
            ],
        )
