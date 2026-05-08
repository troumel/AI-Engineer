"""Pydantic schemas for the AI agent API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """JSON-schema-style description of a single tool parameter."""

    name: str
    type: str
    description: str
    required: bool = True


class ToolInfo(BaseModel):
    """Public description of a registered tool."""

    name: str
    description: str
    parameters: list[ToolParameter]


class ToolsResponse(BaseModel):
    """List of available tools."""

    count: int
    tools: list[ToolInfo]


class ToolInvocationRequest(BaseModel):
    """Direct invocation of a tool (bypasses the agent loop)."""

    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolInvocationResponse(BaseModel):
    """Result of a direct tool invocation."""

    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None


class AgentStep(BaseModel):
    """One step recorded during an agent run."""

    iteration: int
    thought: str
    tool_name: Optional[str] = None
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    observation: Optional[Any] = None
    error: Optional[str] = None


class ChatRequest(BaseModel):
    """Request payload for the chat endpoint."""

    message: str = Field(min_length=1, max_length=4000)
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response payload returned by the chat endpoint."""

    conversation_id: str
    answer: str
    iterations: int
    steps: list[AgentStep]
    finished: bool


class MessageRecord(BaseModel):
    """A single conversation message."""

    role: str
    content: str
    created_at: str


class ConversationInfo(BaseModel):
    """Conversation metadata."""

    conversation_id: str
    created_at: str
    updated_at: str
    message_count: int


class ConversationDetail(ConversationInfo):
    """Full conversation including all messages."""

    messages: list[MessageRecord]


class ConversationListResponse(BaseModel):
    """List of stored conversations."""

    count: int
    conversations: list[ConversationInfo]


class HealthCheckResponse(BaseModel):
    """Health check payload."""

    status: str
    planner_backend: str
    tool_count: int
    conversation_count: int
    max_iterations: int
