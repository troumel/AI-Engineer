"""Tool listing and direct invocation endpoints."""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_agent_service
from app.models.schemas import (
    ToolInvocationRequest,
    ToolInvocationResponse,
    ToolsResponse,
)
from app.services.agent_service import AgentService


router = APIRouter(prefix="/tools", tags=["Tools"])


@router.get("", response_model=ToolsResponse, summary="List registered tools")
async def list_tools(
    service: AgentService = Depends(get_agent_service),
) -> ToolsResponse:
    """Return the catalogue of tools available to the agent."""
    return await asyncio.to_thread(service.list_tools)


@router.post(
    "/invoke",
    response_model=ToolInvocationResponse,
    summary="Invoke a tool directly (debugging)",
)
async def invoke_tool(
    request: ToolInvocationRequest,
    service: AgentService = Depends(get_agent_service),
) -> ToolInvocationResponse:
    """Invoke a tool by name. Useful for debugging without running the planner."""
    response = await asyncio.to_thread(
        service.invoke_tool,
        tool_name=request.tool_name,
        arguments=request.arguments,
    )
    if not response.success and response.error and "not registered" in response.error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=response.error)
    return response
