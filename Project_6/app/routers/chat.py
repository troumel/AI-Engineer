"""Chat and conversation endpoints."""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_agent_service
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationDetail,
    ConversationListResponse,
)
from app.services.agent_service import AgentService


router = APIRouter(tags=["Chat"])


@router.post("/chat", response_model=ChatResponse, summary="Talk to the agent")
async def chat(
    request: ChatRequest,
    service: AgentService = Depends(get_agent_service),
) -> ChatResponse:
    """Send a message; the agent reasons, calls tools, and returns an answer."""
    try:
        return await asyncio.to_thread(
            service.chat,
            message=request.message,
            conversation_id=request.conversation_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.get(
    "/conversations",
    response_model=ConversationListResponse,
    summary="List stored conversations",
)
async def list_conversations(
    service: AgentService = Depends(get_agent_service),
) -> ConversationListResponse:
    """List all stored conversations."""
    return await asyncio.to_thread(service.list_conversations)


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetail,
    summary="Get a conversation transcript",
)
async def get_conversation(
    conversation_id: str,
    service: AgentService = Depends(get_agent_service),
) -> ConversationDetail:
    """Return the full message history for a conversation."""
    try:
        return await asyncio.to_thread(service.get_conversation, conversation_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
