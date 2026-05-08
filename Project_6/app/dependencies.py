"""Dependency registration and singleton initialization for Project 6."""

from typing import Optional

from app.config import settings
from app.services.agent_service import AgentService


_agent_service: Optional[AgentService] = None


def initialize_services() -> None:
    """Initialize the singleton agent service once at startup."""
    global _agent_service

    if _agent_service is not None:
        return

    _agent_service = AgentService(
        conversations_file=settings.conversations_file,
        sqlite_database=settings.sqlite_database,
        planner_backend=settings.planner_backend,
        max_iterations=settings.max_agent_iterations,
    )


def get_agent_service() -> AgentService:
    """Return the singleton agent service instance."""
    if _agent_service is None:
        raise RuntimeError(
            "AgentService not initialized. Call initialize_services() at app startup."
        )
    return _agent_service
