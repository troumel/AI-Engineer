"""Usage and Prometheus-style metrics routes."""

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from app.dependencies import get_llm_service
from app.models.schemas import UsageReport
from app.services.llm_service import LLMService

router = APIRouter(tags=["observability"])


@router.get("/usage", response_model=UsageReport)
def usage(
    llm_service: LLMService = Depends(get_llm_service),
) -> UsageReport:
    return llm_service.usage_report()


@router.get("/metrics", response_class=PlainTextResponse)
def metrics(
    llm_service: LLMService = Depends(get_llm_service),
) -> str:
    return llm_service.metrics_text()
