"""Main FastAPI entry point for the basic RAG system."""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.dependencies import initialize_services
from app.routers import documents, health, qa


logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared services during startup and log shutdown."""
    logger.info("Starting up application")
    logger.info("Environment: %s", settings.environment)
    logger.info("API: %s v%s", settings.app_name, settings.app_version)
    initialize_services()
    yield
    logger.info("Shutting down application")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Document Q&A API that indexes uploaded files, retrieves relevant chunks, "
        "and generates grounded answers with source citations."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log each request with status and latency."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %s in %.2fms",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


app.include_router(health.router)
app.include_router(documents.router)
app.include_router(qa.router)


@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    """Return basic API metadata and entry points."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "upload": "/documents/upload",
        "ask": "/qa/ask",
    }