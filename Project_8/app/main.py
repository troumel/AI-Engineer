"""FastAPI application entry point for Project 8."""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.dependencies import initialize_services, shutdown_services
from app.routers import chat, completions, health, models, observability

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("custom_llm_api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Initializing LLMService (backend=%s)...", settings.llm_backend)
    await initialize_services()
    logger.info("LLM inference API ready on port %s.", settings.api_port)
    try:
        yield
    finally:
        logger.info("Shutting down LLM inference API.")
        await shutdown_services()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
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
async def access_log(request: Request, call_next):
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
app.include_router(models.router)
app.include_router(chat.router)
app.include_router(completions.router)
app.include_router(observability.router)
