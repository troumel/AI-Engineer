"""
Main FastAPI application.

This is the entry point for the API - similar to Program.cs in .NET.
It:
1. Creates the FastAPI app
2. Configures middleware (CORS, logging)
3. Registers routers (controllers)
4. Sets up startup/shutdown events
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.dependencies import initialize_services
from app.routers import health, predictions


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.

    This is called when the app starts and stops.
    Similar to Program.cs in .NET where you configure services before app.Run()

    Startup:
        - Initialize services
        - Load ML model

    Shutdown:
        - Clean up resources (if needed)
    """
    # Startup
    logger.info("Starting up application...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"API: {settings.app_name} v{settings.app_version}")

    # Initialize services (load ML model)
    initialize_services()

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down application...")


# Create FastAPI application instance
# This is like WebApplication.CreateBuilder() in .NET
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="API for predicting house prices using a trained ML model",
    lifespan=lifespan  # Startup/shutdown events
)


# Configure CORS (Cross-Origin Resource Sharing)
# Allows browsers to call this API from different domains
# Similar to app.UseCors() in .NET
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


# Register routers (like app.MapControllers() in .NET)
# Each router handles a specific set of endpoints
app.include_router(health.router)
app.include_router(predictions.router)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - returns basic API information.

    Useful for:
    - Checking if API is accessible
    - Getting API version info
    """
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }
