"""
Leeroopedia Content Service

FastAPI service that provides authenticated access to wiki content.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import get_settings
from .routes import health, me, pages, export


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    settings = get_settings()
    print(f"Leeroopedia Content Service starting...")
    print(f"  Wiki data path: {settings.wiki_data_path}")
    print(f"  Database: {settings.db_user}@{settings.db_host}/{settings.db_name}")
    print(f"  Export rate limit: {settings.export_rate_limit}/hour")

    yield

    # Shutdown
    print("Leeroopedia Content Service shutting down...")


app = FastAPI(
    title="Leeroopedia Content API",
    description="API for accessing Leeroopedia wiki content with authentication",
    version="1.0.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(me.router, tags=["User"])
app.include_router(pages.router, tags=["Pages"])
app.include_router(export.router, tags=["Export"])


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "Leeroopedia Content API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
