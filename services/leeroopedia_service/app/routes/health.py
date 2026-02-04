"""
Health check endpoint.
"""

from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel

from ..config import get_settings
from ..services.wiki_reader import get_wiki_reader


router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    wiki_path_exists: bool
    page_count: int
    timestamp: str


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check service health.

    Returns service status and basic statistics.
    """
    settings = get_settings()
    reader = get_wiki_reader()

    # Check if wiki path exists
    wiki_exists = reader.wiki_path.exists()

    # Count pages (only if path exists)
    page_count = 0
    if wiki_exists:
        try:
            pages = reader.list_pages()
            page_count = len(pages)
        except Exception:
            pass

    return HealthResponse(
        status="healthy",
        wiki_path_exists=wiki_exists,
        page_count=page_count,
        timestamp=datetime.utcnow().isoformat() + "Z"
    )
