"""
Wiki export endpoint with rate limiting.
"""

import json
import re
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, Request, Query
from pydantic import BaseModel

from ..auth.api_key import get_current_user, AuthenticatedUser
from ..services.wiki_reader import get_wiki_reader
from ..dependencies import check_export_rate_limit, get_export_rate_limiter
from ..config import get_settings


router = APIRouter()


class ExportPageItem(BaseModel):
    """Page in export response."""

    title: str
    namespace: str
    namespace_id: int
    content: str


class ExportResponse(BaseModel):
    """Response for wiki export."""

    pages: List[ExportPageItem]
    total: int
    rate_limit_remaining: int
    saved_to: Optional[str] = None


def sanitize_filename(filename: str) -> Optional[str]:
    """
    Sanitize a filename to prevent path traversal.

    Returns None if the filename is invalid.
    """
    # Remove any directory components
    filename = Path(filename).name

    # Only allow alphanumeric, underscores, hyphens, dots
    if not re.match(r"^[a-zA-Z0-9_\-\.]+$", filename):
        return None

    # Block traversal patterns
    if ".." in filename:
        return None

    # Ensure it ends with .json
    if not filename.endswith(".json"):
        filename = f"{filename}.json"

    return filename


@router.get("/v1/export", response_model=ExportResponse)
async def export_wiki(
    request: Request,
    user: AuthenticatedUser = Depends(get_current_user),
    output_path: Optional[str] = Query(
        None,
        description="Optional filename to save the export (e.g., 'my_export.json'). File will be saved to the exports directory."
    ),
):
    """
    Export all wiki content as JSON.

    Rate limited to 10 requests per hour per user.
    Returns all pages with their content.

    Optionally saves to a file if output_path is provided.
    """
    # Check rate limit (will raise 429 if exceeded)
    await check_export_rate_limit(request)

    reader = get_wiki_reader()
    pages = reader.export_all()

    # Get remaining rate limit
    limiter = get_export_rate_limiter()
    remaining = limiter.get_remaining(user.user_id)

    # Build response
    export_pages = [
        ExportPageItem(
            title=p["title"],
            namespace=p["namespace"],
            namespace_id=p["namespace_id"],
            content=p["content"]
        )
        for p in pages
    ]

    saved_to = None

    # Save to file if output_path is provided
    if output_path:
        settings = get_settings()
        safe_filename = sanitize_filename(output_path)

        if safe_filename:
            output_dir = Path(settings.export_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            file_path = output_dir / safe_filename

            # Verify path stays within export directory
            try:
                resolved = file_path.resolve()
                dir_resolved = output_dir.resolve()
                if str(resolved).startswith(str(dir_resolved)):
                    # Save the export
                    export_data = {
                        "pages": [p.model_dump() for p in export_pages],
                        "total": len(export_pages),
                        "exported_by": user.username,
                        "user_id": user.user_id
                    }
                    file_path.write_text(
                        json.dumps(export_data, indent=2, ensure_ascii=False),
                        encoding="utf-8"
                    )
                    saved_to = str(file_path)
            except (OSError, ValueError):
                pass  # Silently fail on save errors, still return data

    return ExportResponse(
        pages=export_pages,
        total=len(export_pages),
        rate_limit_remaining=remaining,
        saved_to=saved_to
    )
