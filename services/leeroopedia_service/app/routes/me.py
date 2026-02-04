"""
User info endpoint.
"""

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..auth.api_key import get_current_user, AuthenticatedUser


router = APIRouter()


class UserResponse(BaseModel):
    """User info response."""

    user_id: int
    username: str
    company: Optional[str]


@router.get("/v1/me", response_model=UserResponse)
async def get_me(user: AuthenticatedUser = Depends(get_current_user)):
    """
    Get current user info.

    Returns the authenticated user's information.
    """
    return UserResponse(
        user_id=user.user_id,
        username=user.username,
        company=user.company
    )
