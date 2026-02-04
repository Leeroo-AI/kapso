"""
FastAPI dependencies for rate limiting and other shared functionality.
"""

import time
from collections import defaultdict
from typing import Dict, List, Tuple

from fastapi import HTTPException, Request
from .config import get_settings


class RateLimiter:
    """
    In-memory rate limiter for per-user request tracking.

    Tracks requests by user_id and enforces limits within a time window.
    """

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # user_id -> list of request timestamps
        self._requests: Dict[int, List[float]] = defaultdict(list)

    def _cleanup_old_requests(self, user_id: int, current_time: float) -> None:
        """Remove requests outside the current window."""
        cutoff = current_time - self.window_seconds
        self._requests[user_id] = [
            ts for ts in self._requests[user_id] if ts > cutoff
        ]

    def check_rate_limit(self, user_id: int) -> Tuple[bool, int]:
        """
        Check if user is within rate limit.

        Returns:
            Tuple of (allowed: bool, retry_after_seconds: int)
        """
        current_time = time.time()
        self._cleanup_old_requests(user_id, current_time)

        request_count = len(self._requests[user_id])

        if request_count >= self.max_requests:
            # Calculate retry-after based on oldest request in window
            if self._requests[user_id]:
                oldest = min(self._requests[user_id])
                retry_after = int(oldest + self.window_seconds - current_time)
                return False, max(retry_after, 1)
            return False, self.window_seconds

        return True, 0

    def record_request(self, user_id: int) -> None:
        """Record a request for the user."""
        self._requests[user_id].append(time.time())

    def get_remaining(self, user_id: int) -> int:
        """Get remaining requests for user in current window."""
        current_time = time.time()
        self._cleanup_old_requests(user_id, current_time)
        return max(0, self.max_requests - len(self._requests[user_id]))


# Global rate limiter for export endpoint
_export_rate_limiter: RateLimiter | None = None


def get_export_rate_limiter() -> RateLimiter:
    """Get or create the export rate limiter."""
    global _export_rate_limiter
    if _export_rate_limiter is None:
        settings = get_settings()
        _export_rate_limiter = RateLimiter(
            max_requests=settings.export_rate_limit,
            window_seconds=settings.export_rate_window
        )
    return _export_rate_limiter


async def check_export_rate_limit(request: Request) -> None:
    """
    Dependency to check export rate limit.

    Must be used after get_current_user dependency.
    Raises HTTPException 429 if rate limit exceeded.
    """
    # Get user from request state (set by get_current_user)
    user = getattr(request.state, "user", None)
    if user is None:
        # No user means auth failed elsewhere
        return

    limiter = get_export_rate_limiter()
    allowed, retry_after = limiter.check_rate_limit(user.user_id)

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limit_exceeded",
                "message": f"Export rate limit exceeded. Maximum {limiter.max_requests} requests per hour.",
                "retry_after_seconds": retry_after
            },
            headers={"Retry-After": str(retry_after)}
        )

    # Record this request
    limiter.record_request(user.user_id)
