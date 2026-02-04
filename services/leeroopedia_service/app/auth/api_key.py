"""
API key authentication for Leeroopedia Content Service.

API keys are stored in user preferences in MediaWiki's database.
Format: lp_<user_id>_<32_hex_chars>
"""

import re
import pymysql
from typing import Optional
from dataclasses import dataclass

from fastapi import HTTPException, Security, Request
from fastapi.security import APIKeyHeader

from ..config import get_settings


@dataclass
class AuthenticatedUser:
    """Represents an authenticated wiki user."""

    user_id: int
    username: str
    company: Optional[str] = None


# API key header definition
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class DatabaseAuthenticator:
    """
    Authenticates API keys against MediaWiki database.

    Queries the user and user_properties tables directly.
    """

    def __init__(self):
        self.settings = get_settings()

    def _get_connection(self):
        """Create a database connection."""
        return pymysql.connect(
            host=self.settings.db_host,
            user=self.settings.db_user,
            password=self.settings.db_password,
            database=self.settings.db_name,
            cursorclass=pymysql.cursors.DictCursor
        )

    def validate_api_key(self, api_key: str) -> Optional[AuthenticatedUser]:
        """
        Validate an API key against the database.

        Args:
            api_key: The API key to validate (format: lp_<user_id>_<hex>)

        Returns:
            AuthenticatedUser if valid, None otherwise
        """
        # Parse API key format: lp_<user_id>_<32_hex_chars>
        match = re.match(r"^lp_(\d+)_([a-f0-9]{32})$", api_key)
        if not match:
            return None

        user_id = int(match.group(1))

        try:
            conn = self._get_connection()
            with conn:
                with conn.cursor() as cursor:
                    # Query user info and API key from user_properties
                    cursor.execute("""
                        SELECT
                            u.user_id,
                            u.user_name,
                            api_key.up_value as api_key,
                            company.up_value as company
                        FROM user u
                        LEFT JOIN user_properties api_key
                            ON u.user_id = api_key.up_user
                            AND api_key.up_property = 'leeroopedia_api_key'
                        LEFT JOIN user_properties company
                            ON u.user_id = company.up_user
                            AND company.up_property = 'companyname'
                        WHERE u.user_id = %s
                    """, (user_id,))

                    row = cursor.fetchone()

                    if not row:
                        return None

                    # Compare API keys (decode bytes if needed)
                    stored_key = row.get("api_key", b"")
                    if isinstance(stored_key, bytes):
                        stored_key = stored_key.decode("utf-8")
                    if stored_key != api_key:
                        return None

                    # Decode username and company
                    username = row["user_name"]
                    if isinstance(username, bytes):
                        username = username.decode("utf-8")

                    company = row.get("company")
                    if isinstance(company, bytes):
                        company = company.decode("utf-8")

                    return AuthenticatedUser(
                        user_id=row["user_id"],
                        username=username,
                        company=company
                    )

        except pymysql.Error:
            return None


# Global authenticator instance
_authenticator: Optional[DatabaseAuthenticator] = None


def get_authenticator() -> DatabaseAuthenticator:
    """Get or create the global authenticator."""
    global _authenticator
    if _authenticator is None:
        _authenticator = DatabaseAuthenticator()
    return _authenticator


async def get_current_user(
    request: Request,
    api_key: Optional[str] = Security(api_key_header)
) -> AuthenticatedUser:
    """
    FastAPI dependency to get the current authenticated user.

    Validates the X-API-Key header against MediaWiki database.
    Raises HTTPException 401 if authentication fails.
    """
    settings = get_settings()

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "missing_api_key",
                "message": "X-API-Key header is required. To get your API key: 1) Go to https://leeroopedia.com/ and sign up, 2) Click on your username (top right), 3) Go to Preferences, 4) Find your Leeroopedia API Key in the Personal info section.",
                "signup_url": "https://leeroopedia.com/"
            }
        )

    authenticator = get_authenticator()
    user = authenticator.validate_api_key(api_key)

    if user is None:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "invalid_api_key",
                "message": "Invalid or expired API key. To get your API key: 1) Go to https://leeroopedia.com/ and sign up, 2) Click on your username (top right), 3) Go to Preferences, 4) Find your Leeroopedia API Key in the Personal info section.",
                "signup_url": "https://leeroopedia.com/"
            }
        )

    # Store user in request state for rate limiting
    request.state.user = user
    return user
